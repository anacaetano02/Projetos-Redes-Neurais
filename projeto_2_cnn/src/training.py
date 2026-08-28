"""
Módulo src/training.py
Orquestra o loop de treinamento PyTorch (CNN/LSTM) — mixed precision,
gradient clipping real, checkpoint retomável entre reinícios de runtime,
early stopping, TensorBoard — e a persistência de experimentos.
"""
import json
import os
import random
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data import CLASSES
from src.utils import log_etapa, log_nota

_experimentos = []

def fixar_seeds(seed: int = 42, priorizar_velocidade: bool = True) -> None:
    """
    Fixa a inicialização de pesos, ordem de shuffle do DataLoader e
    dropout mask — as fontes de aleatoriedade que de fato causam
    variância grande entre execuções.

    priorizar_velocidade=True (padrão): deixa cudnn.benchmark ligado —
    a escolha do algoritmo de convolução mais rápido pela GPU ainda pode
    variar entre execuções (diferenças numéricas mínimas, irrelevantes
    para comparar hiperparâmetros), mas evita o custo de performance
    (2-3x mais lento) do modo determinístico total.

    priorizar_velocidade=False: reprodutibilidade bit-a-bit exata, ao
    custo de velocidade — só vale a pena para comparar execuções byte a
    byte, não para o dia a dia de ablação de hiperparâmetro.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if priorizar_velocidade:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Semente {seed} configurada (priorizar_velocidade={priorizar_velocidade}).")

def computar_pesos_classe(df_train: pl.DataFrame) -> torch.Tensor:
    """
    Peso por classe = total_amostras / (num_classes * contagem_da_classe).
    Classes raras recebem peso > 1 (penalizadas mais forte quando
    erradas), classes dominantes recebem peso < 1. A ordem do tensor
    retornado segue CLASSES (data.py), não a ordem do group_by — isso
    importa para o CrossEntropyLoss bater com os índices usados no Dataset.
    """
    contagens = df_train.group_by("dx").agg(pl.len().alias("n")).to_dict(as_series=False)
    contagem_por_classe = dict(zip(contagens["dx"], contagens["n"]))

    total = df_train.height
    num_classes = len(CLASSES)

    pesos = [
        total / (num_classes * contagem_por_classe.get(classe, 1))
        for classe in CLASSES
    ]
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

    log_etapa(
        "computar_pesos_classe",
        "Pesos por classe: " + ", ".join(f"{c}={p:.2f}" for c, p in zip(CLASSES, pesos)),
    )

    return pesos_tensor

def treinar_modelo(
    modelo: nn.Module,
    dataloaders: dict,
    pesos_classe: torch.Tensor,
    nome_experimento: str,
    checkpoint_dir: str,
    dir_runs: str,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    paciencia_early_stopping: int = 7,
    device: str = None,
    forcar_do_zero: bool = False,
    clip_grad_norm_max: float = 5.0,
) -> dict:
    """
    Treina 'modelo' com early stopping por val_loss. Salva o melhor
    checkpoint (menor val_loss) em checkpoint_dir e loga métricas por
    época no TensorBoard (dir_runs/nome_experimento).

    Retoma automaticamente: se existir um estado salvo de uma sessão
    anterior deste mesmo experimento, carrega e continua de onde parou
    em vez de treinar do zero — protege contra desconexão do Colab no
    meio do treino. forcar_do_zero=True ignora esse estado.

    Retorna um dict com o histórico (train_loss, val_loss, grad_norm,
    batches_overflow por época) — pronto para plot_curvas_loss/
    plot_gradient_norm e para registrar_experimento.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)

    criterio = nn.CrossEntropyLoss(weight=pesos_classe.to(device))
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", patience=3, factor=0.5)

    # Mixed precision (AMP): acelera bastante em GPU sem sacrificar a
    # reprodutibilidade controlada por fixar_seeds() — são preocupações
    # independentes (precisão numérica vs. fonte de aleatoriedade).
    usa_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device) if usa_amp else None

    checkpoint_path = os.path.join(checkpoint_dir, f"{nome_experimento}.pt")
    checkpoint_estado_path = os.path.join(checkpoint_dir, f"{nome_experimento}_estado.pt")
    writer = SummaryWriter(log_dir=os.path.join(dir_runs, nome_experimento))

    historico = {"train_loss": [], "val_loss": [], "grad_norm": [], "batches_overflow": []}
    melhor_val_loss = float("inf")
    epocas_sem_melhora = 0
    epoca_inicial = 1

    if os.path.exists(checkpoint_estado_path) and not forcar_do_zero:
        estado = torch.load(checkpoint_estado_path, map_location=device)
        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        scheduler.load_state_dict(estado["scheduler"])
        historico = estado["historico"]
        melhor_val_loss = estado["melhor_val_loss"]
        epocas_sem_melhora = estado["epocas_sem_melhora"]
        epoca_inicial = estado["epoca"] + 1
        log_nota(
            f"Retomando '{nome_experimento}' da época {epoca_inicial} "
            f"(estado salvo encontrado em '{checkpoint_estado_path}')."
        )

    log_etapa(
        "treinar_modelo",
        f"Treino de '{nome_experimento}' em {device} (AMP {'ativo' if usa_amp else 'inativo'}), "
        f"até {epochs} épocas (retomando da época {epoca_inicial}).",
    )

    try:
        for epoca in range(epoca_inicial, epochs + 1):
            t0 = time.time()

            # ---- treino ----
            modelo.train()
            loss_acumulada = 0.0
            grad_norms_epoca = []
            n_batches_overflow = 0

            for imgs, rotulos in dataloaders["train"]:
                imgs = imgs.to(device, non_blocking=True)
                rotulos = rotulos.to(device, non_blocking=True)

                otimizador.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=usa_amp):
                    saida = modelo(imgs)
                    loss = criterio(saida, rotulos)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(otimizador)  # necessário ANTES de medir a norma real do gradiente
                else:
                    loss.backward()

                # max_norm real (não mais 'inf') quando clip_grad_norm_max é
                # passado: limita a norma ANTES do overflow acontecer, em
                # vez de só medir depois.
                max_norm = clip_grad_norm_max if clip_grad_norm_max is not None else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=max_norm)
                grad_norm_valor = grad_norm.item()

                if not torch.isfinite(grad_norm).item():
                    n_batches_overflow += 1
                else:
                    grad_norms_epoca.append(grad_norm_valor)

                if scaler:
                    scaler.step(otimizador)
                    scaler.update()
                else:
                    otimizador.step()

                loss_acumulada += loss.item() * imgs.size(0)

            train_loss = loss_acumulada / len(dataloaders["train"].dataset)
            # Média só sobre batches finitos — um único overflow não deveria
            # contaminar o diagnóstico da época inteira; a contagem de
            # overflow vira sinal próprio, reportado à parte.
            grad_norm_medio = (
                sum(grad_norms_epoca) / len(grad_norms_epoca) if grad_norms_epoca else float("nan")
            )

            # ---- validação ----
            modelo.eval()
            loss_val_acumulada = 0.0
            with torch.no_grad():
                for imgs, rotulos in dataloaders["val"]:
                    imgs = imgs.to(device, non_blocking=True)
                    rotulos = rotulos.to(device, non_blocking=True)

                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=usa_amp):
                        saida = modelo(imgs)
                        loss = criterio(saida, rotulos)

                    loss_val_acumulada += loss.item() * imgs.size(0)

            val_loss = loss_val_acumulada / len(dataloaders["val"].dataset)

            scheduler.step(val_loss)

            historico["train_loss"].append(train_loss)
            historico["val_loss"].append(val_loss)
            historico["grad_norm"].append(grad_norm_medio)
            historico["batches_overflow"].append(n_batches_overflow)

            writer.add_scalar("Loss/train", train_loss, epoca)
            writer.add_scalar("Loss/val", val_loss, epoca)
            writer.add_scalar("GradNorm/train", grad_norm_medio, epoca)
            writer.add_scalar("LR", otimizador.param_groups[0]["lr"], epoca)

            dt = time.time() - t0
            overflow_str = f" overflow={n_batches_overflow}batches" if n_batches_overflow else ""
            print(
                f"[{nome_experimento}] época {epoca}/{epochs} — "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"grad_norm={grad_norm_medio:.4f}{overflow_str} ({dt:.1f}s)"
            )

            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                epocas_sem_melhora = 0
                torch.save(modelo.state_dict(), checkpoint_path)
            else:
                epocas_sem_melhora += 1

            # Salva o estado completo A CADA ÉPOCA (não só quando melhora) —
            # é o que permite retomar depois de uma desconexão sem perder
            # o progresso das épocas já treinadas.
            torch.save(
                {
                    "modelo": modelo.state_dict(),
                    "otimizador": otimizador.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "historico": historico,
                    "melhor_val_loss": melhor_val_loss,
                    "epocas_sem_melhora": epocas_sem_melhora,
                    "epoca": epoca,
                },
                checkpoint_estado_path,
            )

            if epocas_sem_melhora >= paciencia_early_stopping:
                log_nota(
                    f"Early stopping em '{nome_experimento}' na época {epoca} "
                    f"(sem melhora por {paciencia_early_stopping} épocas). "
                    f"Melhor val_loss={melhor_val_loss:.4f}."
                )
                break
    finally:
        writer.close()

    # treino concluído (ou interrompido por early stopping) — o estado
    # retomável não faz mais sentido; só o melhor checkpoint importa daqui pra frente.
    if os.path.exists(checkpoint_estado_path):
        os.remove(checkpoint_estado_path)

    log_etapa(
        "treinar_modelo",
        f"Treino de '{nome_experimento}' concluído. Melhor val_loss={melhor_val_loss:.4f}, "
        f"checkpoint salvo em '{checkpoint_path}'.",
    )

    return historico

def carregar_melhor_modelo(modelo: nn.Module, checkpoint_path: str, device: str = None) -> nn.Module:
    """Recarrega o melhor checkpoint salvo por treinar_modelo (state_dict puro)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)
    modelo.load_state_dict(state_dict)
    modelo.to(device)
    modelo.eval()
    print(f"Modelo recarregado com sucesso de '{checkpoint_path}'.")
    return modelo

# ---------------------------------------------------------------------------
# Registro de experimentos
# ---------------------------------------------------------------------------

_CHAVES_METRICA_NAO_TABULAVEIS = {"relatorio_por_classe", "matriz_confusao"}

def registrar_experimento(nome: str, tipo: str, hiperparametros: dict, metricas: dict, notas: str = "") -> None:
    """
    Registra o experimento na tabela global. tipo: 'baseline' | 'cnn' |
    'lstm_patches' (ou 'busca_<tipo>' para tentativas comparadas só por
    validação, que ficam fora da tabela de resultados finais).
    """
    _experimentos.append({
        "data": str(np.datetime64("now")),
        "nome": nome,
        "tipo": tipo,
        "hiperparametros": hiperparametros,
        "metricas": metricas,
        "notas": notas,
    })

def limpar_experimentos() -> None:
    """Zera a lista de experimentos em memória — útil ao re-rodar o notebook do zero."""
    _experimentos.clear()

def _tabela_metricas(apenas_busca: bool | None) -> pl.DataFrame:
    registros = _experimentos
    if apenas_busca is True:
        registros = [r for r in registros if r["tipo"].startswith("busca_")]
    elif apenas_busca is False:
        registros = [r for r in registros if not r["tipo"].startswith("busca_")]

    linhas = []
    for r in registros:
        linha = {"nome": r["nome"], "tipo": r["tipo"]}
        linha.update({
            f"metrica_{k}": v for k, v in r["metricas"].items()
            if k not in _CHAVES_METRICA_NAO_TABULAVEIS
        })
        linha["notas"] = r["notas"]
        linhas.append(linha)
    return pl.DataFrame(linhas) if linhas else pl.DataFrame()

def _tabela_hiperparametros(apenas_busca: bool | None) -> pl.DataFrame:
    registros = _experimentos
    if apenas_busca is True:
        registros = [r for r in registros if r["tipo"].startswith("busca_")]
    elif apenas_busca is False:
        registros = [r for r in registros if not r["tipo"].startswith("busca_")]

    linhas = []
    for r in registros:
        linha = {"nome": r["nome"], "tipo": r["tipo"]}
        linha.update({f"hp_{k}": str(v) for k, v in r["hiperparametros"].items()})
        linhas.append(linha)
    return pl.DataFrame(linhas) if linhas else pl.DataFrame()

def _tabela_para_markdown(df: pl.DataFrame) -> str:
    """Converte um pl.DataFrame numa tabela markdown (GitHub-flavored)."""
    if df.is_empty():
        return "_(nenhum registro)_\n"

    colunas = df.columns
    linhas = ["| " + " | ".join(colunas) + " |", "|" + "|".join(["---"] * len(colunas)) + "|"]
    for row in df.iter_rows(named=True):
        valores = []
        for c in colunas:
            v = row[c]
            if v is None:
                valores.append("")
            elif isinstance(v, float):
                valores.append(f"{v:.4f}")
            else:
                valores.append(str(v))
        linhas.append("| " + " | ".join(valores) + " |")
    return "\n".join(linhas) + "\n"

def exportar_experimentos(
    path_json: str = "outputs/report_assets/experimentos.json",
    path_md: str = "outputs/report_assets/log_experimentos.md",
) -> None:
    """
    Exporta o histórico: JSON (dado bruto — default=str cobre matriz de
    confusão/numpy que não são serializáveis nativamente) + markdown
    (tabela legível, pronta para o relatório). matriz_confusao/
    relatorio_por_classe ficam de fora da tabela markdown (são grandes
    demais para uma célula) — inspecione via log_etapa ou plotar_matriz_confusao.
    """
    diretorio_json = os.path.dirname(path_json)
    if diretorio_json:
        os.makedirs(diretorio_json, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(_experimentos, f, indent=2, ensure_ascii=False, default=str)

    linhas_md = ["# Log de Experimentos — Projeto 2 (CNN, HAM10000)\n"]

    linhas_md.append("## Resultados finais (teste)\n")
    linhas_md.append("### Métricas\n")
    linhas_md.append(_tabela_para_markdown(_tabela_metricas(apenas_busca=False)))
    linhas_md.append("")
    linhas_md.append("### Hiperparâmetros\n")
    linhas_md.append(_tabela_para_markdown(_tabela_hiperparametros(apenas_busca=False)))
    linhas_md.append("")

    linhas_md.append("## Testes de busca de hiperparâmetros (validação)\n")
    linhas_md.append("### Métricas\n")
    linhas_md.append(_tabela_para_markdown(_tabela_metricas(apenas_busca=True)))
    linhas_md.append("")
    linhas_md.append("### Hiperparâmetros\n")
    linhas_md.append(_tabela_para_markdown(_tabela_hiperparametros(apenas_busca=True)))
    linhas_md.append("")

    diretorio_md = os.path.dirname(path_md)
    if diretorio_md:
        os.makedirs(diretorio_md, exist_ok=True)
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_md))

    print(f"Experimentos exportados: '{path_json}' e '{path_md}' ({len(_experimentos)} registros)")
