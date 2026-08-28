"""
Módulo src/training.py
Orquestra o loop de treinamento PyTorch, Early Stopping, decaimento de peso adaptativo
com AdamW, ReduceLROnPlateau, logs do TensorBoard e persistência de experimentos.
"""
import os
import random
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.tensorboard import SummaryWriter
from src.utils import (
    log_etapa,
    log_nota,
    plot_curvas_loss,
    plot_gradient_norm,
    capturar_distribuicao_ativacoes,
    avaliar_classificacao,
    avaliar_regressao,
)
from src.models import MLP

_experimentos = []

def fixar_seeds(seed: int = 42) -> None:
    """Garante o determinismo científico desativando o autotuning do cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Semente {seed} configurada de forma estável e determinística!")

def calcular_gradient_norm(modelo: nn.Module) -> float:
    """Calcula a norma L2 acumulativa dos tensores ativos .grad."""
    total_norm = 0.0
    for p in modelo.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def treinar_epoca(modelo, loader, loss_fn, otimizador, device: str, clip_grad_norm_max: float | None = 5.0) -> tuple[float, float]:
    modelo.train()
    loss_total = 0.0
    grad_norm_total = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        otimizador.zero_grad()
        out = modelo(X_batch)
        loss = loss_fn(out, y_batch)
        loss.backward()

        # Captura grad_norm na janela correta (pós-backward, pré-step). Com
        # clip_grad_norm_max definido, clip_grad_norm_ também limita a
        # norma ANTES do step — não só mede o gradiente explosivo que
        # plot_gradient_norm já alerta, evita de fato o passo de otimização
        # correspondente.
        if clip_grad_norm_max is not None:
            gn = torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=clip_grad_norm_max).item()
        else:
            gn = calcular_gradient_norm(modelo)
        grad_norm_total += gn

        otimizador.step()
        loss_total += loss.item()
        n_batches += 1

    assert n_batches > 0, (
        "loader não produziu nenhum batch — com drop_last=True (padrão em "
        "preparar_dataloaders), isso acontece se o conjunto de treino tiver "
        "menos linhas que batch_size. Reduza batch_size ou confirme o "
        "tamanho do conjunto de treino."
    )
    return loss_total / n_batches, grad_norm_total / n_batches

@torch.no_grad()
def avaliar(modelo, loader, loss_fn, device: str) -> float:
    modelo.eval()
    loss_total = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = modelo(X_batch)
        loss = loss_fn(out, y_batch)
        loss_total += loss.item()
        n_batches += 1
    return loss_total / n_batches

def treinar_modelo(modelo, loader_treino, loader_val, loss_fn, otimizador, epocas: int, device: str = "cpu", checkpoint_path: str = "melhor_modelo.pt", scheduler=None, nome_run: str = "run", paciencia_early_stopping: int = 5, dir_runs: str = "runs", clip_grad_norm_max: float | None = 5.0) -> dict:
    """Executa o loop principal aplicando Early Stopping e checkpointing via state_dict."""
    diretorio_checkpoint = os.path.dirname(checkpoint_path)
    if diretorio_checkpoint:
        os.makedirs(diretorio_checkpoint, exist_ok=True)

    writer = SummaryWriter(os.path.join(dir_runs, nome_run))
    best_val_loss = float("inf")
    counter_es = 0

    historico = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}

    try:
        for epoch in range(1, epocas + 1):
            trn_loss, avg_gn = treinar_epoca(modelo, loader_treino, loss_fn, otimizador, device, clip_grad_norm_max=clip_grad_norm_max)
            val_loss = avaliar(modelo, loader_val, loss_fn, device)

            current_lr = otimizador.param_groups[0]["lr"]

            historico["train_loss"].append(trn_loss)
            historico["val_loss"].append(val_loss)
            historico["grad_norm"].append(avg_gn)
            historico["lr"].append(current_lr)

            writer.add_scalar("Loss/Treino", trn_loss, epoch)
            writer.add_scalar("Loss/Validação", val_loss, epoch)
            writer.add_scalar("GradientNorm", avg_gn, epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

            print(f"Época {epoch:02d} | Train Loss: {trn_loss:.4f} | Val Loss: {val_loss:.4f} | GN: {avg_gn:.3f} | LR: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter_es = 0
                # Salvando checkpoint físico estável — "config" torna o
                # checkpoint autodescritivo (arquitetura recuperável sem
                # precisar lembrar os hiperparâmetros usados na chamada
                # que o gerou), quando o modelo expõe .resumo() (MLP expõe).
                torch.save({
                    "epoca": epoch,
                    "state_dict": modelo.state_dict(),
                    "val_loss": val_loss,
                    "config": modelo.resumo() if hasattr(modelo, "resumo") else None,
                }, checkpoint_path)
            else:
                counter_es += 1
                if counter_es >= paciencia_early_stopping:
                    print(f"Early Stopping disparado na época {epoch}!")
                    break

            if scheduler is not None:
                # Sincronismo de paciências: scheduler enxerga o platô antes do Early Stopping
                scheduler.step(val_loss)
    finally:
        writer.close()

    return historico

def carregar_melhor_modelo(modelo, checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.to(device)
    print(f"Modelo recarregado com sucesso! Época: {checkpoint['epoca']} | Val Loss: {checkpoint['val_loss']:.4f}")
    return modelo

def mostrar_historico(historico: dict, nome_modelo: str = "") -> pl.DataFrame:
    """Exibe o histórico de treino (train_loss, val_loss, grad_norm, lr por época) como tabela — mais fácil de revisar/exportar do que rolar pelos prints época a época."""
    n_epocas = len(historico["train_loss"])
    tabela = pl.DataFrame({
        "epoca": list(range(1, n_epocas + 1)),
        "train_loss": historico["train_loss"],
        "val_loss": historico["val_loss"],
        "grad_norm": historico["grad_norm"],
        "lr": historico["lr"],
    })

    titulo = f"Histórico de treino - {nome_modelo}" if nome_modelo else "Histórico de treino"
    log_etapa(titulo, tabela)
    return tabela

def registrar_experimento(nome: str, tipo: str, hiperparametros: dict, metricas: dict, notas: str = ""):
    """Registra o experimento na tabela global."""
    _experimentos.append({
        "data": str(np.datetime64("now")),
        "nome": nome,
        "tipo": tipo,
        "hiperparametros": hiperparametros,
        "metricas": metricas,
        "notas": notas
    })

def limpar_experimentos() -> None:
    """Zera a lista de experimentos em memória — útil ao re-rodar o notebook do zero."""
    _experimentos.clear()

def tabela_experimentos(tipo_filtro: str | None = None) -> pl.DataFrame:
    """Monta uma tabela comparável a partir de TODOS os experimentos registrados (hiperparâmetros e métricas achatados em colunas hp_/metrica_), sem separar por família classificação/regressão."""
    registros = _experimentos
    if tipo_filtro:
        registros = [r for r in registros if r["tipo"] == tipo_filtro]

    linhas = []
    for r in registros:
        linha = {"nome": r["nome"], "tipo": r["tipo"], "data": r["data"]}
        linha.update({f"hp_{k}": str(v) for k, v in r["hiperparametros"].items()})
        linha.update({f"metrica_{k}": v for k, v in r["metricas"].items()})
        linha["notas"] = r["notas"]
        linhas.append(linha)

    df = pl.DataFrame(linhas) if linhas else pl.DataFrame()
    titulo = "Tabela de experimentos" + (f" - {tipo_filtro}" if tipo_filtro else "")
    log_etapa(titulo, df)
    return df

def comparar_com_baseline(tipo_problema: str, metrica_chave: str) -> pl.DataFrame:
    """
    Compara todos os experimentos de um tipo (ex.: "classificacao") contra
    o baseline correspondente (tipo="baseline_<tipo_problema>"), calculando
    a diferença (delta) na métrica-chave informada. Se houver mais de um
    baseline registrado, usa o mais recente.
    """
    baseline = [r for r in _experimentos if r["tipo"] == f"baseline_{tipo_problema}"]
    modelos = [r for r in _experimentos if r["tipo"] == tipo_problema]

    if not baseline:
        log_nota(
            f"Nenhum baseline registrado para '{tipo_problema}' ainda — "
            f"registre com registrar_experimento(tipo='baseline_{tipo_problema}', ...)."
        )
        return pl.DataFrame()

    valor_baseline = baseline[-1]["metricas"].get(metrica_chave)

    linhas = []
    for r in modelos:
        valor_modelo = r["metricas"].get(metrica_chave)
        delta = (valor_modelo - valor_baseline) if (valor_modelo is not None and valor_baseline is not None) else None
        linhas.append({
            "nome": r["nome"],
            metrica_chave: valor_modelo,
            f"baseline_{metrica_chave}": valor_baseline,
            "delta": delta,
        })

    df = pl.DataFrame(linhas) if linhas else pl.DataFrame()
    log_etapa(f"Comparação com baseline - {tipo_problema} ({metrica_chave})", df)
    return df

def _tabela_metricas(familia: str, apenas_busca: bool | None) -> pl.DataFrame:
    registros = [r for r in _experimentos if familia in r["tipo"]]
    if apenas_busca is True:
        registros = [r for r in registros if r["tipo"].startswith("busca_")]
    elif apenas_busca is False:
        registros = [r for r in registros if not r["tipo"].startswith("busca_")]

    linhas = []
    for r in registros:
        linha = {"nome": r["nome"], "tipo": r["tipo"]}
        linha.update({f"metrica_{k}": v for k, v in r["metricas"].items()})
        linha["notas"] = r["notas"]
        linhas.append(linha)
    return pl.DataFrame(linhas) if linhas else pl.DataFrame()

def _tabela_hiperparametros(familia: str, apenas_busca: bool | None) -> pl.DataFrame:
    registros = [r for r in _experimentos if familia in r["tipo"]]
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
    """Exporta o histórico: JSON (dado bruto) + markdown (tabela legível, pronta para o relatório)."""
    import json
    diretorio_json = os.path.dirname(path_json)
    if diretorio_json:
        os.makedirs(diretorio_json, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(_experimentos, f, indent=2, ensure_ascii=False)

    linhas_md = ["# Log de Experimentos\n"]
    for familia, titulo in [("classificacao", "Classificação"), ("regressao", "Regressão")]:
        linhas_md.append(f"## {titulo}\n")

        linhas_md.append("### Resultados finais (teste)\n")
        linhas_md.append("#### Métricas\n")
        linhas_md.append(_tabela_para_markdown(_tabela_metricas(familia, apenas_busca=False)))
        linhas_md.append("")
        linhas_md.append("#### Hiperparâmetros\n")
        linhas_md.append(_tabela_para_markdown(_tabela_hiperparametros(familia, apenas_busca=False)))
        linhas_md.append("")

        linhas_md.append("### Testes de busca de hiperparâmetros (validação)\n")
        linhas_md.append("#### Métricas\n")
        linhas_md.append(_tabela_para_markdown(_tabela_metricas(familia, apenas_busca=True)))
        linhas_md.append("")
        linhas_md.append("#### Hiperparâmetros\n")
        linhas_md.append(_tabela_para_markdown(_tabela_hiperparametros(familia, apenas_busca=True)))
        linhas_md.append("")

    diretorio_md = os.path.dirname(path_md)
    if diretorio_md:
        os.makedirs(diretorio_md, exist_ok=True)
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_md))

    print(f"Experimentos exportados: '{path_json}' e '{path_md}' ({len(_experimentos)} registros)")

def treinar_variacao(
    tipo_problema: str,
    input_size: int,
    dados: dict,
    device: str,
    nome: str,
    camadas_ocultas: list[int],
    ativacao: type[nn.Module] = nn.ReLU,
    dropout: float = 0.2,
    usar_batchnorm: bool = True,
    inicializacao: str | None = "he",
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    pos_weight=None,
    epocas: int = 30,
    paciencia_early_stopping: int = 5,
    scheduler_patience: int = 2,
    checkpoint_dir: str = "outputs/checkpoints",
    dir_runs: str = "runs",
    dir_figuras: str = "outputs/report_assets",
    clip_grad_norm_max: float | None = 5.0,
) -> dict:
    """
    Treina UMA configuração e a compara pela loss de VALIDAÇÃO — nunca
    avalia no conjunto de teste. Registra em registrar_experimento com
    tipo="busca_<tipo_problema>" (não "classificacao"/"regressao" puro),
    para não misturar essas medições de busca com a avaliação final
    (ver avaliar_modelo_final), que sozinha entra na comparação com o
    baseline.

    scheduler_patience deve ser MENOR que paciencia_early_stopping, com
    folga suficiente para o novo LR ter algumas épocas de efeito antes do
    treino parar (ex.: scheduler_patience=2, paciencia_early_stopping=5).
    """
    modelo = MLP(
        input_size=input_size,
        camadas_ocultas=camadas_ocultas,
        ativacao=ativacao,
        dropout=dropout,
        usar_batchnorm=usar_batchnorm,
        inicializacao=inicializacao,
    ).to(device)

    if tipo_problema == "classificacao":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loaders = dados["classificacao"]
    elif tipo_problema == "regressao":
        loss_fn = nn.MSELoss()
        loaders = dados["regressao"]
    else:
        raise ValueError(f"tipo_problema deve ser 'classificacao' ou 'regressao', recebido: {tipo_problema}")

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=scheduler_patience)

    checkpoint_path = os.path.join(checkpoint_dir, f"{nome}.pt")
    historico = treinar_modelo(
        modelo=modelo,
        loader_treino=loaders["train"],
        loader_val=loaders["val"],
        loss_fn=loss_fn,
        otimizador=otimizador,
        epocas=epocas,
        device=device,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        nome_run=nome,
        paciencia_early_stopping=paciencia_early_stopping,
        dir_runs=dir_runs,
        clip_grad_norm_max=clip_grad_norm_max,
    )
    modelo = carregar_melhor_modelo(modelo, checkpoint_path, device)

    plot_curvas_loss(historico, nome, salvar_dir=dir_figuras)
    plot_gradient_norm(historico, nome, salvar_dir=dir_figuras)
    capturar_distribuicao_ativacoes(modelo, loaders["val"], device, nome, salvar_dir=dir_figuras)

    melhor_val_loss = min(historico["val_loss"])
    registrar_experimento(
        nome=nome,
        tipo=f"busca_{tipo_problema}",
        hiperparametros={
            **modelo.resumo(),
            "lr": lr,
            "weight_decay": weight_decay,
        },
        metricas={"melhor_val_loss": melhor_val_loss, "val_loss_final": historico["val_loss"][-1]},
        notas="Comparação por validação — teste ainda não avaliado.",
    )

    return {
        "modelo": modelo,
        "historico": historico,
        "checkpoint_path": checkpoint_path,
        "melhor_val_loss": melhor_val_loss,
    }

def buscar_melhor_configuracao(
    tipo_problema: str,
    input_size: int,
    dados: dict,
    device: str,
    grade_configuracoes: list[dict],
    pos_weight=None,
    epocas: int = 30,
    prefixo_nome: str = "busca",
    checkpoint_dir: str = "outputs/checkpoints",
    dir_runs: str = "runs",
    dir_figuras: str = "outputs/report_assets",
    clip_grad_norm_max: float | None = 5.0,
) -> dict:
    """
    Treina uma lista de configurações em sequência (treinar_variacao) e
    retorna a de menor melhor_val_loss. Nenhuma chamada toca o conjunto
    de teste — comparação inteira por validação, mesma disciplina de
    treinar_variacao.

    grade_configuracoes: lista de dicts com os kwargs aceitos por
    treinar_variacao (camadas_ocultas, ativacao, dropout, usar_batchnorm,
    inicializacao, lr...). Chave 'nome' é opcional; se ausente, gera
    "{prefixo_nome}_{i}". checkpoint_dir/dir_runs/dir_figuras/
    clip_grad_norm_max valem para todas as configurações da grade
    (repassados a cada treinar_variacao).
    """
    resultados = []
    for i, config_original in enumerate(grade_configuracoes):
        config = dict(config_original)
        nome = config.pop("nome", f"{prefixo_nome}_{i}")

        log_nota(f"Busca [{i + 1}/{len(grade_configuracoes)}]: '{nome}' - {config}")
        resultado = treinar_variacao(
            tipo_problema=tipo_problema,
            input_size=input_size,
            dados=dados,
            device=device,
            nome=nome,
            pos_weight=pos_weight,
            clip_grad_norm_max=clip_grad_norm_max,
            epocas=epocas,
            checkpoint_dir=checkpoint_dir,
            dir_runs=dir_runs,
            dir_figuras=dir_figuras,
            **config,
        )
        resultados.append({"nome": nome, "config": config, **resultado})

    melhor = min(resultados, key=lambda r: r["melhor_val_loss"])
    log_nota(
        f"Busca concluída - {len(grade_configuracoes)} configurações testadas. "
        f"Vencedora: '{melhor['nome']}' (melhor_val_loss={melhor['melhor_val_loss']:.4f})."
    )

    return {"resultados": resultados, "melhor": melhor}

def avaliar_modelo_final(
    tipo_problema: str,
    modelo,
    dados: dict,
    device: str,
    nome: str,
    hiperparametros: dict,
    notas: str = "",
    dir_figuras: str = "outputs/report_assets",
) -> dict:
    """
    Chamar UMA VEZ, só na configuração vencedora (menor melhor_val_loss
    entre as variações testadas com treinar_variacao/buscar_melhor_configuracao).
    É aqui, e só aqui, que o conjunto de TESTE é avaliado — a escolha da
    vencedora não foi influenciada pelo teste, preservando a validade da
    comparação. Registra com tipo="<tipo_problema>" (sem prefixo
    "busca_"), entrando na comparação final com o baseline.
    """
    if tipo_problema == "classificacao":
        metricas = avaliar_classificacao(modelo, dados["classificacao"]["test"], device, nome, salvar_dir=dir_figuras)
    elif tipo_problema == "regressao":
        metricas = avaliar_regressao(modelo, dados["regressao"]["test"], device, nome)
    else:
        raise ValueError(f"tipo_problema deve ser 'classificacao' ou 'regressao', recebido: {tipo_problema}")

    registrar_experimento(
        nome=nome,
        tipo=tipo_problema,
        hiperparametros=hiperparametros,
        metricas=metricas,
        notas=notas or "Configuração vencedora, selecionada por loss de validação. Avaliação final no teste.",
    )

    return metricas
