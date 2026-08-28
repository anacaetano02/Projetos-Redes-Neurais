"""
Módulo src/utils.py
Consolida as ferramentas de logging de etapas, EDA visual de imagens e
diagnósticos de treino/avaliação (classificação multiclasse).

Não importa nada de src/data.py de propósito — data.py importa log_etapa/
log_nota daqui, então a direção inversa criaria import circular. Onde uma
função precisaria de uma constante ou helper "de dados" (a lista de
classes, uma transform), ela recebe isso como parâmetro em vez de
importar diretamente.
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix, f1_score

_log_relatorio = []

def log_etapa(titulo: str, conteudo) -> None:
    """Imprime no notebook e guarda em markdown para exportar depois."""
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}")
    print(conteudo)
    _log_relatorio.append(f"## {titulo}\n\n```\n{conteudo}\n```\n")

def log_nota(texto: str) -> None:
    """Guarda observações e decisões sem imprimir blocos de tabela."""
    print(f"\nNOTA: {texto}")
    _log_relatorio.append(f"> **Nota:** {texto}\n")

def exportar_log(path: str = "outputs/report_assets/log_eda.md") -> None:
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_relatorio))
    print(f"\nLog exportado para {path} ({len(_log_relatorio)} entradas)")

def limpar_log() -> None:
    """Reinicia o acumulador — útil ao re-rodar o notebook do zero."""
    _log_relatorio.clear()

DIR_FIGURAS = "outputs/report_assets"

# ---------------------------------------------------------------------------
# EDA visual e estatística das imagens
# ---------------------------------------------------------------------------

def amostra_visual_por_classe(
    df: pl.DataFrame,
    coluna_classe: str = "dx",
    coluna_caminho: str = "caminho",
    n_por_classe: int = 3,
    seed: int = 42,
) -> None:
    """Plota uma grade com n_por_classe imagens de exemplo para cada classe em coluna_classe."""
    random.seed(seed)
    classes = sorted(df[coluna_classe].unique().to_list())

    fig, axes = plt.subplots(
        len(classes), n_por_classe, figsize=(3 * n_por_classe, 3 * len(classes))
    )
    if len(classes) == 1:
        axes = axes.reshape(1, -1)

    for i, classe in enumerate(classes):
        caminhos_classe = (
            df.filter(pl.col(coluna_classe) == classe)[coluna_caminho]
            .drop_nulls()
            .to_list()
        )
        amostrados = random.sample(caminhos_classe, min(n_por_classe, len(caminhos_classe)))

        for j in range(n_por_classe):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(amostrados):
                try:
                    img = Image.open(amostrados[j])
                    ax.imshow(img)
                except (UnidentifiedImageError, OSError):
                    ax.set_title("erro ao abrir", fontsize=8, color="red")
            if j == 0:
                ax.set_ylabel(classe, fontsize=10)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    log_etapa(
        "amostra_visual_por_classe",
        f"Grade de {n_por_classe} imagens x {len(classes)} classes exibida (seed={seed}).",
    )

def checar_imagens_corrompidas(df: pl.DataFrame, coluna_caminho: str = "caminho") -> list[str]:
    """Tenta abrir todas as imagens do inventário. Retorna a lista de caminhos que falharam."""
    caminhos = df[coluna_caminho].drop_nulls().to_list()
    corrompidas = []

    for caminho in caminhos:
        try:
            with Image.open(caminho) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            corrompidas.append(caminho)

    log_etapa(
        "checar_imagens_corrompidas",
        f"{len(caminhos)} imagens verificadas, {len(corrompidas)} corrompidas/ilegíveis.",
    )
    if corrompidas:
        log_nota(f"Imagens corrompidas encontradas: {corrompidas[:10]}{'...' if len(corrompidas) > 10 else ''}")

    return corrompidas

def distribuicao_dimensoes(df: pl.DataFrame, coluna_caminho: str = "caminho", amostra: int | None = 500) -> pl.DataFrame:
    """Lê as dimensões (largura, altura, modo de cor) de uma amostra de imagens."""
    caminhos = df[coluna_caminho].drop_nulls().to_list()
    if amostra is not None and amostra < len(caminhos):
        caminhos = random.sample(caminhos, amostra)

    registros = []
    for caminho in caminhos:
        try:
            with Image.open(caminho) as img:
                registros.append(
                    {"caminho": caminho, "largura": img.width, "altura": img.height, "modo": img.mode}
                )
        except (UnidentifiedImageError, OSError):
            continue

    df_dim = pl.DataFrame(registros)

    dims_unicas = df_dim.select(["largura", "altura", "modo"]).unique()
    log_etapa(
        "distribuicao_dimensoes",
        f"{len(caminhos)} imagens inspecionadas. "
        f"{dims_unicas.height} combinação(ões) única(s) de (largura, altura, modo):\n{dims_unicas}",
    )

    return df_dim

def estatisticas_pixel(df: pl.DataFrame, coluna_caminho: str = "caminho", amostra: int = 200, seed: int = 42) -> dict:
    """Média e desvio padrão de pixel por canal (RGB) sobre uma amostra — insumo para a normalização do preparar_dataloaders."""
    random.seed(seed)
    caminhos = df[coluna_caminho].drop_nulls().to_list()
    amostrados = random.sample(caminhos, min(amostra, len(caminhos)))

    somas = np.zeros(3)
    somas_sq = np.zeros(3)
    n_pixels = 0

    for caminho in amostrados:
        try:
            with Image.open(caminho) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        except (UnidentifiedImageError, OSError):
            continue
        somas += arr.sum(axis=(0, 1))
        somas_sq += (arr ** 2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

    media = somas / n_pixels
    variancia = (somas_sq / n_pixels) - media ** 2
    desvio = np.sqrt(np.maximum(variancia, 0))

    resultado = {
        "media_rgb": tuple(media.round(4)),
        "desvio_rgb": tuple(desvio.round(4)),
        "n_imagens_amostradas": len(amostrados),
    }

    log_etapa(
        "estatisticas_pixel",
        f"Sobre {len(amostrados)} imagens: média RGB={resultado['media_rgb']}, "
        f"desvio RGB={resultado['desvio_rgb']}.",
    )

    return resultado

# ---------------------------------------------------------------------------
# Diagnóstico de treino (curvas de loss / gradiente) — mesmo padrão do
# Projeto 1, reaproveitando historico["train_loss"/"val_loss"/"grad_norm"]
# ---------------------------------------------------------------------------

def plot_curvas_loss(historico: dict, nome_modelo: str, salvar_dir: str = DIR_FIGURAS) -> str:
    """Plota treino vs. validação e sugere um rótulo heurístico de convergência."""
    os.makedirs(salvar_dir, exist_ok=True)
    perdas_treino = historico["train_loss"]
    perdas_val = historico["val_loss"]
    epocas = range(1, len(perdas_treino) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, perdas_treino, label="Treino", marker="o")
    plt.plot(epocas, perdas_val, label="Validação", marker="s")
    plt.title(f"Curvas de Loss - {nome_modelo}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle=":")

    caminho = f"{salvar_dir}/loss_{nome_modelo}.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()

    metade = max(1, len(perdas_treino) // 2)
    gap_final = perdas_val[-1] - perdas_treino[-1]
    gap_metade = perdas_val[metade] - perdas_treino[metade] if metade < len(perdas_val) else gap_final
    tendencia_val = perdas_val[-1] - perdas_val[metade] if metade < len(perdas_val) else 0
    variancia_val = float(np.std(np.diff(perdas_val))) if len(perdas_val) > 2 else 0.0

    if gap_final > gap_metade * 1.5 and tendencia_val >= 0:
        rotulo = "Overfitting (gap treino-validação crescendo, validação estagnada ou subindo)"
    elif perdas_treino[-1] > perdas_treino[0] * 0.9 and perdas_val[-1] > perdas_val[0] * 0.9:
        rotulo = "Underfitting (loss de treino e validação permanecem altas, pouca queda)"
    elif variancia_val > np.mean(perdas_val) * 0.1:
        rotulo = "Instabilidade (alta variância época a época na validação)"
    else:
        rotulo = "Convergência aparentemente saudável"

    log_nota(
        f"[{nome_modelo}] Rótulo de loss sugerido: {rotulo}. Loss final — "
        f"treino: {perdas_treino[-1]:.4f}, validação: {perdas_val[-1]:.4f}, "
        f"gap: {gap_final:.4f}. Confirme visualmente antes de citar no relatório."
    )
    return caminho

def plot_gradient_norm(historico: dict, nome_modelo: str, salvar_dir: str = DIR_FIGURAS) -> str:
    """Plota a norma dos gradientes por época e sinaliza vanishing/exploding gradients."""
    if "grad_norm" not in historico or not historico["grad_norm"]:
        return ""
    os.makedirs(salvar_dir, exist_ok=True)
    grad_norm = [g for g in historico["grad_norm"] if np.isfinite(g)]
    if not grad_norm:
        log_nota(f"[{nome_modelo}] Nenhum valor finito de grad_norm no histórico — gráfico não gerado.")
        return ""
    epocas_validas = range(1, len(historico["grad_norm"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epocas_validas, historico["grad_norm"], label="Gradient Norm", color="purple", marker="^")
    plt.title(f"Evolução da Norma dos Gradientes - {nome_modelo}")
    plt.xlabel("Épocas")
    plt.ylabel("Norma L2 (pós-clipping)")
    plt.legend()
    plt.grid(True, linestyle=":")

    caminho = f"{salvar_dir}/grad_norm_{nome_modelo}.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()

    media = float(np.mean(grad_norm))
    maximo = float(np.max(grad_norm))
    minimo = float(np.min(grad_norm))

    alertas = []
    if minimo < media * 0.01:
        alertas.append("possível vanishing gradient (mínimo muito próximo de zero)")
    if maximo > media * 5:
        alertas.append(f"possível exploding gradient (pico de {maximo:.2f}, ~{maximo / media:.1f}x a média)")

    n_overflow = sum(historico.get("batches_overflow", []))
    if n_overflow:
        alertas.append(f"{n_overflow} batch(es) com overflow de AMP ao longo do treino (grad_norm não finita)")

    log_nota(
        f"[{nome_modelo}] Gradient norm - média: {media:.4f}, min: {minimo:.4f}, "
        f"máx: {maximo:.4f}. " + ("; ".join(alertas) if alertas else "Sem sinal de vanishing/exploding pelas regras heurísticas.")
    )
    return caminho

# ---------------------------------------------------------------------------
# Avaliação (classificação multiclasse) e interpretação de erro
# ---------------------------------------------------------------------------

def metricas_a_partir_de_predicoes(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """F1-macro + relatório completo por classe + matriz de confusão."""
    # labels=range(len(classes)) força classification_report/confusion_matrix
    # a sempre considerar as 7 classes, mesmo que alguma (tipicamente 'df'/
    # 'vasc', as mais raras) não apareça em y_true/y_pred de um lote/split
    # específico — sem isso, target_names (7 nomes) e as classes REALMENTE
    # observadas podem divergir em contagem, e o sklearn lança ValueError.
    labels = list(range(len(classes)))
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro")
    relatorio = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
    )
    matriz = confusion_matrix(y_true, y_pred, labels=labels)

    log_etapa(
        "metricas",
        f"F1-macro={f1_macro:.4f}\n"
        + classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0),
    )

    return {"f1_macro": f1_macro, "accuracy": relatorio["accuracy"], "relatorio_por_classe": relatorio, "matriz_confusao": matriz}

def avaliar_baseline(resultado_baseline: dict, X: np.ndarray, y_true: np.ndarray, classes: list[str]) -> dict:
    """Avalia o modelo do baseline (dict retornado por models.treinar_baseline) sobre features já extraídas (data.construir_matriz_features)."""
    X_scaled = resultado_baseline["scaler"].transform(X)
    y_pred = resultado_baseline["modelo"].predict(X_scaled)
    return metricas_a_partir_de_predicoes(y_true, y_pred, classes)

def avaliar_cnn(modelo: torch.nn.Module, dataloader, classes: list[str], device: str = None) -> dict:
    """Avalia a CNN/LSTM (em modo eval, sem dropout) num DataLoader."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    modelo.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, rotulos in dataloader:
            imgs = imgs.to(device)
            saida = modelo(imgs)
            preditos = saida.argmax(dim=1).cpu().numpy()

            y_true.extend(rotulos.numpy())
            y_pred.extend(preditos)

    return metricas_a_partir_de_predicoes(np.array(y_true), np.array(y_pred), classes)

def plotar_matriz_confusao(matriz: np.ndarray, classes: list[str], titulo: str = "Matriz de Confusão", salvar_dir: str | None = DIR_FIGURAS, nome_modelo: str = "modelo") -> str:
    """Heatmap da matriz de confusão, com rótulos das classes. Salva em disco se salvar_dir for informado."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()

    caminho = ""
    if salvar_dir:
        os.makedirs(salvar_dir, exist_ok=True)
        caminho = f"{salvar_dir}/matriz_confusao_{nome_modelo}.png"
        plt.savefig(caminho, bbox_inches="tight")
    plt.show()
    plt.close()
    return caminho

def exemplos_classificados_errado(
    modelo: torch.nn.Module,
    df_eval: pl.DataFrame,
    transform,
    classes: list[str],
    n_por_par: int = 3,
    device: str = None,
) -> pl.DataFrame:
    """
    Roda o modelo sobre df_eval, identifica os pares (classe_real, classe_predita)
    mais frequentes de ERRO, e plota exemplos visuais de cada um dos 3 pares
    mais comuns. Retorna o DataFrame de erros completo, não só os plotados.

    transform: a mesma transform (resize + normalização) usada no treino —
    monte via data.montar_transform(media_rgb, desvio_rgb) e passe aqui,
    em vez desta função importar de data.py (evita import circular).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    modelo.eval()

    caminhos = df_eval["caminho"].to_list()

    predicoes = []
    with torch.no_grad():
        for caminho in caminhos:
            img = Image.open(caminho).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            saida = modelo(img_t)
            pred_idx = saida.argmax(dim=1).item()
            predicoes.append(classes[pred_idx])

    df_resultado = df_eval.with_columns(pl.Series("predito", predicoes)).with_columns(
        (pl.col("dx") != pl.col("predito")).alias("erro")
    )

    df_erros = df_resultado.filter(pl.col("erro"))
    pares_frequentes = (
        df_erros.group_by(["dx", "predito"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )

    log_etapa(
        "exemplos_classificados_errado",
        f"{df_erros.height} erros de {df_resultado.height} avaliados. "
        f"Pares (real, predito) mais frequentes:\n{pares_frequentes.head(10)}",
    )

    top_pares = pares_frequentes.head(3).to_dicts()
    if not top_pares:
        log_nota("Nenhum erro encontrado — não há pares para exibir.")
        return df_resultado

    fig, axes = plt.subplots(len(top_pares), n_por_par, figsize=(3 * n_por_par, 3 * len(top_pares)))
    if len(top_pares) == 1:
        axes = axes.reshape(1, -1)

    for i, par in enumerate(top_pares):
        exemplos = df_erros.filter(
            (pl.col("dx") == par["dx"]) & (pl.col("predito") == par["predito"])
        )["caminho"].to_list()[:n_por_par]

        for j in range(n_por_par):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(exemplos):
                ax.imshow(Image.open(exemplos[j]))
            if j == 0:
                ax.set_title(f"real={par['dx']} / predito={par['predito']} (n={par['n']})", fontsize=9)

    plt.tight_layout()
    plt.show()

    return df_resultado
