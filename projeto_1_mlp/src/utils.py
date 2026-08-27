"""
Módulo src/utils.py
Consolida as ferramentas de logging de etapas e de diagnósticos visuais
e analíticos.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

_log_relatorio = []

def log_etapa(titulo: str, conteudo) -> None:
    """Imprime no notebook e guarda em markdown para exportar depois."""
    print(f"\n{'=' * 60}\n{titulo}\n{'=' * 60}")
    print(conteudo)
    _log_relatorio.append(f"## {titulo}\n\n\n{conteudo}\n\n")

def log_nota(texto: str) -> None:
    """Guarda observações e decisões sem imprimir blocos de tabela."""
    print(f"\nNOTA: {texto}")
    _log_relatorio.append(f">  **Nota:**  {texto}\n")

def exportar_log(path: str = "outputs/report_assets/log_eda.md") -> None:
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_relatorio))
    print(f"\nLog exportado para {path} ({len(_log_relatorio)} entradas)")

def limpar_log() -> None:
    _log_relatorio.clear()

DIR_FIGURAS = "outputs/report_assets"

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

    # Heurística relativa à escala da própria loss, em vez de limiares
    # absolutos (que não fazem sentido comparando BCE com MSE): olha o
    # crescimento do gap treino-validação entre a metade e o fim do
    # treino, e a variância época a época da validação.
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
    grad_norm = historico["grad_norm"]
    epocas = range(1, len(grad_norm) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, grad_norm, label="Gradient Norm", color="purple", marker="^")
    plt.title(f"Evolução da Norma dos Gradientes - {nome_modelo}")
    plt.xlabel("Épocas")
    plt.ylabel("Norma L2")
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

    log_nota(
        f"[{nome_modelo}] Gradient norm - média: {media:.4f}, min: {minimo:.4f}, "
        f"máx: {maximo:.4f}. " + ("; ".join(alertas) if alertas else "Sem sinal de vanishing/exploding pelas regras heurísticas.")
    )
    return caminho

def capturar_distribuicao_ativacoes(modelo, loader, device: str, nome_modelo: str, salvar_dir: str = DIR_FIGURAS) -> dict:
    """Avalia a distribuição de ativações ocultas para mapear neurônios mortos."""
    modelo.eval()
    X_amostra, _ = next(iter(loader))
    X_amostra = X_amostra.to(device)

    try:
        with torch.no_grad():
            out, ativacoes = modelo(X_amostra, retornar_ativacoes=True)
    except TypeError:
        log_nota("O modelo não suporta retornar_ativacoes=True em seu forward pass.")
        return {}

    os.makedirs(salvar_dir, exist_ok=True)
    relatorio = {}
    for nome_camada, valores in ativacoes.items():
        val_np = valores.cpu().numpy().flatten()
        pct_zeros = float((val_np == 0.0).mean() * 100)
        relatorio[nome_camada] = {
            "pct_zeros": pct_zeros,
            "media": float(val_np.mean()),
            "std": float(val_np.std()),
        }

        caminho = f"{salvar_dir}/ativacoes_{nome_modelo}_{nome_camada}.png"
        plt.figure(figsize=(8, 4))
        plt.hist(val_np, bins=50)
        plt.title(f"Ativações - {nome_modelo}/{nome_camada} ({pct_zeros:.1f}% em zero)")
        plt.savefig(caminho, bbox_inches="tight")
        plt.close()

    log_etapa(f"Neurônios Mortos (Ativações em Zero) - {nome_modelo}", relatorio)
    for nome_camada, stats in relatorio.items():
        if stats["pct_zeros"] > 40:
            log_nota(
                f"[{nome_modelo}/{nome_camada}] {stats['pct_zeros']:.1f}% das ativações em "
                f"zero — possível indício de neurônios mortos (ReLU). Candidato a testar "
                f"LeakyReLU nessa configuração."
            )
    return relatorio

def avaliar_classificacao(modelo, loader, device: str, nome_modelo: str, limiar: float = 0.5, salvar_dir: str = DIR_FIGURAS) -> dict:
    """Roda a classificação sobre o loader de teste e calcula métricas completas."""
    modelo.eval()
    probs, reais = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = modelo(X_batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob)
            reais.extend(y_batch.numpy())

    probs = np.array(probs).reshape(-1)
    reais = np.array(reais).reshape(-1)
    preds = (probs >= limiar).astype(int)

    cm = confusion_matrix(reais, preds, labels=[0, 1])
    auc = roc_auc_score(reais, probs)
    rep = classification_report(reais, preds, output_dict=True)

    # Normalização de classes em classificação
    # Verificando as chaves das métricas do dict do classification_report
    key_class_1 = "1.0" if "1.0" in rep else "1"

    os.makedirs(salvar_dir, exist_ok=True)
    caminho_cm = f"{salvar_dir}/matriz_confusao_{nome_modelo}.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.colorbar()
    plt.savefig(caminho_cm, bbox_inches="tight")
    plt.close()

    vn, fp, fn, vp = cm.ravel()
    resultados = {
        "accuracy": rep["accuracy"],
        "precision": rep[key_class_1]["precision"] if key_class_1 in rep else 0.0,
        "recall": rep[key_class_1]["recall"] if key_class_1 in rep else 0.0,
        "f1": rep[key_class_1]["f1-score"] if key_class_1 in rep else 0.0,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "falsos_negativos": int(fn),
        "falsos_positivos": int(fp),
    }

    log_etapa(f"Métricas de Classificação - {nome_modelo}", resultados)
    log_nota(
        f"[{nome_modelo}] Falsos negativos: {fn} (inadimplente previsto como bom "
        f"pagador — erro mais custoso no contexto de crédito). Falsos positivos: {fp}. "
        f"Figura salva em {caminho_cm}."
    )
    return resultados

def avaliar_regressao(modelo, loader, device: str, nome_modelo: str) -> dict:
    """Avalia o modelo de regressão calculando MAE, RMSE e R2."""
    modelo.eval()
    preds, reais = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = modelo(X_batch)
            preds.extend(out.cpu().numpy())
            reais.extend(y_batch.numpy())

    preds = np.array(preds).reshape(-1)
    reais = np.array(reais).reshape(-1)

    mae = mean_absolute_error(reais, preds)
    mse = mean_squared_error(reais, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(reais, preds)

    resultados = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

    log_etapa(f"Métricas de Regressão - {nome_modelo}", resultados)
    return resultados

def perfil_distribuicao_categorica(df: pl.DataFrame, coluna: str) -> pl.DataFrame:
    """Contagem + percentual de uma coluna categórica, já logado."""
    resultado = (
        df[coluna]
        .value_counts()
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).round(2).alias("%")
        )
        .sort("count", descending=True)
    )
    log_etapa(f"Distribuição de {coluna}", resultado)
    return resultado

def perfil_skew_zero_pct(df: pl.DataFrame, colunas: list[str]) -> pl.DataFrame:
    """Skew e percentual de zeros para uma lista de colunas numéricas — apoia a decisão de transformação (ver aplicar_transformacoes_por_regra)."""
    skew_df = df.select([
        pl.col(c).skew().alias(c) for c in colunas
    ]).transpose(include_header=True, header_name="coluna", column_names=["skew"])

    zero_pcts = df.select([
        ((pl.col(c) == 0).sum() / pl.len() * 100).alias(c) for c in colunas
    ]).transpose(include_header=True, header_name="coluna", column_names=["zero_pct"])

    resumo = skew_df.join(zero_pcts, on="coluna").sort("skew", descending=True)
    log_etapa("Perfil de skew e percentual de zeros", resumo)
    return resumo

def analisar_distribuicao_int_rate(df: pl.DataFrame, salvar_figura: str | None = f"{DIR_FIGURAS}/hist_int_rate.png") -> None:
    """describe() + histograma de target_regressao (int_rate) — achado relevante para a profundidade da arquitetura de regressão."""
    log_etapa("target_regressao (int_rate) - describe()", df["target_regressao"].describe())

    if salvar_figura:
        diretorio = os.path.dirname(salvar_figura)
        if diretorio:
            os.makedirs(diretorio, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.hist(df["target_regressao"].to_numpy(), bins=80, edgecolor="black", alpha=0.7)
        plt.xlabel("int_rate (%)")
        plt.ylabel("Frequência")
        plt.title("Distribuição de int_rate")
        plt.savefig(salvar_figura, bbox_inches="tight")
        plt.close()
        log_nota(f"Histograma de int_rate salvo em {salvar_figura}.")

    log_nota(
        "int_rate pode apresentar distribuição multimodal (picos recorrentes), "
        "consistente com o pricing histórico do Lending Club por sub_grade — "
        "variável removida do conjunto de features por vazamento condicional. "
        "Se confirmado visualmente no histograma acima, é argumento para maior "
        "profundidade na arquitetura de regressão, já que uma rede rasa tende a "
        "suavizar os picos numa aproximação contínua média."
    )

def checar_correlacao(df: pl.DataFrame, colunas: list[str], limiar: float = 0.85) -> pl.DataFrame:
    """Pares de colunas numéricas com |correlação de Pearson| acima do limiar."""
    corr_matrix = df.select(colunas).corr()

    pares = []
    for i, col_i in enumerate(colunas):
        for j, col_j in enumerate(colunas):
            if j <= i:
                continue
            r = corr_matrix[col_i][j]
            if r is not None and abs(r) > limiar:
                pares.append({"coluna_1": col_i, "coluna_2": col_j, "correlacao": round(r, 3)})

    pares_df = pl.DataFrame(pares) if pares else pl.DataFrame({"coluna_1": [], "coluna_2": [], "correlacao": []})
    if len(pares_df) > 0:
        pares_df = pares_df.sort("correlacao", descending=True)

    log_etapa(f"Pares de features com |r| > {limiar}", pares_df)
    log_nota(
        f"{len(pares_df)} pares excederam o limiar de {limiar}. "
        f"Decisão a documentar: remover uma de cada par redundante, "
        f"ou manter todas e delegar a redundância à regularização (weight decay)."
    )
    return pares_df
