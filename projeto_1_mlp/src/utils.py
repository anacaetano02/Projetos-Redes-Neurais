"""
Módulo src/utils.py
Consolida as ferramentas de logging de etapas e de diagnósticos visuais
e analíticos.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
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
    
    # Heurística de primeira leitura para rótulos
    final_trn = perdas_treino[-1]
    final_val = perdas_val[-1]
    gap = final_val - final_trn
    
    if final_val > 1.10 and final_trn > 0.90:
        rotulo = "Underfitting"
    elif gap > 0.15:
        rotulo = "Overfitting"
    else:
        rotulo = "Convergência Saudável"
        
    log_nota(f"Rótulo de Loss sugerido para {nome_modelo}: {rotulo}")
    return caminho

def plot_gradient_norm(historico: dict, nome_modelo: str, salvar_dir: str = DIR_FIGURAS) -> str:
    """Plota a norma dos gradientes por época para auditoria de gradientes."""
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
    return caminho

def capturar_distribuicao_ativacoes(modelo, loader, device: str, nome_modelo: str) -> dict:
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

    relatorio_zeros = {}
    for nome_camada, valores in ativacoes.items():
        val_np = valores.cpu().numpy()
        total = val_np.size
        zeros = np.sum(val_np == 0.0)
        pct_zeros = (zeros / total) * 100
        relatorio_zeros[nome_camada] = pct_zeros
        
    log_etapa(f"Neurônios Mortos (Ativações em Zero) - {nome_modelo}", relatorio_zeros)
    return relatorio_zeros

def avaliar_classificacao(modelo, loader, device: str, nome_modelo: str, limiar: float = 0.5) -> dict:
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
    
    cm = confusion_matrix(reais, preds)
    auc = roc_auc_score(reais, probs)
    rep = classification_report(reais, preds, output_dict=True)
    
    # Normalização de classes em classificação
    # Verificando as chaves das métricas do dict do classification_report
    key_class_1 = "1.0" if "1.0" in rep else "1"
    
    resultados = {
        "accuracy": rep["accuracy"],
        "precision": rep[key_class_1]["precision"] if key_class_1 in rep else 0.0,
        "recall": rep[key_class_1]["recall"] if key_class_1 in rep else 0.0,
        "f1": rep[key_class_1]["f1-score"] if key_class_1 in rep else 0.0,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }
    
    log_etapa(f"Métricas de Classificação - {nome_modelo}", resultados)
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