from typing import List, Dict, Optional
import os
import matplotlib.pyplot as plt


def plotar_curvas_de_perda(
    historico: List[Dict[str, float]], 
    tarefa: str = "classification", 
    caminho_salvamento: Optional[str] = None
) -> None:
    """
    Gera e salva o gráfico das curvas de perda (Train Loss vs Val Loss).
    """
    perdas_treino = [item['train_loss'] for item in historico]
    perdas_val = [item['val_loss'] for item in historico]
    epocas = range(1, len(perdas_treino) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    ax.plot(epocas, perdas_treino, label="Perda de Treino (Train Loss)", 
            color="#1f77b4", linewidth=2.5, marker='o', markersize=5, alpha=0.9)
    
    ax.plot(epocas, perdas_val, label="Perda de Validação (Val Loss)", 
            color="#ff7f0e", linewidth=2.5, linestyle="--", marker='s', markersize=5, alpha=0.9)
    
    nome_tarefa = "Classificação Binária" if tarefa == "classification" else "Regressão Contínua"
    tipo_perda = "Cross-Entropy Loss" if tarefa == "classification" else "Mean Squared Error (MSE)"
    
    ax.set_title(f"Curvas de Perda - {nome_tarefa}\nMLP Lending Club", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Épocas", fontsize=10)
    ax.set_ylabel(tipo_perda, fontsize=10)
    ax.set_xticks(list(epocas))
    ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    
    plt.tight_layout()
    if caminho_salvamento:
        diretorio = os.path.dirname(caminho_salvamento)
        if diretorio and not os.path.exists(diretorio):
            os.makedirs(diretorio, exist_ok=True)
        plt.savefig(caminho_salvamento, bbox_inches='tight', dpi=300)
        print(f"[Visualization] Gráfico salvo em: '{caminho_salvamento}'")
    plt.close()
