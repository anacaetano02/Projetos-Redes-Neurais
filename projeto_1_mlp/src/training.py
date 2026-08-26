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
from src.utils import log_etapa, log_nota, plot_curvas_loss, plot_gradient_norm, avaliar_classificacao, avaliar_regressao

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

def treinar_epoca(modelo, loader, loss_fn, otimizador, device: str) -> tuple[float, float]:
    modelo.train()
    loss_total = 0.0
    grad_norm_total = 0.0
    n_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        otimizador.zero_grad()
        out = modelo(X_batch).squeeze()
        loss = loss_fn(out, y_batch)
        loss.backward()
        
        # Captura grad_norm na janela correta (pós-backward, pré-step)
        gn = calcular_gradient_norm(modelo)
        grad_norm_total += gn
        
        otimizador.step()
        loss_total += loss.item()
        n_batches += 1
        
    return loss_total / n_batches, grad_norm_total / n_batches

@torch.no_grad()
def avaliar(modelo, loader, loss_fn, device: str) -> float:
    modelo.eval()
    loss_total = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = modelo(X_batch).squeeze()
        loss = loss_fn(out, y_batch)
        loss_total += loss.item()
        n_batches += 1
    return loss_total / n_batches

def treinar_modelo(modelo, loader_treino, loader_val, loss_fn, otimizador, epocas: int, device: str = "cpu", checkpoint_path: str = "melhor_modelo.pt", scheduler=None, nome_run: str = "run", paciencia_early_stopping: int = 5) -> dict:
    """Executa o loop principal aplicando Early Stopping e checkpointing via state_dict."""
    diretorio_checkpoint = os.path.dirname(checkpoint_path)
    if diretorio_checkpoint:
        os.makedirs(diretorio_checkpoint, exist_ok=True)

    writer = SummaryWriter(f"runs/{nome_run}")
    best_val_loss = float("inf")
    counter_es = 0

    historico = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}

    try:
        for epoch in range(1, epocas + 1):
            trn_loss, avg_gn = treinar_epoca(modelo, loader_treino, loss_fn, otimizador, device)
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
                # Salvando checkpoint físico estável
                torch.save({
                    "epoca": epoch,
                    "state_dict": modelo.state_dict(),
                    "val_loss": val_loss
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

def exportar_experimentos(path_json: str = "outputs/report_assets/experimentos.json"):
    import json
    diretorio = os.path.dirname(path_json)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(_experimentos, f, indent=2, ensure_ascii=False)
    print(f"Experimentos registrados e persistidos em '{path_json}'")