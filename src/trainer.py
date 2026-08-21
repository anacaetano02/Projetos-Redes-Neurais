from typing import Dict, Any, Literal, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score
)


TaskMode = Literal["classification", "regression"]


def _calcular_metricas(
    targets: np.ndarray, 
    predictions: np.ndarray, 
    loss: float, 
    mode: TaskMode
) -> Dict[str, float]:
    """
    Função utilitária interna para calcular métricas estatísticas finais por época.
    """
    metrics = {"loss": float(loss)}
    if mode == "classification":
        metrics["precision"] = float(precision_score(targets, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(targets, predictions, zero_division=0))
        metrics["f1"] = float(f1_score(targets, predictions, zero_division=0))
    elif mode == "regression":
        metrics["mae"] = float(mean_absolute_error(targets, predictions))
        metrics["mse"] = float(loss)
        metrics["rmse"] = float(np.sqrt(loss))
        metrics["r2"] = float(r2_score(targets, predictions))
    return metrics


def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    mode: TaskMode = "classification"
) -> Dict[str, float]:
    """
    Executa o treinamento do modelo por uma época.
    Otimizado para evitar sincronizações GPU-CPU por batch.
    """
    model.train()
    running_loss = 0.0
    all_targets_list: List[torch.Tensor] = []
    all_preds_list: List[torch.Tensor] = []
    total_grad_norm = 0.0
    num_batches = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if mode == "regression":
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets)
            all_preds_list.append(outputs.detach())
        else:
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            all_preds_list.append(preds.detach())
            
        loss.backward()
        
        # Vetorização do cálculo da norma do gradiente (evita loops manuais por parâmetro)
        grads = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack(grads)).item() if grads else 0.0
        total_grad_norm += grad_norm
        num_batches += 1
        
        optimizer.step()
        
        running_loss += loss.item()
        all_targets_list.append(targets.detach())
        
    epoch_loss = running_loss / len(dataloader)
    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
    
    # Concatenação e sincronização GPU->CPU realizada apenas UMA vez no final da época
    all_predictions = torch.cat(all_preds_list).cpu().numpy()
    all_targets = torch.cat(all_targets_list).cpu().numpy()
    
    metrics = _calcular_metricas(all_targets, all_predictions, epoch_loss, mode)
    metrics["grad_norm"] = avg_grad_norm
    return metrics


def validate_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    mode: TaskMode = "classification"
) -> Dict[str, float]:
    """
    Executa a avaliação do modelo no conjunto de validação por uma época.
    """
    model.eval()
    running_loss = 0.0
    all_targets_list: List[torch.Tensor] = []
    all_preds_list: List[torch.Tensor] = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            
            if mode == "regression":
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, targets)
                all_preds_list.append(outputs)
            else:
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                all_preds_list.append(preds)
                
            running_loss += loss.item()
            all_targets_list.append(targets)
            
    epoch_loss = running_loss / len(dataloader)
    
    # Sincronização única por época
    all_predictions = torch.cat(all_preds_list).cpu().numpy()
    all_targets = torch.cat(all_targets_list).cpu().numpy()
    
    return _calcular_metricas(all_targets, all_predictions, epoch_loss, mode)


def evaluate_on_test_set(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    mode: TaskMode = "classification"
) -> None:
    """
    Avaliação cega final no conjunto de teste.
    Imprime Matriz de Confusão, Classification Report e métricas de regressão.
    """
    model.eval()
    all_targets_list: List[torch.Tensor] = []
    all_preds_list: List[torch.Tensor] = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            
            if mode == "classification":
                preds = outputs.argmax(dim=1)
                all_preds_list.append(preds)
            elif mode == "regression":
                outputs = outputs.squeeze(-1)
                all_preds_list.append(outputs)
            
            all_targets_list.append(targets)
                
    all_predictions = torch.cat(all_preds_list).cpu().numpy()
    all_targets = torch.cat(all_targets_list).cpu().numpy()

    print(f"\n========================================================")
    print(f"   RELATÓRIO DE DESEMPENHO NO CONJUNTO DE TESTE ({mode.upper()})")
    print(f"========================================================")
    
    if mode == "classification":
        print("\nMatriz de Confusão:")
        print(confusion_matrix(all_targets, all_predictions))
        
        print("\nRelatório de Métricas:")
        print(classification_report(
            all_targets, 
            all_predictions, 
            target_names=["Fully Paid (0)", "Charged Off (1)"]
        ))
    elif mode == "regression":
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        
        print(f"Mean Absolute Error (MAE):     {mae:.6f}")
        print(f"Mean Squared Error (MSE):      {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Coeficiente R² (Determinação):  {r2:.4f}")
    print(f"========================================================\n")
