"""
Módulo de treino, validação e avaliação para o modelo LendingClubMLP.
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


TaskMode = Literal["classification", "regression"]


def _calcular_metricas(
    targets: np.ndarray,
    predictions: np.ndarray,
    loss: float,
    mode: TaskMode,
) -> Dict[str, float]:
    """
    Função utilitária interna para calcular métricas estatísticas finais por época.
    """
    metrics: Dict[str, float] = {"loss": float(loss)}
    if mode == "classification":
        metrics["precision"] = float(precision_score(targets, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(targets, predictions, zero_division=0))
        metrics["f1"] = float(f1_score(targets, predictions, zero_division=0))
    elif mode == "regression":
        mse = float(mean_squared_error(targets, predictions))
        metrics["mae"] = float(mean_absolute_error(targets, predictions))
        metrics["mse"] = mse
        metrics["rmse"] = float(np.sqrt(mse))
        metrics["r2"] = float(r2_score(targets, predictions))
    return metrics


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: TaskMode = "classification",
    optimizer: Optional[torch.optim.Optimizer] = None,
    track_grad_norm: bool = False,
) -> Dict[str, float]:
    """
    Executa uma época de treino (se `optimizer` for fornecido) ou de validação
    (se `optimizer` for None).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    all_targets_list: List[torch.Tensor] = []
    all_preds_list: List[torch.Tensor] = []
    total_grad_norm = 0.0
    num_batches = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)

            if mode == "regression":
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, targets)
                preds = outputs.detach() if is_training else outputs
            else:
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                preds = preds.detach() if is_training else preds

            if is_training:
                loss.backward()

                if track_grad_norm:
                    grads = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                    grad_norm = torch.norm(torch.stack(grads)).item() if grads else 0.0
                    total_grad_norm += grad_norm

                optimizer.step()

            num_batches += 1
            running_loss += loss.item()
            all_preds_list.append(preds)
            all_targets_list.append(targets.detach() if is_training else targets)

    epoch_loss = running_loss / len(dataloader)

    # Concatenação e sincronização GPU->CPU realizada apenas UMA vez no final da época
    all_predictions = torch.cat(all_preds_list).cpu().numpy()
    all_targets = torch.cat(all_targets_list).cpu().numpy()

    metrics = _calcular_metricas(all_targets, all_predictions, epoch_loss, mode)

    if is_training and track_grad_norm:
        metrics["grad_norm"] = total_grad_norm / num_batches if num_batches > 0 else 0.0

    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: TaskMode = "classification",
    track_grad_norm: bool = True,
) -> Dict[str, float]:
    """Executa o treinamento do modelo por uma época."""
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        mode=mode,
        optimizer=optimizer,
        track_grad_norm=track_grad_norm,
    )


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: TaskMode = "classification",
) -> Dict[str, float]:
    """Executa a avaliação do modelo no conjunto de validação por uma época."""
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        mode=mode,
        optimizer=None,
        track_grad_norm=False,
    )


def evaluate_on_test_set(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: TaskMode = "classification",
) -> Dict[str, Any]:
    """
    Avaliação cega final no conjunto de teste.

    Acumula predições e alvos na GPU e sincroniza com a CPU uma única vez, no final,
    para evitar overhead de sincronizações repetidas por batch.
    Imprime Matriz de Confusão, Classification Report e métricas de regressão,
    e retorna um dicionário com as métricas para uso posterior (logs, JSON, etc.).
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

    print("\n========================================================")
    print(f"   RELATÓRIO DE DESEMPENHO NO CONJUNTO DE TESTE ({mode.upper()})")
    print("========================================================")

    metrics: Dict[str, Any] = {"mode": mode}

    if mode == "classification":
        cm = confusion_matrix(all_targets, all_predictions)
        print("\nMatriz de Confusão:")
        print(cm)
        print("\nRelatório de Métricas:")
        print(
            classification_report(
                all_targets,
                all_predictions,
                target_names=["Fully Paid (0)", "Charged Off (1)"],
            )
        )
        metrics.update(
            {
                "confusion_matrix": cm.tolist(),
                "accuracy": float((all_predictions == all_targets).mean()),
                "precision": float(precision_score(all_targets, all_predictions, zero_division=0)),
                "recall": float(recall_score(all_targets, all_predictions, zero_division=0)),
                "f1": float(f1_score(all_targets, all_predictions, zero_division=0)),
            }
        )
    elif mode == "regression":
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        print(f"Mean Absolute Error (MAE):      {mae:.6f}")
        print(f"Mean Squared Error (MSE):       {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Coeficiente R² (Determinação):  {r2:.4f}")
        metrics.update(
            {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
            }
        )

    print("========================================================\n")
    return metrics
