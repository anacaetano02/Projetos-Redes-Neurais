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
    precision_recall_curve,
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


def _predict_probs_and_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: TaskMode = "classification",
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Roda o modelo em modo avaliação e retorna:
      - classificação: a probabilidade da classe positiva (1 = Charged Off), via softmax.
      - regressão: a predição contínua diretamente.
    Sincroniza GPU->CPU apenas uma vez, no final.
    """
    model.eval()
    all_targets_list: List[torch.Tensor] = []
    all_outputs_list: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)

            if mode == "classification":
                probs_classe_positiva = torch.softmax(outputs, dim=1)[:, 1]
                all_outputs_list.append(probs_classe_positiva)
            else:
                all_outputs_list.append(outputs.squeeze(-1))

            all_targets_list.append(targets)

    all_outputs = torch.cat(all_outputs_list).cpu().numpy()
    all_targets = torch.cat(all_targets_list).cpu().numpy()
    return all_outputs, all_targets


def buscar_melhor_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    estrategia: Literal["f1", "recall_minimo"] = "f1",
    recall_minimo: float = 0.5,
) -> Dict[str, float]:
    """
    Varre a curva precisão-recall para encontrar o melhor threshold de decisão.

    IMPORTANTE: `dataloader` deve ser o de VALIDAÇÃO, nunca o de teste — escolher o
    threshold olhando o teste é uma forma de vazamento (overfitting na escolha do
    ponto de operação), mesmo que o modelo em si não seja retreinado.

    Args:
        estrategia: "f1" maximiza o F1-score da classe positiva (Charged Off).
                    "recall_minimo" maximiza a precisão entre os thresholds que
                    atingem pelo menos `recall_minimo` de recall.
        recall_minimo: usado apenas quando estrategia="recall_minimo".

    Returns:
        Dict com "threshold", "precision" e "recall" no ponto escolhido.
    """
    probs, targets = _predict_probs_and_targets(model, dataloader, device, mode="classification")
    precisions, recalls, thresholds = precision_recall_curve(targets, probs)

    # precision_recall_curve retorna 1 ponto a mais que thresholds (o último ponto
    # é recall=0/precision=1, sem threshold correspondente) — descartamos esse último.
    precisions_t = precisions[:-1]
    recalls_t = recalls[:-1]

    if len(thresholds) == 0:
        # Caso degenerado (praticamente nunca ocorre com dados reais)
        return {"threshold": 0.5, "precision": float(precisions_t[0]) if len(precisions_t) else 0.0,
                "recall": float(recalls_t[0]) if len(recalls_t) else 0.0}

    if estrategia == "recall_minimo":
        candidatos = np.where(recalls_t >= recall_minimo)[0]
        if len(candidatos) == 0:
            idx = int(np.argmax(recalls_t))  # nenhum threshold atinge o mínimo pedido
        else:
            idx = candidatos[np.argmax(precisions_t[candidatos])]
    else:  # "f1"
        denom = precisions_t + recalls_t
        f1s = np.where(denom > 0, 2 * precisions_t * recalls_t / np.maximum(denom, 1e-12), 0.0)
        idx = int(np.argmax(f1s))

    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precisions_t[idx]),
        "recall": float(recalls_t[idx]),
    }


def evaluate_on_test_set(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: TaskMode = "classification",
    threshold: Optional[float] = None,
    dataset_label: str = "TESTE",
) -> Dict[str, Any]:
    """
    Avaliação cega no conjunto informado (por padrão rotulado como "TESTE" no print,
    mas pode ser reutilizada em VALIDAÇÃO passando dataset_label="VALIDAÇÃO",
    por exemplo durante grid search).

    Args:
        threshold: em classificação, ponto de corte sobre a probabilidade da classe
            positiva (1 = Charged Off). Se None, usa 0.5 (equivalente a argmax puro).
            Encontre esse valor via `buscar_melhor_threshold` no conjunto de VALIDAÇÃO,
            nunca no de teste.

    Acumula predições e alvos na GPU e sincroniza com a CPU uma única vez, no final.
    Imprime Matriz de Confusão, Classification Report e métricas de regressão,
    e retorna um dicionário com as métricas para uso posterior (logs, JSON, etc.).
    """
    if mode == "classification":
        probs, all_targets = _predict_probs_and_targets(model, dataloader, device, mode="classification")
        cutoff = threshold if threshold is not None else 0.5
        all_predictions = (probs >= cutoff).astype(np.int64)
    else:
        all_predictions, all_targets = _predict_probs_and_targets(model, dataloader, device, mode="regression")
        cutoff = None

    print("\n========================================================")
    print(f"   RELATÓRIO DE DESEMPENHO NO CONJUNTO DE {dataset_label} ({mode.upper()})")
    if mode == "classification":
        print(f"   Threshold de decisão: {cutoff:.4f}")
    print("========================================================")

    metrics: Dict[str, Any] = {"mode": mode}
    if mode == "classification":
        metrics["threshold"] = float(cutoff)

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
                zero_division=0,
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
