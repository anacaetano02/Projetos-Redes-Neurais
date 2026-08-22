"""
Rastreamento leve de execuções (experiment tracking), baseado em um arquivo JSON append-only.

Cada chamada a `log_run` registra uma execução (tarefa, nome do modelo, métricas de teste,
timestamp e observações opcionais) em um histórico persistente. O `report_generator.py`
lê esse histórico para montar comparações e o texto de discussão automaticamente.

Uso típico ao final de um notebook/script de treino:

    from src.experiment_tracker import log_run

    test_metrics_class = evaluate_on_test_set(model, test_loader_class, device, mode="classification")
    log_run(
        task="classification",
        model_name="mlp_v3",
        metrics=test_metrics_class,
        notes="class_weight adicionado ao CrossEntropyLoss",
    )
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

TaskName = Literal["classification", "regression"]

DEFAULT_HISTORY_PATH = "results/run_history.json"


def log_run(
    task: TaskName,
    model_name: str,
    metrics: Dict[str, Any],
    notes: Optional[str] = None,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> Dict[str, Any]:
    """
    Registra uma execução no histórico persistente (append-only).

    Args:
        task: "classification" ou "regression".
        model_name: identificador estável do modelo/abordagem
            (ex: "baseline_logistic_regression", "mlp_v1", "mlp_v3_class_weight").
            Use o MESMO nome entre execuções da mesma abordagem para permitir
            comparar a evolução dela ao longo do tempo.
        metrics: dicionário de métricas já calculadas (ex: retorno de evaluate_on_test_set).
        notes: observação livre sobre o que mudou nesta execução
            (ex: "adicionado class_weight balanceado").
        history_path: caminho do arquivo JSON de histórico.

    Returns:
        O registro (entry) que foi adicionado ao histórico.
    """
    parent = os.path.dirname(history_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    history: List[Dict[str, Any]] = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Histórico corrompido/vazio: recomeça em vez de travar o treino por causa do relatório
            history = []

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task,
        "model_name": model_name,
        "metrics": metrics,
        "notes": notes,
    }
    history.append(entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[ExperimentTracker] Execução registrada: task='{task}' model='{model_name}' -> {history_path}")
    return entry


def load_history(history_path: str = DEFAULT_HISTORY_PATH) -> List[Dict[str, Any]]:
    """Carrega o histórico completo de execuções, ou lista vazia se ainda não existir."""
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def latest_run_per_model(
    task: TaskName,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> Dict[str, Dict[str, Any]]:
    """
    Retorna a execução mais recente de cada `model_name` para uma dada `task`.
    Ex: {"baseline_logistic_regression": {...}, "mlp_v3": {...}}
    """
    history = load_history(history_path)
    latest: Dict[str, Dict[str, Any]] = {}
    for entry in history:
        if entry.get("task") != task:
            continue
        model = entry.get("model_name", "desconhecido")
        # Histórico é append-only e cronológico: a última ocorrência é sempre a mais recente
        latest[model] = entry
    return latest


def runs_for_model(
    task: TaskName,
    model_name: str,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> List[Dict[str, Any]]:
    """Retorna todas as execuções históricas de um modelo específico numa tarefa, em ordem cronológica."""
    history = load_history(history_path)
    return [e for e in history if e.get("task") == task and e.get("model_name") == model_name]
