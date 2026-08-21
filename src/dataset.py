from typing import Literal
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


TaskMode = Literal["classification", "regression"]


class LendingClubDataset(Dataset):
    """
    Carrega os DataFrames do Polars e fornece tensores otimizados para a rede PyTorch.
    Pré-converte todos os alvos e características para tensores PyTorch na inicialização,
    eliminando alocações redundantes dentro do loop de treinamento.
    """
    def __init__(self, polars_df: pl.DataFrame, mode: TaskMode = "classification"):
        self.mode = mode
        
        # Isola as características removendo colunas de alvo e controle
        exclude_cols = ["classification_target", "int_rate", "loan_status"]
        feature_cols = [c for c in polars_df.columns if c not in exclude_cols]
        
        features_np = polars_df.select(feature_cols).to_numpy().astype(np.float32)
        self.X_tensor = torch.from_numpy(features_np)
        self.features = self.X_tensor

        
        # Pré-converte o vetor de resposta para Tensor PyTorch
        if mode == "classification":
            y_np = polars_df["classification_target"].to_numpy().astype(np.int64)
            self.y_tensor = torch.from_numpy(y_np)
        elif mode == "regression":
            y_np = polars_df["int_rate"].to_numpy().astype(np.float32)
            self.y_tensor = torch.from_numpy(y_np)
        else:
            raise ValueError(f"Modo '{mode}' inválido. Escolha 'classification' ou 'regression'.")
        
    def __len__(self) -> int:
        return len(self.X_tensor)
        
    def __getitem__(self, idx: int):
        return self.X_tensor[idx], self.y_tensor[idx]
