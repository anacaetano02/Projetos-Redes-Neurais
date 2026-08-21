from typing import Optional, Literal
import torch
import torch.nn as nn


ActivationType = Literal["relu", "leaky_relu", "tanh"]
NormType = Literal["batch", "layer"]


class LendingClubMLP(nn.Module):
    """
    Rede neural MLP totalmente conectada com ativações dinâmicas, Dropout
    e suporte a Normalização por Lote (BatchNorm) ou Normalização por Camada (LayerNorm).
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int | list[int], 
        num_classes: int, 
        activation_type: ActivationType = "relu", 
        dropout_rate: float = 0.3, 
        use_norm: Optional[NormType] = None
    ):
        super().__init__()

        # Normaliza entrada para lista de tamanhos ocultos
        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = hidden_size

        # Escolhe a classe de ativação
        if activation_type == "relu":
            activation_fn = nn.ReLU
        elif activation_type == "leaky_relu":
            activation_fn = lambda: nn.LeakyReLU(negative_slope=0.01)
        elif activation_type == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Ativação '{activation_type}' não suportada. Use 'relu', 'leaky_relu' ou 'tanh'.")

        layers: list[nn.Module] = []
        in_dim = input_size
        for idx, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h))
            if use_norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif use_norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(activation_fn() if callable(activation_fn) else activation_fn)
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h

        # Camada de saída
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
