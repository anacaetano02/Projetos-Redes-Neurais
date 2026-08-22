from typing import Optional, Literal, Union, List
import torch
import torch.nn as nn

ActivationType = Literal["relu", "leaky_relu", "tanh"]
NormType = Literal["batch", "layer"]


def _build_activation(activation_type: ActivationType) -> nn.Module:
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif activation_type == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Ativação '{activation_type}' não suportada. Use 'relu', 'leaky_relu' ou 'tanh'."
    )


class LendingClubMLP(nn.Module):
    """
    MLP totalmente conectado com suporte a múltiplas camadas ocultas,
    ativações configuráveis, Dropout e Normalização (BatchNorm ou LayerNorm).
    Mantém as camadas acessíveis individualmente (via self.blocks) para
    facilitar introspecção, hooks e depuração.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, List[int]],
        num_classes: int,
        activation_type: ActivationType = "relu",
        dropout_rate: float = 0.3,
        use_norm: Optional[NormType] = None,
    ):
        super().__init__()

        hidden_sizes = [hidden_size] if isinstance(hidden_size, int) else hidden_size
        if use_norm is not None and use_norm not in ("batch", "layer"):
            raise ValueError(f"Normalização '{use_norm}' não suportada. Use 'batch', 'layer' ou None.")

        self.blocks = nn.ModuleList()
        in_dim = input_size
        for h in hidden_sizes:
            block = nn.ModuleDict({
                "linear": nn.Linear(in_dim, h),
                "norm": (
                    nn.BatchNorm1d(h) if use_norm == "batch"
                    else nn.LayerNorm(h) if use_norm == "layer"
                    else nn.Identity()
                ),
                "activation": _build_activation(activation_type),
                "dropout": nn.Dropout(p=dropout_rate),
            })
            self.blocks.append(block)
            in_dim = h

        self.output_layer = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block["linear"](out)
            out = block["norm"](out)
            out = block["activation"](out)
            out = block["dropout"](out)
        return self.output_layer(out)
