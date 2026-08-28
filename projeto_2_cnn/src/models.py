"""
Módulo src/models.py
Agrupa as arquiteturas do Projeto 2 (CNN de referência, a variante
LSTM-sobre-patches da Etapa 2) e o baseline não trivial de scikit-learn.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils import log_etapa

# ---------------------------------------------------------------------------
# CNN de referência (v1)
# ---------------------------------------------------------------------------

class BlocoConvolucional(nn.Module):
    """conv3x3 -> BN -> ativação -> conv3x3 -> BN -> ativação -> pool -> Dropout2d."""

    def __init__(
        self,
        canais_entrada: int,
        canais_saida: int,
        usar_batchnorm: bool = True,
        dropout2d: float = 0.1,
    ):
        super().__init__()

        camadas = [
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1),
        ]
        if usar_batchnorm:
            camadas.append(nn.BatchNorm2d(canais_saida))
        camadas.append(nn.LeakyReLU(inplace=True))

        camadas.append(nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1))
        if usar_batchnorm:
            camadas.append(nn.BatchNorm2d(canais_saida))
        camadas.append(nn.LeakyReLU(inplace=True))

        camadas.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if dropout2d > 0:
            camadas.append(nn.Dropout2d(dropout2d))

        self.bloco = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloco(x)

class CNNLesaoPele(nn.Module):
    """
    CNN de referência para classificação das 7 classes do HAM10000.

    - Filtros: progressão conservadora (padrão [16, 32, 64, 128]) — cada
      valor da lista é um bloco convolucional (Conv->BN->Ativação x2 ->
      MaxPool -> Dropout2d espacial, já que pixels vizinhos em mapas de
      feature são correlacionados; dropout por pixel isolado é pouco eficaz).
    - Global Average Pooling em vez de Flatten+Dense, para não gerar
      milhões de parâmetros numa única camada densa com um dataset modesto
      (~7000 imagens).
    - Dropout comum só depois do GAP, antes da camada de saída.

    filtros: lista com o nº de canais de saída de cada bloco. Tamanho da
    lista = número de blocos (profundidade).
    """

    def __init__(
        self,
        num_classes: int = 7,
        filtros: list = None,
        usar_batchnorm: bool = True,
        dropout2d: float = 0.1,
        dropout_final: float = 0.3,
        canais_entrada: int = 3,
        inicializacao: str = "he",
    ):
        super().__init__()

        filtros = filtros or [16, 32, 64, 128]

        blocos = []
        canais_anterior = canais_entrada
        for n_filtros in filtros:
            blocos.append(
                BlocoConvolucional(canais_anterior, n_filtros, usar_batchnorm, dropout2d)
            )
            canais_anterior = n_filtros

        self.blocos_conv = nn.Sequential(*blocos)

        # Global Average Pooling: reduz (C, H, W) -> (C, 1, 1) -> (C,)
        # em vez de Flatten, que geraria C * H * W parâmetros na próxima camada.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classificador = nn.Sequential(
            nn.Dropout(dropout_final),
            nn.Linear(canais_anterior, num_classes),
        )

        self._inicializar_pesos(inicializacao)

    def _inicializar_pesos(self, metodo: str) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if metodo == "he":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                elif metodo == "xavier":
                    nn.init.xavier_normal_(m.weight)
                # metodo == "padrao": não faz nada, mantém a inicialização default do PyTorch
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocos_conv(x)        # (B, C_final, H_final, W_final)
        x = self.gap(x)                # (B, C_final, 1, 1)
        x = x.flatten(1)               # (B, C_final) — só achata o GAP, não o mapa inteiro
        return self.classificador(x)   # (B, num_classes) — logits, sem softmax (aplicar na loss)

def resumo_parametros(modelo: nn.Module) -> dict:
    """
    Conta parâmetros treináveis, separando quanto vem dos blocos
    convolucionais vs. do classificador final — confirma na prática
    o quanto o GAP economizou em relação a um Flatten+Dense.
    """
    total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    conv = sum(p.numel() for p in modelo.blocos_conv.parameters() if p.requires_grad)
    classificador = sum(p.numel() for p in modelo.classificador.parameters() if p.requires_grad)

    return {"total": total, "blocos_conv": conv, "classificador": classificador}

def criar_cnn_padrao(num_classes: int = 7) -> CNNLesaoPele:
    return CNNLesaoPele(
        num_classes=num_classes,
        filtros=[16, 32, 64, 128],
        usar_batchnorm=True,
        dropout2d=0.1,
        dropout_final=0.3,
        inicializacao="he",
    )

# ---------------------------------------------------------------------------
# Variante "Etapa 2" — LSTM sobre sequência de patches (teste ilustrativo)
# ---------------------------------------------------------------------------

class PatchSequenceLSTM(nn.Module):
    """
    Implementação simplificada da estratégia de Abohashish et al. (2025):
    divide a imagem em patches, embute cada patch num vetor (aqui, uma
    única camada linear — simplificado de propósito em relação ao artigo
    original, que usa uma CNN por patch), processa a sequência com uma
    LSTM e classifica a partir do estado oculto final.

    SIMPLIFICADO DE PROPÓSITO: suficiente para o teste ilustrativo
    (confirmar que a estruturação em sequência funciona e observar o
    comportamento de treino), não para buscar um resultado competitivo
    com a CNN de referência — esse não é o objetivo deste modelo.
    """

    def __init__(
        self,
        num_classes: int = 7,
        tamanho_imagem: int = 128,
        tamanho_patch: int = 16,
        canais_entrada: int = 3,
        dim_embedding: int = 128,
        hidden_lstm: int = 128,
        num_camadas_lstm: int = 1,
        dropout_final: float = 0.3,
    ):
        super().__init__()

        assert tamanho_imagem % tamanho_patch == 0, (
            f"tamanho_imagem ({tamanho_imagem}) precisa ser divisível por "
            f"tamanho_patch ({tamanho_patch})."
        )

        self.tamanho_patch = tamanho_patch
        self.canais_entrada = canais_entrada
        patch_dim = canais_entrada * tamanho_patch * tamanho_patch
        self.n_patches = (tamanho_imagem // tamanho_patch) ** 2

        self.embedding_patch = nn.Linear(patch_dim, dim_embedding)

        self.lstm = nn.LSTM(
            input_size=dim_embedding,
            hidden_size=hidden_lstm,
            num_layers=num_camadas_lstm,
            batch_first=True,
        )

        self.classificador = nn.Sequential(
            nn.Dropout(dropout_final),
            nn.Linear(hidden_lstm, num_classes),
        )

    def _extrair_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N_patches, C*patch*patch), ordem de varredura linha a linha."""
        B, C, H, W = x.shape
        p = self.tamanho_patch

        patches = x.unfold(2, p, p).unfold(3, p, p)             # (B, C, H/p, W/p, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)      # (B, C, N, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()    # (B, N, C, p, p)
        patches = patches.view(B, self.n_patches, -1)            # (B, N, C*p*p)

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._extrair_patches(x)               # (B, N, patch_dim)
        embeddings = self.embedding_patch(patches)        # (B, N, dim_embedding)

        _, (h_n, _) = self.lstm(embeddings)                # h_n: (num_camadas, B, hidden_lstm)
        estado_final = h_n[-1]                             # última camada: (B, hidden_lstm)

        return self.classificador(estado_final)            # (B, num_classes)

def criar_lstm_patches_padrao(num_classes: int = 7, tamanho_imagem: int = 128) -> PatchSequenceLSTM:
    return PatchSequenceLSTM(
        num_classes=num_classes,
        tamanho_imagem=tamanho_imagem,
        tamanho_patch=16,
        dim_embedding=128,
        hidden_lstm=128,
        num_camadas_lstm=1,
        dropout_final=0.3,
    )

# ---------------------------------------------------------------------------
# Baseline não trivial (scikit-learn)
# ---------------------------------------------------------------------------

def treinar_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> dict:
    """
    Regressão logística multinomial com class_weight='balanced' sobre
    features artesanais de cor+textura (ver data.construir_matriz_features)
    — comparação "conhecimento de domínio codificado à mão" vs. "features
    aprendidas automaticamente pela CNN", não só "simples vs. complexo".

    Se X_val/y_val forem passados, calcula acurácia também na validação
    (diagnóstico rápido antes de ir para o teste).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelo = LogisticRegression(class_weight="balanced", max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    acc_train = modelo.score(X_train_scaled, y_train)
    log_etapa("treinar_baseline", f"Acurácia no treino: {acc_train:.4f}")

    resultado = {"modelo": modelo, "scaler": scaler, "acc_train": acc_train}

    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        acc_val = modelo.score(X_val_scaled, y_val)
        resultado["acc_val"] = acc_val
        log_etapa("treinar_baseline", f"Acurácia na validação: {acc_val:.4f}")

    return resultado
