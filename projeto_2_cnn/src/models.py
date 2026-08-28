"""
Módulo src/models.py
Agrupa as arquiteturas do Projeto 2 (CNN de referência, a variante
LSTM-sobre-patches da Etapa 2) e o baseline não trivial de scikit-learn.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Variante residual — teste controlado contra a CNN de referência
# ---------------------------------------------------------------------------

class _BlocoConvSimples(nn.Module):
    """conv3x3(stride=1, sem bias) -> BN -> ReLU -> Dropout2d — não altera resolução espacial (pooling/stride ficam em CNNLesaoPeleResidual.forward, não aqui)."""

    def __init__(self, canais_entrada: int, canais_saida: int, dropout2d: float = 0.25):
        super().__init__()
        self.conv = nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(canais_saida)
        self.dropout = nn.Dropout2d(p=dropout2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.bn(self.conv(x))))

class CNNLesaoPeleResidual(nn.Module):
    """
    Variante residual da CNN de referência (ver CNNLesaoPele) — portada de
    um teste externo para comparação controlada dentro deste pipeline
    (mesmo split, mesma ponderação de classe, mesmo loop de treino e
    avaliação de CNNLesaoPele). Preserva a topologia original do teste
    externo (não é uma reinterpretação "mais limpa" de um ResNet clássico):
    duas convoluções por estágio, com uma conexão de atalho (projeção 1x1)
    somada a cada 2 blocos — não bloco a bloco.

    Duas diferenças estruturais em relação à referência, ambas deliberadas:
    (1) conexões de atalho somadas a cada estágio, para o gradiente fluir
    por uma rede mais funda sem se dissipar; (2) GAP seguido de uma camada
    densa oculta (256 -> 128) antes da saída, em vez de GAP direto pra
    saída — mais parâmetros no classificador, mais capacidade de decisão
    não-linear no topo.

    HIPÓTESE A TESTAR (não teórica — decisão empírica): essa capacidade
    extra melhora o ajuste ou piora por overfitting, dado que o HAM10000
    tem só ~10 mil imagens? Ver seção 10.5 do notebook para o resultado
    medido no conjunto de teste, sob a MESMA infraestrutura de treino da
    referência (nada de comparar contra um loop de treino diferente).
    """

    def __init__(self, num_classes: int = 7, canais_entrada: int = 3, dropout2d: float = 0.25, inicializacao: str = "he"):
        super().__init__()

        self.conv_entrada = nn.Conv2d(canais_entrada, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_entrada = nn.BatchNorm2d(32)

        self.bloco1 = _BlocoConvSimples(32, 64, dropout2d=0.2)
        self.bloco2 = _BlocoConvSimples(64, 128, dropout2d=dropout2d)
        self.atalho1 = nn.Conv2d(32, 128, kernel_size=1, stride=2, bias=False)
        self.bn_atalho1 = nn.BatchNorm2d(128)

        self.bloco3 = _BlocoConvSimples(128, 256, dropout2d=dropout2d + 0.05)
        self.bloco4 = _BlocoConvSimples(256, 256, dropout2d=dropout2d + 0.05)
        self.atalho2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.bn_atalho2 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_oculta = nn.Linear(256, 128)
        self.classificador = nn.Linear(128, num_classes)

        self._inicializar_pesos(inicializacao)

    def _inicializar_pesos(self, metodo: str) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if metodo == "he":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif metodo == "xavier":
                    nn.init.xavier_normal_(m.weight)
                # metodo == "padrao": não faz nada, mantém a inicialização default do PyTorch
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_entrada(self.conv_entrada(x)))
        identidade1 = x

        x = self.bloco1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bloco2(x)

        projecao1 = self.bn_atalho1(self.atalho1(identidade1))
        x = F.relu(x + projecao1)
        identidade2 = x

        x = self.bloco3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bloco4(x)

        projecao2 = self.bn_atalho2(self.atalho2(identidade2))
        x = F.relu(x + projecao2)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_oculta(x))
        return self.classificador(x)

def criar_cnn_residual(num_classes: int = 7) -> CNNLesaoPeleResidual:
    return CNNLesaoPeleResidual(num_classes=num_classes, dropout2d=0.25, inicializacao="he")

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
