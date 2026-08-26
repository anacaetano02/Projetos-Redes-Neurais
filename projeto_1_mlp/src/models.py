"""
Módulo src/models.py
Agrupa a arquitetura MLP em PyTorch e as referências estatísticas lineares do Scikit-learn.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.utils import log_nota

_NONLINEARITY_HE = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.Tanh: "tanh",
    nn.SELU: "selu",
}

class MLP(nn.Module):
    """
    Perceptron Multicamadas parametrizável.
    Evita o uso de nn.Sequential automático para permitir auditoria explícita de sinais e ativações.
    """
    def __init__(self, input_size: int, camadas_ocultas: list[int], ativacao: type[nn.Module] = nn.ReLU, dropout: float = 0.2, usar_batchnorm: bool = True, inicializacao: str | None = "he"):
        super().__init__()
        
        # Uso de ModuleList para registro explícito de parâmetros no otimizador
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if usar_batchnorm else None
        
        last_dim = input_size
        for dim in camadas_ocultas:
            self.layers.append(nn.Linear(last_dim, dim))
            if usar_batchnorm:
                self.bns.append(nn.BatchNorm1d(dim))
            last_dim = dim
            
        self.output_layer = nn.Linear(last_dim, 1)
        self.activation = ativacao()
        self.dropout = nn.Dropout(p=dropout)
        
        # Inicialização de pesos orientada por teoria matemática.
        # Aplicada só nas camadas ocultas: output_layer não tem ativação
        # subsequente, então a premissa do He/Xavier não se sustenta nela.
        if inicializacao == "he":
            nonlinearity = _NONLINEARITY_HE.get(ativacao, "relu")
            for layer in self.layers:
                self._init_he(layer, nonlinearity)
        elif inicializacao == "xavier":
            for layer in self.layers:
                self._init_xavier(layer)

    def _init_he(self, m, nonlinearity: str = "relu"):
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    def _init_xavier(self, m):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x, retornar_ativacoes: bool = False):
        ativacoes = {}
        out = x
        
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.bns is not None:
                out = self.bns[i](out)
            out = self.activation(out)
            if retornar_ativacoes:
                ativacoes[f"camada_{i}"] = out.clone()
            out = self.dropout(out)
            
        logits = self.output_layer(out)
        if retornar_ativacoes:
            return logits, ativacoes
        return logits

def criar_mlp_classificacao_padrao(input_size: int) -> MLP:
    return MLP(input_size=input_size, camadas_ocultas=[128, 64], ativacao=nn.ReLU, dropout=0.2, usar_batchnorm=True, inicializacao="he")

def criar_mlp_regressao_padrao(input_size: int) -> MLP:
    return MLP(input_size=input_size, camadas_ocultas=[128, 64, 32], ativacao=nn.ReLU, dropout=0.2, usar_batchnorm=True, inicializacao="he")

# Baselines Sklearn
def treinar_baseline_classificacao(X_train, y_train, X_test, y_test) -> dict:
    """Regressão logística como baseline não trivial para risco de crédito."""
    log_nota("Treinando baseline linear de Classificação (Regressão Logística)...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    
    from sklearn.metrics import classification_report, roc_auc_score
    rep = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs)
    
    key_class_1 = "1.0" if "1.0" in rep else "1"
    
    return {
        "accuracy": rep["accuracy"],
        "precision": rep[key_class_1]["precision"] if key_class_1 in rep else 0.0,
        "recall": rep[key_class_1]["recall"] if key_class_1 in rep else 0.0,
        "f1": rep[key_class_1]["f1-score"] if key_class_1 in rep else 0.0,
        "auc": auc
    }

def treinar_baseline_regressao(X_train, y_train, X_test, y_test) -> dict:
    """Regressão linear multivariada como baseline não trivial para taxa de juros."""
    log_nota("Treinando baseline de Regressão (Mínimos Quadrados Clássicos)...")
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    preds = reg.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }