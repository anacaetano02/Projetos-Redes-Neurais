"""
Módulo src/data.py
Consolida todo o carregamento Polars, Engenharia de Atributos, One-Hot Encoding,
Particionamento Temporal, Padronização e conversão para DataLoader do PyTorch.
"""
import os
import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils import log_etapa, log_nota

_NAO_FEATURES = {"id", "issue_d", "target_classificacao", "target_regressao"}
_CATEGORICAS = {
    "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status", "application_type", "disbursement_method",
}

def carregar_df_raw(path: str, limiar_nulos_pct: float = 50.0) -> pl.DataFrame:
    """Carrega o DataFrame de forma preguiçosa e remove colunas com alto índice de nulos."""
    lf = pl.scan_csv(path, null_values=["", "NA", "NaN"])
    total_linhas = lf.select(pl.len()).collect().item()
    contagem_nulos = lf.select(pl.all().null_count()).collect()

    colunas_validas = [
        col for col in contagem_nulos.columns
        if (contagem_nulos[col][0] / total_linhas) * 100 < limiar_nulos_pct
    ]

    return lf.select(colunas_validas).collect()

def colunas_features_modelo(df: pl.DataFrame) -> list[str]:
    """Lista todas as colunas de entrada pós-processamento e OHE."""
    return [c for c in df.columns if c not in _NAO_FEATURES]

def colunas_numericas_continuas(df: pl.DataFrame) -> list[str]:
    """Filtra colunas contínuas para análise de skew e normalização."""
    return [
        c for c in df.columns 
        if df[c].dtype in (pl.Float64, pl.Float32) and c not in _NAO_FEATURES
    ]

def tratar_earliest_cr_line(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula o tempo de histórico de crédito em anos de forma robusta."""
    df = df.filter(pl.col("earliest_cr_line").is_not_null())
    df = df.with_columns(
        (((pl.col("issue_d").str.to_date("%b-%Y") - pl.col("earliest_cr_line").str.to_date("%b-%Y"))
          .dt.total_days() / 365.25)
         .clip(lower_bound=0.0))
        .alias("anos_historico_credito")
    )
    return df.drop("earliest_cr_line")

def agrupar_home_ownership(df: pl.DataFrame) -> pl.DataFrame:
    """Consolida classes de propriedade de moradia raras em 'OTHER'."""
    return df.with_columns(
        pl.when(pl.col("home_ownership").is_in(["OTHER", "NONE", "ANY"]))
        .then(pl.lit("OTHER"))
        .otherwise(pl.col("home_ownership"))
        .alias("home_ownership")
    )

def extrair_term_meses(df: pl.DataFrame) -> pl.DataFrame:
    """Converte o prazo do empréstimo em string ('36 months'/'60 months') para numérico."""
    return df.with_columns(
        pl.col("term").str.extract(r"(\d+)").cast(pl.Int32).alias("term_meses")
    ).drop("term")

def mapear_emp_length(df: pl.DataFrame) -> pl.DataFrame:
    """Mapeia tempo de emprego para escala ordinal, criando flag para nulos."""
    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10
    }
    return df.with_columns([
        pl.col("emp_length").replace_strict(emp_map, default=None).cast(pl.Int32).alias("emp_length_num"),
        pl.col("emp_length").is_null().cast(pl.Int8).alias("emp_length_missing")
    ]).drop("emp_length")

def tratar_dti(df: pl.DataFrame) -> pl.DataFrame:
    """Unifica valores sentinelas de dti, aplica imputação e clipping."""
    return df.with_columns([
        pl.when(pl.col("dti").is_in([999, -1])).then(None).otherwise(pl.col("dti")).alias("dti"),
        pl.col("dti").is_null().cast(pl.Int8).alias("dti_missing")
    ])

def tratar_annual_inc(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica transformação log1p e clipping inferior na renda anual para estabilizar skew."""
    mediana = df["annual_inc"].median()
    return df.with_columns(
        pl.col("annual_inc").fill_null(mediana)
    ).with_columns(
        pl.col("annual_inc").clip(lower_bound=100.0).log1p().alias("annual_inc")
    )

def aplicar_transformacoes_por_regra(df: pl.DataFrame) -> pl.DataFrame:
    """Tratamento sistemático de variáveis altamente assimétricas por regras matemáticas."""
    # Para colunas com skew elevado, aplica o log1p para homogeneizar a escala
    for col in df.columns:
        if col in _CATEGORICAS or col in _NAO_FEATURES:
            continue
        if df[col].dtype in (pl.Float64, pl.Int64, pl.Int32):
            if df[col].skew() >= 1.0:
                df = df.with_columns(pl.col(col).log1p())
    return df

def codificar_categoricas(df: pl.DataFrame) -> pl.DataFrame:
    """Executa One-Hot Encoding sínclito para evitar divergência de dimensões entre splits."""
    categoricas_existentes = [c for c in df.columns if c in _CATEGORICAS]
    return df.to_dummies(columns=categoricas_existentes)

def split_temporal(df: pl.DataFrame, frac_treino: float = 0.70, frac_val: float = 0.15) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Realiza o particionamento temporal cronológico baseado em issue_d para simular produção."""
    df_sorted = df.sort("issue_d")
    total_linhas = len(df_sorted)
    
    end_train = int(frac_treino * total_linhas)
    end_val = end_train + int(frac_val * total_linhas)
    
    df_train = df_sorted.slice(0, end_train)
    df_val_set = df_sorted.slice(end_train, end_val - end_train)
    df_test = df_sorted.slice(end_val)
    
    return df_train, df_val_set, df_test

def tratar_nulos_residuais(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Imputa pela mediana de treino qualquer nulo que tenha sobrevivido."""
    for col in df_treino.columns:
        if col in _NAO_FEATURES:
            continue
        tem_nulo = (
            df_treino[col].null_count() > 0
            or df_val[col].null_count() > 0
            or df_test[col].null_count() > 0
        )
        if tem_nulo:
            mediana = df_treino[col].median()
            df_treino = df_treino.with_columns(pl.col(col).fill_null(mediana))
            df_val = df_val.with_columns(pl.col(col).fill_null(mediana))
            df_test = df_test.with_columns(pl.col(col).fill_null(mediana))
    return df_treino, df_val, df_test

def padronizar_numericas(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Aplica normalização Z-score com parâmetros estritamente extraídos do treino."""
    stats = {}
    colunas_cont = colunas_numericas_continuas(df_treino)
    
    for col in colunas_cont:
        mean = df_treino[col].mean()
        std = df_treino[col].std()
        std = std if std != 0 else 1.0
        stats[col] = {"mean": mean, "std": std}
        
        df_treino = df_treino.with_columns(((pl.col(col) - mean) / std).alias(col))
        df_val = df_val.with_columns(((pl.col(col) - mean) / std).alias(col))
        df_test = df_test.with_columns(((pl.col(col) - mean) / std).alias(col))
        
    return df_treino, df_val, df_test, stats

def extrair_arrays(df: pl.DataFrame, colunas_features: list[str]):
    """Converte o DataFrame Polars para arrays NumPy em float32 para o PyTorch."""
    X = df.select(colunas_features).to_numpy().astype("float32")
    y_clf = df["target_classificacao"].to_numpy().astype("float32")
    y_reg = df["target_regressao"].to_numpy().astype("float32")
    return X, y_clf, y_reg

def validar_tensores(tensores: dict[str, torch.Tensor], input_size_esperado: int) -> None:
    """Verifica integridade de dimensões, NaNs e valores infinitos."""
    for nome, tensor in tensores.items():
        assert tensor.shape[1] == input_size_esperado, f"{nome} possui tamanho de colunas incompatível."
        assert not torch.isnan(tensor).any(), f"{nome} possui valores NaNs impeditivos."
        assert not torch.isinf(tensor).any(), f"{nome} possui valores infinitos."

def preparar_dataloaders(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame, batch_size: int = 512) -> dict:
    """Instancia e expõe os DataLoader de treino, validação e teste de forma síncrona."""
    features = colunas_features_modelo(df_treino)
    input_size = len(features)
    
    X_train, y_train_clf, y_train_reg = extrair_arrays(df_treino, features)
    X_val, y_val_clf, y_val_reg = extrair_arrays(df_val, features)
    X_test, y_test_clf, y_test_reg = extrair_arrays(df_test, features)
    
    # Validação cruzada de tensores
    validar_tensores({
        "X_train": torch.from_numpy(X_train),
        "X_val": torch.from_numpy(X_val),
        "X_test": torch.from_numpy(X_test)
    }, input_size)
    
    # Datasets
    ds_train_clf = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_clf))
    ds_train_reg = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_reg))
    
    ds_val_clf = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_clf))
    ds_val_reg = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_reg))
    
    ds_test_clf = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test_clf))
    ds_test_reg = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test_reg))
    
    return {
        "input_size": input_size,
        "classificacao": {
            "train": DataLoader(ds_train_clf, batch_size=batch_size, shuffle=True, drop_last=True),
            "val": DataLoader(ds_val_clf, batch_size=batch_size, shuffle=False),
            "test": DataLoader(ds_test_clf, batch_size=batch_size, shuffle=False),
        },
        "regressao": {
            "train": DataLoader(ds_train_reg, batch_size=batch_size, shuffle=True, drop_last=True),
            "val": DataLoader(ds_val_reg, batch_size=batch_size, shuffle=False),
            "test": DataLoader(ds_test_reg, batch_size=batch_size, shuffle=False),
        }
    }