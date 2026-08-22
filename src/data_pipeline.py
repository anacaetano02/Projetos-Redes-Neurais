import os
from typing import Tuple, Dict, Any
import polars as pl
import subprocess
import torch

def download_kaggle_dataset(
    dataset_handle: str = "wordsforthewise/lending-club",
    target_dir: str = "data/raw",
    filename: str = "accepted_2007_to_2018Q4.csv"
) -> str:
    """
    Baixa o dataset do Lending Club diretamente do Kaggle para `data/raw` (ou `../data/raw`).
    Suporta o comando CLI `kaggle datasets download`.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Lista de possíveis locais onde o arquivo final pode ser encontrado
    possible_paths = [
        os.path.join(target_dir, filename),
        os.path.join(target_dir, "accepted_2007_to_2018q4.csv", filename),
        os.path.join(target_dir, "accepted_2007_to_2018Q4.csv"),
        os.path.join("..", "data", "raw", filename),
        os.path.join("..", "data", "raw", "accepted_2007_to_2018q4.csv", filename),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[DataPipeline] Arquivo local encontrado em '{path}'. Ignorando download.")
            return path

    print(f"[DataPipeline] Baixando dataset '{dataset_handle}' do Kaggle em '{target_dir}'...")
    
    # Comando CLI `kaggle datasets download -d ... -p ... --unzip`
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_handle, "-p", target_dir, "--unzip"]
        subprocess.run(cmd, check=True)
        print("Download concluído!")
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
    except Exception as e:
        print(f"[DataPipeline] Não foi possível executar o download via CLI do Kaggle ({e}).")

    raise FileNotFoundError(
        f"Não foi possível localizar ou baixar o dataset '{dataset_handle}' em nenhum dos caminhos esperados."
    )

def pipeline_dados_lending_club_completo(
    csv_path: str, 
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Carrega, limpa e divide a base em 70% treino, 15% validação e 15% teste.
    Utiliza avaliação preguiçosa (LazyFrame) do Polars para otimização de memória
    e aplica escala Z-score somente no treino.
    """
    
    selected_cols = [
        'loan_amnt',                # Valor solicitado
        'installment',              # Parcela mensal
        'annual_inc',               # Renda anual
        'dti',                      # Debt-to-income ratio
        'delinq_2yrs',              # Inadimplências nos últimos 2 anos
        'fico_range_low',           # Score FICO (limite inferior)
        'open_acc',                 # Contas abertas
        'pub_rec',                  # Registros públicos negativos
        'revol_bal',                # Saldo rotativo
        'revol_util',               # Utilização crédito rotativo (%)
        'total_acc',                # Total de linhas de crédito
        'mort_acc',                 # Número de hipotecas
        'pub_rec_bankruptcies',     # Falências registradas
        'term',                     # Prazo (36 ou 60 meses)
        'grade',                    # Nota de crédito (A-G)
        'emp_length',               # Tempo de emprego
        'home_ownership',           # Situação habitacional
        'verification_status',      # Status de verificação de renda
        'purpose',                  # Finalidade do empréstimo
        'application_type',         # Individual ou joint
        'loan_status',              # Status atual/final do empréstimo
        'int_rate'                  # Taxa de juros
    ]
    
    num_features = [
        'loan_amnt', 
        'installment', 
        'annual_inc', 
        'dti', 
        'delinq_2yrs',
        'fico_range_low', 
        'open_acc', 
        'pub_rec', 
        'revol_bal', 
        'revol_util',
        'total_acc', 
        'mort_acc', 
        'pub_rec_bankruptcies'
    ]
    
    cat_features = [
        'term', 
        'grade', 
        'emp_length', 
        'home_ownership', 
        'verification_status', 
        'purpose', 
        'application_type'
    ]
    
    
    # 1. Pipeline Preguiçoso (LazyFrame) - Otimiza filtragem e transformações antes do carregamento em memória
    lazy_df = (
        pl.scan_csv(csv_path, ignore_errors=True)
        .select(selected_cols)
        .filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off"]))
        .with_columns([
            pl.col(col).fill_null("unknown").cast(pl.Utf8) for col in cat_features
        ])
        .with_columns(
            pl.when(pl.col("loan_status") == "Fully Paid").then(0).otherwise(1).alias("classification_target")
        )
    )
    
    # 2. One-Hot Encoding (OHE) no DataFrame
    df_encoded = lazy_df.collect().to_dummies(columns=cat_features)

    # 3. Divisão Física Estrita (70% Treino / 15% Validação / 15% Teste)
    df_shuffled = df_encoded.sample(fraction=1.0, seed=seed)

    total_rows = len(df_shuffled)
    train_end = int(0.70 * total_rows)
    val_end = train_end + int(0.15 * total_rows)
    
    df_train = df_shuffled.slice(0, train_end)
    df_val = df_shuffled.slice(train_end, val_end - train_end)
    df_test = df_shuffled.slice(val_end)
    
    # 4. Normalização e Imputação Numérica somente na base treino
    train_stats: Dict[str, Dict[str, Any]] = {}
    for col in num_features:
        std_val = df_train[col].std()
        train_stats[col] = {
            "median": df_train[col].median(),
            "mean": df_train[col].mean(),
            "std": std_val if (std_val is not None and std_val != 0) else 1.0
        }
    

    def aplicar_imputacao_e_escala(df_partition: pl.DataFrame, stats: Dict[str, Dict[str, Any]]) -> pl.DataFrame:
        # Executa imputação de mediana e Z-score em uma única passada de expressões paralelas
        expressions = [
            ((pl.col(col).fill_null(stats[col]["median"]) - stats[col]["mean"]) / stats[col]["std"]).alias(col)
            for col in num_features
        ]
        return df_partition.with_columns(expressions)
        
    df_train_scaled = aplicar_imputacao_e_escala(df_train, train_stats)
    df_val_scaled = aplicar_imputacao_e_escala(df_val, train_stats)
    df_test_scaled = aplicar_imputacao_e_escala(df_test, train_stats)
    
    print("\n--- Pipeline de Dados Polars Concluído (Lazy Otimizado) ---")
    print(f"Colunas totais pós-OHE: {df_train_scaled.width}")
    print(f"Linhas -> Treino: {df_train_scaled.height} | Val: {df_val_scaled.height} | Teste: {df_test_scaled.height}")
        
    return df_train_scaled, df_val_scaled, df_test_scaled



def calcular_pesos_classe(
    df_train: pl.DataFrame,
    target_col: str = "classification_target",
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Calcula pesos de classe inversamente proporcionais à frequência de cada classe
    no conjunto de TREINO, para uso em nn.CrossEntropyLoss(weight=...).
 
    Fórmula (balanceamento "inverse frequency"):
        peso_da_classe = total_amostras / (num_classes * amostras_da_classe)
 
    Isso faz com que erros na classe minoritária ("Charged Off") pesem mais na loss,
    corrigindo a tendência do modelo de simplesmente prever sempre a classe majoritária.
 
    IMPORTANTE: calcule sempre a partir de `df_train` (nunca de val/teste), para não
    vazar informação sobre a distribuição de val/teste no treino.
    """
    total = df_train.height
    weights = []
    counts_log: Dict[int, int] = {}
 
    for class_id in range(num_classes):
        n_class = df_train.filter(pl.col(target_col) == class_id).height
        n_class_safe = max(n_class, 1)  # evita divisão por zero se a classe não aparecer no treino
        weight = total / (num_classes * n_class_safe)
        weights.append(weight)
        counts_log[class_id] = n_class
 
    print(f"[DataPipeline] Contagem por classe (treino): {counts_log}")
    print(f"[DataPipeline] Pesos de classe calculados: {dict(enumerate(weights))}")
 
    return torch.tensor(weights, dtype=torch.float32)
