import os
import shutil
import subprocess
from typing import Tuple, Dict, Any
import polars as pl


SELECTED_COLS = [
    'loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
    'fico_range_low', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'term', 'grade',
    'emp_length', 'home_ownership', 'verification_status', 'purpose',
    'application_type', 'loan_status', 'int_rate'
]

NUMERICAL_FEATURES = [
    'loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
    'fico_range_low', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'mort_acc', 'pub_rec_bankruptcies'
]

CATEGORICAL_FEATURES = [
    'term', 'grade', 'emp_length', 'home_ownership', 
    'verification_status', 'purpose', 'application_type'
]


def download_kaggle_dataset(
    dataset_handle: str = "wordsforthewise/lending-club",
    target_dir: str = "data/raw",
    filename: str = "accepted_2007_to_2018Q4.csv"
) -> str:
    """
    Baixa o dataset do Lending Club diretamente do Kaggle para `data/raw` (ou `../data/raw`).
    Suporta o comando CLI `kaggle datasets download`, extração e fallback para base simulada.
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
    
    # 1. Tenta comando CLI `kaggle datasets download -d ... -p ... --unzip`
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_handle, "-p", target_dir, "--unzip"]
        subprocess.run(cmd, check=True)
        print("Download concluído!")
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
    except Exception as e:
        print(f"[DataPipeline] Não foi possível executar o download via CLI do Kaggle ({e}).")

    # 2. Tenta via kagglehub como alternativa
    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_handle)
        print(f"[DataPipeline] Dataset baixado via kagglehub em: {path}")
        
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            downloaded_csv = os.path.join(path, csv_files[0])
            for f in csv_files:
                if "accepted" in f.lower():
                    downloaded_csv = os.path.join(path, f)
                    break
            
            target_path = os.path.join(target_dir, filename)
            shutil.copyfile(downloaded_csv, target_path)
            print(f"[DataPipeline] Arquivo CSV copiado para: '{target_path}'")
            return target_path
    except Exception as e:
        print(f"[DataPipeline] Não foi possível baixar via kagglehub ({e}).")

    # 3. Fallback: Base Simulada para garantia de execução
    print("[DataPipeline] AVISO: Credenciais do Kaggle não configuradas. Gerando base simulada como fallback...")
    fallback_path = os.path.join(target_dir, "accepted_mock_22_cols.csv")
    return gerar_base_simulada(fallback_path)


def gerar_base_simulada(csv_path: str = "data/accepted_mock_22_cols.csv") -> str:
    """
    Gera um arquivo CSV mock para testes do pipeline se ele ainda não existir.
    """
    if os.path.exists(csv_path):
        return csv_path

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    mock_data = {
        'loan_amnt': [10000.0, 15000.0, 20000.0, 5000.0, 35000.0] * 200,
        'installment': [322.0, 480.0, 600.0, 160.0, 1100.0] * 200,
        'annual_inc': [60000.0, 72000.0, 110000.0, 45000.0, 150000.0] * 200,
        'dti': [15.5, 20.1, 8.2, 5.0, 28.4] * 200,
        'delinq_2yrs': [0.0, 1.0, 0.0, 0.0, 2.0] * 200,
        'fico_range_low': [720.0, 680.0, 750.0, 710.0, 640.0] * 200,
        'open_acc': [10.0, 12.0, 15.0, 6.0, 22.0] * 200,
        'pub_rec': [0.0, 0.0, 0.0, 1.0, 0.0] * 200,
        'revol_bal': [12000.0, 18000.0, 25000.0, 3000.0, 54000.0] * 200,
        'revol_util': [45.2, 60.5, 12.1, 15.0, 89.9] * 200,
        'total_acc': [24.0, 30.0, 35.0, 12.0, 50.0] * 200,
        'mort_acc': [1.0, 2.0, 4.0, 0.0, 5.0] * 200,
        'pub_rec_bankruptcies': [0.0, 0.0, 0.0, 1.0, 0.0] * 200,
        'term': [" 36 months", " 60 months", " 36 months", " 36 months", " 60 months"] * 200,
        'grade': ["B", "C", "A", "B", "E"] * 200,
        'emp_length': ["10+ years", "2 years", "3 years", "< 1 year", "10+ years"] * 200,
        'home_ownership': ["MORTGAGE", "RENT", "OWN", "RENT", "MORTGAGE"] * 200,
        'verification_status': ["Source Verified", "Verified", "Not Verified", "Source Verified", "Verified"] * 200,
        'purpose': ["debt_consolidation", "credit_card", "home_improvement", "other", "debt_consolidation"] * 200,
        'application_type': ["Individual", "Individual", "Joint App", "Individual", "Individual"] * 200,
        'loan_status': ["Fully Paid", "Charged Off", "Fully Paid", "Fully Paid", "Charged Off"] * 200,
        'int_rate': [0.115, 0.142, 0.079, 0.089, 0.185] * 200
    }
    pl.DataFrame(mock_data).write_csv(csv_path)
    print(f"[DataPipeline] Base simulada gravada com sucesso em: '{csv_path}'")
    return csv_path


def pipeline_dados_lending_club_completo(
    csv_path: str, 
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Carrega, limpa e divide a base em 70% treino, 15% validação e 15% teste.
    Utiliza avaliação preguiçosa (LazyFrame) do Polars para otimização de memória
    e aplica escala Z-score sem vazamento de informação (data leakage).
    """
    # 1. Pipeline Preguiçoso (LazyFrame) - Otimiza filtragem e transformações antes do carregamento em memória
    lazy_df = (
        pl.scan_csv(csv_path, ignore_errors=True)
        .select(SELECTED_COLS)
        .filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off"]))
        .with_columns([
            pl.col(col).fill_null("unknown").cast(pl.Utf8) for col in CATEGORICAL_FEATURES
        ])
        .with_columns(
            pl.when(pl.col("loan_status") == "Fully Paid").then(0).otherwise(1).alias("classification_target")
        )
    )
    
    # Materializa apenas após a construção do plano otimizado
    df = lazy_df.collect()
    
    # 2. One-Hot Encoding (OHE) no DataFrame
    df_encoded = df.to_dummies(columns=CATEGORICAL_FEATURES)
    
    # 3. Divisão Física Estrita (70% Treino / 15% Validação / 15% Teste)
    df_shuffled = df_encoded.sample(fraction=1.0, seed=seed)
    
    total_rows = len(df_shuffled)
    train_end = int(0.70 * total_rows)
    val_end = train_end + int(0.15 * total_rows)
    
    df_train = df_shuffled.slice(0, train_end)
    df_val = df_shuffled.slice(train_end, val_end - train_end)
    df_test = df_shuffled.slice(val_end)
    
    # 4. Normalização e Imputação Numérica Isenta de Data Leakage
    train_stats: Dict[str, Dict[str, Any]] = {}
    for col in NUMERICAL_FEATURES:
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
            for col in NUMERICAL_FEATURES
        ]
        return df_partition.with_columns(expressions)
        
    df_train_scaled = aplicar_imputacao_e_escala(df_train, train_stats)
    df_val_scaled = aplicar_imputacao_e_escala(df_val, train_stats)
    df_test_scaled = aplicar_imputacao_e_escala(df_test, train_stats)
    
    print("\n--- Pipeline de Dados Polars Concluído (Lazy Otimizado) ---")
    print(f"Colunas totais pós-OHE: {df_train_scaled.width}")
    print(f"Linhas -> Treino: {df_train_scaled.height} | Val: {df_val_scaled.height} | Teste: {df_test_scaled.height}")
    
    return df_train_scaled, df_val_scaled, df_test_scaled

