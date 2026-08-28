"""
Módulo src/data.py
Consolida a aquisição do dataset HAM10000 (Kaggle), montagem do inventário
particionado por lesão (evitando vazamento entre imagens da mesma lesão),
extração de features do baseline e conversão para DataLoader do PyTorch.
"""
import os
import shutil
import time
import zipfile

import numpy as np
import polars as pl
import torch
from PIL import Image
from skimage.feature import local_binary_pattern
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import log_etapa, log_nota

DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"

# Ordem fixa das classes — define o mapeamento string -> índice inteiro.
# Fixar isso explicitamente (em vez de deixar polars/sklearn inferir a
# ordem sozinho) garante que o mapeamento não mude entre execuções.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASSE_PARA_INDICE = {c: i for i, c in enumerate(CLASSES)}
INDICE_PARA_CLASSE = {i: c for c, i in CLASSE_PARA_INDICE.items()}

TAMANHO_RESIZE = 128
TAMANHO_PADRONIZADO_BASELINE = 128  # mesmo tamanho de entrada da CNN, pra comparação justa
BINS_HISTOGRAMA_COR = 16
RAIO_LBP = 1
PONTOS_LBP = 8  # LBP uniforme com 8 pontos gera 8+2=10 bins

# ---------------------------------------------------------------------------
# Aquisição (Kaggle) e sincronização local
# ---------------------------------------------------------------------------

def autenticar_kaggle(token_path: str):
    """
    Lê o token novo do Kaggle (formato KGAT_...) e autentica.

    O pacote `kaggle` se autentica sozinho no `import`, ao encontrar
    KAGGLE_API_TOKEN no ambiente, e consome essa variável no processo.
    Por isso usamos o objeto `kaggle.api` (já autenticado) em vez de criar
    uma instância própria de KaggleApi().
    """
    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"Token do Kaggle não encontrado em '{token_path}'. "
            "Gere um novo token em kaggle.com/settings/api e salve o arquivo nesse caminho."
        )

    with open(token_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token.startswith("KGAT_"):
        log_nota(
            "Token não começa com 'KGAT_' (formato novo esperado) — "
            "verifique se não é um kaggle.json legado por engano."
        )

    os.environ["KAGGLE_API_TOKEN"] = token

    import kaggle  # import tardio: autentica automaticamente aqui

    log_etapa("autenticacao_kaggle", "Autenticado via KAGGLE_API_TOKEN (formato novo).")
    return kaggle.api

def download_kaggle_dataset(
    dest_dir: str,
    metadata_csv: str,
    token_path: str,
    dataset_slug: str = DATASET_SLUG,
    forcar_novo_download: bool = False,
) -> str:
    """
    Baixa e extrai o HAM10000, se ainda não estiver presente.
    Marca "já presente" pela existência do CSV de metadados, não de uma
    pasta específica — esse dataset extrai direto em dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(metadata_csv) and not forcar_novo_download:
        log_etapa(
            "download_dataset",
            f"Dataset já presente ({os.path.basename(metadata_csv)} encontrado em "
            f"'{dest_dir}'), pulando download.",
        )
        return dest_dir

    api = autenticar_kaggle(token_path)

    log_etapa("download_dataset", f"Baixando '{dataset_slug}' do Kaggle...")
    api.dataset_download_files(dataset_slug, path=dest_dir, quiet=False)

    zips = sorted(
        (os.path.join(dest_dir, f) for f in os.listdir(dest_dir) if f.endswith(".zip")),
        key=os.path.getmtime,
    )
    if not zips:
        raise FileNotFoundError(
            f"Download aparentemente concluiu mas nenhum .zip foi encontrado em '{dest_dir}'."
        )
    zip_path = zips[-1]

    log_etapa("extracao_dataset", f"Extraindo {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if not os.path.exists(metadata_csv):
        arquivos_encontrados = os.listdir(dest_dir)
        raise FileNotFoundError(
            f"Extração concluída mas '{os.path.basename(metadata_csv)}' não foi "
            f"encontrado em '{dest_dir}'. Conteúdo real da pasta: {arquivos_encontrados}. "
            f"Ajuste metadata_csv se o nome do arquivo vier diferente."
        )

    log_nota(
        f"Dataset extraído em '{dest_dir}'. Zip mantido em '{zip_path}' "
        f"(pode apagar depois de confirmar integridade)."
    )
    return dest_dir

def sincronizar_para_local(
    metadata_csv_origem: str,
    images_dir_1_origem: str,
    images_dir_2_origem: str,
    local_dir: str,
    forcar: bool = False,
) -> dict:
    """
    Copia metadata.csv + as duas pastas de imagens do Drive para o disco
    local do runtime (/content). Evita ler milhares de arquivos pequenos
    direto do Drive via FUSE, que é ordens de magnitude mais lento.

    Efêmero por design: some ao reiniciar o runtime, mas é barato refazer.
    Retorna os caminhos locais equivalentes, prontos para uso em
    carregar_inventario.
    """
    local_metadata_csv = os.path.join(local_dir, os.path.basename(metadata_csv_origem))
    local_images_dir_1 = os.path.join(local_dir, os.path.basename(images_dir_1_origem))
    local_images_dir_2 = os.path.join(local_dir, os.path.basename(images_dir_2_origem))

    ja_sincronizado = os.path.exists(local_metadata_csv) and os.path.isdir(local_images_dir_1)
    if ja_sincronizado and not forcar:
        log_etapa("sincronizar_para_local", f"Já sincronizado em '{local_dir}', pulando.")
        return {
            "metadata_csv": local_metadata_csv,
            "images_dir_1": local_images_dir_1,
            "images_dir_2": local_images_dir_2,
        }

    os.makedirs(local_dir, exist_ok=True)
    t0 = time.time()

    shutil.copy2(metadata_csv_origem, local_metadata_csv)
    shutil.copytree(images_dir_1_origem, local_images_dir_1, dirs_exist_ok=True)
    shutil.copytree(images_dir_2_origem, local_images_dir_2, dirs_exist_ok=True)

    dt = time.time() - t0
    log_etapa("sincronizar_para_local", f"Copiado para '{local_dir}' em {dt:.0f}s.")

    return {
        "metadata_csv": local_metadata_csv,
        "images_dir_1": local_images_dir_1,
        "images_dir_2": local_images_dir_2,
    }

# ---------------------------------------------------------------------------
# Inventário + split por lesão
# ---------------------------------------------------------------------------

def _localizar_imagem(image_id: str, pastas_imagens: list[str]) -> str | None:
    """Procura o arquivo da imagem nas pastas informadas, na ordem dada (ex.: local antes de Drive)."""
    for pasta in pastas_imagens:
        candidato = os.path.join(pasta, f"{image_id}.jpg")
        if os.path.exists(candidato):
            return candidato
    return None

def carregar_inventario(metadata_csv: str, pastas_imagens: list[str]) -> pl.DataFrame:
    """
    Lê o CSV de metadados e resolve o caminho físico de cada imagem,
    procurando em todas as pastas informadas (não assume em qual delas
    cada imagem está — verifica de fato no disco).
    """
    df = pl.read_csv(metadata_csv)

    caminhos = [_localizar_imagem(img_id, pastas_imagens) for img_id in df["image_id"].to_list()]
    df = df.with_columns(pl.Series("caminho", caminhos, dtype=pl.Utf8))

    n_total = df.height
    n_sem_arquivo = df.filter(pl.col("caminho").is_null()).height
    n_lesoes_unicas = df["lesion_id"].n_unique()

    log_etapa(
        "carregar_inventario",
        f"{n_total} linhas no metadata.csv, {n_lesoes_unicas} lesion_id "
        f"únicos (múltiplas imagens por lesão são esperadas e documentadas "
        f"pelo dataset).",
    )

    if n_sem_arquivo > 0:
        log_nota(
            f"{n_sem_arquivo} linhas do metadata.csv não têm arquivo de "
            f"imagem correspondente em nenhuma das pastas informadas — checar "
            f"antes de seguir (pode ser download/extração incompleta)."
        )

    return df

def montar_split_por_lesao(
    df: pl.DataFrame,
    proporcoes: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> pl.DataFrame:
    """
    Cria a coluna 'split' (train/val/test), garantindo que todas as
    imagens de um mesmo lesion_id caiam no MESMO split — sem isso, o
    mesmo tipo de vazamento do dataset de pneumonia se repetiria aqui.

    Estratégia: embaralha a lista de lesion_id únicos (não as linhas!)
    com seed fixa, e corta essa lista nas proporções pedidas.

    Nota: não estratifica por classe (dx) neste corte simples — vale
    conferir a distribuição de classes por split depois de rodar isso.
    """
    train_frac, val_frac, test_frac = proporcoes
    assert abs(sum(proporcoes) - 1.0) < 1e-9, "Proporções devem somar 1.0"

    lesoes = df["lesion_id"].unique().sort()
    lesoes_embaralhadas = lesoes.shuffle(seed=seed)

    n = len(lesoes_embaralhadas)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    lesoes_train = set(lesoes_embaralhadas[:n_train].to_list())
    lesoes_val = set(lesoes_embaralhadas[n_train:n_train + n_val].to_list())
    # o resto (n_train+n_val em diante) vai pra teste

    def _rotular(lesion_id: str) -> str:
        if lesion_id in lesoes_train:
            return "train"
        elif lesion_id in lesoes_val:
            return "val"
        return "test"

    df = df.with_columns(
        pl.col("lesion_id").map_elements(_rotular, return_dtype=pl.Utf8).alias("split")
    )

    contagem = df.group_by("split").agg(
        pl.len().alias("n_imagens"),
        pl.col("lesion_id").n_unique().alias("n_lesoes"),
    )
    log_etapa("montar_split_por_lesao", f"Split criado agrupado por lesion_id (seed={seed}):\n{contagem}")

    return df

def checar_vazamento_split(df: pl.DataFrame) -> dict[str, set]:
    """Confirma que nenhum lesion_id aparece em mais de um split."""
    lesoes_por_split = {
        split: set(df.filter(pl.col("split") == split)["lesion_id"])
        for split in df["split"].unique().to_list()
    }

    splits = list(lesoes_por_split.keys())
    vazamentos = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            intersecao = lesoes_por_split[a] & lesoes_por_split[b]
            if intersecao:
                vazamentos[f"{a}_vs_{b}"] = intersecao

    if vazamentos:
        detalhes = ", ".join(f"{k}: {len(v)} lesões" for k, v in vazamentos.items())
        log_nota(
            f"VAZAMENTO DETECTADO no split construído — {detalhes}. "
            f"Isso não deveria acontecer com montar_split_por_lesao; revisar antes de treinar."
        )
    else:
        log_etapa(
            "checagem_vazamento_split",
            "Nenhum vazamento de lesion_id entre splits — split construído corretamente.",
        )

    return vazamentos

def salvar_inventario_particionado(df: pl.DataFrame, path: str) -> None:
    """Persiste o inventário com a coluna 'split' já definida."""
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
    df.write_parquet(path)
    log_etapa("salvar_inventario", f"Inventário particionado salvo em '{path}' ({df.height} linhas).")

def carregar_inventario_particionado(path: str) -> pl.DataFrame:
    """Carrega o inventário particionado salvo, sem precisar refazer o split."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' não existe ainda — rode salvar_inventario_particionado() "
            f"primeiro (ou o pipeline completo: carregar_inventario -> "
            f"montar_split_por_lesao -> salvar_inventario_particionado)."
        )
    df = pl.read_parquet(path)
    log_etapa("carregar_inventario_particionado", f"{df.height} linhas carregadas de '{path}'.")
    return df

# ---------------------------------------------------------------------------
# Ponte para o PyTorch
# ---------------------------------------------------------------------------

class HAM10000Dataset(Dataset):
    """Dataset PyTorch sobre o inventário Polars (uma linha = uma imagem)."""

    def __init__(self, df: pl.DataFrame, transform: transforms.Compose):
        classes_presentes = set(df["dx"].unique().to_list())
        classes_desconhecidas = classes_presentes - set(CLASSES)
        if classes_desconhecidas:
            raise ValueError(
                f"Classes não mapeadas em CLASSES: {classes_desconhecidas}. "
                f"Atualize a constante CLASSES em data.py."
            )

        self.caminhos = df["caminho"].to_list()
        self.rotulos = [CLASSE_PARA_INDICE[c] for c in df["dx"].to_list()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, idx: int):
        img = Image.open(self.caminhos[idx]).convert("RGB")
        img = self.transform(img)
        rotulo = torch.tensor(self.rotulos[idx], dtype=torch.long)
        return img, rotulo

def montar_transform(media_rgb: tuple, desvio_rgb: tuple, tamanho: int = TAMANHO_RESIZE, aumentar_dados: bool = False) -> transforms.Compose:
    """
    Resize direto pro quadrado (distorce a proporção — decisão de
    simplicidade, documentar no relatório) + normalização com as
    estatísticas reais do dataset (calculadas na EDA, não os valores
    padrão do ImageNet).

    aumentar_dados=True acrescenta augmentation geométrico (flip
    horizontal/vertical, rotação até 45°, translação/escala leves) —
    lesões de pele não têm orientação canônica, então essas
    transformações preservam o rótulo. Só faz sentido no split de
    TREINO (ver preparar_dataloaders); aplicar em val/test faria a
    métrica de avaliação variar entre execuções à toa.
    """
    passos = [transforms.Resize((tamanho, tamanho))]
    if aumentar_dados:
        passos += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    passos += [
        transforms.ToTensor(),
        transforms.Normalize(mean=media_rgb, std=desvio_rgb),
    ]
    return transforms.Compose(passos)

def preparar_dataloaders(
    df: pl.DataFrame,
    media_rgb: tuple,
    desvio_rgb: tuple,
    batch_size: int = 32,
    tamanho: int = TAMANHO_RESIZE,
    num_workers: int = 2,
    aumentar_treino: bool = True,
) -> dict:
    """
    Recebe o inventário completo (com coluna 'split') e devolve um dict
    {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.

    shuffle=True só no train — val/test mantêm ordem fixa, para
    diagnóstico e comparação entre execuções não ficarem embaralhados à toa.

    aumentar_treino=True (padrão): o split de treino usa um transform com
    augmentation geométrico (ver montar_transform); val/test usam sempre
    o transform sem augmentation — todo modelo treinado a partir do
    'train' destes DataLoaders (CNN de referência, variante residual,
    LSTM sobre patches) se beneficia igualmente, então uma comparação
    entre arquiteturas continua isolando só a arquitetura.
    """
    transform_treino = montar_transform(media_rgb, desvio_rgb, tamanho, aumentar_dados=aumentar_treino)
    transform_avaliacao = montar_transform(media_rgb, desvio_rgb, tamanho, aumentar_dados=False)

    dataloaders = {}
    for split in ["train", "val", "test"]:
        df_split = df.filter(pl.col("split") == split)
        transform = transform_treino if split == "train" else transform_avaliacao
        dataset = HAM10000Dataset(df_split, transform)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    log_etapa(
        "preparar_dataloaders",
        f"DataLoaders prontos — batch_size={batch_size}, tamanho={tamanho}x{tamanho}, "
        f"augmentation no treino={'ativo' if aumentar_treino else 'inativo'}. "
        f"Tamanhos: " + ", ".join(f"{k}={len(v.dataset)}" for k, v in dataloaders.items()),
    )

    return dataloaders

# ---------------------------------------------------------------------------
# Features artesanais do baseline (cor + textura)
# ---------------------------------------------------------------------------

def extrair_features(caminho_imagem: str) -> np.ndarray:
    """
    Vetor de features = histograma de cor (R, G, B, 16 bins cada,
    normalizado) + histograma de textura LBP (10 bins, normalizado).
    Total: 48 + 10 = 58 features.
    """
    img = Image.open(caminho_imagem).convert("RGB").resize(
        (TAMANHO_PADRONIZADO_BASELINE, TAMANHO_PADRONIZADO_BASELINE)
    )
    arr = np.asarray(img)

    features_cor = []
    for canal in range(3):
        hist, _ = np.histogram(arr[:, :, canal], bins=BINS_HISTOGRAMA_COR, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-8
        features_cor.append(hist)
    features_cor = np.concatenate(features_cor)

    cinza = np.asarray(img.convert("L"))
    lbp = local_binary_pattern(cinza, P=PONTOS_LBP, R=RAIO_LBP, method="uniform")
    n_bins_lbp = PONTOS_LBP + 2
    hist_lbp, _ = np.histogram(lbp, bins=n_bins_lbp, range=(0, n_bins_lbp))
    hist_lbp = hist_lbp.astype(np.float64)
    hist_lbp /= hist_lbp.sum() + 1e-8

    return np.concatenate([features_cor, hist_lbp])

def construir_matriz_features(df: pl.DataFrame) -> tuple:
    """Extrai features de todas as linhas do df. Retorna (X, y)."""
    caminhos = df["caminho"].to_list()
    rotulos_str = df["dx"].to_list()

    X = np.stack([extrair_features(c) for c in caminhos])
    y = np.array([CLASSE_PARA_INDICE[r] for r in rotulos_str])

    log_etapa("construir_matriz_features", f"Features extraídas: X.shape={X.shape}, {len(np.unique(y))} classes.")

    return X, y
