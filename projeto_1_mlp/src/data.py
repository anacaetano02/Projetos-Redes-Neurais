"""
Módulo src/data.py
Consolida o carregamento Polars, a montagem do dataset de trabalho
(seleção de colunas + criação dos targets), engenharia de atributos,
particionamento temporal, padronização e conversão para DataLoader do
PyTorch. Não cobre aquisição (download do Kaggle) — isso fica no notebook.
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

STATUS_DESFECHO_DEFINIDO = [
    "Fully Paid", "Charged Off", "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]
STATUS_INADIMPLENCIA = [
    "Charged Off", "Does not meet the credit policy. Status:Charged Off", "Default",
]

def carregar_df_raw(path: str, limiar_nulos_pct: float = 50.0) -> pl.DataFrame:
    """Carrega o DataFrame de forma preguiçosa e remove colunas com alto índice de nulos."""
    # ignore_errors=True: o export do Lending Club tem linhas de
    # rodapé/resumo (ex.: "Total amount funded in policy code...") que não
    # batem com o schema das colunas de dado — sem isso, o parser quebra ao
    # encontrar a primeira linha malformada em vez de só descartá-la.
    lf = pl.scan_csv(path, null_values=["", "NA", "NaN"], ignore_errors=True)
    total_linhas = lf.select(pl.len()).collect().item()
    contagem_nulos = lf.select(pl.all().null_count()).collect()

    colunas_validas = [
        col for col in contagem_nulos.columns
        if (contagem_nulos[col][0] / total_linhas) * 100 < limiar_nulos_pct
    ]

    log_nota(
        "ignore_errors=True está ativo no scan_csv — linhas malformadas "
        "(ex.: rodapés/resumos do export do Lending Club) são descartadas "
        f"silenciosamente durante o parsing. Total de linhas efetivamente "
        f"lidas: {total_linhas}."
    )

    return lf.select(colunas_validas).collect()

def montar_dataset_bruto(df_raw: pl.DataFrame, colunas_trabalho: list[str]) -> pl.DataFrame:
    """
    Seleciona a lista de colunas curada e cria os targets a partir de
    loan_status (inadimplência) e int_rate (taxa de juros) — o CSV bruto
    do Lending Club não tem target_classificacao/target_regressao prontos.

    Mantém só empréstimos com desfecho definido (Fully Paid/Charged Off/
    Default e as variantes "Does not meet the credit policy") — Current,
    Late e In Grace Period não têm rótulo válido de inadimplência ainda.

    colunas_trabalho precisa incluir 'loan_status' e 'int_rate' (consumidos
    aqui e descartados) além de 'id'/'issue_d' (preservados — id para
    rastreabilidade, issue_d para anos_historico_credito e o split temporal).
    """
    colunas_disponiveis = [c for c in colunas_trabalho if c in df_raw.columns]
    colunas_faltando = sorted(set(colunas_trabalho) - set(colunas_disponiveis))
    if colunas_faltando:
        log_nota(
            f"{len(colunas_faltando)} coluna(s) da lista curada não estavam em df_raw "
            f"(removidas pelo filtro de nulos ou ausentes no schema): {colunas_faltando}"
        )

    dist_status = df_raw.group_by("loan_status").len().sort("len", descending=True)
    log_etapa("Distribuição de loan_status (antes do filtro de desfecho definido)", dist_status)

    df = (
        df_raw
        .select(colunas_disponiveis)
        .filter(pl.col("loan_status").is_in(STATUS_DESFECHO_DEFINIDO))
        .with_columns(
            pl.col("loan_status").is_in(STATUS_INADIMPLENCIA).cast(pl.Int8).alias("target_classificacao"),
            pl.col("int_rate").alias("target_regressao"),
        )
        .drop(["loan_status", "int_rate"])
    )

    log_etapa("Dataset de trabalho após seleção de colunas e criação dos targets", f"shape: {df.shape}")
    return df

def colunas_features_modelo(df: pl.DataFrame) -> list[str]:
    """Lista todas as colunas de entrada pós-processamento e OHE."""
    return [c for c in df.columns if c not in _NAO_FEATURES]

def colunas_numericas_continuas(df: pl.DataFrame, excluir_extra: set[str] | None = None) -> list[str]:
    """
    Lista as colunas numéricas CONTÍNUAS (só Float64) — deliberadamente
    exclui Int8 (flags binárias: emp_length_missing, dti_missing,
    *_indisponivel, *_teve_evento) e as dummies do one-hot (UInt8/Int8):
    correlação de Pearson entre indicadores esparsos infla o coeficiente
    pela coincidência estrutural de zeros, não pela redundância real de
    informação, e não faz sentido padronizar (z-score) uma variável 0/1.
    """
    excluir = _NAO_FEATURES | _CATEGORICAS | (excluir_extra or set())
    return [c for c in df.columns if df[c].dtype == pl.Float64 and c not in excluir]

# ---------------------------------------------------------------------------
# Limpeza e features derivadas
# ---------------------------------------------------------------------------

def tratar_earliest_cr_line(df: pl.DataFrame) -> pl.DataFrame:
    """
    Deriva anos_historico_credito a partir de (issue_d - earliest_cr_line).
    Remove linhas com earliest_cr_line nula. Clipping em [0, 70] anos —
    piso contra earliest_cr_line posterior a issue_d (erro de digitação
    nos dados-fonte); teto contra valores fisicamente implausíveis.
    """
    n_antes = df.height
    df = df.filter(pl.col("earliest_cr_line").is_not_null())
    log_nota(
        f"earliest_cr_line nula: {n_antes - df.height} linhas removidas "
        f"({(n_antes - df.height) / n_antes * 100:.3f}% da base)"
    )

    df = df.with_columns(
        ((pl.col("issue_d").str.to_date("%b-%Y") - pl.col("earliest_cr_line").str.to_date("%b-%Y"))
         .dt.total_days() / 365.25)
        .alias("anos_historico_credito")
    )

    n_outliers = df.filter(pl.col("anos_historico_credito") > 50).height
    log_nota(
        f"{n_outliers} linhas com anos_historico_credito > 50 anos "
        f"({n_outliers / df.height * 100:.3f}%) — clipping aplicado em [0, 70]."
    )

    df = df.with_columns(pl.col("anos_historico_credito").clip(lower_bound=0.0, upper_bound=70.0))
    log_etapa("anos_historico_credito - describe() final", df["anos_historico_credito"].describe())

    return df.drop("earliest_cr_line")

def agrupar_home_ownership(df: pl.DataFrame) -> pl.DataFrame:
    """Consolida ANY/OTHER/NONE em uma única categoria 'OTHER'."""
    antes = df["home_ownership"].value_counts()
    df = df.with_columns(
        pl.when(pl.col("home_ownership").is_in(["OTHER", "NONE", "ANY"]))
        .then(pl.lit("OTHER"))
        .otherwise(pl.col("home_ownership"))
        .alias("home_ownership")
    )
    log_etapa("home_ownership - antes do agrupamento", antes)
    log_etapa("home_ownership - depois do agrupamento", df["home_ownership"].value_counts())
    return df

def extrair_term_meses(df: pl.DataFrame) -> pl.DataFrame:
    """Converte o prazo do empréstimo em string ('36 months'/'60 months') para numérico."""
    return df.with_columns(
        pl.col("term").str.extract(r"(\d+)").cast(pl.Int32).alias("term_meses")
    ).drop("term")

def mapear_emp_length(df: pl.DataFrame) -> pl.DataFrame:
    """
    emp_length mapeada para escala ordinal (preserva ordem — diferente de
    One-Hot, que destruiria a noção de "mais"/"menos" tempo de emprego).
    Nulos E categorias não reconhecidas viram emp_length_num=0 + flag
    emp_length_missing=1 (só nulos verdadeiros marcam a flag; a coluna
    'bogus'/inesperada some junto do cast, mas isso não deveria acontecer
    com o schema conhecido do Lending Club).
    """
    emp_length_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10,
    }

    n_nulos = df["emp_length"].null_count()
    log_nota(
        f"emp_length: {n_nulos} nulos ({n_nulos / df.height * 100:.2f}%) "
        f"tratados como flag emp_length_missing, não como imputação silenciosa."
    )

    return df.with_columns(
        pl.col("emp_length").is_null().cast(pl.Int8).alias("emp_length_missing"),
        pl.col("emp_length").replace(emp_length_map).cast(pl.Float64, strict=False)
            .fill_null(0).alias("emp_length_num"),
    ).drop("emp_length")

def tratar_dti(df: pl.DataFrame) -> pl.DataFrame:
    """
    dti: sentinelas (999, -1) e nulos verdadeiros unificados em null, com
    flag + imputação pela mediana, seguido de clipping no p999 para
    outliers genuínos de cálculo (renda declarada próxima de zero).
    """
    skew_inicial = df["dti"].skew()
    n_sentinelas = df.filter(pl.col("dti").is_in([999, -1])).height

    df = df.with_columns(
        pl.when(pl.col("dti").is_in([999, -1])).then(None).otherwise(pl.col("dti")).alias("dti")
    )
    df = df.with_columns(
        pl.col("dti").is_null().cast(pl.Int8).alias("dti_missing"),
        pl.col("dti").fill_null(pl.col("dti").median()).alias("dti"),
    )

    p999 = df["dti"].quantile(0.999)
    df = df.with_columns(pl.col("dti").clip(upper_bound=p999))
    skew_final = df["dti"].skew()

    log_nota(
        f"dti: skew inicial {skew_inicial:.2f} -> {n_sentinelas} sentinelas (999/-1) "
        f"removidos -> clipping em p999={p999:.2f} -> skew final {skew_final:.2f}."
    )
    return df

def tratar_annual_inc(df: pl.DataFrame) -> pl.DataFrame:
    """
    annual_inc: cauda superior genuína, tratada com log1p. log1p sozinho
    não resolve — sobra uma distorção na cauda INFERIOR (rendas declaradas
    próximas de zero), tratada com clipping no percentil 0,1% já em escala
    log. Nulos residuais (se houver) ficam para tratar_nulos_residuais,
    igual a qualquer outra coluna — imputação de nulo não é preocupação
    desta função, só o formato da distribuição.
    """
    skew_antes = df["annual_inc"].skew()
    df = df.with_columns(pl.col("annual_inc").log1p().alias("annual_inc"))
    skew_pos_log = df["annual_inc"].skew()

    p_baixo = df["annual_inc"].quantile(0.001)
    df = df.with_columns(pl.col("annual_inc").clip(lower_bound=p_baixo))
    skew_final = df["annual_inc"].skew()

    log_nota(
        f"annual_inc: skew {skew_antes:.2f} -> {skew_pos_log:.2f} (log1p) -> "
        f"{skew_final:.2f} (clipping inferior em p_baixo={p_baixo:.2f})."
    )
    return df

# ---------------------------------------------------------------------------
# Transformação genérica por regra (skew / zero_pct)
# ---------------------------------------------------------------------------

_JA_TRATADAS = {
    "dti", "annual_inc", "fico_range_low", "fico_range_high",
    "anos_historico_credito", "term_meses", "emp_length_num",
    "emp_length_missing", "dti_missing",
}

def aplicar_transformacoes_por_regra(df: pl.DataFrame) -> pl.DataFrame:
    """
    Regra sistemática para as numéricas restantes (não tratadas
    individualmente acima):
        zero_pct > 50% e |skew| >= 1 -> flag binária + log1p, em colunas
            separadas ({col}_teve_evento, {col}_log) — evento raro: o
            log1p sozinho comprimiria demais o sinal dos poucos casos
            "que aconteceram" junto com a massa de zeros.
        zero_pct <= 50% e |skew| >= 1 -> log1p direto (cauda longa genuína).
        |skew| < 1 -> sem transformação.
    """
    numericas = [
        c for c in df.columns
        if df[c].dtype in (pl.Float64, pl.Int64, pl.Int32)
        and c not in _CATEGORICAS and c not in _NAO_FEATURES and c not in _JA_TRATADAS
    ]

    skew_vals = df.select([pl.col(c).skew().alias(c) for c in numericas]).row(0, named=True)
    zero_pcts = df.select([
        ((pl.col(c) == 0).sum() / pl.len() * 100).alias(c) for c in numericas
    ]).row(0, named=True)

    cols_flag_log = [c for c in numericas if (zero_pcts[c] or 0) > 50 and abs(skew_vals[c] or 0) >= 1]
    cols_log_direto = [c for c in numericas if (zero_pcts[c] or 0) <= 50 and abs(skew_vals[c] or 0) >= 1]

    for col in cols_flag_log:
        df = df.with_columns(
            (pl.col(col) > 0).cast(pl.Int8).alias(f"{col}_teve_evento"),
            pl.col(col).log1p().alias(f"{col}_log"),
        )
    df = df.drop(cols_flag_log)

    for col in cols_log_direto:
        df = df.with_columns(pl.col(col).log1p().alias(col))

    log_etapa(f"Transformação por regra - flag+log1p ({len(cols_flag_log)} colunas)", cols_flag_log)
    log_etapa(f"Transformação por regra - log1p direto ({len(cols_log_direto)} colunas)", cols_log_direto)

    return df

def resolver_redundancia_correlacionada(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica decisões tomadas a partir de checar_correlacao(limiar=0.85):

    - fico_range_low/high: r=1.0 (piso e teto do mesmo intervalo).
      Combinadas em fico_score_medio.
    - acc_now_delinq_teve_evento/_log e pub_rec_teve_evento/_log:
      correlação alta reflete principalmente a massa compartilhada de
      zeros (evento raro), não redundância de informação por si só —
      versões _log descartadas nesses dois casos.
    - delinq_amnt_teve_evento/_log e delinq_2yrs_teve_evento/_log: mesma
      correlação alta, mas com variância real de severidade dentro do
      subconjunto com evento=1 — ambas as versões mantidas.

    Só remove colunas que existirem no df (robusto a rodar sobre um
    subconjunto de features onde alguma dessas não foi gerada).
    """
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df = df.with_columns(
            ((pl.col("fico_range_low") + pl.col("fico_range_high")) / 2).alias("fico_score_medio")
        ).drop(["fico_range_low", "fico_range_high"])

    a_remover = [c for c in ("acc_now_delinq_log", "pub_rec_log") if c in df.columns]
    if a_remover:
        df = df.drop(a_remover)

    log_nota(
        "Redundância resolvida: fico_range_low/high combinadas em "
        "fico_score_medio; acc_now_delinq_log e pub_rec_log descartadas "
        "(flag já captura o sinal); delinq_amnt_log e delinq_2yrs_log "
        "mantidas (variância real de severidade além da flag)."
    )
    return df

def codificar_categoricas(df: pl.DataFrame) -> pl.DataFrame:
    """
    One-Hot Encoding das categóricas, sobre o dataset inteiro (antes do
    split temporal) para garantir que treino/val/teste compartilhem
    exatamente as mesmas colunas dummy.

    drop_first=True: omite uma categoria por variável, evitando a
    dependência linear perfeita entre as k colunas de um one-hot completo
    (elas sempre somam 1) — mesma preocupação de multicolinearidade da
    etapa de correlação.
    """
    categoricas_existentes = [c for c in df.columns if c in _CATEGORICAS]
    n_antes = df.width
    df = df.to_dummies(columns=categoricas_existentes, drop_first=True)
    log_nota(
        f"One-Hot Encoding: {len(categoricas_existentes)} colunas categóricas -> "
        f"{df.width - n_antes + len(categoricas_existentes)} colunas dummy (drop_first=True). "
        f"Total de colunas: {n_antes} -> {df.width}."
    )
    return df

def split_temporal(df: pl.DataFrame, frac_treino: float = 0.70, frac_val: float = 0.15) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Corta cronologicamente em treino (mais antigo) / validação / teste
    (mais recente), nas proporções frac_treino/frac_val/resto.

    Corte por VALOR de data (issue_d parseada), não por posição de linha
    após o sort — issue_d tem granularidade mensal, e cortar por posição
    fragmentaria um mesmo mês entre duas partições. As frações resultantes
    são aproximadas, não exatas.

    issue_d continua como string "%b-%Y" no restante do pipeline — ordenar
    pela coluna crua seria alfabético por nome de mês (ex.: "Aug-2018"
    antes de "Dec-2007"), não cronológico. Por isso parseamos aqui, só
    para decidir os cortes, e não alteramos a coluna original.
    """
    if frac_treino + frac_val >= 1.0:
        raise ValueError("frac_treino + frac_val deve ser menor que 1.0")

    df = df.with_columns(pl.col("issue_d").str.to_date("%b-%Y").alias("_data_split"))
    df_ordenado = df.sort("_data_split")

    n_corte_treino = int(df.height * frac_treino)
    n_corte_val = int(df.height * (frac_treino + frac_val))

    data_corte_treino = df_ordenado["_data_split"][n_corte_treino]
    data_corte_val = df_ordenado["_data_split"][n_corte_val]

    df_treino = df.filter(pl.col("_data_split") < data_corte_treino).drop("_data_split")
    df_val = df.filter(
        (pl.col("_data_split") >= data_corte_treino) & (pl.col("_data_split") < data_corte_val)
    ).drop("_data_split")
    df_teste = df.filter(pl.col("_data_split") >= data_corte_val).drop("_data_split")

    total = df.height
    log_nota(
        f"Split temporal: cortes em {data_corte_treino} e {data_corte_val} (issue_d).\n"
        f"Treino: {df_treino.height} linhas ({df_treino.height / total:.1%}).\n"
        f"Validação: {df_val.height} linhas ({df_val.height / total:.1%}).\n"
        f"Teste: {df_teste.height} linhas ({df_teste.height / total:.1%})."
    )

    for nome, parte in [("treino", df_treino), ("validação", df_val), ("teste", df_teste)]:
        dist = (
            parte["target_classificacao"].value_counts()
            .with_columns((pl.col("count") / pl.col("count").sum() * 100).round(2).alias("%"))
        )
        log_etapa(f"target_classificacao - distribuição no {nome}", dist)
    log_nota(
        "Diferenças na distribuição do target entre as três partições são "
        "esperadas em split temporal (refletem mudança de política de "
        "crédito/comportamento macroeconômico entre safras), não erro de amostragem."
    )

    return df_treino, df_val, df_teste

def tratar_nulos_residuais(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_teste: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Trata nulos residuais que sobreviveram a todas as etapas anteriores —
    tipicamente colunas de bureau de crédito "avançado" que o Lending Club
    só passou a coletar a partir de certa safra (ausência estrutural, não
    aleatória), mais alguns nulos esparsos genuínos.

    Para cada coluna com nulo (em qualquer split), cria uma flag
    {coluna}_indisponivel (preserva a informação de que o dado não
    existia na origem) e imputa: MEDIANA para colunas contínuas, MODA
    para flags Int8 (fill_null com mediana promoveria uma coluna 0/1 para
    float, quebrando o dtype). Estatísticas calculadas exclusivamente no
    treino, aplicadas sem recálculo a validação/teste.
    """
    colunas_com_nulo = [
        c for c in df_treino.columns
        if c not in _NAO_FEATURES and (
            df_treino[c].null_count() > 0 or df_val[c].null_count() > 0 or df_teste[c].null_count() > 0
        )
    ]

    if not colunas_com_nulo:
        log_nota("Nenhum nulo residual encontrado - nada a tratar.")
        return df_treino, df_val, df_teste

    colunas_flag = [c for c in colunas_com_nulo if df_treino[c].dtype == pl.Int8]
    colunas_continuas = [c for c in colunas_com_nulo if c not in colunas_flag]

    medianas = {}
    if colunas_continuas:
        medianas = df_treino.select([pl.col(c).median().alias(c) for c in colunas_continuas]).row(0, named=True)

    modas = {}
    for c in colunas_flag:
        valores_moda = df_treino[c].drop_nulls().mode().sort()
        modas[c] = int(valores_moda[0]) if len(valores_moda) > 0 else 0

    if colunas_flag:
        log_nota(
            f"{len(colunas_flag)} coluna(s) Int8 (flags) com nulo residual tratadas "
            f"pela MODA (preservando dtype), não pela mediana: {colunas_flag}"
        )

    def _aplicar(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_indisponivel") for c in colunas_com_nulo
        ])
        if colunas_continuas:
            df = df.with_columns([pl.col(c).fill_null(medianas[c]).alias(c) for c in colunas_continuas])
        if colunas_flag:
            df = df.with_columns([pl.col(c).fill_null(modas[c]).cast(pl.Int8).alias(c) for c in colunas_flag])
        return df

    n_antes = len(colunas_features_modelo(df_treino))
    df_treino = _aplicar(df_treino)
    df_val = _aplicar(df_val)
    df_teste = _aplicar(df_teste)
    n_depois = len(colunas_features_modelo(df_treino))

    log_etapa(f"Nulos residuais tratados ({len(colunas_com_nulo)} colunas)", colunas_com_nulo)
    log_nota(f"input_size: {n_antes} -> {n_depois} (+{n_depois - n_antes} flags de indisponibilidade).")

    return df_treino, df_val, df_teste

def padronizar_numericas(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_teste: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """
    Padronização Z-score, com média/desvio calculados exclusivamente no
    treino e aplicados às três partições sem recálculo — calcular com o
    dataframe inteiro vazaria estatísticas de períodos futuros.

    Colunas com desvio padrão zero no treino (constantes) são deixadas
    SEM transformação (em vez de aplicar com um desvio de 1.0 arbitrário)
    e reportadas explicitamente.
    """
    colunas = colunas_numericas_continuas(df_treino)

    medias = df_treino.select([pl.col(c).mean().alias(c) for c in colunas]).row(0, named=True)
    desvios = df_treino.select([pl.col(c).std().alias(c) for c in colunas]).row(0, named=True)

    colunas_constantes = [c for c in colunas if not desvios[c]]
    colunas_a_padronizar = [c for c in colunas if desvios[c]]

    if colunas_constantes:
        log_nota(
            f"{len(colunas_constantes)} coluna(s) com desvio padrão zero no treino "
            f"NÃO foram padronizadas, para evitar divisão por zero: {colunas_constantes}"
        )

    def _aplicar(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            ((pl.col(c) - medias[c]) / desvios[c]).alias(c) for c in colunas_a_padronizar
        ])

    df_treino = _aplicar(df_treino)
    df_val = _aplicar(df_val)
    df_teste = _aplicar(df_teste)

    stats = {"medias": medias, "desvios": desvios, "colunas": colunas_a_padronizar}

    log_etapa("Validação - média por coluna no TREINO (esperado: ~0)", df_treino.select(colunas_a_padronizar).mean())
    log_etapa("Validação - média por coluna na VALIDAÇÃO (esperado: != 0, prova de ausência de vazamento)", df_val.select(colunas_a_padronizar).mean())
    log_etapa("Validação - média por coluna no TESTE (esperado: != 0, prova de ausência de vazamento)", df_teste.select(colunas_a_padronizar).mean())
    log_nota(f"Padronização (Z-score) aplicada a {len(colunas_a_padronizar)} colunas contínuas, estatística do treino ({df_treino.height} linhas).")

    return df_treino, df_val, df_teste, stats

# ---------------------------------------------------------------------------
# Cache em parquet (resiliência a reinício de runtime)
# ---------------------------------------------------------------------------

def salvar_particoes(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_teste: pl.DataFrame, diretorio: str) -> None:
    """
    Salva as três partições já pré-processadas (split + nulos residuais +
    padronização) em parquet — permite recarregar em segundos após um
    crash/reinício do runtime, sem reprocessar o CSV bruto do zero.
    """
    os.makedirs(diretorio, exist_ok=True)
    df_treino.write_parquet(os.path.join(diretorio, "treino.parquet"))
    df_val.write_parquet(os.path.join(diretorio, "validacao.parquet"))
    df_teste.write_parquet(os.path.join(diretorio, "teste.parquet"))
    log_nota(f"Partições salvas em '{diretorio}' (treino/validacao/teste.parquet).")

def carregar_particoes(diretorio: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Recarrega as partições salvas por salvar_particoes(), evitando reprocessar o pipeline inteiro."""
    caminho_treino = os.path.join(diretorio, "treino.parquet")
    if not os.path.exists(caminho_treino):
        raise FileNotFoundError(f"'{caminho_treino}' não existe ainda — rode salvar_particoes() primeiro.")
    df_treino = pl.read_parquet(caminho_treino)
    df_val = pl.read_parquet(os.path.join(diretorio, "validacao.parquet"))
    df_teste = pl.read_parquet(os.path.join(diretorio, "teste.parquet"))
    log_nota(f"Partições recarregadas de '{diretorio}' - treino: {df_treino.shape}, validação: {df_val.shape}, teste: {df_teste.shape}.")
    return df_treino, df_val, df_teste

# ---------------------------------------------------------------------------
# Ponte para o PyTorch
# ---------------------------------------------------------------------------

def extrair_arrays(df: pl.DataFrame, colunas_features: list[str]):
    """Converte o DataFrame Polars para arrays NumPy em float32 para o PyTorch."""
    X = df.select(colunas_features).to_numpy().astype("float32")
    y_clf = df["target_classificacao"].to_numpy().astype("float32")
    y_reg = df["target_regressao"].to_numpy().astype("float32")
    return X, y_clf, y_reg

def validar_tensores(tensores: dict[str, torch.Tensor], input_size_esperado: int) -> None:
    """Verifica integridade de dimensões, NaNs e valores infinitos antes de qualquer treino."""
    for nome, tensor in tensores.items():
        assert tensor.shape[1] == input_size_esperado, (
            f"{nome} tem {tensor.shape[1]} colunas, esperado {input_size_esperado}. "
            f"Confira se o pipeline de pré-processamento mudou desde a última contagem."
        )
        assert not torch.isnan(tensor).any(), f"{nome} contém NaN."
        assert not torch.isinf(tensor).any(), f"{nome} contém Inf."
    log_etapa(
        "Validação dos tensores de entrada (shape / NaN / Inf)",
        {nome: f"shape={tuple(t.shape)}, OK" for nome, t in tensores.items()},
    )

def preparar_dataloaders(df_treino: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame, batch_size: int = 512) -> dict:
    """
    Instancia e expõe os DataLoader de treino, validação e teste.

    y é convertido com unsqueeze(1) — shape (N, 1), compatível com a saída
    de 1 neurônio do MLP. Sem isso, BCEWithLogitsLoss/MSELoss fariam
    broadcasting silencioso entre (N,) e (N, 1) e produziriam uma loss
    matematicamente errada sem erro algum (e dependeriam de um .squeeze()
    do lado do modelo para "consertar" de volta, frágil com batch de
    tamanho 1). X é o MESMO tensor para os dois problemas — só o y muda.

    drop_last=True nos loaders de treino: os MLPs usam BatchNorm1d, que
    não aceita um batch de tamanho 1 durante o treino (o último batch de
    uma época pode ter exatamente 1 amostra se o tamanho do treino não
    for múltiplo de batch_size).
    """
    features = colunas_features_modelo(df_treino)
    input_size = len(features)

    n_val = len(colunas_features_modelo(df_val))
    n_test = len(colunas_features_modelo(df_test))
    assert input_size == n_val == n_test, (
        f"Número de features difere entre partições: treino={input_size}, "
        f"val={n_val}, teste={n_test}. Confirme que codificar_categoricas "
        f"rodou antes de split_temporal."
    )

    X_train, y_train_clf, y_train_reg = extrair_arrays(df_treino, features)
    X_val, y_val_clf, y_val_reg = extrair_arrays(df_val, features)
    X_test, y_test_clf, y_test_reg = extrair_arrays(df_test, features)

    validar_tensores({
        "X_train": torch.from_numpy(X_train),
        "X_val": torch.from_numpy(X_val),
        "X_test": torch.from_numpy(X_test),
    }, input_size)

    X_train_t, X_val_t, X_test_t = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(X_test)

    def _y(arr):
        return torch.from_numpy(arr).unsqueeze(1)

    ds_train_clf = TensorDataset(X_train_t, _y(y_train_clf))
    ds_train_reg = TensorDataset(X_train_t, _y(y_train_reg))
    ds_val_clf = TensorDataset(X_val_t, _y(y_val_clf))
    ds_val_reg = TensorDataset(X_val_t, _y(y_val_reg))
    ds_test_clf = TensorDataset(X_test_t, _y(y_test_clf))
    ds_test_reg = TensorDataset(X_test_t, _y(y_test_reg))

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
        },
    }
