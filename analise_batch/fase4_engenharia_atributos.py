#!/usr/bin/env python3
"""
Fase 4 — Engenharia de atributos para classificação supervisionada.

Produz dois datasets semanais (2023-2025, Reddit MG × crimes):
  - sem_toxicidade : features autoregressivas de crimes + calendário
  - com_toxicidade : idem + toxicidade defasada (T-1 e MM3)

NÃO treina nenhum classificador. Para após gerar e reportar os datasets.

Design:
  Alvo        : alvo ∈ {0,1}, corte na mediana de n_crimes
  Toxicidade  : features de T-1 NUNCA da semana T (evita data leakage)
  Validação   : reservada para script seguinte (TimeSeriesSplit)
"""

# ── ADAPTADO PARA EXECUCAO LOCAL ──────────────────────────────────────────────
# Este script foi adaptado para rodar sem Google Cloud Storage.
# Coloque os arquivos CSV em:   analise_batch/dados/
# Os resultados sao salvos em:  analise_batch/saida/
# Mais detalhes:                analise_batch/README_analise.md
# ──────────────────────────────────────────────────────────────────────────────


import io
import sys

import numpy as np
import pandas as pd
from gcs_local import gcs_client, storage
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
# KEY_FILE      = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET        = "lgbtminas-dados"
REDDIT_PREFIX = "rede social/reddit/analysis/tybyria/"
REDDIT_BLOB   = "rede social/reddit/analysis/tybyria/reddit_tybyria_RC_total.csv"
CRIME_BLOB    = (
    "criminal/processed/"
    "DIS - Envolvidos - Eventos de LGBTQIAfobia - Jan 2023 a Dez 2025.csv"
)
OUT_SEM_TOX = "analysis/fase4_dataset_sem_toxicidade.csv"
OUT_COM_TOX = "analysis/fase4_dataset_com_toxicidade.csv"

MG_SUBREDDITS = frozenset([
    "BeloHorizonte", "MinasGerais", "juizdefora",
    "Uberlandia", "Uberaba", "OuroPreto", "montesclaros_",
])
SEMANAS_ALVO = pd.period_range("2023-01-01", "2025-12-31", freq="W")

SEP  = "=" * 70
SEP2 = "-" * 70
def section(t): print(f"\n{SEP}\n{t}\n{SEP}")
def sub(t):     print(f"\n{SEP2}\n{t}\n{SEP2}")

# gcs_client() importada de gcs_local

def read_blob_csv(bkt, name, **kw):
    return pd.read_csv(io.BytesIO(bkt.blob(name).download_as_bytes()), **kw)

def upload_csv(bkt, name, df):
    b = df.to_csv(index=False).encode("utf-8")
    bkt.blob(name).upload_from_string(b, content_type="text/csv")
    print(f"  [OK] gs://{BUCKET}/{name}  ({len(b)/1024:.1f} KB)")

client = gcs_client()
bucket = client.bucket(BUCKET)


# ═════════════════════════════════════════════════════════════════════════════
# 1. REDDIT MG — AGREGAÇÃO SEMANAL
# ═════════════════════════════════════════════════════════════════════════════
section("1. REDDIT MG — AGREGAÇÃO SEMANAL DE tybyria_score")

print(f"\n  Carregando arquivo consolidado: {REDDIT_BLOB}")
print("  (pode demorar ~30s) ...")
df_reddit = read_blob_csv(bucket, REDDIT_BLOB, low_memory=False)
df_reddit = df_reddit[df_reddit["subreddit"].isin(MG_SUBREDDITS)].copy()
print(f"  Linhas após filtro MG: {len(df_reddit):,}")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_reddit["ts"]  = pd.to_datetime(
        dict(year=df_reddit["ano"], month=df_reddit["mes"], day=df_reddit["dia"])
    )
    df_reddit["sem"] = df_reddit["ts"].dt.to_period("W")

reddit_sem = (
    df_reddit[df_reddit["sem"].isin(SEMANAS_ALVO)]
    .groupby("sem")
    .agg(
        n_posts          = ("id",            "count"),
        media_tybyria    = ("tybyria_score",  "mean"),
        mediana_tybyria  = ("tybyria_score",  "median"),
        pct_toxicos      = ("tybyria_score",  lambda s: (s >= 0.30).mean() * 100),
    )
    .reindex(SEMANAS_ALVO, fill_value=np.nan)
)
reddit_sem.index.name = "sem"

# Semanas com dados
n_com_reddit = reddit_sem["n_posts"].notna().sum()
n_sem_total  = len(SEMANAS_ALVO)
print(f"  Semanas MG com posts: {n_com_reddit}/{n_sem_total}")
print(f"  media_tybyria  — min={reddit_sem['media_tybyria'].min():.4f}  "
      f"max={reddit_sem['media_tybyria'].max():.4f}  "
      f"média={reddit_sem['media_tybyria'].mean():.4f}")
print(f"  pct_toxicos    — min={reddit_sem['pct_toxicos'].min():.1f}%  "
      f"max={reddit_sem['pct_toxicos'].max():.1f}%  "
      f"média={reddit_sem['pct_toxicos'].mean():.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 2. CRIMES — AGREGAÇÃO SEMANAL
# ═════════════════════════════════════════════════════════════════════════════
section("2. CRIMES — AGREGAÇÃO SEMANAL")

df_crime = read_blob_csv(
    bucket, CRIME_BLOB,
    encoding="iso-8859-1", sep=";", on_bad_lines="skip", low_memory=False,
)
df_crime.columns = [c.replace("\xad", "").strip() for c in df_crime.columns]
df_crime = df_crime.dropna(subset=["ID Ocorrência"]).copy()
df_crime["dt"] = pd.to_datetime(
    df_crime["Data/Hora Ocorrência"], format="%d/%m/%Y %H:%M", errors="coerce"
)
df_crime = df_crime[df_crime["dt"].notna()].copy()
# Arquivo Envolvidos: deduplicar por ID Ocorrência para contar eventos únicos
df_crime = df_crime.drop_duplicates(subset=["ID Ocorrência"]).copy()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_crime["sem"] = df_crime["dt"].dt.to_period("W")

crimes_sem = (
    df_crime[df_crime["sem"].isin(SEMANAS_ALVO)]
    .groupby("sem")["ID Ocorrência"]
    .count()
    .reindex(SEMANAS_ALVO, fill_value=0)
)
crimes_sem.index.name = "sem"

print(f"\n  Semanas com crimes: {(crimes_sem > 0).sum()}/{len(SEMANAS_ALVO)}")
print(f"  n_crimes — min={crimes_sem.min()}  max={crimes_sem.max()}  "
      f"média={crimes_sem.mean():.2f}  mediana={crimes_sem.median():.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. VARIÁVEL-ALVO — CORTE NA MEDIANA
# ═════════════════════════════════════════════════════════════════════════════
section("3. VARIÁVEL-ALVO — CORTE NA MEDIANA")

mediana_calculada = float(crimes_sem.median())
MEDIANA = mediana_calculada

print(f"\n  Mediana calculada dos dados: {mediana_calculada}")

print(f"\n  Corte: alvo=0 se n_crimes ≤ {MEDIANA:.0f}, alvo=1 se > {MEDIANA:.0f}")
n_classe0 = (crimes_sem <= MEDIANA).sum()
n_classe1 = (crimes_sem >  MEDIANA).sum()
print(f"  Classe 0 (baixo) : {n_classe0} semanas  ({n_classe0/len(crimes_sem)*100:.1f}%)")
print(f"  Classe 1 (alto)  : {n_classe1} semanas  ({n_classe1/len(crimes_sem)*100:.1f}%)")
print(f"  Balanço          : {min(n_classe0,n_classe1)/max(n_classe0,n_classe1):.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUÇÃO DO DATASET BASE (índice temporal completo)
# ═════════════════════════════════════════════════════════════════════════════
section("4. ENGENHARIA DE ATRIBUTOS")

# Junta as duas séries no índice de semanas
df = pd.DataFrame({
    "n_crimes":        crimes_sem,
    "media_tybyria":   reddit_sem["media_tybyria"],
    "pct_toxicos":     reddit_sem["pct_toxicos"],
    "n_posts_reddit":  reddit_sem["n_posts"],
}, index=SEMANAS_ALVO)
df.index.name = "sem"

# ── Alvo ─────────────────────────────────────────────────────────────────────
df["alvo"] = (df["n_crimes"] > MEDIANA).astype(int)

# ── Features autoregressivas de crimes ───────────────────────────────────────
# Todas baseadas em t-1 ou anterior — nunca na semana T
df["crimes_t1"]  = df["n_crimes"].shift(1)
df["crimes_t2"]  = df["n_crimes"].shift(2)
# Média das 3 semanas anteriores: shift(1) move T→T-1, rolling(3) cobre T-1,T-2,T-3
df["crimes_mm3"] = df["n_crimes"].shift(1).rolling(3, min_periods=3).mean()

# ── Features de toxicidade defasadas (T-1 e MM3) ─────────────────────────────
# pct_toxicos (% posts com tybyria ≥ 0.30)
df["toxicidade_t1"]  = df["pct_toxicos"].shift(1)
df["toxicidade_mm3"] = df["pct_toxicos"].shift(1).rolling(3, min_periods=3).mean()

# media_tybyria defasada (feature alternativa mais granular)
df["tybyria_t1"]  = df["media_tybyria"].shift(1)
df["tybyria_mm3"] = df["media_tybyria"].shift(1).rolling(3, min_periods=3).mean()

# volume de posts defasado (T-1) — sem vazamento temporal
df["n_posts_t1"]  = df["n_posts_reddit"].shift(1)

# ── Features de calendário ────────────────────────────────────────────────────
# Extraídas do início da semana (Monday = dia 0 na convenção pandas W)
df["mes"]       = df.index.map(lambda p: p.start_time.month).astype(int)
df["trimestre"] = df.index.map(lambda p: p.start_time.quarter).astype(int)
df["sem_ano"]   = df.index.map(lambda p: p.start_time.isocalendar()[1]).astype(int)

sub("4A. Semanas perdidas por defasagem insuficiente")
n_total_antes = len(df)

# A feature que requer mais histórico é crimes_mm3 / toxicidade_mm3:
# precisa de 3 semanas passadas → as primeiras 3 semanas ficam com NaN
nan_por_col = df[["crimes_t1","crimes_t2","crimes_mm3",
                   "toxicidade_t1","toxicidade_mm3"]].isnull().sum()
print(f"\n  NaN por feature (antes do descarte):")
for col, n in nan_por_col.items():
    print(f"    {col:<20}: {n} semanas com NaN")

# Linhas com qualquer NaN nas features de crimes (necessárias nos dois datasets)
base_features = ["crimes_t1", "crimes_t2", "crimes_mm3", "mes", "trimestre"]
mask_valida   = df[base_features].notna().all(axis=1)
n_descartadas = (~mask_valida).sum()
n_final       = mask_valida.sum()

print(f"\n  Semanas descartadas (histórico insuficiente) : {n_descartadas}")
print(f"  Semanas mantidas (n final disponível)        : {n_final}")
print(f"  Período resultante: {df.index[mask_valida][0]} → {df.index[mask_valida][-1]}")
print(f"\n  Decisão: DESCARTE (não imputação) — as primeiras {n_descartadas} semanas")
print(f"  não têm contexto temporal suficiente e imputação introduziria ruído artificial.")
print(f"  Impacto: {n_descartadas}/{n_total_antes} semanas = {n_descartadas/n_total_antes*100:.1f}%")

df_valido = df[mask_valida].copy()


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATASETS FINAIS
# ═════════════════════════════════════════════════════════════════════════════
section("5. DATASETS FINAIS")

# Colunas base (ambos os datasets)
COLS_BASE = [
    "sem",           # índice como coluna para o CSV
    "n_crimes",      # referência — NÃO é feature
    "alvo",          # variável-alvo
    "crimes_t1",     # feature: autoregressive lag 1
    "crimes_t2",     # feature: autoregressive lag 2
    "crimes_mm3",    # feature: média móvel 3 semanas anteriores
    "mes",           # feature: calendário
    "trimestre",     # feature: calendário
    "sem_ano",       # feature: número da semana ISO (sazonalidade anual)
]

# Colunas de toxicidade (apenas dataset com_toxicidade)
COLS_TOX = [
    "toxicidade_t1",   # pct_toxicos na semana T-1
    "toxicidade_mm3",  # pct_toxicos média das semanas T-1 a T-3
    "tybyria_t1",      # media_tybyria na semana T-1 (feature alternativa)
    "tybyria_mm3",     # media_tybyria MM3 (feature alternativa)
    "n_posts_t1",      # volume de posts na semana T-1 (defasado, sem vazamento)
]

df_valido_reset = df_valido.reset_index()
df_valido_reset["sem"] = df_valido_reset["sem"].astype(str)

df_sem_tox = df_valido_reset[COLS_BASE].copy()
df_com_tox = df_valido_reset[COLS_BASE + COLS_TOX].copy()

# Verificar se há NaN nas features de toxicidade no df_com_tox
nan_tox = df_com_tox[COLS_TOX].isnull().sum()
if nan_tox.any():
    print(f"\n  [AVISO] NaN restantes em features de toxicidade (semanas sem posts Reddit):")
    for col, n in nan_tox[nan_tox > 0].items():
        print(f"    {col}: {n} semanas")
    print(f"  Essas linhas serão válidas no dataset 'sem_toxicidade' mas terão NaN")
    print(f"  no dataset 'com_toxicidade'. Reportado abaixo por dataset.")
else:
    print(f"\n  [OK] Nenhum NaN nas features de toxicidade.")

sub("5A. Dataset SEM toxicidade")
print(f"\n  Shape  : {df_sem_tox.shape}  (linhas × colunas)")
print(f"\n  Colunas:")
for c in df_sem_tox.columns:
    dtype = df_sem_tox[c].dtype
    n_null = df_sem_tox[c].isnull().sum()
    print(f"    {c:<20} dtype={str(dtype):<10} nulls={n_null}")

print(f"\n  Distribuição do alvo:")
vc0 = df_sem_tox["alvo"].value_counts().sort_index()
for cls, cnt in vc0.items():
    label = "baixo (≤11 crimes)" if cls == 0 else "alto  (>11 crimes)"
    print(f"    Classe {cls} [{label}]: {cnt} semanas  ({cnt/len(df_sem_tox)*100:.1f}%)")

print(f"\n  Primeiras 5 linhas:")
print(df_sem_tox.head(5).to_string(index=False))

sub("5B. Dataset COM toxicidade")
print(f"\n  Shape  : {df_com_tox.shape}  (linhas × colunas)")
print(f"\n  Colunas adicionais (toxicidade):")
for c in COLS_TOX:
    n_null = df_com_tox[c].isnull().sum()
    print(f"    {c:<22} nulls={n_null}")

print(f"\n  Distribuição do alvo:")
vc1 = df_com_tox["alvo"].value_counts().sort_index()
for cls, cnt in vc1.items():
    label = "baixo (≤11 crimes)" if cls == 0 else "alto  (>11 crimes)"
    print(f"    Classe {cls} [{label}]: {cnt} semanas  ({cnt/len(df_com_tox)*100:.1f}%)")

print(f"\n  Primeiras 5 linhas (todas as colunas):")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(df_com_tox.head(5).to_string(index=False))

sub("5C. Estatísticas das features (dataset com toxicidade)")
feat_cols = [c for c in df_com_tox.columns
             if c not in ("sem", "n_crimes", "alvo")]
print(df_com_tox[feat_cols].describe().round(4).to_string())

sub("5D. Correlação das features com o alvo")
print(f"\n  {'Feature':<22} {'Pearson r':>10} {'Dir.'}")
print("  " + "-" * 40)
for col in feat_cols:
    s = df_com_tox[col].dropna()
    if len(s) < 10:
        continue
    mask = df_com_tox[col].notna()
    r = float(np.corrcoef(df_com_tox.loc[mask, col], df_com_tox.loc[mask, "alvo"])[0, 1])
    sinal = "↑" if r > 0.05 else ("↓" if r < -0.05 else "~")
    print(f"  {col:<22} {r:>+10.4f} {sinal}")
print(f"\n  [ATENÇÃO] Correlações calculadas sobre TODO o dataset (sem split temporal).")
print(f"  Servem apenas para sanidade — não para seleção de features.")


# ═════════════════════════════════════════════════════════════════════════════
# 6. SALVAR NO GCS
# ═════════════════════════════════════════════════════════════════════════════
section("6. SALVANDO DATASETS NO GCS")
upload_csv(bucket, OUT_SEM_TOX, df_sem_tox)
upload_csv(bucket, OUT_COM_TOX, df_com_tox)

print(f"""
RESUMO PARA CONFIRMAÇÃO:
  n semanas disponíveis (após descarte das {n_descartadas} iniciais) : {n_final}
  Dataset sem_toxicidade : {df_sem_tox.shape[0]} linhas × {df_sem_tox.shape[1]} colunas
  Dataset com_toxicidade : {df_com_tox.shape[0]} linhas × {df_com_tox.shape[1]} colunas
  Variável-alvo          : binária, corte na mediana = {MEDIANA:.0f}
  Balanço de classes     : {n_classe0} (baixo) vs {n_classe1} (alto)
  Features de crimes     : crimes_t1, crimes_t2, crimes_mm3
  Features de calendário : mes, trimestre, sem_ano
  Features de toxicidade : toxicidade_t1, toxicidade_mm3, tybyria_t1, tybyria_mm3
  Estratégia de leakage  : toxicidade nunca da semana T, sempre de T-1 ou anterior
  Próximo passo          : treinamento com TimeSeriesSplit (Fase C)
""")
