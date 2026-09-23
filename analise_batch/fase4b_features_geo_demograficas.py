#!/usr/bin/env python3
"""
Fase 4B — Engenharia de atributos geográficos, demográficos e Instagram.

Replicação da metodologia do artigo "Real-time crime prediction using social media":
  - Features geográficas (lat/lon dos CRIMES, dispersão espacial, municípios)
  - Features demográficas (etnia, gênero das vítimas/envolvidos — dados REDS)
  - Features do Instagram (Tybyria + VADER, defasadas T-1)
  - Integração com features já existentes (Reddit + autoregressivas)

NOTA METODOLÓGICA:
  As features geográficas vêm EXCLUSIVAMENTE dos dados criminais geocodificados.
  Os dados de redes sociais (Reddit e Instagram) NÃO têm coordenadas geográficas —
  a dimensão regional é capturada por inferência: subreddits de MG (BeloHorizonte,
  MinasGerais, etc.) e perfis do Instagram de instituições/veículos mineiros.
  Isso é análogo ao artigo: longitude era atributo dos crimes, não do Twitter.

Produz 3 datasets semanais (2023-2025):
  analysis/fase4_artigo_base.csv       — autoregressivo + calendário (baseline)
  analysis/fase4_artigo_midia.csv      — base + Reddit + Instagram
  analysis/fase4_artigo_completo.csv   — midia + geográfico + demográfico

AVISO IMPORTANTE: Este script imprime as colunas brutas dos arquivos criminais
na primeira execução para permitir mapeamento correto dos campos demográficos.
"""

# ── ADAPTADO PARA EXECUCAO LOCAL ──────────────────────────────────────────────
# Este script foi adaptado para rodar sem Google Cloud Storage.
# Coloque os arquivos CSV em:   analise_batch/dados/
# Os resultados sao salvos em:  analise_batch/saida/
# Mais detalhes:                analise_batch/README_analise.md
# ──────────────────────────────────────────────────────────────────────────────


import io
import sys
import warnings

import numpy as np
import pandas as pd
from gcs_local import gcs_client, storage
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
# KEY_FILE         = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET           = "lgbtminas-dados"

# Dados criminais
GEOCODED_BLOB    = (
    "criminal/processed/"
    "DIS - Registros - Eventos de LGBTQIAfobia - Jan 2023 a Jan 2025_geocoded.csv"
)
ENVOLVIDOS_BLOB  = (
    "criminal/processed/"
    "DIS - Envolvidos - Eventos de LGBTQIAfobia - Jan 2023 a Dez 2025.csv"
)

# Reddit (já processado pelo fase4_engenharia_atributos)
REDDIT_BLOB      = "rede social/reddit/analysis/tybyria/reddit_tybyria_RC_total.csv"

# Instagram
INSTA_BLOB       = (
    "rede social/instagram/processed/"
    "instagram_comments_2023_2025_limpo_normalizado.csv"
)

# Datasets fase4 já gerados (base para join)
BLOB_BASE_EXISTE = "analysis/fase4_dataset_sem_toxicidade.csv"
BLOB_TOX_EXISTE  = "analysis/fase4_dataset_com_toxicidade.csv"

# Outputs
OUT_BASE      = "analysis/fase4_artigo_base.csv"
OUT_MIDIA     = "analysis/fase4_artigo_midia.csv"
OUT_COMPLETO  = "analysis/fase4_artigo_completo.csv"

MG_SUBREDDITS = frozenset([
    "BeloHorizonte", "MinasGerais", "juizdefora",
    "Uberlandia", "Uberaba", "OuroPreto", "montesclaros_",
])
SEMANAS_ALVO  = pd.period_range("2023-01-01", "2025-12-31", freq="W")
THRESHOLD_TOX = 0.30

SEP  = "=" * 70
SEP2 = "-" * 70
def section(t): print(f"\n{SEP}\n{t}\n{SEP}")
def sub(t):     print(f"\n{SEP2}\n{t}\n{SEP2}")

# ── GCS helpers ───────────────────────────────────────────────────────────────
# gcs_client() importada de gcs_local

def read_blob_csv(bkt, name, **kw):
    return pd.read_csv(io.BytesIO(bkt.blob(name).download_as_bytes()), **kw)

def upload_csv(bkt, name, df):
    b = df.to_csv(index=False).encode("utf-8")
    bkt.blob(name).upload_from_string(b, content_type="text/csv")
    print(f"  [OK] gs://{BUCKET}/{name}  ({len(b)/1024:.1f} KB,  {len(df)} linhas)")

client = gcs_client()
bucket = client.bucket(BUCKET)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADE: parsear coordenadas do REDS (string com pontos embutidos)
# Fonte: fase2_agregacao_analise.py:67-75
# ─────────────────────────────────────────────────────────────────────────────
def fix_coord(s):
    try:
        s = str(s).replace(".", "")
        if s.startswith("-"):
            return float(s[:3] + "." + s[3:])
        return float(s[:2] + "." + s[2:])
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. CARREGAR BASE JÁ EXISTENTE (fase4_dataset_sem_toxicidade)
#    → usamos como âncora temporal para o join
# ═════════════════════════════════════════════════════════════════════════════
section("1. CARREGANDO BASE EXISTENTE (fase4_dataset_sem_toxicidade)")

df_base = read_blob_csv(bucket, BLOB_BASE_EXISTE)
df_base["sem"] = pd.PeriodIndex(df_base["sem"], freq="W")
df_base = df_base.set_index("sem")
print(f"  Semanas disponíveis: {len(df_base)}  "
      f"({df_base.index[0]} → {df_base.index[-1]})")

FEAT_BASE = ["crimes_t1", "crimes_t2", "crimes_mm3", "mes", "trimestre", "sem_ano"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURES DE TOXICIDADE REDDIT (já existem no dataset com toxicidade)
# ═════════════════════════════════════════════════════════════════════════════
section("2. FEATURES REDDIT (do dataset com_toxicidade existente)")

df_tox = read_blob_csv(bucket, BLOB_TOX_EXISTE)
df_tox["sem"] = pd.PeriodIndex(df_tox["sem"], freq="W")
df_tox = df_tox.set_index("sem")

FEAT_REDDIT = ["toxicidade_t1", "toxicidade_mm3", "tybyria_t1", "tybyria_mm3", "n_posts_t1"]
df_reddit_feats = df_tox[FEAT_REDDIT]
print(f"  Features Reddit: {FEAT_REDDIT}")
print(f"  NaN por coluna:")
for c in FEAT_REDDIT:
    print(f"    {c}: {df_reddit_feats[c].isna().sum()} semanas com NaN")


# ═════════════════════════════════════════════════════════════════════════════
# 3. FEATURES DO INSTAGRAM (Tybyria + VADER, defasadas T-1)
# ═════════════════════════════════════════════════════════════════════════════
section("3. INSTAGRAM — agregação semanal e defasagem T-1")

print(f"\n  Carregando: {INSTA_BLOB}")
print("  (pode demorar ~20s) ...")
df_insta = read_blob_csv(bucket, INSTA_BLOB, low_memory=False)
print(f"  Linhas brutas: {len(df_insta):,}")
print(f"\n  Colunas disponíveis no Instagram:")
for c in df_insta.columns:
    print(f"    {c}")

# Identificar coluna de data
date_col = None
for candidate in ["comment_created_at", "created_at", "data", "date", "timestamp"]:
    if candidate in df_insta.columns:
        date_col = candidate
        break
if date_col is None:
    print("  [AVISO] Coluna de data não identificada automaticamente.")
    print("  Colunas com 'date' ou 'time' no nome:")
    for c in df_insta.columns:
        if any(k in c.lower() for k in ["date", "time", "data", "created"]):
            print(f"    {c}")
    date_col = df_insta.columns[0]
    print(f"  Usando '{date_col}' como fallback — ajuste se incorreto.")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_insta["_dt"] = pd.to_datetime(df_insta[date_col], utc=True, errors="coerce")

n_invalido = df_insta["_dt"].isna().sum()
if n_invalido > 0:
    print(f"  [AVISO] {n_invalido} datas inválidas descartadas")
    df_insta = df_insta[df_insta["_dt"].notna()].copy()

df_insta["sem"] = df_insta["_dt"].dt.to_period("W")

# Identificar colunas NLP
tybyria_col = next((c for c in df_insta.columns
                    if "tybyria" in c.lower()), None)
vader_col   = next((c for c in df_insta.columns
                    if "vader_compound" in c.lower() or c.lower() == "vader_compound"), None)
if vader_col is None:
    vader_col = next((c for c in df_insta.columns
                      if "vader" in c.lower() and "compound" in c.lower()), None)

print(f"\n  Coluna Tybyria detectada : {tybyria_col}")
print(f"  Coluna VADER detectada   : {vader_col}")

# Agregação semanal Instagram
agg_dict = {}
if tybyria_col:
    agg_dict["insta_tybyria"]    = (tybyria_col, "mean")
    agg_dict["insta_pct_toxicos"] = (tybyria_col, lambda s: (s >= THRESHOLD_TOX).mean() * 100)
if vader_col:
    agg_dict["insta_vader"] = (vader_col, "mean")
agg_dict["insta_n_comentarios"] = (df_insta.columns[0], "count")

insta_sem = (
    df_insta[df_insta["sem"].isin(SEMANAS_ALVO)]
    .groupby("sem")
    .agg(**agg_dict)
    .reindex(SEMANAS_ALVO, fill_value=np.nan)
)

# Semanas com comentários Instagram
n_com = (insta_sem["insta_n_comentarios"] > 0).sum()
print(f"\n  Semanas com comentários Instagram: {n_com}/{len(SEMANAS_ALVO)}")
if tybyria_col:
    print(f"  insta_tybyria — média={insta_sem['insta_tybyria'].mean():.4f}  "
          f"NaN={insta_sem['insta_tybyria'].isna().sum()}")
if vader_col:
    print(f"  insta_vader   — média={insta_sem['insta_vader'].mean():.4f}  "
          f"NaN={insta_sem['insta_vader'].isna().sum()}")

# Defasar T-1
insta_feats_raw = [c for c in insta_sem.columns]
insta_def = pd.DataFrame(index=SEMANAS_ALVO)
for col in insta_feats_raw:
    insta_def[col + "_t1"] = insta_sem[col].shift(1)

FEAT_INSTA = [c + "_t1" for c in insta_feats_raw]
print(f"\n  Features Instagram (defasadas T-1): {FEAT_INSTA}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. FEATURES GEOGRÁFICAS (crimes geocoded, defasadas T-1)
# ═════════════════════════════════════════════════════════════════════════════
section("4. FEATURES GEOGRÁFICAS (crimes geocoded)")

print(f"\n  Carregando: {GEOCODED_BLOB}")
df_geo = read_blob_csv(
    bucket, GEOCODED_BLOB,
    encoding="iso-8859-1", sep=";", on_bad_lines="skip", low_memory=False,
)
df_geo.columns = [c.replace("\xad", "").strip() for c in df_geo.columns]
print(f"\n  Colunas disponíveis no arquivo geocoded:")
for c in df_geo.columns:
    print(f"    {c}")

# Identificar colunas de lat/lon
lat_col = next((c for c in df_geo.columns
                if any(k in c.lower() for k in ["latitude", "lat"])), None)
lon_col = next((c for c in df_geo.columns
                if any(k in c.lower() for k in ["longitude", "lon", "lng"])), None)
# Preferir coluna com nome do município (Fato) sobre código numérico
mun_col = next(
    (c for c in df_geo.columns
     if any(k in c.lower() for k in ["municipio", "município", "cidade"])
     and "código" not in c.lower() and "codigo" not in c.lower()),
    None
)

print(f"\n  Coluna latitude  : {lat_col}")
print(f"  Coluna longitude : {lon_col}")
print(f"  Coluna município : {mun_col}")

# Parsear data
dt_col_geo = next((c for c in df_geo.columns
                   if "data" in c.lower() or "hora" in c.lower()), "Data/Hora Ocorrência")
df_geo["_dt"] = pd.to_datetime(df_geo[dt_col_geo], format="%d/%m/%Y %H:%M", errors="coerce")
df_geo = df_geo[df_geo["_dt"].notna()].copy()
df_geo = df_geo.drop_duplicates(subset=["ID Ocorrência"] if "ID Ocorrência" in df_geo.columns
                                 else df_geo.columns[:1]).copy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_geo["sem"] = df_geo["_dt"].dt.to_period("W")

# Parsear coordenadas
if lat_col:
    # Tentar direto float primeiro; se falhar, usar fix_coord
    df_geo["_lat"] = pd.to_numeric(df_geo[lat_col], errors="coerce")
    if df_geo["_lat"].isna().mean() > 0.5:
        df_geo["_lat"] = df_geo[lat_col].apply(fix_coord)
else:
    df_geo["_lat"] = np.nan
    print("  [AVISO] Coluna de latitude não encontrada — usando NaN")

if lon_col:
    df_geo["_lon"] = pd.to_numeric(df_geo[lon_col], errors="coerce")
    if df_geo["_lon"].isna().mean() > 0.5:
        df_geo["_lon"] = df_geo[lon_col].apply(fix_coord)
else:
    df_geo["_lon"] = np.nan
    print("  [AVISO] Coluna de longitude não encontrada — usando NaN")

n_geo_valido = df_geo[["_lat", "_lon"]].notna().all(axis=1).sum()
print(f"\n  Registros com coordenadas válidas: {n_geo_valido}/{len(df_geo)}")

# Coordenadas de referência por município (para preencher semanas sem geocoded)
coords_municipio = {}
if mun_col and n_geo_valido > 0:
    df_mun = df_geo[df_geo["_lat"].notna() & df_geo["_lon"].notna()].copy()
    df_mun[mun_col] = df_mun[mun_col].str.strip().str.upper()
    coords_municipio = (
        df_mun.groupby(mun_col)
        .agg(lat=("_lat", "median"), lon=("_lon", "median"))
        .to_dict("index")
    )
    print(f"  Municípios com coordenadas de referência: {len(coords_municipio)}")

# Identificar BH no campo município
bh_patterns = ["BELO HORIZONTE", "B.H.", "BH"]

def _pct_bh(series):
    if mun_col is None:
        return np.nan
    return series.str.strip().str.upper().isin(bh_patterns).mean() * 100

# Agregação semanal geográfica
geo_agg = {}
if n_geo_valido > 0:
    df_geo_valido = df_geo[df_geo["_lat"].notna() & df_geo["_lon"].notna()].copy()
    geo_sem = (
        df_geo_valido[df_geo_valido["sem"].isin(SEMANAS_ALVO)]
        .groupby("sem")
        .agg(
            lat_mean      = ("_lat", "mean"),
            lon_mean      = ("_lon", "mean"),
            lat_std       = ("_lat", "std"),
            lon_std       = ("_lon", "std"),
        )
        .reindex(SEMANAS_ALVO, fill_value=np.nan)
    )

    if mun_col:
        mun_sem = (
            df_geo[df_geo["sem"].isin(SEMANAS_ALVO)]
            .groupby("sem")
            .agg(
                n_municipios = (mun_col, "nunique"),
                pct_bh       = (mun_col, _pct_bh),
            )
            .reindex(SEMANAS_ALVO, fill_value=np.nan)
        )
        geo_sem = geo_sem.join(mun_sem)
else:
    geo_sem = pd.DataFrame(
        index=SEMANAS_ALVO,
        columns=["lat_mean", "lon_mean", "lat_std", "lon_std",
                 "n_municipios", "pct_bh"],
        data=np.nan,
    )
    print("  [AVISO] Sem coordenadas válidas — features geográficas serão NaN")

n_geo_sem = geo_sem["lat_mean"].notna().sum()
print(f"\n  Semanas com dados geográficos: {n_geo_sem}/{len(SEMANAS_ALVO)}")
if n_geo_valido > 0:
    print(f"  lat_mean  — min={geo_sem['lat_mean'].min():.4f}  "
          f"max={geo_sem['lat_mean'].max():.4f}  "
          f"(MG: aprox. -14° a -23°)")
    print(f"  lon_mean  — min={geo_sem['lon_mean'].min():.4f}  "
          f"max={geo_sem['lon_mean'].max():.4f}  "
          f"(MG: aprox. -41° a -50°)")

# Defasar T-1
geo_def = pd.DataFrame(index=SEMANAS_ALVO)
for col in geo_sem.columns:
    geo_def[col + "_t1"] = geo_sem[col].shift(1)

FEAT_GEO = [c + "_t1" for c in geo_sem.columns]
print(f"\n  Features geográficas (defasadas T-1): {FEAT_GEO}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. FEATURES DEMOGRÁFICAS (Envolvidos — etnia, gênero, defasadas T-1)
# ═════════════════════════════════════════════════════════════════════════════
section("5. FEATURES DEMOGRÁFICAS (Envolvidos)")

print(f"\n  Carregando: {ENVOLVIDOS_BLOB}")
df_env = read_blob_csv(
    bucket, ENVOLVIDOS_BLOB,
    encoding="iso-8859-1", sep=";", on_bad_lines="skip", low_memory=False,
)
df_env.columns = [c.replace("\xad", "").strip() for c in df_env.columns]
print(f"\n  Colunas disponíveis no arquivo Envolvidos:")
for c in df_env.columns:
    print(f"    {c}  (amostra: {str(df_env[c].iloc[0])[:60]})")

# Parsear data
dt_col_env = next((c for c in df_env.columns
                   if "data" in c.lower() and "hora" in c.lower()), None)
if dt_col_env is None:
    dt_col_env = next((c for c in df_env.columns if "data" in c.lower()), None)

if dt_col_env:
    df_env["_dt"] = pd.to_datetime(df_env[dt_col_env], format="%d/%m/%Y %H:%M", errors="coerce")
    df_env = df_env[df_env["_dt"].notna()].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_env["sem"] = df_env["_dt"].dt.to_period("W")
else:
    print("  [AVISO] Coluna de data não encontrada no Envolvidos — features demográficas serão NaN")
    df_env["sem"] = None

# Identificar colunas de etnia e gênero (REDS tipicamente usa "Raça/Cor" e "Sexo")
# Prioridade: nome exato do campo REDS, fallback por substring mais estrita
_ETNIA_NAMES = ["Raça/Etnia", "Raça / Etnia", "Cor/Raça", "Cor / Raça", "Etnia"]
etnia_col = next((c for c in df_env.columns if c.strip() in _ETNIA_NAMES), None)
if etnia_col is None:
    etnia_col = next((c for c in df_env.columns
                      if "raça" in c.lower() or "raca" in c.lower()), None)

_GENERO_NAMES = ["Sexo - Código", "Sexo", "Gênero", "Genero"]
genero_col = next((c for c in df_env.columns if c.strip() in _GENERO_NAMES), None)
if genero_col is None:
    genero_col = next((c for c in df_env.columns
                       if any(k in c.lower()
                              for k in ["sexo", "genero", "gênero"])), None)

print(f"\n  Coluna de etnia/raça detectada : {etnia_col}")
print(f"  Coluna de gênero/sexo detectada: {genero_col}")

if etnia_col:
    print(f"\n  Valores únicos em '{etnia_col}':")
    for v in df_env[etnia_col].value_counts().head(10).index:
        print(f"    {v}")

if genero_col:
    print(f"\n  Valores únicos em '{genero_col}':")
    for v in df_env[genero_col].value_counts().head(10).index:
        print(f"    {v}")

# Funções de proporção robustas (fillna evita TypeError com NaN float)
def _pct_value(col, patterns):
    """% de registros cujo valor (str) contém qualquer padrão da lista."""
    regex = "|".join(p.upper() for p in patterns)
    def _fn(series):
        s = series.fillna("").astype(str).str.strip().str.upper()
        return s.str.contains(regex, regex=True, na=False).mean() * 100
    return (col, _fn)

# Detectar colunas LGBTQIA+ específicas presentes no REDS
id_genero_col     = next((c for c in df_env.columns
                          if "identidade" in c.lower() and "g" in c.lower()), None)
orient_col        = next((c for c in df_env.columns
                          if "orienta" in c.lower() and "sexual" in c.lower()), None)
lgbtqia_flag_col  = next((c for c in df_env.columns
                          if "lgbtqia" in c.lower() or "lgbt" in c.lower()), None)

print(f"\n  Coluna Identidade de Gênero : {id_genero_col}")
print(f"  Coluna Orientação Sexual    : {orient_col}")
print(f"  Coluna LGBTQIA+?            : {lgbtqia_flag_col}")

# Padrões esperados no REDS para etnia e gênero
PADROES_BRANCA    = ["BRANCA", "BRANCO"]
PADROES_PARDA     = ["PARDA", "PARDO"]
PADROES_PRETA     = ["PRETA", "PRETO", "NEGRA", "NEGRO"]
PADROES_FEMININO  = ["F"]
PADROES_MASCULINO = ["M"]
PADROES_TRANS     = ["TRANS", "TRAVESTI", "NÃO BINÁRIO", "NAO BINARIO",
                     "TRANSGÊNERO", "TRANSGENERO"]
PADROES_LGBTQIA   = ["SIM"]
PADROES_HOMOBI    = ["HOMOSSEXUAL", "BISSEXUAL", "GAY", "LÉSBICA", "LESBICA"]

demo_agg_dict = {}
if etnia_col:
    demo_agg_dict["pct_branca"]  = _pct_value(etnia_col, PADROES_BRANCA)
    demo_agg_dict["pct_parda"]   = _pct_value(etnia_col, PADROES_PARDA)
    demo_agg_dict["pct_preta"]   = _pct_value(etnia_col, PADROES_PRETA)

if genero_col:
    demo_agg_dict["pct_feminino"]  = _pct_value(genero_col, PADROES_FEMININO)
    demo_agg_dict["pct_masculino"] = _pct_value(genero_col, PADROES_MASCULINO)

if id_genero_col:
    demo_agg_dict["pct_trans"] = _pct_value(id_genero_col, PADROES_TRANS)

if lgbtqia_flag_col:
    demo_agg_dict["pct_lgbtqia"] = _pct_value(lgbtqia_flag_col, PADROES_LGBTQIA)

if orient_col:
    demo_agg_dict["pct_homobi"] = _pct_value(orient_col, PADROES_HOMOBI)

if demo_agg_dict and df_env["sem"].notna().any():
    demo_sem = (
        df_env[df_env["sem"].isin(SEMANAS_ALVO)]
        .groupby("sem")
        .agg(**demo_agg_dict)
        .reindex(SEMANAS_ALVO, fill_value=np.nan)
    )
else:
    demo_sem = pd.DataFrame(index=SEMANAS_ALVO, data=np.nan,
                            columns=list(demo_agg_dict.keys()) or ["pct_branca"])
    print("  [AVISO] Sem dados demográficos processáveis — features serão NaN")

n_demo_sem = demo_sem.notna().any(axis=1).sum()
print(f"\n  Semanas com dados demográficos: {n_demo_sem}/{len(SEMANAS_ALVO)}")
for c in demo_sem.columns:
    print(f"  {c}: NaN={demo_sem[c].isna().sum()}  "
          f"média={demo_sem[c].mean():.1f}%")

# Defasar T-1
demo_def = pd.DataFrame(index=SEMANAS_ALVO)
for col in demo_sem.columns:
    demo_def[col + "_t1"] = demo_sem[col].shift(1)

FEAT_DEMO = [c + "_t1" for c in demo_sem.columns]
print(f"\n  Features demográficas (defasadas T-1): {FEAT_DEMO}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. MONTAR OS 3 DATASETS
# ═════════════════════════════════════════════════════════════════════════════
section("6. CONSTRUINDO OS 3 DATASETS")

# Juntar tudo no índice de semanas da base existente
df_full = df_base[FEAT_BASE + ["n_crimes", "alvo"]].copy()
df_full = df_full.join(df_reddit_feats)
df_full = df_full.join(insta_def)
df_full = df_full.join(geo_def)
df_full = df_full.join(demo_def)

# Resetar índice para CSV
df_full_reset = df_full.reset_index()
df_full_reset["sem"] = df_full_reset["sem"].astype(str)

# 6A. Dataset BASE (apenas autoregressivo + calendário)
COLS_BASE = ["sem", "n_crimes", "alvo"] + FEAT_BASE
df_out_base = df_full_reset[COLS_BASE].copy()

# 6B. Dataset MÍDIA (base + Reddit + Instagram)
FEAT_MIDIA = FEAT_BASE + FEAT_REDDIT + FEAT_INSTA
COLS_MIDIA = ["sem", "n_crimes", "alvo"] + FEAT_MIDIA
df_out_midia = df_full_reset[[c for c in COLS_MIDIA if c in df_full_reset.columns]].copy()

# 6C. Dataset COMPLETO (mídia + geográfico + demográfico)
FEAT_COMPLETO = FEAT_MIDIA + FEAT_GEO + FEAT_DEMO
COLS_COMPLETO = ["sem", "n_crimes", "alvo"] + FEAT_COMPLETO
df_out_completo = df_full_reset[[c for c in COLS_COMPLETO if c in df_full_reset.columns]].copy()

def _relatorio_dataset(nome, df, feat_cols):
    sub(f"Dataset: {nome}  ({len(feat_cols)} features, {len(df)} semanas)")
    print(f"  Shape: {df.shape}")
    print(f"  Alvo — classe 0: {(df['alvo']==0).sum()}  classe 1: {(df['alvo']==1).sum()}")
    nan_feats = [(c, df[c].isna().sum()) for c in feat_cols if c in df.columns]
    nan_com   = [(c, n) for c, n in nan_feats if n > 0]
    if nan_com:
        print(f"  Colunas com NaN (serão imputadas pela mediana no treino):")
        for c, n in nan_com:
            print(f"    {c}: {n} semanas")
    else:
        print(f"  Nenhuma coluna com NaN nas features.")

_relatorio_dataset("BASE", df_out_base, FEAT_BASE)
_relatorio_dataset("MÍDIA", df_out_midia, FEAT_MIDIA)
_relatorio_dataset("COMPLETO", df_out_completo, FEAT_COMPLETO)

print(f"\n  Features por dataset:")
print(f"  BASE     ({len(FEAT_BASE):2d} features): {FEAT_BASE}")
print(f"  MÍDIA    ({len([c for c in FEAT_MIDIA if c in df_out_midia.columns]):2d} features): "
      f"BASE + Reddit + Instagram")
print(f"  COMPLETO ({len([c for c in FEAT_COMPLETO if c in df_out_completo.columns]):2d} features): "
      f"MÍDIA + Geográfico + Demográfico")


# ═════════════════════════════════════════════════════════════════════════════
# 7. SALVAR NO GCS
# ═════════════════════════════════════════════════════════════════════════════
section("7. SALVANDO NO GCS")

upload_csv(bucket, OUT_BASE,     df_out_base)
upload_csv(bucket, OUT_MIDIA,    df_out_midia)
upload_csv(bucket, OUT_COMPLETO, df_out_completo)

print(f"""
RESUMO:
  Dataset BASE     : {df_out_base.shape[0]} semanas × {len(FEAT_BASE)} features
  Dataset MÍDIA    : {df_out_midia.shape[0]} semanas × {len([c for c in FEAT_MIDIA if c in df_out_midia.columns])} features
  Dataset COMPLETO : {df_out_completo.shape[0]} semanas × {len([c for c in FEAT_COMPLETO if c in df_out_completo.columns])} features

  Próximo passo: executar fase4d_classificacao_artigo.py
""")
