#!/usr/bin/env python3
"""
Fase 3 — Reddit MG 2023-2025 + Registros Criminais 2023-2025.

Expande a janela temporal de n=12 (Instagram/Fase 2) para até n=36 meses,
aumentando o poder estatístico da mesma bateria de testes já validada.

CONTROLE DE EXECUÇÃO:
  PARAR_APOS_ETAPA1 = True   → para após mostrar contagens de subreddit (confirme primeiro)
  PARAR_APOS_ETAPA1 = False  → executa a análise completa

  RODAR_VADER = False  → pula análise VADER secundária (salva ~90 MB de download)
  RODAR_VADER = True   → executa VADER com join interno e reporte de cobertura
"""

# ── ADAPTADO PARA EXECUCAO LOCAL ──────────────────────────────────────────────
# Este script foi adaptado para rodar sem Google Cloud Storage.
# Coloque os arquivos CSV em:   analise_batch/dados/
# Os resultados sao salvos em:  analise_batch/saida/
# Mais detalhes:                analise_batch/README_analise.md
# ──────────────────────────────────────────────────────────────────────────────


import io
import math
import sys
import warnings

import numpy as np
import pandas as pd
from gcs_local import gcs_client, storage
from scipy.stats import t as t_dist
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ── Controle de execução ──────────────────────────────────────────────────────
PARAR_APOS_ETAPA1 = False  # confirmado na Etapa 1: subreddits corretos, 195.421 posts MG
RODAR_VADER       = False  # análise secundária; mude para True quando quiser

# ── Configuração ──────────────────────────────────────────────────────────────
# KEY_FILE      = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET        = "lgbtminas-dados"

REDDIT_PREFIX  = "rede social/reddit/analysis/tybyria/"
VADER_PREFIX   = "rede social/reddit/analysis/vader/"
REDDIT_BLOB    = "rede social/reddit/analysis/tybyria/reddit_tybyria_RC_total.csv"
VADER_BLOB     = "rede social/reddit/analysis/vader/reddit_vader_RC_total.csv"
CRIME_BLOB     = (
    "criminal/processed/"
    "DIS - Envolvidos - Eventos de LGBTQIAfobia - Jan 2023 a Dez 2025.csv"
)
INSTA_MES_BLOB = "rede social/instagram/processed/fase2_mensal_instagram.csv"

OUT_REDDIT_MES  = "rede social/reddit/processed/fase3_mensal_reddit_mg.csv"
OUT_CRIME_MES   = "criminal/processed/fase3_mensal_criminal_2023_2025.csv"
OUT_COMBINADO   = "analysis/fase3_serie_mensal_combinada.csv"

THRESHOLD       = 0.30
MESES_ALVO      = pd.period_range(start="2023-01", end="2025-12", freq="M")
MAX_LAG_CCF     = 6   # com n=36, n_eff=30 no lag extremo — aceitável

MG_SUBREDDITS = frozenset([
    "BeloHorizonte", "MinasGerais", "juizdefora",
    "Uberlandia", "Uberaba", "OuroPreto", "montesclaros_",
])

SEP  = "=" * 70
SEP2 = "-" * 70


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES REUTILIZADAS (validadas na Fase 2 — não modificar)
# ─────────────────────────────────────────────────────────────────────────────
def section(t):  print(f"\n{SEP}\n{t}\n{SEP}")
def sub(t):      print(f"\n{SEP2}\n{t}\n{SEP2}")

# gcs_client() importada de gcs_local

def read_blob_csv(bkt, blob_name, **kw):
    data = bkt.blob(blob_name).download_as_bytes()
    return pd.read_csv(io.BytesIO(data), **kw)

def upload_csv(bkt, blob_name, df):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    bkt.blob(blob_name).upload_from_string(csv_bytes, content_type="text/csv")
    print(f"  [OK] gs://{BUCKET}/{blob_name}  ({len(csv_bytes)/1024:.1f} KB)")

def fix_coord(s):
    try:
        s = str(s).replace(".", "")
        return float(s[:3] + "." + s[3:]) if s.startswith("-") else float(s[:2] + "." + s[2:])
    except Exception:
        return None

def _t_cdf(t_val, df):
    if df > 30:
        x = t_val / math.sqrt(1 + t_val * t_val / df)
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return float(t_dist.cdf(t_val, df=df))

def pearson_with_p(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    if n < 3 or abs(r) >= 1.0:
        return r, float("nan"), n
    t_val = r * math.sqrt(n - 2) / math.sqrt(max(1 - r**2, 1e-9))
    return r, 2 * (1 - _t_cdf(abs(t_val), df=n - 2)), n

def spearman_with_p(x, y):
    xr = pd.Series(x, dtype=float).rank()
    yr = pd.Series(y, dtype=float).rank()
    return pearson_with_p(xr.values, yr.values)

def corr_ci_95(r, n):
    if n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return round(math.tanh(z - 1.96 * se), 4), round(math.tanh(z + 1.96 * se), 4)

def ccf_table(x, y, max_lag=6):
    """CCF com t-distribuição exata e n_eff por lag (corrigido na Fase 2)."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        xa = x[: n - lag] if lag > 0 else (x[-lag:] if lag < 0 else x)
        ya = y[lag:]       if lag > 0 else (y[: n + lag] if lag < 0 else y)
        n_eff = len(xa)
        r = float(np.corrcoef(xa, ya)[0, 1])
        if n_eff < 4 or abs(r) >= 1.0:
            rows.append({"lag": lag, "r": round(r, 4), "n_eff": n_eff,
                         "p": float("nan"), "sig": "n/a", "sig_bonf": "n/a"})
            continue
        t_val = r * math.sqrt(n_eff - 2) / math.sqrt(max(1 - r**2, 1e-9))
        p = 2 * (1 - _t_cdf(abs(t_val), df=n_eff - 2))
        n_tests = 2 * max_lag + 1
        p_bonf = min(p * n_tests, 1.0)
        rows.append({
            "lag": lag, "r": round(r, 4), "n_eff": n_eff,
            "p": round(p, 4),
            "sig":      "*" if p < 0.05      else ("†" if p < 0.10      else ""),
            "p_bonf": round(p_bonf, 4),
            "sig_bonf": "*" if p_bonf < 0.05 else ("†" if p_bonf < 0.10 else ""),
        })
    return pd.DataFrame(rows)

def r_critico(n, alpha=0.05, bonferroni_n=1):
    """Mínimo |r| para significância dado n e eventual correção de Bonferroni."""
    alpha_adj = alpha / bonferroni_n
    df = max(n - 2, 1)
    t_crit = t_dist.ppf(1 - alpha_adj / 2, df=df)
    return round(t_crit / math.sqrt(df + t_crit**2), 4)


# ─────────────────────────────────────────────────────────────────────────────
# NOVA FUNÇÃO: diagnóstico ADF + KPSS + ACF1 + DW (amplia Fase 2 com KPSS)
# ─────────────────────────────────────────────────────────────────────────────
def diagnostico_serie(nome, s_arr, trend_vec):
    """
    ADF + KPSS + ACF(1) + Durbin-Watson.
    Decisão: precisa_diff = ADF p > 0.05 AND KPSS p < 0.05 (ambos concordam).
    """
    n = len(s_arr)

    # ADF — H0: raiz unitária (não-estacionária)
    try:
        adf_stat, adf_p, *_ = adfuller(s_arr, autolag="AIC", regression="c")
        adf_est = "SIM" if adf_p < 0.05 else "NÃO"
    except Exception:
        adf_stat, adf_p, adf_est = float("nan"), 1.0, "ERRO"

    # KPSS — H0: série é ESTACIONÁRIA (oposto do ADF)
    # p ≥ 0.10 → falha em rejeitar H0 → estacionária
    # p ≤ 0.01 → rejeita H0 → não-estacionária
    # statsmodels clampea p em [0.01, 0.10]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(s_arr, regression="c", nlags="auto")
        kpss_est = "SIM" if kpss_p > 0.05 else "NÃO"
        kpss_p_str = (">0.10" if kpss_p >= 0.10 else
                      ("<0.01" if kpss_p <= 0.01 else f"{kpss_p:.4f}"))
    except Exception:
        kpss_stat, kpss_p, kpss_est, kpss_p_str = float("nan"), 1.0, "ERRO", "ERRO"

    # ACF lag=1
    acf1 = float(np.corrcoef(s_arr[:-1], s_arr[1:])[0, 1]) if n > 2 else float("nan")

    # DW nos resíduos de OLS(série ~ tendência linear)
    X = np.column_stack([np.ones(n), trend_vec])
    beta = np.linalg.lstsq(X, s_arr, rcond=None)[0]
    dw = float(durbin_watson(s_arr - X @ beta))

    # Ambos os testes devem concordar em NÃO-estacionariedade para diferenciar
    precisa_diff = (adf_p > 0.05 and kpss_p < 0.05)

    return {
        "nome": nome,
        "n": n,
        "adf_stat": round(adf_stat, 4), "adf_p": round(adf_p, 4), "adf_est": adf_est,
        "kpss_stat": round(kpss_stat, 4) if not math.isnan(kpss_stat) else float("nan"),
        "kpss_p_str": kpss_p_str, "kpss_p": kpss_p, "kpss_est": kpss_est,
        "acf1": round(acf1, 4), "dw": round(dw, 4),
        "precisa_diff": precisa_diff,
        "s_arr": s_arr,
        "s_diff": np.diff(s_arr),
    }


def print_diagnostico(resultados):
    """Imprime tabela de diagnóstico e retorna se alguma série precisa diff."""
    print(f"\n{'Série':<18} {'n':>4} {'ADF':>9} {'ADF p':>8} {'ADF est?':>9}"
          f" {'KPSS':>7} {'KPSS p':>7} {'KPSS est?':>10}"
          f" {'ACF(1)':>7} {'DW':>6} {'Diff?':>7}")
    print("-" * 105)
    for d in resultados:
        print(
            f"  {d['nome']:<16} {d['n']:>4} {d['adf_stat']:>9.4f} {d['adf_p']:>8.4f}"
            f" {d['adf_est']:>9} {d['kpss_stat']:>7.4f} {d['kpss_p_str']:>7}"
            f" {d['kpss_est']:>10} {d['acf1']:>+7.4f} {d['dw']:>6.3f}"
            f" {'→ DIFF' if d['precisa_diff'] else 'ok':>7}"
        )
    print("\n  Decisão: diferenciar SOMENTE se ADF (p>0.05) E KPSS (p<0.05) concordam.")
    print("  KPSS p clamped em [0.01, 0.10] pelo statsmodels.")
    any_diff = any(d["precisa_diff"] for d in resultados)
    return any_diff


def print_ccf(ccf_df, label, n_lags):
    """Imprime tabela CCF com p bruto e p corrigido por Bonferroni."""
    n_tests = n_lags * 2 + 1
    print(f"\n  {label}  (Bonferroni: {n_tests} testes, α_adj = {0.05/n_tests:.4f})")
    print(ccf_df[["lag", "r", "n_eff", "p", "sig", "p_bonf", "sig_bonf"]].to_string(index=False))
    validos = ccf_df[ccf_df["p"].notna()]
    if len(validos):
        melhor = validos.loc[validos["r"].abs().idxmax()]
        print(f"  Lag ótimo: {int(melhor['lag']):+d}  r={melhor['r']:+.4f}"
              f"  p={melhor['p']:.4f}{melhor['sig']}"
              f"  p_bonf={melhor['p_bonf']:.4f}{melhor['sig_bonf']}")
    return ccf_df


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR REDDIT — 36 arquivos tybyria + diagnóstico de subreddits
# ─────────────────────────────────────────────────────────────────────────────
section("1. CARREGANDO REDDIT (ARQUIVO CONSOLIDADO 2023-2025)")

client = gcs_client()
bucket = client.bucket(BUCKET)

print(f"\n  Carregando arquivo consolidado: {REDDIT_BLOB}")
print("  (pode demorar ~30s) ...")
df_reddit_raw = read_blob_csv(bucket, REDDIT_BLOB, low_memory=False)
print(f"\n  DataFrame bruto (antes de qualquer filtro): {df_reddit_raw.shape}")

# ── Diagnóstico de subreddits ANTES do filtro ─────────────────────────────────
sub("1A. Valores únicos da coluna 'subreddit' (ANTES do filtro)")
print("\n  IMPORTANTE: confirme que não há variações de case (ex: 'belohorizonte') "
      "que possam causar descarte inadvertido de linhas.\n")

contagem_sub = df_reddit_raw["subreddit"].value_counts()
print(contagem_sub.to_string())

sub_raw_set = set(df_reddit_raw["subreddit"].unique())
nao_na_lista = sub_raw_set - MG_SUBREDDITS - {"SubredditsBrasil"}
if nao_na_lista:
    print(f"\n  [ATENÇÃO] Subreddits encontrados nos dados que NÃO estão em nenhuma lista:")
    for s in sorted(nao_na_lista):
        print(f"    '{s}'  ({int(contagem_sub.get(s, 0)):,} linhas)")
else:
    print("\n  [OK] Todos os subreddits encontrados pertencem a MG_SUBREDDITS ou SubredditsBrasil.")

print(f"\n  SubredditsBrasil (EXCLUÍDO por ser nacional): "
      f"{int(contagem_sub.get('SubredditsBrasil', 0)):,} linhas")

# ── Aplicar filtro MG ─────────────────────────────────────────────────────────
df_reddit = df_reddit_raw[df_reddit_raw["subreddit"].isin(MG_SUBREDDITS)].copy()

sub("1B. Contagem APÓS filtro (somente subreddits de MG)")
contagem_mg = df_reddit["subreddit"].value_counts()
print(contagem_mg.to_string())
print(f"\n  Total MG  : {len(df_reddit):,} linhas")
print(f"  Total bruto: {len(df_reddit_raw):,} linhas")
print(f"  Descartado : {len(df_reddit_raw) - len(df_reddit):,} linhas "
      f"({(1 - len(df_reddit)/len(df_reddit_raw))*100:.1f}%)")

if PARAR_APOS_ETAPA1:
    print(f"""
{SEP}
PAUSA PROGRAMADA — PARAR_APOS_ETAPA1 = True
{SEP}
Confirme acima:
  1. Os subreddits de MG batem exatamente com a lista esperada?
  2. Há algum nome com variação de case ou grafia inesperada?
  3. O total MG ({len(df_reddit):,} linhas) parece razoável?

Para prosseguir: mude PARAR_APOS_ETAPA1 = False no topo do script e execute novamente.
""")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIMPEZA DO REDDIT — datas e nulos
# ─────────────────────────────────────────────────────────────────────────────
section("2. LIMPEZA DO REDDIT")

# Arquivo consolidado usa colunas ano/mes/dia em vez de created_utc (Unix timestamp)
df_reddit["mes"] = pd.to_datetime(
    dict(year=df_reddit["ano"], month=df_reddit["mes"], day=1)
).dt.to_period("M")

n_fora = (~df_reddit["mes"].isin(MESES_ALVO)).sum()
if n_fora:
    print(f"  [AVISO] {n_fora} linhas fora do intervalo 2023-01/2025-12 — removidas.")
    df_reddit = df_reddit[df_reddit["mes"].isin(MESES_ALVO)].copy()

n_null_texto = df_reddit["text_clean"].isnull().sum()
periodo_min = df_reddit["mes"].min()
periodo_max = df_reddit["mes"].max()
print(f"  Período  : {periodo_min} → {periodo_max}")
print(f"  Linhas MG: {len(df_reddit):,}  |  nulos em text_clean: {n_null_texto}")
print(f"  tybyria_score nulos: {df_reddit['tybyria_score'].isnull().sum()}")
print(f"  Range tybyria_score: [{df_reddit['tybyria_score'].min():.4f}, "
      f"{df_reddit['tybyria_score'].max():.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARREGAR REGISTROS CRIMINAIS 2023-2025
# ─────────────────────────────────────────────────────────────────────────────
section("3. CARREGANDO REGISTROS CRIMINAIS 2023-2025")
print(f"\n  Blob: gs://{BUCKET}/{CRIME_BLOB}")

df_crime = read_blob_csv(
    bucket, CRIME_BLOB,
    encoding="iso-8859-1", sep=";", on_bad_lines="skip", low_memory=False,
)
df_crime.columns = [c.replace("\xad", "").strip() for c in df_crime.columns]
print(f"  Shape bruto: {df_crime.shape}  |  nulos em ID Ocorrência: "
      f"{df_crime['ID Ocorrência'].isnull().sum()}")

# Arquivo Envolvidos: múltiplas linhas por ocorrência (uma por pessoa envolvida)
# Precisamos deduplir por ID Ocorrência para contar eventos únicos
df_crime = df_crime.dropna(subset=["ID Ocorrência"]).copy()
n_envolvidos = len(df_crime)

df_crime["dt_ocorrencia"] = pd.to_datetime(
    df_crime["Data/Hora Ocorrência"], format="%d/%m/%Y %H:%M", errors="coerce"
)
df_crime = df_crime[df_crime["dt_ocorrencia"].dt.year.isin([2023, 2024, 2025])].copy()
df_crime["mes"] = df_crime["dt_ocorrencia"].dt.to_period("M")

# Deduplicar: manter uma linha por ocorrência única (evita inflação na contagem)
df_crime_eventos = df_crime.drop_duplicates(subset=["ID Ocorrência"]).copy()

print(f"  Envolvidos (total linhas)  : {n_envolvidos:,}")
print(f"  Ocorrências únicas         : {len(df_crime_eventos):,}")
print(f"  Intervalo   : {df_crime_eventos['dt_ocorrencia'].min()} → {df_crime_eventos['dt_ocorrencia'].max()}")
print(f"  Meses       : {sorted(df_crime_eventos['mes'].unique().astype(str))}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGREGAÇÃO MENSAL
# ─────────────────────────────────────────────────────────────────────────────
section("4. AGREGAÇÃO MENSAL")

# ── 4A. Reddit — série geral MG ───────────────────────────────────────────────
sub("4A. Reddit MG — série mensal (todos os subreddits de MG)")

reddit_mes = (
    df_reddit.groupby("mes")
    .agg(
        n_posts           = ("id",            "count"),
        n_toxicos         = ("tybyria_score",  lambda s: (s >= THRESHOLD).sum()),
        media_tybyria     = ("tybyria_score",  "mean"),
        mediana_tybyria   = ("tybyria_score",  "median"),
        std_tybyria       = ("tybyria_score",  "std"),
        n_lgbt_term       = ("has_lgbt_term",  "sum"),
        n_hate_term       = ("has_hate_term",  "sum"),
        n_mg_city         = ("has_mg_city",    "sum"),
    )
    .reindex(MESES_ALVO, fill_value=0)
)
reddit_mes["pct_toxicos"] = (
    reddit_mes["n_toxicos"] / reddit_mes["n_posts"].replace(0, np.nan) * 100
).round(2)
reddit_mes.index.name = "mes"
print(reddit_mes[["n_posts", "n_toxicos", "pct_toxicos", "media_tybyria",
                   "mediana_tybyria", "std_tybyria"]].to_string())

# ── 4B. Reddit — por subreddit × mês ─────────────────────────────────────────
sub("4B. Reddit — pct_toxicos e n_posts por subreddit × mês")

reddit_sub_mes = (
    df_reddit.groupby(["mes", "subreddit"])
    .agg(
        n_posts       = ("id",           "count"),
        n_toxicos     = ("tybyria_score", lambda s: (s >= THRESHOLD).sum()),
        media_tybyria = ("tybyria_score", "mean"),
    )
    .reset_index()
)
reddit_sub_mes["pct_toxicos"] = (
    reddit_sub_mes["n_toxicos"] / reddit_sub_mes["n_posts"] * 100
).round(2)

pivot_n   = reddit_sub_mes.pivot(index="mes", columns="subreddit", values="n_posts").reindex(MESES_ALVO)
pivot_pct = reddit_sub_mes.pivot(index="mes", columns="subreddit", values="pct_toxicos").reindex(MESES_ALVO)
pivot_tyb = reddit_sub_mes.pivot(index="mes", columns="subreddit", values="media_tybyria").reindex(MESES_ALVO)

print("\nn_posts por subreddit:")
print(pivot_n.to_string())
print(f"\npct_toxicos (tybyria ≥ {THRESHOLD}) por subreddit:")
print(pivot_pct.round(2).to_string())
print("\nmedia_tybyria por subreddit:")
print(pivot_tyb.round(4).to_string())

# ── 4C. Resumo por subreddit (série completa) ─────────────────────────────────
sub("4C. Perfil geral por subreddit (período completo 2023-2025)")
print(
    df_reddit.groupby("subreddit")["tybyria_score"]
    .agg(n="count", media="mean", mediana="median", std="std",
         pct_toxico=lambda s: (s >= THRESHOLD).mean() * 100)
    .round(4).to_string()
)

# ── 4D. Criminal ──────────────────────────────────────────────────────────────
sub("4D. Registros criminais — série mensal 2023-2025")

crime_mes = (
    df_crime_eventos.groupby("mes")
    .agg(
        n_crimes          = ("ID Ocorrência",    "count"),
        municipios_unicos = ("Município (Fato)", "nunique"),
    )
    .reindex(MESES_ALVO, fill_value=0)
)
crime_mes.index.name = "mes"
print(crime_mes.to_string())

sub("4E. Tipos de crime (Natureza Principal — top 10)")
print(df_crime_eventos["Natureza Principal"].value_counts().head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. SÉRIE TEMPORAL COMBINADA
# ─────────────────────────────────────────────────────────────────────────────
section("5. SÉRIE TEMPORAL COMBINADA (Reddit MG × Crimes, 2023-2025)")

df_comb = reddit_mes[["n_posts", "n_toxicos", "pct_toxicos",
                       "media_tybyria", "std_tybyria"]].join(
    crime_mes[["n_crimes", "municipios_unicos"]], how="outer"
).reindex(MESES_ALVO)

df_comb.index.name = "mes"
df_comb = df_comb.reset_index()
df_comb["mes"] = df_comb["mes"].astype(str)

print(df_comb.to_string(index=False))

meses_validos = df_comb[(df_comb["n_posts"] > 0) & (df_comb["n_crimes"] > 0)]
print(f"\n  Meses com dados em ambas as séries: {len(meses_validos)}")
sem_reddit = df_comb[df_comb["n_posts"] == 0]["mes"].tolist()
sem_crime  = df_comb[df_comb["n_crimes"] == 0]["mes"].tolist()
if sem_reddit: print(f"  [AVISO] Meses sem Reddit MG: {sem_reddit}")
if sem_crime:  print(f"  [AVISO] Meses sem crimes   : {sem_crime}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISE ESTATÍSTICA
# ─────────────────────────────────────────────────────────────────────────────
section("6. ANÁLISE ESTATÍSTICA — Reddit MG (n até 36)")

mv = meses_validos.copy()
n_obs = len(mv)
x_pct = mv["pct_toxicos"].values
x_tyb = mv["media_tybyria"].values
x_n   = mv["n_posts"].values
y_cr  = mv["n_crimes"].values
trend_vec = np.arange(1, n_obs + 1, dtype=float)

# Limiares de significância para este n
r_crit_nc = r_critico(n_obs, alpha=0.05, bonferroni_n=1)
r_crit_b  = r_critico(n_obs, alpha=0.05, bonferroni_n=2 * MAX_LAG_CCF + 1)
print(f"""
  n = {n_obs} meses sobrepostos.
  |r| mínimo para p < 0.05 (sem correção)        : {r_crit_nc}
  |r| mínimo para p < 0.05 (Bonferroni, {2*MAX_LAG_CCF+1} testes): {r_crit_b}
  Resultados devem ser interpretados como exploratórios. Correlação ≠ causalidade.
""")

# ── 6.1 Estatísticas descritivas ─────────────────────────────────────────────
sub("6.1 Estatísticas descritivas das séries mensais")
desc_cols = ["n_posts", "pct_toxicos", "media_tybyria", "n_crimes"]
print(df_comb[df_comb["mes"].isin(mv["mes"])][desc_cols].describe().round(4).to_string())

# ── 6.2 Correlações Pearson e Spearman ───────────────────────────────────────
sub("6.2 Correlações de Pearson e Spearman (× n_crimes)")

pairs = [
    ("pct_toxicos",   x_pct, "% tóxicos (tybyria ≥ 0.30)"),
    ("media_tybyria", x_tyb, "Média tybyria_score"),
    ("n_posts",       x_n,   "Volume de posts"),
]
print(f"\n{'Variável':<28} {'Pearson r':>10} {'p':>8} {'IC 95%':>18}"
      f" {'Spearman ρ':>12} {'p':>8}")
print("-" * 90)
for _, x_arr, label in pairs:
    r_p, p_p, n_p = pearson_with_p(x_arr, y_cr)
    r_s, p_s, _   = spearman_with_p(x_arr, y_cr)
    lo, hi         = corr_ci_95(r_p, n_p)
    sp = "*" if p_p < 0.05 else ("†" if p_p < 0.10 else "")
    ss = "*" if p_s < 0.05 else ("†" if p_s < 0.10 else "")
    print(f"{label:<28} {r_p:>+10.4f} {p_p:>8.4f}{sp:<1}"
          f" [{lo:.3f},{hi:.3f}] {r_s:>+12.4f} {p_s:>8.4f}{ss:<1}")
print("  * p < 0.05   † p < 0.10   IC = Fisher 95%")

# ── 6.3 Diagnóstico ADF + KPSS ───────────────────────────────────────────────
sub("6.3 Diagnóstico de estacionariedade — ADF + KPSS (ampliado da Fase 2)")
print("""
  ADF  H0: raiz unitária (não-estacionária) → rejeitar = estacionária
  KPSS H0: ESTACIONÁRIA (oposto!) → rejeitar = não-estacionária
  Decisão: diferenciar APENAS se ambos concordam (ADF p>0.05 E KPSS p<0.05)
""")

series_para_diag = [
    ("pct_toxicos",   x_pct),
    ("media_tybyria", x_tyb),
    ("n_posts",       x_n),
    ("n_crimes",      y_cr),
]
diag_results = [diagnostico_serie(n, s, trend_vec) for n, s in series_para_diag]
usar_diff = print_diagnostico(diag_results)

# Séries diferenciadas (sempre computadas; usadas somente se usar_diff=True)
d_pct = {d["nome"]: d for d in diag_results}
x_pct_d = d_pct["pct_toxicos"]["s_diff"]
x_tyb_d = d_pct["media_tybyria"]["s_diff"]
x_n_d   = d_pct["n_posts"]["s_diff"]
y_cr_d  = d_pct["n_crimes"]["s_diff"]
ccf_df_d = None

if usar_diff:
    print(f"\n  [INFO] Diferenciação acionada. n_eff = {n_obs - 1} após diff(1).")
else:
    print(f"\n  [INFO] Nenhuma série requer diferenciação.")

# ── 6.3b Volume (n_posts) × crimes — níveis e diferenciado ───────────────────
sub("6.3b Correlação n_posts × n_crimes — antes e depois da diferenciação")
print("  Verifica se r=+0.377* (seção 6.2) sobrevive à remoção de tendência.\n")

d_n = d_pct["n_posts"]
d_c = d_pct["n_crimes"]

r_niv, p_niv, n_niv = pearson_with_p(x_n, y_cr)
lo_niv, hi_niv = corr_ci_95(r_niv, n_niv)
sig_niv = "*" if p_niv < 0.05 else ("†" if p_niv < 0.10 else "ns")

r_dif, p_dif, n_dif = pearson_with_p(x_n_d, y_cr_d)
lo_dif, hi_dif = corr_ci_95(r_dif, n_dif)
sig_dif = "*" if p_dif < 0.05 else ("†" if p_dif < 0.10 else "ns")

print(f"  {'Versão':<22} {'Pearson r':>10} {'p':>8} {'sig':>4} {'IC 95%':>18}")
print("  " + "-" * 65)
print(f"  {'Níveis (n='+str(n_niv)+')':<22} {r_niv:>+10.4f} {p_niv:>8.4f} {sig_niv:>4}"
      f" [{lo_niv:.3f},{hi_niv:.3f}]")
print(f"  {'Diferenciada (n='+str(n_dif)+')':<22} {r_dif:>+10.4f} {p_dif:>8.4f} {sig_dif:>4}"
      f" [{lo_dif:.3f},{hi_dif:.3f}]")

print()
if sig_dif == "*":
    print("  → r significativo PERSISTE após diferenciação — associação de volume robusta.")
elif sig_dif == "†":
    print("  → r marginalmente significativo após diferenciação — interpretar com cautela.")
else:
    print("  → r NÃO significativo após diferenciação — correlação de níveis era espúria"
          " por tendência comum (ambas as séries crescem ao longo do período).")

print(f"\n  Estacionariedade de n_posts: "
      f"ADF p={d_n['adf_p']:.4f}({d_n['adf_est']}) | "
      f"KPSS p={d_n['kpss_p_str']}({d_n['kpss_est']}) | "
      f"ACF(1)={d_n['acf1']:+.4f} | "
      f"{'→ não-estacionária (diff recomendada)' if d_n['precisa_diff'] else 'estacionária'}")

# ── 6.4 CCF pct_toxicos × n_crimes ───────────────────────────────────────────
sub("6.4 CCF — pct_toxicos × n_crimes")
print("""
  lag < 0 → discurso tóxico PRECEDE crimes em |lag| meses (hipótese principal)
  lag = 0 → relação contemporânea
  lag > 0 → crimes PRECEDEM discurso tóxico
""")

ccf_df = None
if n_obs >= 2 * MAX_LAG_CCF + 4:
    print("  [A] NÍVEIS BRUTOS:")
    ccf_df = ccf_table(x_pct, y_cr, max_lag=MAX_LAG_CCF)
    print_ccf(ccf_df, "pct_toxicos × n_crimes", MAX_LAG_CCF)

    if usar_diff:
        print(f"\n  [B] SÉRIES DIFERENCIADAS (n_eff={len(x_pct_d)}):")
        ccf_df_d = ccf_table(x_pct_d, y_cr_d, max_lag=MAX_LAG_CCF - 1)
        print_ccf(ccf_df_d, "Δpct_toxicos × Δn_crimes", MAX_LAG_CCF - 1)
else:
    print(f"  [AVISO] n={n_obs} insuficiente para max_lag={MAX_LAG_CCF}.")

# ── 6.5 CCF media_tybyria × n_crimes ─────────────────────────────────────────
sub("6.5 CCF — media_tybyria × n_crimes")
ccf_tyb = None
if n_obs >= 2 * MAX_LAG_CCF + 4:
    print("  [A] NÍVEIS BRUTOS:")
    ccf_tyb = ccf_table(x_tyb, y_cr, max_lag=MAX_LAG_CCF)
    print_ccf(ccf_tyb, "media_tybyria × n_crimes", MAX_LAG_CCF)

    if usar_diff:
        print(f"\n  [B] SÉRIES DIFERENCIADAS:")
        ccf_tyb_d = ccf_table(x_tyb_d, y_cr_d, max_lag=MAX_LAG_CCF - 1)
        print_ccf(ccf_tyb_d, "Δmedia_tybyria × Δn_crimes", MAX_LAG_CCF - 1)

# ── 6.6 Sensibilidade: BeloHorizonte isolado vs. MG agregado ─────────────────
sub("6.6 SENSIBILIDADE — BeloHorizonte (88,3%) vs. MG Agregado")
print("""
  Objetivo: verificar se qualquer padrão da CCF em 6.4 é impulsionado
  pela dinâmica de BeloHorizonte (172.601 posts, 88,3% do total) ou se
  é consistente mesmo quando analisado isoladamente.

  Interpretação:
    r_MG ≈ r_BH  →  padrão reflete BH (esperado pelo peso); achado robusto dentro de BH
    r_MG >> r_BH  →  outros subreddits menores "amplificam" o sinal; interpretar com cautela
    r_BH significativo, r_MG não  →  sinal de BH diluído pelo ruído dos menores
""")

# Série mensal BeloHorizonte isolada
bh_mes = (
    df_reddit[df_reddit["subreddit"] == "BeloHorizonte"]
    .groupby("mes")
    .agg(
        n_posts_bh       = ("id",            "count"),
        n_toxicos_bh     = ("tybyria_score",  lambda s: (s >= THRESHOLD).sum()),
        media_tybyria_bh = ("tybyria_score",  "mean"),
    )
    .reindex(MESES_ALVO, fill_value=0)
)
bh_mes["pct_toxicos_bh"] = (
    bh_mes["n_toxicos_bh"] / bh_mes["n_posts_bh"].replace(0, np.nan) * 100
).round(2)

# Meses com dados em ambas as séries (BH + crimes)
bh_comb = bh_mes.join(crime_mes["n_crimes"], how="outer").reindex(MESES_ALVO)
mv_bh = bh_comb[(bh_comb["n_posts_bh"] > 0) & (bh_comb["n_crimes"] > 0)]
n_bh = len(mv_bh)
x_pct_bh = mv_bh["pct_toxicos_bh"].values
x_tyb_bh = mv_bh["media_tybyria_bh"].values
y_cr_bh  = mv_bh["n_crimes"].values

print(f"  n BeloHorizonte : {n_bh} meses  |  n MG Agregado: {n_obs} meses")
print(f"  Posts/mês médio BH: {bh_mes['n_posts_bh'].replace(0, np.nan).mean():.0f}"
      f"  |  Posts/mês médio MG: {reddit_mes['n_posts'].replace(0, np.nan).mean():.0f}")

# Estatísticas descritivas BH
print(f"\n  pct_toxicos_bh: média={x_pct_bh.mean():.2f}%  "
      f"std={x_pct_bh.std():.2f}%  "
      f"min={x_pct_bh.min():.2f}%  max={x_pct_bh.max():.2f}%")
print(f"  media_tybyria_bh: média={x_tyb_bh.mean():.4f}  "
      f"std={x_tyb_bh.std():.4f}")

# CCF BH
ccf_bh = None
if n_bh >= 2 * MAX_LAG_CCF + 4:
    ccf_bh = ccf_table(x_pct_bh, y_cr_bh, max_lag=MAX_LAG_CCF)

# Tabela comparativa lado a lado
print(f"\n  {'':6} {'── MG Agregado ──':^30}  {'── BeloHorizonte ──':^30}")
print(f"  {'Lag':>5}  {'r_MG':>8} {'p_MG':>8} {'sig':>4} {'BonfMG':>7}"
      f"   {'r_BH':>8} {'p_BH':>8} {'sig':>4} {'BonfBH':>7}  {'Δr':>7}")
print("  " + "-" * 85)

if ccf_df is not None and ccf_bh is not None:
    for _, row_m in ccf_df.iterrows():
        lag = int(row_m["lag"])
        row_b = ccf_bh[ccf_bh["lag"] == lag]
        if row_b.empty:
            continue
        row_b = row_b.iloc[0]
        delta = row_b["r"] - row_m["r"]
        print(
            f"  {lag:>5}  {row_m['r']:>+8.4f} {row_m['p']:>8.4f} {row_m['sig']:>4}"
            f" {row_m['sig_bonf']:>7}"
            f"   {row_b['r']:>+8.4f} {row_b['p']:>8.4f} {row_b['sig']:>4}"
            f" {row_b['sig_bonf']:>7}  {delta:>+7.4f}"
        )

    # Lag ótimo em cada série
    best_mg = ccf_df.loc[ccf_df["r"].abs().idxmax()]
    best_bh = ccf_bh.loc[ccf_bh["r"].abs().idxmax()]
    print(f"\n  Lag ótimo MG: {int(best_mg['lag']):+d}  "
          f"r={best_mg['r']:+.4f}  p={best_mg['p']:.4f}{best_mg['sig']}")
    print(f"  Lag ótimo BH: {int(best_bh['lag']):+d}  "
          f"r={best_bh['r']:+.4f}  p={best_bh['p']:.4f}{best_bh['sig']}")

    if int(best_mg["lag"]) == int(best_bh["lag"]):
        print("\n  → Lags ótimos CONCORDAM: padrão reflete principalmente dinâmica de BH.")
    else:
        print("\n  → Lags ótimos DIVERGEM: outros subreddits alteram a estrutura temporal.")

    print("\n  CCF media_tybyria — BH isolado:")
    if n_bh >= 2 * MAX_LAG_CCF + 4:
        ccf_tyb_bh = ccf_table(x_tyb_bh, y_cr_bh, max_lag=MAX_LAG_CCF)
        print_ccf(ccf_tyb_bh, "media_tybyria_BH × n_crimes", MAX_LAG_CCF)
elif n_bh < 2 * MAX_LAG_CCF + 4:
    print(f"  [AVISO] n_bh={n_bh} insuficiente para max_lag={MAX_LAG_CCF}.")
else:
    print("  [AVISO] CCF MG (seção 6.4) não disponível — execute sem restrição de n.")


# ─────────────────────────────────────────────────────────────────────────────
# 6.7 TESTE DE CAUSALIDADE DE GRANGER (séries diferenciadas)
# ─────────────────────────────────────────────────────────────────────────────
sub("6.7 Teste de Causalidade de Granger — séries diferenciadas, max_lag=4")
print("""
  Hipótese testada: valores passados de pct_toxicos_diff (ou media_tybyria_diff)
  adicionam poder preditivo sobre n_crimes_diff ALÉM do que a própria história
  da série de crimes já explica?

  H0 (Granger): x_diff NÃO causa y_diff no sentido de Granger
  Rejeitar H0 → x adiciona informação incremental para prever y

  Séries: JÁ DIFERENCIADAS (tendência removida — mesmas usadas na CCF [B])
  Correção de Bonferroni: 4 lags × 2 preditores = 8 testes → α_adj = 0.05/4 = 0.0125
  (Bonferroni aplicado por preditor: 4 lags cada)
""")

from statsmodels.tsa.stattools import grangercausalitytests

MAX_LAG_GC    = 4
ALPHA_GC      = 0.05
ALPHA_GC_BONF = ALPHA_GC / MAX_LAG_GC   # 0.0125

def _run_granger(y_diff, x_diff, label_x, label_y="Δn_crimes"):
    """
    Roda grangercausalitytests e imprime tabela F + p por lag.
    Retorna True se algum lag é significativo após Bonferroni.
    """
    # statsmodels espera array (n, 2): coluna 0 = variável a prever (y), coluna 1 = preditor (x)
    data_gc = np.column_stack([y_diff, x_diff])
    try:
        gc_res = grangercausalitytests(data_gc, maxlag=MAX_LAG_GC, verbose=False)
    except Exception as e:
        print(f"  [ERRO] {e}")
        return False

    print(f"\n  {label_x} → {label_y}")
    print(f"  {'Lag':>4} {'F-stat':>9} {'p (F)':>8} {'sig':>4} {'p_bonf':>8} {'sig_bonf':>9}")
    print("  " + "-" * 52)

    any_sig = False
    for lag in range(1, MAX_LAG_GC + 1):
        f_stat, p_f, df_denom, df_num = gc_res[lag][0]["ssr_ftest"]
        p_bonf = min(p_f * MAX_LAG_GC, 1.0)
        sig      = "*" if p_f    < ALPHA_GC      else ("†" if p_f    < 0.10 else "")
        sig_bonf = "*" if p_bonf < ALPHA_GC      else ("†" if p_bonf < 0.10 else "")
        if p_bonf < ALPHA_GC:
            any_sig = True
        print(f"  {lag:>4} {f_stat:>9.4f} {p_f:>8.4f} {sig:>4} {p_bonf:>8.4f} {sig_bonf:>9}")

    return any_sig

sig_pct = _run_granger(y_cr_d, x_pct_d, "Δpct_toxicos")
sig_tyb = _run_granger(y_cr_d, x_tyb_d, "Δmedia_tybyria")

print(f"\n  α_adj (Bonferroni, 4 lags): {ALPHA_GC_BONF:.4f}")
print("\n  CONCLUSÃO:")
if sig_pct or sig_tyb:
    preditores = []
    if sig_pct: preditores.append("pct_toxicos")
    if sig_tyb: preditores.append("media_tybyria")
    print(f"  Causalidade de Granger DETECTADA em: {', '.join(preditores)}.")
    print("  A toxicidade de discurso adiciona poder preditivo incremental sobre crimes,")
    print("  controlando pela própria história da série de crimes (após diferenciação).")
    print("  ATENÇÃO: causalidade de Granger é preditiva, não causal no sentido estrutural.")
else:
    print("  Nenhuma evidência de causalidade de Granger em nenhuma das duas métricas")
    print("  de toxicidade (pct_toxicos_diff e media_tybyria_diff), controlando pela")
    print("  própria história da série de crimes diferenciada.")
    print("  Conclusão: o discurso tóxico no Reddit MG NÃO adiciona poder preditivo")
    print("  significativo sobre crimes LGBTfóbicos além do que os próprios valores")
    print("  passados de crimes já explicam — após remoção de tendência compartilhada.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANÁLISE VADER SECUNDÁRIA (opcional — RODAR_VADER = True para ativar)
# ─────────────────────────────────────────────────────────────────────────────
section("7. ANÁLISE VADER SECUNDÁRIA")

if not RODAR_VADER:
    print("""
  [PULADA] RODAR_VADER = False.
  Para ativar: mude RODAR_VADER = True no topo do script.

  O que esta seção fará quando ativada:
    - Baixar os 36 arquivos vader (~90 MB)
    - Join interno com tybyria por 'id' (subreddits MG)
    - Reportar n_tybyria, n_vader, % cobertura por mês
    - Agregar vader_compound mensal
    - Correlação contemporânea vader_compound × n_crimes (sem CCF)
    - Aviso explícito de que a cobertura reduzida (~59%) limita interpretação
""")
else:
    sub("7A. Carregando arquivo vader consolidado e fazendo join com tybyria")

    print(f"  Carregando: {VADER_BLOB}")
    df_vader_raw = read_blob_csv(bucket, VADER_BLOB, low_memory=False)
    print(f"  Shape bruto: {df_vader_raw.shape}")

    # Filtro MG e join com tybyria por id
    df_vader_mg = df_vader_raw[df_vader_raw["subreddit"].isin(MG_SUBREDDITS)].copy()
    df_vader_mg["mes"] = pd.to_datetime(
        dict(year=df_vader_mg["ano"], month=df_vader_mg["mes"], day=1)
    ).dt.to_period("M")

    # Join interno: só posts presentes em ambos
    ids_mg_tybyria = set(df_reddit["id"])
    df_joined = df_vader_mg[df_vader_mg["id"].isin(ids_mg_tybyria)].copy()

    n_tyb  = len(df_reddit)
    n_vad  = len(df_vader_mg)
    n_join = len(df_joined)
    print(f"\n  [COBERTURA]")
    print(f"    Posts tybyria MG         : {n_tyb:,}")
    print(f"    Posts vader  MG          : {n_vad:,}  ({n_vad/n_tyb*100:.1f}% do tybyria)")
    print(f"    Após join interno (by id): {n_join:,}  ({n_join/n_tyb*100:.1f}% do tybyria)")
    print(f"\n  AVISO: a análise vader cobre apenas {n_join/n_tyb*100:.1f}% dos posts.")
    print("  Métricas abaixo NÃO devem ser comparadas diretamente com os resultados")
    print("  do tybyria (seção 6), pois representam um subconjunto diferente.\n")

    vader_mes = (
        df_joined.groupby("mes")
        .agg(
            n_join     = ("id",             "count"),
            mean_vader = ("vader_compound", "mean"),
            std_vader  = ("vader_compound", "std"),
        )
        .reindex(MESES_ALVO, fill_value=np.nan)
    )
    vader_mes.index.name = "mes"

    sub("7B. Série mensal vader_compound (cobertura parcial)")
    print(vader_mes.to_string())

    sub("7C. Correlação vader_compound × n_crimes (APENAS contemporânea)")
    mv_v = vader_mes.join(crime_mes["n_crimes"]).dropna()
    if len(mv_v) >= 6:
        rv, pv, nv = pearson_with_p(mv_v["mean_vader"].values, mv_v["n_crimes"].values)
        rv_s, pv_s, _ = spearman_with_p(mv_v["mean_vader"].values, mv_v["n_crimes"].values)
        lo, hi = corr_ci_95(rv, nv)
        print(f"\n  n={nv} | Pearson r={rv:+.4f} p={pv:.4f} IC=[{lo},{hi}]"
              f"  |  Spearman ρ={rv_s:+.4f} p={pv_s:.4f}")
        print("  [AVISO: resultado baseado em subconjunto filtrado — interpretar com cautela]")
    else:
        print(f"  [AVISO] n={len(mv_v)} insuficiente para correlação.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPARAÇÃO REDDIT (n=36) vs. INSTAGRAM (n=12)
# ─────────────────────────────────────────────────────────────────────────────
section("8. COMPARAÇÃO PLATAFORMAS: Reddit MG (n=36) vs. Instagram (n=12)")

insta_mes = read_blob_csv(bucket, INSTA_MES_BLOB, low_memory=False)
insta_mes["mes"] = pd.PeriodIndex(insta_mes["mes"], freq="M")

# CCF Instagram (2025) já foi calculada na Fase 2; recalcular aqui para garantir paridade
meses_2025 = pd.period_range("2025-01", "2025-12", freq="M")
insta_mv = insta_mes[insta_mes["mes"].isin(meses_2025)].copy()
crime_2025 = crime_mes[crime_mes.index.isin(meses_2025)]
insta_comb = insta_mv.set_index("mes").join(crime_2025["n_crimes"], how="inner")
insta_comb = insta_comb.dropna(subset=["pct_toxicos", "n_crimes"])

sub("8A. Séries descritivas por plataforma (2025 — período de sobreposição)")
print(f"\n  {'Métrica':<22} {'Instagram':>12} {'Reddit MG':>12}")
print("  " + "-" * 48)
mv_2025 = meses_validos[meses_validos["mes"].str.startswith("2025")]
for col, label in [("pct_toxicos", "pct_toxicos média"),
                   ("media_tybyria", "media_tybyria"),
                   ("n_posts", "posts/mês médio"),
                   ("n_crimes", "crimes/mês médio")]:
    i_val = insta_comb[col].mean() if col in insta_comb.columns else float("nan")
    r_val = mv_2025[col].mean() if col in mv_2025.columns else float("nan")
    print(f"  {label:<22} {i_val:>12.3f} {r_val:>12.3f}")

sub("8B. CCF lado a lado — pct_toxicos × n_crimes")
print(f"\n  {'Lag':>5} | {'Reddit r':>10} {'p':>8} {'Bonf*':>6} | {'Instagram r':>12} {'p':>8} {'Bonf*':>6}")
print("  " + "-" * 65)

MAX_LAG_COMP = min(4, len(insta_comb) // 3)
if ccf_df is not None and len(insta_comb) >= 6:
    ccf_insta = ccf_table(
        insta_comb["pct_toxicos"].values,
        insta_comb["n_crimes"].values,
        max_lag=MAX_LAG_COMP,
    )
    ccf_reddit_sub = ccf_df[ccf_df["lag"].between(-MAX_LAG_COMP, MAX_LAG_COMP)]

    for _, row_r in ccf_reddit_sub.iterrows():
        lag = int(row_r["lag"])
        row_i = ccf_insta[ccf_insta["lag"] == lag]
        if row_i.empty:
            continue
        row_i = row_i.iloc[0]
        print(f"  {lag:>5} | {row_r['r']:>+10.4f} {row_r['p']:>8.4f} {row_r['sig_bonf']:>6}"
              f" | {row_i['r']:>+12.4f} {row_i['p']:>8.4f} {row_i['sig_bonf']:>6}")
    print("  (* Bonferroni)")

sub("8C. Interpretação comparativa")
print("""
  Leitura sugerida da tabela 8B:
  - Se Reddit e Instagram mostram o mesmo lag ótimo com sinal concordante
    → padrão replicado entre plataformas (mais robusto, mesmo que n pequeno)
  - Se lag ótimo diverge entre plataformas
    → padrão plataforma-específico; não generalizar
  - Se nenhuma plataforma mostra significância após Bonferroni
    → ausência de evidência com os dados disponíveis; ampliar janela ou n
""")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMÁRIO E SALVAR NO GCS
# ─────────────────────────────────────────────────────────────────────────────
section("9. SUMÁRIO E SALVANDO OUTPUTS NO GCS")

r_cont, p_cont, _ = pearson_with_p(x_pct, y_cr)
ccf_melhor = (ccf_df.loc[ccf_df["r"].abs().idxmax()]
              if ccf_df is not None and len(ccf_df) else None)

print(f"""
RESULTADOS PRINCIPAIS:
  Reddit MG     : {len(df_reddit):,} posts válidos | {n_obs} meses sobrepostos com crimes
  Crimes 2023-25: {len(df_crime):,} registros válidos

  Correlação contemporânea pct_toxicos × n_crimes:
    Pearson r = {r_cont:+.4f}  p = {p_cont:.4f}
    {"SIGNIFICATIVO — interpretar com cautela (n limitado)." if p_cont < 0.05
     else "Não significativo (p > 0.05)."}
""")

if ccf_melhor is not None:
    lag_ot = int(ccf_melhor["lag"])
    print(f"  CCF (níveis) — lag ótimo: {lag_ot:+d} mês(es)"
          f"  r = {ccf_melhor['r']:+.4f}"
          f"  p = {ccf_melhor['p']:.4f}{ccf_melhor['sig']}"
          f"  p_bonf = {ccf_melhor['p_bonf']:.4f}{ccf_melhor['sig_bonf']}")
    if lag_ot < 0:
        print(f"  → Discurso tóxico PRECEDE crimes em {abs(lag_ot)} mês(es) [hipótese exploratória]")
    elif lag_ot == 0:
        print("  → Relação contemporânea mais forte")
    else:
        print("  → Crimes precedem discurso tóxico")

print("\nESTACIONARIEDADE (ADF + KPSS):")
for d in diag_results:
    print(f"  {d['nome']:<18}: ADF p={d['adf_p']:.4f}({d['adf_est']}) | "
          f"KPSS p={d['kpss_p_str']}({d['kpss_est']}) | "
          f"{'→ DIFF' if d['precisa_diff'] else 'estacionária (ambos)'}")

if usar_diff and ccf_df_d is not None:
    melhor_d = ccf_df_d.loc[ccf_df_d["r"].abs().idxmax()]
    diff_sig  = melhor_d["sig_bonf"] == "*"
    nivel_sig = ccf_melhor is not None and ccf_melhor["sig_bonf"] == "*"
    print("\nCONCLUSÃO (níveis vs. diferenciadas):")
    if nivel_sig and diff_sig:
        print("  Resultado PERSISTE após diferenciação — sinal robusto (mas n pequeno).")
    elif nivel_sig and not diff_sig:
        print("  Resultado dos NÍVEIS não sobrevive à diferenciação → possível correlação espúria.")
    else:
        print("  Nenhuma associação significativa após Bonferroni em ambas as versões.")
elif not usar_diff:
    print("\nCONCLUSÃO: séries estacionárias — CCF em níveis válida.")
    if ccf_melhor is not None and ccf_melhor["sig_bonf"] == "*":
        print("  Resultado significativo após Bonferroni — reportar como hipótese exploratória.")
    else:
        print("  Nenhuma associação temporal significativa após correção de Bonferroni.")

print(f"""
LIMITAÇÕES:
  1. n={n_obs} meses: maior que Fase 2 (n=12), mas ADF/CCF ainda limitados para
     inferência confirmatória (mínimo recomendado: ~50 obs para CCF).
  2. Reddit MG cobre apenas {len(MG_SUBREDDITS)} subreddits — viés de seleção desconhecido.
  3. Crimes: subnotificação e heterogeneidade das categorias de natureza.
  4. Sem controle de sazonalidade, tendência, ou variáveis de confusão.
  5. TybyrIA não foi validado especificamente para o domínio LGBTQIA+ em MG.
""")

# ── Salvar no GCS ─────────────────────────────────────────────────────────────
section("SALVANDO OUTPUTS NO GCS")
reddit_mes_out = reddit_mes.reset_index()
reddit_mes_out["mes"] = reddit_mes_out["mes"].astype(str)
upload_csv(bucket, OUT_REDDIT_MES, reddit_mes_out)

crime_mes_out = crime_mes.reset_index()
crime_mes_out["mes"] = crime_mes_out["mes"].astype(str)
upload_csv(bucket, OUT_CRIME_MES, crime_mes_out)

df_comb_out = df_comb.copy()
upload_csv(bucket, OUT_COMBINADO, df_comb_out)
