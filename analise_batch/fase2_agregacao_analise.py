#!/usr/bin/env python3
"""
Fase 2 — Limpeza, agregação mensal e análise estatística exploratória.

Premissas confirmadas (Fase 1):
  - Registros criminais: ~622 válidos (linhas nulas do export REDS descartadas)
  - Duplicatas de comment_id Instagram: MANTIDAS (análise por perfil preservada)
  - Corte temporal: 2025-01-01 a 2025-12-31 (ambos os datasets)
  - comment_created_at: ISO 8601 UTC → pd.to_datetime(..., utc=True)
  - Data/Hora Ocorrência: '%d/%m/%Y %H:%M' → pd.to_datetime(format=...)
  - Coordenadas criminais: string com dois pontos — aplicar fix_coord()
  - Limiar toxicidade (tybyria_score): 0.30

Outputs salvos no GCS:
  rede social/instagram/processed/fase2_mensal_instagram.csv
  criminal/processed/fase2_mensal_criminal.csv
  analysis/fase2_serie_mensal_combinada.csv
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

import numpy as np
import pandas as pd
from gcs_local import gcs_client, storage
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ── Configuração ──────────────────────────────────────────────────────────────
# KEY_FILE       = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET         = "lgbtminas-dados"
INSTA_BLOB     = "rede social/instagram/processed/instagram_comments_2023_2025_limpo_normalizado.csv"
CRIME_BLOB     = (
    "criminal/processed/"
    "DIS - Envolvidos - Eventos de LGBTQIAfobia - Jan 2023 a Dez 2025.csv"
)
OUT_INSTA_MES  = "rede social/instagram/processed/fase2_mensal_instagram.csv"
OUT_CRIME_MES  = "criminal/processed/fase2_mensal_criminal.csv"
OUT_COMBINADO  = "analysis/fase2_serie_mensal_combinada.csv"

THRESHOLD      = 0.30
MESES_ALVO     = pd.period_range(start="2023-01", end="2025-12", freq="M")

SEP  = "=" * 70
SEP2 = "-" * 70


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(t):  print(f"\n{SEP}\n{t}\n{SEP}")
def sub(t):      print(f"\n{SEP2}\n{t}\n{SEP2}")

# gcs_client() importada de gcs_local

def read_blob_csv(bucket, blob_name, **kw):
    data = bucket.blob(blob_name).download_as_bytes()
    return pd.read_csv(io.BytesIO(data), **kw)

def upload_csv(bucket, blob_name, df):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    bucket.blob(blob_name).upload_from_string(csv_bytes, content_type="text/csv")
    print(f"  [OK] gs://{BUCKET}/{blob_name}  ({len(csv_bytes)/1024:.1f} KB)")

def fix_coord(s):
    """Converte coordenadas do REDS (string com dois pontos) para float."""
    try:
        s = str(s).replace(".", "")
        if s.startswith("-"):
            return float(s[:3] + "." + s[3:])
        return float(s[:2] + "." + s[2:])
    except Exception:
        return None

# ── Estatísticas sem scipy ────────────────────────────────────────────────────
def _t_cdf(t_val, df):
    """CDF da distribuição t via aproximação normal quando df > 30, beta incompleta otherwise."""
    if df > 30:
        x = t_val / math.sqrt(1 + t_val * t_val / df)
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    # Approximation para df <= 30 via série de Abramowitz & Stegun
    try:
        from scipy.stats import t as t_dist
        return float(t_dist.cdf(t_val, df=df))
    except ImportError:
        return 0.5 * (1 + math.erf(t_val / math.sqrt(2)))

def pearson_with_p(x, y):
    """Pearson r e p-valor bicaudal."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    if n < 3 or abs(r) >= 1.0:
        return r, float("nan"), n
    t_val = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
    p = 2 * (1 - _t_cdf(abs(t_val), df=n - 2))
    return r, p, n

def spearman_with_p(x, y):
    """Spearman rho via ranking + Pearson."""
    x_r = pd.Series(x, dtype=float).rank()
    y_r = pd.Series(y, dtype=float).rank()
    return pearson_with_p(x_r.values, y_r.values)

def ccf_table(x, y, max_lag=4):
    """Cross-correlation function com p-valores (lag negativo = x lidera y).

    n_eff = len(xa) = n - |lag|, pois é o número real de pares sobrepostos.
    Usar n em vez de n_eff inflaria a significância dos lags maiores.
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xa = x[: n - lag] if lag > 0 else x
            ya = y[lag:] if lag > 0 else y
        else:
            xa = x[-lag:]
            ya = y[: n + lag]
        n_eff = len(xa)          # pares reais disponíveis neste lag
        r = float(np.corrcoef(xa, ya)[0, 1])
        if n_eff < 4 or abs(r) >= 1.0:
            rows.append({"lag_meses": lag, "r": round(r, 4), "n_eff": n_eff,
                         "p": float("nan"), "sig": "n/a"})
            continue
        t_val = r * math.sqrt(n_eff - 2) / math.sqrt(max(1 - r ** 2, 1e-9))
        p = 2 * (1 - _t_cdf(abs(t_val), df=n_eff - 2))
        rows.append({"lag_meses": lag, "r": round(r, 4), "n_eff": n_eff,
                     "p": round(p, 4),
                     "sig": "*" if p < 0.05 else ("†" if p < 0.10 else "")})
    return pd.DataFrame(rows)

def corr_ci_95(r, n):
    """IC 95% para r de Pearson via transformação de Fisher."""
    if n < 4:
        return float("nan"), float("nan")
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    lo = math.tanh(z - 1.96 * se)
    hi = math.tanh(z + 1.96 * se)
    return round(lo, 4), round(hi, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────────────────────
section("1. CARREGANDO DADOS DO GCS")

client = gcs_client()
bucket = client.bucket(BUCKET)

print(f"  Carregando Instagram: {INSTA_BLOB}")
df_insta = read_blob_csv(bucket, INSTA_BLOB, low_memory=False)
print(f"  Instagram consolidado : {df_insta.shape}")

df_crime = read_blob_csv(
    bucket, CRIME_BLOB,
    encoding="iso-8859-1", sep=";", on_bad_lines="skip", low_memory=False,
)
df_crime.columns = [c.replace("\xad", "").strip() for c in df_crime.columns]
print(f"  Registros criminais   : {df_crime.shape} (antes da limpeza)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIMPEZA E FILTRAGEM
# ─────────────────────────────────────────────────────────────────────────────
section("2. LIMPEZA E FILTRAGEM")

# ── Instagram ─────────────────────────────────────────────────────────────────
sub("2A. Instagram")

# Arquivo 2023-2025 já tem colunas comment_ano e comment_mes — usar diretamente
df_insta["mes"] = pd.to_datetime(
    dict(year=df_insta["comment_ano"], month=df_insta["comment_mes"], day=1)
).dt.to_period("M")

n_fora = (~df_insta["mes"].isin(MESES_ALVO)).sum()
if n_fora:
    print(f"  [AVISO] {n_fora} linhas fora do intervalo 2023-01/2025-12 — removidas.")
    df_insta = df_insta[df_insta["mes"].isin(MESES_ALVO)].copy()

print(f"  Total comentários   : {len(df_insta):,}")
print(f"  Duplicatas mantidas : {int(df_insta.duplicated('comment_id').sum())} (análise por perfil preservada)")
print(f"  Intervalo de meses  : {df_insta['mes'].min()} → {df_insta['mes'].max()}")
print(f"  Meses com dados     : {sorted(df_insta['mes'].unique().astype(str))}")

# ── Criminal ──────────────────────────────────────────────────────────────────
sub("2B. Registros Criminais (Envolvidos 2023-2025)")

n_raw = len(df_crime)
df_crime = df_crime.dropna(subset=["ID Ocorrência"]).copy()
n_valido = len(df_crime)
print(f"  Linhas brutas (envolvidos) : {n_raw:,}")
print(f"  Após remover nulos         : {n_valido:,}")

df_crime["dt_ocorrencia"] = pd.to_datetime(
    df_crime["Data/Hora Ocorrência"], format="%d/%m/%Y %H:%M", errors="coerce"
)
n_parse_ok = df_crime["dt_ocorrencia"].notna().sum()
print(f"  Datas parseadas OK         : {n_parse_ok:,} / {n_valido:,}")

df_crime = df_crime[df_crime["dt_ocorrencia"].dt.year.isin([2023, 2024, 2025])].copy()
df_crime["mes"] = df_crime["dt_ocorrencia"].dt.to_period("M")

# Arquivo Envolvidos: deduplicar por ID Ocorrência para contar eventos únicos
df_crime = df_crime.drop_duplicates(subset=["ID Ocorrência"]).copy()
print(f"  Ocorrências únicas (2023-2025): {len(df_crime):,}")
print(f"  Intervalo de datas     : {df_crime['dt_ocorrencia'].min()} → {df_crime['dt_ocorrencia'].max()}")
print(f"  Meses com dados        : {sorted(df_crime['mes'].unique().astype(str))}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGREGAÇÃO MENSAL
# ─────────────────────────────────────────────────────────────────────────────
section("3. AGREGAÇÃO MENSAL")

# ── Instagram: série mensal geral ─────────────────────────────────────────────
sub("3A. Instagram — série mensal (todos os perfis)")

insta_mes = (
    df_insta
    .groupby("mes")
    .agg(
        n_comentarios     = ("comment_id",     "count"),
        n_toxicos         = ("tybyria_score",  lambda s: (s >= THRESHOLD).sum()),
        media_tybyria     = ("tybyria_score",  "mean"),
        mediana_tybyria   = ("tybyria_score",  "median"),
        std_tybyria       = ("tybyria_score",  "std"),
        media_vader       = ("vader_compound", "mean"),
        mediana_vader     = ("vader_compound", "median"),
        n_vader_pos       = ("vader_compound", lambda s: (s > 0.05).sum()),
        n_vader_neg       = ("vader_compound", lambda s: (s < -0.05).sum()),
    )
    .reindex(MESES_ALVO, fill_value=0)
)
insta_mes["pct_toxicos"] = (insta_mes["n_toxicos"] / insta_mes["n_comentarios"].replace(0, np.nan) * 100).round(2)
insta_mes.index.name = "mes"
print(insta_mes.to_string())

# ── Instagram: série mensal por perfil ────────────────────────────────────────
sub("3B. Instagram — n_comentarios e pct_toxicos por perfil × mês")

insta_perfil_mes = (
    df_insta
    .groupby(["mes", "perfil"])
    .agg(
        n_comentarios = ("comment_id",    "count"),
        n_toxicos     = ("tybyria_score", lambda s: (s >= THRESHOLD).sum()),
        media_tybyria = ("tybyria_score", "mean"),
    )
    .reset_index()
)
insta_perfil_mes["pct_toxicos"] = (
    insta_perfil_mes["n_toxicos"] / insta_perfil_mes["n_comentarios"] * 100
).round(2)

pivot_n = insta_perfil_mes.pivot(index="mes", columns="perfil", values="n_comentarios").reindex(MESES_ALVO)
pivot_pct = insta_perfil_mes.pivot(index="mes", columns="perfil", values="pct_toxicos").reindex(MESES_ALVO)
pivot_tyb = insta_perfil_mes.pivot(index="mes", columns="perfil", values="media_tybyria").reindex(MESES_ALVO)

print(f"\nContagem de comentários por perfil:")
print(pivot_n.to_string())
print(f"\n% tóxicos (tybyria >= {THRESHOLD}) por perfil:")
print(pivot_pct.round(2).to_string())
print(f"\nMédia tybyria_score por perfil:")
print(pivot_tyb.round(4).to_string())

# ── Criminal: série mensal ────────────────────────────────────────────────────
sub("3C. Registros criminais — série mensal")

crime_mes = (
    df_crime
    .groupby("mes")
    .agg(
        n_crimes          = ("ID Ocorrência",    "count"),
        municipios_unicos = ("Município (Fato)", "nunique"),
    )
    .reindex(MESES_ALVO, fill_value=0)
)
crime_mes.index.name = "mes"
print(crime_mes.to_string())

# ── Distribuição por tipo de crime ────────────────────────────────────────────
sub("3D. Tipos de crime (Natureza Principal)")
print(df_crime["Natureza Principal"].value_counts().to_string())

# ── Distribuição por causa presumida ─────────────────────────────────────────
sub("3E. Causa presumida")
print(df_crime["Causa Presumida"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 4. SÉRIE TEMPORAL COMBINADA
# ─────────────────────────────────────────────────────────────────────────────
section("4. SÉRIE TEMPORAL COMBINADA (mês a mês)")

df_comb = insta_mes[["n_comentarios", "n_toxicos", "pct_toxicos",
                      "media_tybyria", "std_tybyria", "media_vader"]].join(
    crime_mes[["n_crimes", "municipios_unicos"]], how="outer"
).reindex(MESES_ALVO)

df_comb.index.name = "mes"
df_comb = df_comb.reset_index()
df_comb["mes"] = df_comb["mes"].astype(str)

print(df_comb.to_string(index=False))

# Meses sem sobreposição
sem_insta = df_comb[df_comb["n_comentarios"] == 0]["mes"].tolist()
sem_crime = df_comb[df_comb["n_crimes"] == 0]["mes"].tolist()
if sem_insta:
    print(f"\n  [AVISO] Meses sem comentários Instagram: {sem_insta}")
if sem_crime:
    print(f"  [AVISO] Meses sem registros criminais  : {sem_crime}")

n_overlap = int((df_comb["n_comentarios"] > 0).sum() & (df_comb["n_crimes"] > 0).sum())
meses_validos = df_comb[(df_comb["n_comentarios"] > 0) & (df_comb["n_crimes"] > 0)]
print(f"\n  Meses com dados em ambas as séries: {len(meses_validos)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISE ESTATÍSTICA
# ─────────────────────────────────────────────────────────────────────────────
section("5. ANÁLISE ESTATÍSTICA")

print(f"""
NOTA METODOLÓGICA (leia antes dos resultados):
  n = {len(meses_validos)} meses de sobreposição.
  Com n ≤ 12, o poder estatístico é baixo. Para Pearson a p < 0.05 bicaudal,
  é necessário |r| > {abs(np.corrcoef([1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,8,9,10,11,12])[0,1]):.0f}
  (limiar real com n={len(meses_validos)}: |r| > ~{0.576 if len(meses_validos)==12 else 0.514:.3f}).
  Resultados devem ser interpretados como exploratórios, não confirmatórios.
  Correlação ≠ causalidade.
""")

# Séries para análise (apenas meses com dados nos dois datasets)
mv = meses_validos.copy()
x_pct    = mv["pct_toxicos"].values
x_tyb    = mv["media_tybyria"].values
x_vader  = mv["media_vader"].values
x_n      = mv["n_comentarios"].values
y_crimes = mv["n_crimes"].values

sub("5.1 Estatísticas descritivas das séries mensais")

desc_cols = ["n_comentarios", "pct_toxicos", "media_tybyria", "media_vader", "n_crimes"]
print(df_comb[df_comb["mes"].isin(mv["mes"])][desc_cols].describe().round(4).to_string())

sub("5.2 Correlações de Pearson e Spearman (% tóxicos × crimes)")

pairs = [
    ("pct_toxicos",    x_pct,   "% comentários tóxicos (tybyria ≥ 0.30)"),
    ("media_tybyria",  x_tyb,   "Média tybyria_score"),
    ("media_vader",    x_vader, "Média vader_compound"),
    ("n_comentarios",  x_n,     "Volume total de comentários"),
]

print(f"\n{'Variável':<28} {'Pearson r':>10} {'p':>8} {'IC 95%':>18} {'Spearman ρ':>12} {'p':>8}")
print("-" * 90)
for nome, x_arr, label in pairs:
    r_p, p_p, n_p = pearson_with_p(x_arr, y_crimes)
    r_s, p_s, _   = spearman_with_p(x_arr, y_crimes)
    lo, hi         = corr_ci_95(r_p, n_p)
    sig_p = "*" if p_p < 0.05 else ("†" if p_p < 0.10 else "")
    sig_s = "*" if p_s < 0.05 else ("†" if p_s < 0.10 else "")
    ic_str = f"[{lo:.3f}, {hi:.3f}]"
    print(f"{label:<28} {r_p:>+10.4f} {p_p:>8.4f}{sig_p:<1} {ic_str:>18} {r_s:>+12.4f} {p_s:>8.4f}{sig_s:<1}")

print("\n  * p < 0.05   † p < 0.10   IC = intervalo de confiança de Fisher (95%)")

# ─────────────────────────────────────────────────────────────────────────────
sub("5.2.5 Diagnóstico de estacionariedade e autocorrelação")
# ─────────────────────────────────────────────────────────────────────────────
print("""
  OBJETIVO: verificar se as séries são estacionárias ANTES de interpretar a CCF.
  Séries não-estacionárias com tendência comum produzem correlações espúrias
  (Granger & Newbold, 1974). Com n=12 o ADF tem poder baixo — os três testes
  são usados em conjunto para a decisão de diferenciação.

  Testes:
    ADF  — Augmented Dickey-Fuller (H0: raiz unitária / não-estacionária)
           rejeitar H0 a 5% → série estacionária
    ACF1 — autocorrelação de lag 1 (|ACF(1)| > 0.5 = memória forte)
    DW   — Durbin-Watson nos resíduos de regressão contra tendência linear
           DW ≈ 2 = sem autocorrelação; DW < 1.5 = autocorrelação positiva forte

  Critério de diferenciação: ADF p > 0.05  OU  |ACF(1)| > 0.5
""")

series_diag = {
    "pct_toxicos"  : x_pct,
    "media_tybyria": x_tyb,
    "media_vader"  : x_vader,
    "n_crimes"     : y_crimes,
}
trend_vec = np.arange(1, len(x_pct) + 1, dtype=float)

diag_results = {}
hdr = (f"{'Série':<18} {'ADF stat':>10} {'ADF p':>8} {'Estac?':>7}"
       f" {'ACF(1)':>8} {'|>0.5|?':>8} {'DW':>6} {'DW<1.5?':>8} {'Diff?':>12}")
print(hdr)
print("-" * len(hdr))

for nome, s in series_diag.items():
    s_arr = np.array(s, dtype=float)

    # ADF — maxlag=2 porque n=12 não suporta lags maiores
    try:
        adf_stat, adf_p, *_ = adfuller(s_arr, maxlag=2, autolag="AIC", regression="c")
        estac = "SIM" if adf_p < 0.05 else "NÃO"
    except Exception:
        adf_stat, adf_p, estac = float("nan"), 1.0, "ERRO"

    # ACF lag=1 manual
    acf1 = float(np.corrcoef(s_arr[:-1], s_arr[1:])[0, 1])
    acf1_flag = "SIM" if abs(acf1) > 0.5 else "não"

    # Durbin-Watson nos resíduos de OLS(série ~ trend)
    X = np.column_stack([np.ones(len(s_arr)), trend_vec])
    beta = np.linalg.lstsq(X, s_arr, rcond=None)[0]
    residuos = s_arr - X @ beta
    dw = float(durbin_watson(residuos))
    dw_flag = "SIM" if dw < 1.5 else "não"

    precisa_diff = (adf_p > 0.05 or abs(acf1) > 0.5)
    flag = "→ DIFF" if precisa_diff else "ok"

    diag_results[nome] = {
        "s_arr": s_arr, "adf_stat": adf_stat, "adf_p": adf_p, "estac": estac,
        "acf1": acf1, "dw": dw, "precisa_diff": precisa_diff,
        "s_diff": np.diff(s_arr),
    }
    print(f"{nome:<18} {adf_stat:>10.4f} {adf_p:>8.4f} {estac:>7}"
          f" {acf1:>+8.4f} {acf1_flag:>8} {dw:>6.3f} {dw_flag:>8} {flag:>12}")

series_com_diff = [n for n, d in diag_results.items() if d["precisa_diff"]]
print(f"\n  Séries que acionam diferenciação: "
      f"{series_com_diff if series_com_diff else 'nenhuma'}")

# Decisão: diferenciar CCF se qualquer série de toxicidade OU crimes precisar
usar_diff = (
    diag_results["pct_toxicos"]["precisa_diff"] or
    diag_results["n_crimes"]["precisa_diff"]
)

# Séries diferenciadas — sempre computadas para CCF, usadas só se usar_diff=True
x_pct_d    = diag_results["pct_toxicos"]["s_diff"]
x_tyb_d    = diag_results["media_tybyria"]["s_diff"]
y_crimes_d = diag_results["n_crimes"]["s_diff"]
ccf_df_d   = None   # será preenchido na 5.3 se usar_diff

if usar_diff:
    print(f"\n  [INFO] Séries diferenciadas têm n={len(x_pct_d)} (1 obs perdida por diff).")
    print("  [INFO] A CCF será exibida em NÍVEIS e em DIFERENCIADAS — comparação na seção 5.3.")
else:
    print("\n  [INFO] Nenhuma série requer diferenciação — CCF 5.3 usa apenas os níveis brutos.")

# ─────────────────────────────────────────────────────────────────────────────
sub("5.3 Função de correlação cruzada (CCF) — pct_toxicos × n_crimes")
# ─────────────────────────────────────────────────────────────────────────────
print("""
  Interpretação dos lags:
    lag < 0 → discurso tóxico PRECEDE crimes em |lag| meses (hipótese principal)
    lag = 0 → relação contemporânea
    lag > 0 → crimes PRECEDEM discurso tóxico
""")

MAX_LAG_CCF = min(4, len(meses_validos) // 3)
ccf_df = None

if len(meses_validos) >= 6:
    print("  [A] NÍVEIS BRUTOS:")
    ccf_df = ccf_table(x_pct, y_crimes, max_lag=MAX_LAG_CCF)
    print(ccf_df.to_string(index=False))
    melhor = ccf_df.loc[ccf_df["r"].abs().idxmax()]
    print(f"  Lag com maior |r|: {int(melhor['lag_meses'])} mês(es)  "
          f"r = {melhor['r']:.4f}  p = {melhor['p']:.4f}{melhor['sig']}")

    if usar_diff:
        max_lag_d = min(3, len(x_pct_d) // 3)
        print(f"\n  [B] SÉRIES DIFERENCIADAS (Δ mês a mês, n={len(x_pct_d)}):")
        ccf_df_d = ccf_table(x_pct_d, y_crimes_d, max_lag=max_lag_d)
        print(ccf_df_d.to_string(index=False))
        melhor_d = ccf_df_d.loc[ccf_df_d["r"].abs().idxmax()]
        print(f"  Lag com maior |r|: {int(melhor_d['lag_meses'])} mês(es)  "
              f"r = {melhor_d['r']:.4f}  p = {melhor_d['p']:.4f}{melhor_d['sig']}")

        print("\n  COMPARAÇÃO DIRETA (lag ótimo de cada versão):")
        print(f"    [Níveis]        lag={int(melhor['lag_meses']):+d}  "
              f"r={melhor['r']:+.4f}  p={melhor['p']:.4f}{melhor['sig']}")
        print(f"    [Diferenciadas] lag={int(melhor_d['lag_meses']):+d}  "
              f"r={melhor_d['r']:+.4f}  p={melhor_d['p']:.4f}{melhor_d['sig']}")
else:
    print(f"  [AVISO] Apenas {len(meses_validos)} meses sobrepostos — CCF não confiável.")

sub("5.4 CCF — media_tybyria × n_crimes")
if len(meses_validos) >= 6:
    print("  [A] NÍVEIS BRUTOS:")
    ccf_tyb = ccf_table(x_tyb, y_crimes, max_lag=MAX_LAG_CCF)
    print(ccf_tyb.to_string(index=False))

    if usar_diff:
        max_lag_d = min(3, len(x_tyb_d) // 3)
        print(f"\n  [B] SÉRIES DIFERENCIADAS (n={len(x_tyb_d)}):")
        ccf_tyb_d = ccf_table(x_tyb_d, y_crimes_d, max_lag=max_lag_d)
        print(ccf_tyb_d.to_string(index=False))

sub("5.5 Análise por perfil — média tybyria_score × meses")
print(f"\nMédia tybyria_score por perfil (2023-2025):")
print(
    df_insta.groupby("perfil")["tybyria_score"]
    .agg(n="count", media="mean", mediana="median", std="std",
         pct_toxico=lambda s: (s >= THRESHOLD).mean() * 100)
    .round(4)
    .to_string()
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SALVAR OUTPUTS NO GCS
# ─────────────────────────────────────────────────────────────────────────────
section("6. SALVANDO OUTPUTS NO GCS")

# Mensal Instagram (índice como coluna)
insta_mes_out = insta_mes.reset_index()
insta_mes_out["mes"] = insta_mes_out["mes"].astype(str)
upload_csv(bucket, OUT_INSTA_MES, insta_mes_out)

# Mensal criminal
crime_mes_out = crime_mes.reset_index()
crime_mes_out["mes"] = crime_mes_out["mes"].astype(str)
upload_csv(bucket, OUT_CRIME_MES, crime_mes_out)

# Série combinada
upload_csv(bucket, OUT_COMBINADO, df_comb)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMÁRIO INTERPRETATIVO
# ─────────────────────────────────────────────────────────────────────────────
section("7. SUMÁRIO E PRÓXIMOS PASSOS")

r_principal, p_principal, _ = pearson_with_p(x_pct, y_crimes)
r_ccf_melhor = ccf_df.loc[ccf_df["r"].abs().idxmax()] if ccf_df is not None else None

print(f"""
RESULTADOS PRINCIPAIS:
  Dataset Instagram : {len(df_insta):,} comentários válidos (2023-2025)
  Dataset Criminal  : {len(df_crime):,} ocorrências únicas válidas (2023-2025)
  Sobreposição      : {len(meses_validos)} meses

  Correlação contemporânea (pct_toxicos × n_crimes):
    Pearson r = {r_principal:+.4f}  p = {p_principal:.4f}
    {'ATENÇÃO: resultado significativo — interpretar com cautela dado n pequeno.' if p_principal < 0.05 else 'Não significativo ao nível 0.05 (esperado com n pequeno).'}
""")

if r_ccf_melhor is not None:
    lag_otimo = int(r_ccf_melhor["lag_meses"])
    print(f"  CCF NÍVEIS — Lag ótimo: {lag_otimo} mês(es)  "
          f"r = {r_ccf_melhor['r']:.4f}  p = {r_ccf_melhor['p']:.4f}{r_ccf_melhor['sig']}")
    if lag_otimo < 0:
        print(f"  → Padrão sugestivo (níveis): discurso tóxico precede crimes em "
              f"{abs(lag_otimo)} mês(es) [hipótese exploratória]")
    elif lag_otimo == 0:
        print("  → Relação contemporânea mais forte nos níveis")
    else:
        print("  → Crimes precedem discurso tóxico nos níveis — possível reação online")

# ── Conclusão sobre estacionariedade ─────────────────────────────────────────
print("\nDIAGNÓSTICO DE ESTACIONARIEDADE (seção 5.2.5):")
for nome, d in diag_results.items():
    status = "estacionária" if d["adf_p"] < 0.05 else "não-estacionária (ADF)"
    print(f"  {nome:<18}: ADF p={d['adf_p']:.4f} ({status}) | "
          f"ACF(1)={d['acf1']:+.4f} | DW={d['dw']:.3f}")

print()
if usar_diff:
    print("  → Diferenciação aplicada. CCF recalculada com Δ(pct_toxicos) × Δ(n_crimes).")
    if ccf_df_d is not None:
        melhor_d = ccf_df_d.loc[ccf_df_d["r"].abs().idxmax()]
        lag_d = int(melhor_d["lag_meses"])
        sig_d = melhor_d["sig"]
        p_d   = melhor_d["p"]
        r_d   = melhor_d["r"]
        print(f"  CCF DIFERENCIADAS — Lag ótimo: {lag_d} mês(es)  r = {r_d:.4f}  p = {p_d:.4f}{sig_d}")

        # Conclusão explícita sobre mudança de resultado
        print()
        nivs_sig = r_ccf_melhor is not None and str(r_ccf_melhor.get("sig","")) == "*"
        diff_sig = sig_d == "*"
        if nivs_sig and diff_sig:
            print("  CONCLUSÃO: o resultado da CCF PERSISTE após diferenciação (p < 0.05 em ambos).")
            print("  Interpretar como hipótese exploratória robusta — requer validação com n maior.")
        elif nivs_sig and not diff_sig:
            print("  CONCLUSÃO: o resultado significativo nos NÍVEIS DESAPARECE após diferenciação.")
            print("  Evidência de correlação espúria por tendência comum ou memória de longo prazo.")
            print("  A associação temporal observada nos níveis NÃO deve ser reportada como achado.")
        elif not nivs_sig and diff_sig:
            print("  CONCLUSÃO: a diferenciação REVELA associação não visível nos níveis.")
            print("  Sinal presente nas variações mês a mês — reportar com cautela dado n pequeno.")
        else:
            print("  CONCLUSÃO: nenhuma associação significativa em nenhuma das versões.")
            print("  Resultado consistente: ausência de sinal temporal com estes dados.")
else:
    print("  → Nenhuma série requereu diferenciação.")
    print("  CONCLUSÃO: CCF em níveis brutos é válida para as séries analisadas.")
    if r_ccf_melhor is not None and str(r_ccf_melhor.get("sig","")) == "*":
        print("  O resultado significativo nos níveis não é suspeito de correlação espúria.")
    else:
        print("  Nenhuma associação temporal significativa encontrada (esperado com n=12).")

print(f"""
LIMITAÇÕES METODOLÓGICAS:
  1. n = {len(meses_validos)} meses: poder estatístico insuficiente para inferência confirmatória.
  2. ADF com n=12 tem poder muito baixo — não rejeitar H0 não é diagnóstico definitivo.
  3. Instagram cobre apenas {df_insta['perfil'].nunique()} perfis (selecionados, não amostra aleatória).
  4. mgtown_minas tem {len(df_insta[df_insta['perfil']=='mgtown_minas']):} comentários (2023-2025) — peso analítico marginal.
  5. Registros criminais refletem subnotificação desconhecida (viés de captura).
  6. Não foram controladas variáveis de confusão (eventos, sazonalidade, cobertura midiática).
  7. TybyrIA foi treinado em dados genéricos — pode ter viés para o domínio LGBTQIA+.

PRÓXIMOS PASSOS SUGERIDOS:
  → Fase 3: incorporar dados Reddit (maior volume → mais poder estatístico para ADF e CCF)
  → Fase 3: análise por município (geoespacial) onde n por célula for suficiente
  → Fase 3: expandir janela temporal (2023–2025) com dataset Reddit completo
  → Fase 3: controlar sazonalidade (diferenciação ou dummies mensais)
  → Fase 3: considerar teste de cointegração de Engle-Granger se séries > 30 obs
""")
