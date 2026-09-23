#!/usr/bin/env python3
"""
Fase 5 — Análise de correlação semanal (séries temporais).

Complementa a classificação supervisionada (fase4d) com análise de correlação:
  1. Estacionariedade (ADF + KPSS) das séries
  2. Pearson e Spearman com p-valores em múltiplos lags (0–8 semanas)
  3. CCF (Função de Correlação Cruzada) com IC 95%
  4. Causalidade de Granger (max_lag=6, correção de Bonferroni)
  5. Correlação parcial (controlando o histórico autorregressivo de crimes)

Usa os datasets semanais gerados pela fase4b:
  analysis/fase4_artigo_completo.csv  (155 semanas, 2023-2025)

Lógica de reconstrução das séries contemporâneas:
  O arquivo fase4b armazena features em T-1. Para análise de correlação em
  múltiplos lags, reconstruímos a série contemporânea de cada variável
  de redes sociais fazendo shift(-1) na coluna _t1, obtendo o valor real
  no tempo T de cada semana.

Salva resultados em:
  analysis/fase5_correlacao_semanal.csv   (tabela de correlações por lag)
  analysis/fase5_granger_semanal.csv      (resultados dos testes de Granger)
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
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, ccf
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_OK = True
except ImportError:
    print("[ERRO] statsmodels não encontrado. Instale com: pip install statsmodels")
    sys.exit(1)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# KEY_FILE      = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET        = "lgbtminas-dados"
BLOB_COMPLETO = "analysis/fase4_artigo_completo.csv"
OUT_CORR      = "analysis/fase5_correlacao_semanal.csv"
OUT_GRANGER   = "analysis/fase5_granger_semanal.csv"

MAX_LAG       = 8     # lags máximos para CCF e Pearson/Spearman
MAX_LAG_GR    = 6     # lags máximos para Granger
ALPHA         = 0.05  # nível de significância

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
# 1. CARREGAR E RECONSTRUIR SÉRIES CONTEMPORÂNEAS
# ═════════════════════════════════════════════════════════════════════════════
section("1. CARREGANDO DADOS E RECONSTRUINDO SÉRIES")

df = read_blob_csv(bucket, BLOB_COMPLETO)
df["sem"] = pd.PeriodIndex(df["sem"], freq="W")
df = df.set_index("sem").sort_index()
print(f"  Semanas: {len(df)}  ({df.index[0]} → {df.index[-1]})")

# Séries de redes sociais que vamos analisar
# As colunas _t1 armazenam o valor da semana T-1.
# Para obter a série contemporânea no tempo T:
#   valor_em_T = coluna_t1.shift(-1)
# Isso move cada "valor da semana anterior" para a semana a que realmente pertence.
SERIES_MIDIA = {
    "reddit_tybyria":    "tybyria_t1",
    "reddit_toxicidade": "toxicidade_t1",
    "reddit_n_posts":    "n_posts_t1",
    "insta_vader":       "insta_vader_t1",
    "insta_tybyria":     "insta_tybyria_t1",
    "insta_pct_toxicos": "insta_pct_toxicos_t1",
    "insta_n_coment":    "insta_n_comentarios_t1",
}

df_raw = pd.DataFrame(index=df.index)
df_raw["n_crimes"]    = df["n_crimes"]
df_raw["crimes_t1"]   = df["crimes_t1"]   # mantido para correlação parcial
df_raw["crimes_mm3"]  = df["crimes_mm3"]

for nome, col_t1 in SERIES_MIDIA.items():
    if col_t1 in df.columns:
        df_raw[nome] = df[col_t1].shift(-1)
    else:
        print(f"  [AVISO] Coluna {col_t1} não encontrada — {nome} ignorada")

# Remover NaN das bordas após o shift
df_raw = df_raw.dropna(subset=["n_crimes"])
n_obs = len(df_raw)
print(f"  Observações após alinhamento: {n_obs}")

print(f"\n  Séries disponíveis para análise:")
for nome in SERIES_MIDIA:
    if nome in df_raw.columns:
        n_nan = df_raw[nome].isna().sum()
        print(f"    {nome:<25}  n_nan={n_nan}  média={df_raw[nome].mean():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. ESTACIONARIEDADE — ADF + KPSS
# ═════════════════════════════════════════════════════════════════════════════
section("2. TESTE DE ESTACIONARIEDADE (ADF + KPSS)")

print("""
  ADF H0: série tem raiz unitária (não-estacionária)
    p < 0.05 → rejeita H0 → série estacionária
  KPSS H0: série é estacionária
    p < 0.05 → rejeita H0 → série não-estacionária
  Concordância: ambos apontam na mesma direção → diagnóstico robusto
""")

print(f"  {'Série':<25}  {'ADF p':>8}  {'KPSS p':>8}  {'Estacionária?':>16}  {'Difer. necesária?':>18}")
print("  " + "─" * 80)

estacionariedade = {}
for nome in ["n_crimes"] + list(SERIES_MIDIA.keys()):
    if nome not in df_raw.columns:
        continue
    s = df_raw[nome].dropna()
    if len(s) < 20:
        continue
    try:
        adf_p  = adfuller(s, autolag="AIC")[1]
    except Exception:
        adf_p  = float("nan")
    try:
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = float("nan")

    adf_stat  = "I(0)" if adf_p  < ALPHA else "I(1)?"
    kpss_stat = "I(0)" if kpss_p > ALPHA else "I(1)?"
    estac = (adf_p < ALPHA) and (kpss_p > ALPHA)
    difer = "NÃO" if estac else "SIM (recomendado)"
    estacionariedade[nome] = estac
    print(f"  {nome:<25}  {adf_p:>8.4f}  {kpss_p:>8.4f}  "
          f"{'SIM' if estac else 'NÃO':>16}  {difer:>18}")

# Séries não-estacionárias → diferenciar para Granger
print(f"\n  Preparando séries diferenciadas (para Granger e CCF robusto):")
df_diff = df_raw.diff().dropna()
print(f"  Observações após diferenciação: {len(df_diff)}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. PEARSON E SPEARMAN EM MÚLTIPLOS LAGS
# ═════════════════════════════════════════════════════════════════════════════
section("3. PEARSON E SPEARMAN — CORRELAÇÃO EM MÚLTIPLOS LAGS (0–8 semanas)")

print("""
  Lag k: correlação entre polarity na semana T e crimes na semana T+k.
  Lag k positivo → polarity ANTECEDE crimes (hipótese do artigo).
  Lag k negativo → crimes antecedem polarity (causalidade inversa).
  Correção de Bonferroni: α_adj = 0.05 / (n_series × n_lags)
""")

SERIES_CORR = [s for s in SERIES_MIDIA if s in df_raw.columns
               and s not in ("reddit_n_posts", "insta_n_coment")]
N_LAGS_CORR = list(range(0, MAX_LAG + 1))
ALPHA_BONF  = ALPHA / (len(SERIES_CORR) * len(N_LAGS_CORR))

print(f"  α ajustado (Bonferroni): {ALPHA_BONF:.5f}  "
      f"({len(SERIES_CORR)} séries × {len(N_LAGS_CORR)} lags)\n")

corr_rows = []
for nome in SERIES_CORR:
    sub(f"Série: {nome}")
    print(f"  {'Lag':>4}  {'Pearson r':>10}  {'p Pearson':>10}  "
          f"{'Spearman ρ':>11}  {'p Spearman':>11}  {'Sig?':>6}")
    print(f"  {'─'*60}")

    for k in N_LAGS_CORR:
        # Crimes na semana T+k correlacionados com polarity em T
        # → shift crimes por -k (move crimes para o passado)
        crimes_shifted = df_raw["n_crimes"].shift(-k)
        s_pol  = df_raw[nome].dropna()
        s_crim = crimes_shifted.reindex(s_pol.index).dropna()
        s_pol  = s_pol.reindex(s_crim.index)

        if len(s_crim) < 15:
            continue

        r_p, p_p = stats.pearsonr(s_pol, s_crim)
        r_s, p_s = stats.spearmanr(s_pol, s_crim)
        sig = "*" if min(p_p, p_s) < ALPHA_BONF else (
              "." if min(p_p, p_s) < ALPHA else " ")

        print(f"  {k:>4}  {r_p:>+10.4f}  {p_p:>10.4f}  "
              f"{r_s:>+11.4f}  {p_s:>11.4f}  {sig:>6}")

        corr_rows.append({
            "serie":       nome,
            "lag_semanas": k,
            "pearson_r":   round(r_p, 4),
            "pearson_p":   round(p_p, 5),
            "spearman_r":  round(r_s, 4),
            "spearman_p":  round(p_s, 5),
            "sig_bonf":    int(min(p_p, p_s) < ALPHA_BONF),
            "sig_alpha05": int(min(p_p, p_s) < ALPHA),
            "n":           len(s_crim),
        })

df_corr = pd.DataFrame(corr_rows)

# Resumo: lag mais forte por série
print(f"\n  RESUMO — lag com maior |Pearson r| por série:")
print(f"  {'Série':<25}  {'Melhor lag':>10}  {'Pearson r':>10}  {'p':>8}  {'Sig?':>6}")
print(f"  {'─'*65}")
for nome in SERIES_CORR:
    sub_df = df_corr[df_corr["serie"] == nome]
    if sub_df.empty:
        continue
    best = sub_df.loc[sub_df["pearson_r"].abs().idxmax()]
    sig  = "**" if best["sig_bonf"] else ("*" if best["sig_alpha05"] else "ns")
    print(f"  {nome:<25}  lag={int(best['lag_semanas']):>7}  "
          f"{best['pearson_r']:>+10.4f}  {best['pearson_p']:>8.5f}  {sig:>6}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. CCF — FUNÇÃO DE CORRELAÇÃO CRUZADA (statsmodels)
# ═════════════════════════════════════════════════════════════════════════════
section("4. CCF — FUNÇÃO DE CORRELAÇÃO CRUZADA COM IC 95%")

print("""
  CCF(k) = correlação entre polarity em T e crimes em T+k.
  IC 95% = ±1.96/√n (aproximação para série estacionária).
  Valores fora do IC indicam correlação significativa naquele lag.
  Usamos séries diferenciadas para séries não-estacionárias.
""")

ccf_rows = []
for nome in SERIES_CORR:
    s_pol  = df_diff[nome].dropna()
    s_crim = df_diff["n_crimes"].reindex(s_pol.index).dropna()
    s_pol  = s_pol.reindex(s_crim.index).fillna(0)

    if len(s_crim) < 20:
        continue

    n_ccf = len(s_crim)
    ic_95 = 1.96 / np.sqrt(n_ccf)

    # ccf do statsmodels: ccf(x, y)[k] = corr(x[t], y[t+k])
    # Queremos: corr(polarity[t], crimes[t+k]) → x=polarity, y=crimes
    ccf_vals = ccf(s_pol.values, s_crim.values, nlags=MAX_LAG, alpha=None)

    print(f"\n  {nome}  (n={n_ccf}, IC95=±{ic_95:.4f})")
    print(f"  {'Lag':>4}  {'CCF':>8}  {'|CCF|>IC':>10}  {'Direção'}")
    print(f"  {'─'*40}")
    for k, val in enumerate(ccf_vals[:MAX_LAG + 1]):
        sig = "SIM ***" if abs(val) > ic_95 else "—"
        dir_str = "polarity→crimes" if val > 0 else "inverso"
        print(f"  {k:>4}  {val:>+8.4f}  {sig:>10}  {dir_str if abs(val) > ic_95 else ''}")
        ccf_rows.append({
            "serie":       nome,
            "lag_semanas": k,
            "ccf":         round(val, 4),
            "ic_95":       round(ic_95, 4),
            "sig":         int(abs(val) > ic_95),
        })

df_ccf = pd.DataFrame(ccf_rows)


# ═════════════════════════════════════════════════════════════════════════════
# 5. GRANGER CAUSALITY — polarity → crimes
# ═════════════════════════════════════════════════════════════════════════════
section("5. CAUSALIDADE DE GRANGER (semanal, max_lag=6)")

print(f"""
  Testa se a série de polarity MELHORA a previsão de crimes além do que
  o histórico de crimes já prevê sozinho.
  H0: polarity NÃO Granger-causa crimes.
  p < {ALPHA:.2f} → rejeita H0 → evidência de causalidade de Granger.

  Correção de Bonferroni: α_adj = {ALPHA}/{len(SERIES_CORR)}/{MAX_LAG_GR} = {ALPHA/(len(SERIES_CORR)*MAX_LAG_GR):.5f}
  Séries diferenciadas (D1) para garantir estacionariedade.
""")

granger_rows = []
ALPHA_GR_BONF = ALPHA / (len(SERIES_CORR) * MAX_LAG_GR)

for nome in SERIES_CORR:
    s_pol  = df_diff[nome].dropna()
    s_crim = df_diff["n_crimes"].reindex(s_pol.index).dropna()
    valid  = s_pol.notna() & s_crim.notna()
    s_pol  = s_pol[valid]
    s_crim = s_crim[valid]

    if len(s_crim) < MAX_LAG_GR + 10:
        print(f"  {nome}: n insuficiente ({len(s_crim)}) — pulado")
        continue

    data_granger = pd.DataFrame({
        "crimes":  s_crim.values,
        "polarity": s_pol.values,
    })

    print(f"\n  {nome}  (n={len(data_granger)} semanas D1)")
    print(f"  {'Lag':>4}  {'F-stat':>8}  {'p-valor':>9}  {'p_bonf':>9}  {'Granger?':>10}")
    print(f"  {'─'*50}")

    try:
        results = grangercausalitytests(data_granger, maxlag=MAX_LAG_GR, verbose=False)
        melhor_p = 1.0
        melhor_lag = None
        for lag, res in results.items():
            # Usar F-test (mais robusto que chi2 para amostras pequenas)
            f_stat = res[0]["ssr_ftest"][0]
            p_val  = res[0]["ssr_ftest"][1]
            p_bonf = min(p_val * MAX_LAG_GR, 1.0)
            sig    = "SIM **" if p_bonf < ALPHA_GR_BONF else (
                     "SIM *"  if p_val  < ALPHA           else "—")
            print(f"  {lag:>4}  {f_stat:>8.3f}  {p_val:>9.5f}  "
                  f"{p_bonf:>9.5f}  {sig:>10}")
            if p_val < melhor_p:
                melhor_p = p_val
                melhor_lag = lag
            granger_rows.append({
                "serie":   nome,
                "lag":     lag,
                "f_stat":  round(f_stat, 4),
                "p_valor": round(p_val, 5),
                "p_bonf":  round(p_bonf, 5),
                "sig_alpha05":  int(p_val  < ALPHA),
                "sig_bonf":     int(p_bonf < ALPHA_GR_BONF),
            })
        print(f"  → melhor lag={melhor_lag}  p={melhor_p:.5f}  "
              f"({'SIGNIFICATIVO' if melhor_p < ALPHA else 'não significativo'})")
    except Exception as e:
        print(f"  [ERRO] {e}")

df_granger = pd.DataFrame(granger_rows)


# ═════════════════════════════════════════════════════════════════════════════
# 6. CORRELAÇÃO PARCIAL — controlando histórico autorregressivo de crimes
# ═════════════════════════════════════════════════════════════════════════════
section("6. CORRELAÇÃO PARCIAL (controlando crimes_t1 e crimes_mm3)")

print("""
  Método: regressão dos resíduos (Frisch-Waugh).
    1. Regride n_crimes em {crimes_t1, crimes_mm3} → resíduos de crimes
    2. Regride polarity_t1 em {crimes_t1, crimes_mm3} → resíduos de polarity
    3. Correlaciona os dois conjuntos de resíduos
  Isso responde: a polarity tem relação com os crimes ALÉM do que o
  histórico autorregressivo já explica?
""")

# Usar os dados originais (não diferenciados) para correlação parcial com lags T-1
df_pc = df[["n_crimes", "crimes_t1", "crimes_mm3"]].copy()
for nome, col_t1 in SERIES_MIDIA.items():
    if col_t1 in df.columns:
        df_pc[nome + "_t1"] = df[col_t1]

df_pc = df_pc.dropna()
n_pc  = len(df_pc)

controles = ["crimes_t1", "crimes_mm3"]
X_ctrl    = add_constant(df_pc[controles])

# Resíduos de n_crimes após remover efeito autorregressivo
res_crimes = OLS(df_pc["n_crimes"], X_ctrl).fit().resid

print(f"  n = {n_pc} semanas  |  Controles: {controles}")
print(f"  R² do modelo autorregressivo (crimes ~ crimes_t1 + crimes_mm3): "
      f"{OLS(df_pc['n_crimes'], X_ctrl).fit().rsquared:.4f}\n")
print(f"  {'Série':<25}  {'r parcial':>10}  {'p-valor':>10}  {'Sig?':>6}  {'Interpretação'}")
print(f"  {'─'*75}")

parcial_rows = []
for nome in SERIES_CORR:
    col = nome + "_t1"
    if col not in df_pc.columns:
        continue
    # Resíduos da polarity após remover efeito dos controles
    res_pol = OLS(df_pc[col], X_ctrl).fit().resid
    r_p, p_p = stats.pearsonr(res_pol, res_crimes)
    sig = "**" if p_p < ALPHA / len(SERIES_CORR) else ("*" if p_p < ALPHA else "ns")
    interp = ("polarity ↑ → crimes ↑" if r_p > 0 else "polarity ↑ → crimes ↓")
    print(f"  {nome:<25}  {r_p:>+10.4f}  {p_p:>10.5f}  {sig:>6}  {interp if sig != 'ns' else ''}")
    parcial_rows.append({
        "serie":    nome,
        "r_parcial": round(r_p, 4),
        "p_valor":  round(p_p, 5),
        "sig":      sig,
    })

df_parcial = pd.DataFrame(parcial_rows)


# ═════════════════════════════════════════════════════════════════════════════
# 7. TABELA RESUMO EXECUTIVA
# ═════════════════════════════════════════════════════════════════════════════
section("7. TABELA RESUMO — TODOS OS TESTES POR SÉRIE")

print("""
  Síntese para escrita acadêmica.
  Pearson@melhor_lag = correlação mais forte encontrada (lags 0-8).
  Granger sig = se ao menos um lag tem p < 0.05 no teste de Granger.
  r_parcial = correlação controlando histórico autorregressivo de crimes.
  Legenda: ** p<Bonferroni  * p<0.05  ns = não significativo
""")

print(f"  {'Série':<25}  {'Pearson@lag':>12}  {'lag':>4}  "
      f"{'Granger':>8}  {'r_parcial':>10}  {'Interpretação geral'}")
print(f"  {'─'*85}")

for nome in SERIES_CORR:
    # Melhor Pearson
    sub_c = df_corr[df_corr["serie"] == nome]
    if not sub_c.empty:
        best_c = sub_c.loc[sub_c["pearson_r"].abs().idxmax()]
        pr_str = f"{best_c['pearson_r']:+.3f}"
        lag_str = str(int(best_c["lag_semanas"]))
        pr_sig  = "**" if best_c["sig_bonf"] else ("*" if best_c["sig_alpha05"] else "ns")
        pr_str += f" ({pr_sig})"
    else:
        pr_str = "n/a"; lag_str = "—"

    # Granger
    sub_g = df_granger[df_granger["serie"] == nome]
    if not sub_g.empty:
        gr_str = "SIM *" if sub_g["sig_alpha05"].any() else "não"
        if sub_g["sig_bonf"].any():
            gr_str = "SIM **"
    else:
        gr_str = "n/a"

    # Parcial
    sub_p = df_parcial[df_parcial["serie"] == nome]
    if not sub_p.empty:
        rp = sub_p.iloc[0]
        rp_str = f"{rp['r_parcial']:+.3f} ({rp['sig']})"
    else:
        rp_str = "n/a"

    # Interpretação
    if "vader" in nome:
        interp = "Sentimento geral (VADER)"
    elif "tybyria" in nome and "insta" in nome:
        interp = "Toxicidade Instagram (TybyrIA)"
    elif "tybyria" in nome:
        interp = "Toxicidade Reddit (TybyrIA)"
    elif "toxicidade" in nome:
        interp = "% posts tóxicos Reddit"
    elif "pct_tox" in nome:
        interp = "% comentários tóxicos Instagram"
    else:
        interp = nome

    print(f"  {nome:<25}  {pr_str:>12}  {lag_str:>4}  "
          f"{gr_str:>8}  {rp_str:>10}  {interp}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. SALVAR NO GCS
# ═════════════════════════════════════════════════════════════════════════════
section("8. SALVANDO RESULTADOS NO GCS")

# Combinar todas as tabelas em um único CSV
df_corr["tipo"]    = "pearson_spearman"
df_ccf_out = df_ccf.rename(columns={"ccf": "pearson_r"})
df_ccf_out["tipo"] = "ccf"

upload_csv(bucket, OUT_CORR,    df_corr)
upload_csv(bucket, OUT_GRANGER, df_granger)

print(f"""
RESUMO FINAL — FASE 5:
  n observações (semanal)   : {n_obs}
  n observações (D1 Granger): {len(df_diff)}
  Séries analisadas         : {len(SERIES_CORR)}
  Lags testados (Pearson)   : 0–{MAX_LAG} semanas
  Lags testados (Granger)   : 1–{MAX_LAG_GR} semanas
  α Bonferroni (Pearson)    : {ALPHA_BONF:.5f}
  α Bonferroni (Granger)    : {ALPHA_GR_BONF:.5f}

  Para a metodologia:
    - Correlações de Pearson e Spearman com p-valores corrigidos por Bonferroni
    - CCF para identificar o lag ótimo da associação temporal
    - Granger para testar se polarity Granger-causa crimes (séries D1)
    - Correlação parcial para isolar efeito da polarity do autorregressivo

  Arquivos:
    {OUT_CORR}
    {OUT_GRANGER}
""")
