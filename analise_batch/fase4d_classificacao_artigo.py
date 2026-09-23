#!/usr/bin/env python3
"""
Fase 4D — Classificação supervisionada: replicação do artigo.

Metodologia replicada de: "Real-time crime prediction using social media"
  - 6 classificadores base: RandomForest, XGBoost, DecisionTree, SVC, KNN, MLP
  - 3 conjuntos de features (BASE, MÍDIA, COMPLETO)
  - Feature Importance: ranking comparativo por conjunto
  - Quantifica contribuição da polaridade das redes sociais vs. variáveis
    geográficas e demográficas (como o artigo faz com Twitter, longitude e etnia)

Requer execução prévia de:
  fase4_engenharia_atributos.py  (gera dataset base)
  fase4b_features_geo_demograficas.py  (gera 3 datasets enriquecidos)
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
import math

import numpy as np
import pandas as pd
from gcs_local import gcs_client, storage
missing = []
try:
    import xgboost as xgb
except ImportError:
    missing.append("xgboost")
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    missing.append("shap (opcional)")

if "xgboost" in missing:
    print(f"[ERRO] xgboost não encontrado. Instale com: pip install xgboost")
    sys.exit(1)
if missing:
    print(f"[AVISO] Pacotes opcionais ausentes: {[m for m in missing if 'xgboost' not in m]}")

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from xgboost import XGBClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────
# KEY_FILE   = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET     = "lgbtminas-dados"

BLOB_BASE     = "analysis/fase4_artigo_base.csv"
BLOB_MIDIA    = "analysis/fase4_artigo_midia.csv"
BLOB_COMPLETO = "analysis/fase4_artigo_completo.csv"
OUT_RESULT    = "analysis/fase4_resultados_artigo.csv"
OUT_IMP       = "analysis/fase4_importancia_artigo.csv"

HOLDOUT_FRAC = 0.20
N_SPLITS_CV  = 5
N_ITER       = 30
SEED         = 42
N_BOOTSTRAP  = 1000

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
# 1. CARREGAR OS 3 DATASETS
# ═════════════════════════════════════════════════════════════════════════════
section("1. CARREGANDO DATASETS DO ARTIGO")

df_base     = read_blob_csv(bucket, BLOB_BASE)
df_midia    = read_blob_csv(bucket, BLOB_MIDIA)
df_completo = read_blob_csv(bucket, BLOB_COMPLETO)

# Validar alinhamento temporal
assert list(df_base["sem"]) == list(df_midia["sem"]) == list(df_completo["sem"]), \
    "Datasets com ordens temporais diferentes — re-execute fase4b."

META_COLS = ["sem", "n_crimes", "alvo"]

def get_feat_cols(df):
    return [c for c in df.columns if c not in META_COLS]

feat_base     = get_feat_cols(df_base)
feat_midia    = get_feat_cols(df_midia)
feat_completo = get_feat_cols(df_completo)

print(f"\n  BASE     : {len(df_base)} semanas × {len(feat_base)} features")
print(f"  MÍDIA    : {len(df_midia)} semanas × {len(feat_midia)} features")
print(f"  COMPLETO : {len(df_completo)} semanas × {len(feat_completo)} features")

print(f"\n  Features BASE    : {feat_base}")
print(f"\n  Features MÍDIA   : {feat_midia}")
print(f"\n  Features COMPLETO: {feat_completo}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. DIVISÃO TEMPORAL TREINO / TESTE (80/20)
# ═════════════════════════════════════════════════════════════════════════════
section("2. DIVISÃO TEMPORAL TREINO/TESTE")

n_total  = len(df_base)
n_teste  = max(1, round(n_total * HOLDOUT_FRAC))
n_treino = n_total - n_teste

print(f"""
  Total de semanas : {n_total}
  Treino+val       : {n_treino}  ({df_base['sem'].iloc[0]} → {df_base['sem'].iloc[n_treino-1]})
  Teste holdout    : {n_teste}   ({df_base['sem'].iloc[n_treino]} → {df_base['sem'].iloc[-1]})
""")

y_all    = df_base["alvo"].values
y_treino = y_all[:n_treino]
y_teste  = y_all[n_treino:]

for nome, y in [("Treino+val", y_treino), ("Teste", y_teste)]:
    n0, n1 = (y==0).sum(), (y==1).sum()
    print(f"  {nome:<12}: n={len(y)}  classe0={n0} ({n0/len(y)*100:.0f}%)  "
          f"classe1={n1} ({n1/len(y)*100:.0f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE DE TREINO
# ═════════════════════════════════════════════════════════════════════════════
section("3. SETUP DOS MODELOS")

tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])

MODELOS = {
    "RandomForest": make_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    "XGBoost":      make_pipeline(XGBClassifier(random_state=SEED, eval_metric="logloss",
                                                verbosity=0, n_jobs=-1)),
    "DecisionTree": make_pipeline(DecisionTreeClassifier(random_state=SEED)),
    "SVC":          make_pipeline(SVC(probability=True, random_state=SEED)),
    "KNN":          make_pipeline(KNeighborsClassifier()),
    "MLP":          make_pipeline(MLPClassifier(random_state=SEED, max_iter=500)),
}

PARAM_GRIDS = {
    "RandomForest": {
        "clf__n_estimators":      [100, 200, 300, 500],
        "clf__max_depth":         [None, 5, 10, 15, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "clf__n_estimators":     [50, 100, 200, 300],
        "clf__max_depth":        [3, 4, 5, 6, 8],
        "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
    "DecisionTree": {
        "clf__max_depth":         [None, 3, 5, 7, 10, 15],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf":  [1, 2, 4, 8],
        "clf__criterion":         ["gini", "entropy"],
    },
    "SVC": {
        "clf__C":      [0.01, 0.1, 1, 10, 100],
        "clf__kernel": ["rbf", "linear", "poly"],
        "clf__gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "clf__weights":     ["uniform", "distance"],
        "clf__metric":      ["euclidean", "manhattan", "minkowski"],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        "clf__activation":         ["relu", "tanh"],
        "clf__alpha":              [0.0001, 0.001, 0.01, 0.1],
        "clf__learning_rate":      ["constant", "adaptive"],
    },
}

def avaliar(nome, pipe, X_test, y_test, y_pred=None):
    if y_pred is None:
        y_pred = pipe.predict(X_test)
    return {
        "modelo":   nome,
        "acuracia": round(accuracy_score(y_test, y_pred), 4),
        "precisao": round(precision_score(y_test, y_pred, average="binary",
                                          pos_label=1, zero_division=0), 4),
        "recall":   round(recall_score(y_test, y_pred, average="binary",
                                       pos_label=1, zero_division=0), 4),
        "f1":       round(f1_score(y_test, y_pred, average="binary",
                                   pos_label=1, zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0), 4),
        "cm":       confusion_matrix(y_test, y_pred).tolist(),
        "y_pred":   y_pred.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. TREINO NOS 3 DATASETS
# ═════════════════════════════════════════════════════════════════════════════
section("4. TREINO — 6 CLASSIFICADORES × 3 DATASETS")

todos_resultados = []
tuned_por_ds = {}   # ds_label → {nome_modelo: pipe_otimizado}

DATASETS = [
    ("BASE",     df_base,     feat_base),
    ("MÍDIA",    df_midia,    feat_midia),
    ("COMPLETO", df_completo, feat_completo),
]

for ds_label, df_ds, feats in DATASETS:
    sub(f"Dataset: {ds_label}  ({len(feats)} features)")

    X = df_ds[feats].values
    y = df_ds["alvo"].values
    X_train, X_test = X[:n_treino], X[n_treino:]
    y_train, y_test = y[:n_treino], y[n_treino:]

    tuned_pipes = {}

    # Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    res_d = avaliar(f"Dummy_most_frequent", dummy, X_test, y_test)
    res_d["dataset"] = ds_label
    todos_resultados.append(res_d)
    print(f"  Dummy (most_frequent) → F1={res_d['f1']:.4f}  Acc={res_d['acuracia']:.4f}")

    for nome, pipe_def in MODELOS.items():
        # Default
        pipe_d = clone(pipe_def)
        pipe_d.fit(X_train, y_train)
        res = avaliar(f"{nome}_default", pipe_d, X_test, y_test)
        res["dataset"] = ds_label
        todos_resultados.append(res)

        # Otimizado
        pipe_opt = clone(pipe_def)
        rscv = RandomizedSearchCV(
            pipe_opt, PARAM_GRIDS[nome],
            n_iter=N_ITER, cv=tscv, scoring="f1",
            refit=True, random_state=SEED, n_jobs=-1, error_score=0,
        )
        rscv.fit(X_train, y_train)
        best_pipe = rscv.best_estimator_
        tuned_pipes[nome] = best_pipe

        res_opt = avaliar(f"{nome}_otimizado", best_pipe, X_test, y_test)
        res_opt["dataset"]     = ds_label
        res_opt["cv_best"]     = round(rscv.best_score_, 4)
        res_opt["best_params"] = str(rscv.best_params_)
        todos_resultados.append(res_opt)

        print(f"  {nome:<13} default→F1={res['f1']:.4f}  "
              f"otimizado→F1={res_opt['f1']:.4f}  (cv={rscv.best_score_:.4f})")

    tuned_por_ds[ds_label] = {"feats": feats, "X_test": X_test, "y_test": y_test,
                               "tuned": tuned_pipes}


# ═════════════════════════════════════════════════════════════════════════════
# 5. TABELA COMPARATIVA: F1 POR DATASET × MODELO
# ═════════════════════════════════════════════════════════════════════════════
section("5. TABELA COMPARATIVA — F1 POR DATASET (replicação da tabela do artigo)")

print("\n  Métrica principal: F1 binário (pos_label=1, classe 'alto')")
print("  Critério de seleção: cv_best (TimeSeriesSplit F1 na fase de treino)\n")

# Pivot: linhas = modelos, colunas = datasets
modelos_base = ["Dummy_most_frequent"] + [f"{n}_otimizado" for n in MODELOS.keys()]
pivot_rows = []
for mod in modelos_base:
    row = {"modelo": mod}
    for ds_label, _, _ in DATASETS:
        match = next((r for r in todos_resultados
                      if r["modelo"] == mod and r["dataset"] == ds_label), None)
        if match:
            row[f"F1_{ds_label}"]      = match["f1"]
            row[f"F1mac_{ds_label}"]   = match["f1_macro"]
            row[f"Acc_{ds_label}"]     = match["acuracia"]
    pivot_rows.append(row)

df_pivot = pd.DataFrame(pivot_rows)
print(f"  {'Modelo':<22}", end="")
for ds_label, _, _ in DATASETS:
    print(f"  {ds_label:>12}(F1)", end="")
print()
print("  " + "─" * 65)
for _, row in df_pivot.iterrows():
    print(f"  {row['modelo']:<22}", end="")
    for ds_label, _, _ in DATASETS:
        val = row.get(f"F1_{ds_label}", float("nan"))
        print(f"  {val:>15.4f}", end="")
    print()

# Ganho de F1 relativo ao baseline
print(f"\n  Ganho de F1 vs. BASE (com_midia - base  /  completo - base):")
print(f"  {'Modelo':<22}  {'Δ MÍDIA':>10}  {'Δ COMPLETO':>12}")
print("  " + "─" * 50)
for _, row in df_pivot.iterrows():
    f1_b = row.get("F1_BASE", float("nan"))
    f1_m = row.get("F1_MÍDIA", float("nan"))
    f1_c = row.get("F1_COMPLETO", float("nan"))
    dm = f"{f1_m - f1_b:+.4f}" if not (np.isnan(f1_m) or np.isnan(f1_b)) else "n/a"
    dc = f"{f1_c - f1_b:+.4f}" if not (np.isnan(f1_c) or np.isnan(f1_b)) else "n/a"
    print(f"  {row['modelo']:<22}  {dm:>10}  {dc:>12}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE — RANKING COMPARATIVO (tabela central do artigo)
# ═════════════════════════════════════════════════════════════════════════════
section("6. FEATURE IMPORTANCE — RANKING COMPARATIVO POR DATASET")

print("""
  Replica a análise de Estimativa de Importância de Recursos do artigo.
  Para cada dataset, extrai importância do melhor modelo (selecionado por cv_best).
  Método: feature_importances_ (RF/XGBoost/DT) ou permutation_importance (SVC/KNN/MLP).
""")

# Categorias de features para o relatório comparativo
def categorizar_feature(nome):
    nome_l = nome.lower()
    if any(k in nome_l for k in ["lat", "lon", "municipio", "pct_bh", "geo"]):
        return "GEOGRÁFICO"
    if any(k in nome_l for k in ["branca", "parda", "preta", "feminino",
                                  "masculino", "trans", "lgb", "demo"]):
        return "DEMOGRÁFICO"
    if any(k in nome_l for k in ["tybyria", "toxicidade", "vader",
                                  "insta", "n_posts", "comentarios"]):
        return "MÍDIA SOCIAL"
    if any(k in nome_l for k in ["crimes", "mes", "trimestre", "sem_ano"]):
        return "CRIME/CALENDÁRIO"
    return "OUTRO"

all_importance_rows = []

for ds_label, df_ds, feats in DATASETS:
    sub(f"Feature Importance — {ds_label}")

    info = tuned_por_ds[ds_label]
    X_te = info["X_test"]
    y_te = info["y_test"]
    tuned = info["tuned"]

    # Selecionar melhor modelo por cv_best
    cands = [r for r in todos_resultados
             if r["dataset"] == ds_label and r.get("cv_best") is not None]
    if not cands:
        print(f"  [AVISO] Sem modelos otimizados para {ds_label}")
        continue

    best_row  = max(cands, key=lambda x: x["cv_best"])
    nome_best = best_row["modelo"].replace("_otimizado", "")

    # Garantir que temos o pipe treinado
    if nome_best not in tuned:
        nome_best = "RandomForest"
    best_pipe = tuned[nome_best]

    print(f"  Modelo selecionado: {best_row['modelo']}")
    print(f"  cv_best={best_row['cv_best']:.4f}  F1 holdout={best_row['f1']:.4f}")

    clf = best_pipe.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        imp     = clf.feature_importances_
        imp_std = np.zeros_like(imp)
        metodo  = "feature_importances_"
    elif hasattr(clf, "coef_"):
        imp     = np.abs(clf.coef_[0])
        imp_std = np.zeros_like(imp)
        metodo  = "|coef_|"
    else:
        perm = permutation_importance(
            best_pipe, X_te, y_te,
            n_repeats=100, scoring="f1", random_state=SEED,
        )
        imp     = perm.importances_mean
        imp_std = perm.importances_std
        metodo  = "permutation_importance"

    fi = pd.DataFrame({
        "feature":    feats,
        "importance": imp,
        "std":        imp_std,
        "categoria":  [categorizar_feature(f) for f in feats],
        "dataset":    ds_label,
        "modelo":     best_row["modelo"],
        "metodo":     metodo,
    })
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1

    print(f"\n  Método: {metodo}")
    print(f"  {'Rank':>4}  {'Feature':<28}  {'Importância':>12}  {'Categoria':<18}")
    print(f"  {'─'*65}")
    for _, row in fi.iterrows():
        print(f"  {int(row['rank']):>4}  {row['feature']:<28}  "
              f"{row['importance']:>12.4f}  {row['categoria']:<18}")

    # Importância média por categoria
    cat_mean = fi.groupby("categoria")["importance"].sum().sort_values(ascending=False)
    print(f"\n  Importância total por categoria:")
    for cat, val in cat_mean.items():
        pct = val / fi["importance"].sum() * 100 if fi["importance"].sum() > 0 else 0
        print(f"    {cat:<20}: {val:.4f}  ({pct:.1f}% do total)")

    all_importance_rows.append(fi)

# Tabela consolidada para exportação
df_imp_all = pd.concat(all_importance_rows, ignore_index=True) if all_importance_rows else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# 7. COMPARAÇÃO DE IMPORTÂNCIA ENTRE DATASETS (análise do artigo)
# ═════════════════════════════════════════════════════════════════════════════
section("7. ANÁLISE COMPARATIVA — POLARIDADE vs. GEOGRÁFICO vs. DEMOGRÁFICO")

print("""
  Replica a descoberta central do artigo:
  "Longitude e etnia branca foram fatores influentes,
   enquanto a polaridade do sentimento mostrou relação
   indireta mas significativa com a previsão de crimes."
""")

if not df_imp_all.empty:
    # Features de mídia social presentes no dataset COMPLETO
    fi_completo = df_imp_all[df_imp_all["dataset"] == "COMPLETO"].copy()

    if len(fi_completo) > 0:
        print("  Ranking no Dataset COMPLETO (todas as categorias):\n")
        print(f"  {'Rank':>4}  {'Categoria':<20}  {'Feature':<28}  {'Importância':>12}")
        print(f"  {'─'*68}")
        for _, row in fi_completo.iterrows():
            print(f"  {int(row['rank']):>4}  {row['categoria']:<20}  "
                  f"{row['feature']:<28}  {row['importance']:>12.4f}")

        # Onde está a polaridade no ranking?
        midia_rows = fi_completo[fi_completo["categoria"] == "MÍDIA SOCIAL"]
        geo_rows   = fi_completo[fi_completo["categoria"] == "GEOGRÁFICO"]
        demo_rows  = fi_completo[fi_completo["categoria"] == "DEMOGRÁFICO"]

        print(f"\n  RESUMO COMPARATIVO (replicação da tabela do artigo):")
        print(f"  {'Categoria':<22}  {'Rank médio':>11}  {'Importância total':>18}  {'n features':>11}")
        print(f"  {'─'*70}")
        for cat, rows in [("MÍDIA SOCIAL", midia_rows), ("GEOGRÁFICO", geo_rows),
                          ("DEMOGRÁFICO", demo_rows), ("CRIME/CALENDÁRIO",
                           fi_completo[fi_completo["categoria"] == "CRIME/CALENDÁRIO"])]:
            if len(rows) == 0:
                print(f"  {cat:<22}  {'(sem dados)':>11}")
                continue
            rank_med = rows["rank"].mean()
            imp_tot  = rows["importance"].sum()
            print(f"  {cat:<22}  {rank_med:>11.1f}  {imp_tot:>18.4f}  {len(rows):>11}")

        # Interpretação
        print(f"""
  INTERPRETAÇÃO:
    Rank médio menor = features com maior poder preditivo em geral.
    Importância total = soma da contribuição de toda a categoria.

    Se GEOGRÁFICO ou DEMOGRÁFICO têm rank médio menor que MÍDIA SOCIAL,
    replicamos o achado do artigo: variáveis espaciais/demográficas são
    mais preditivas que a polaridade do texto nas redes sociais.

    Se MÍDIA SOCIAL tem importância total relevante (> 5%), confirma-se
    que a polaridade tem relação indireta mas significativa com os crimes.
""")
    else:
        print("  [AVISO] Sem resultados para o dataset COMPLETO.")


# ═════════════════════════════════════════════════════════════════════════════
# 8. BOOTSTRAP CI DO F1 — COMPARAÇÃO ENTRE DATASETS
# ═════════════════════════════════════════════════════════════════════════════
section("8. BOOTSTRAP CI DO F1 (n=1000)")

def bootstrap_f1(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = np.array(y_true)[idx], np.array(y_pred)[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(f1_score(yt, yp, zero_division=0))
    return np.array(scores)

y_test_arr = df_base["alvo"].values[n_treino:]

print(f"\n  Comparação: quantifica se social media e geo/demo melhoram a predição.")
print(f"  IC 95% bootstrap (n={N_BOOTSTRAP} reamostragens com reposição, n_teste={n_teste}).\n")

boot_results = {}
for ds_label, _, _ in DATASETS:
    cands = [r for r in todos_resultados
             if r["dataset"] == ds_label and r.get("cv_best") is not None]
    if not cands:
        continue
    best = max(cands, key=lambda x: x["cv_best"])
    sc   = bootstrap_f1(y_test_arr, best["y_pred"])
    lo, hi = np.percentile(sc, [2.5, 97.5])
    boot_results[ds_label] = {"row": best, "lo": lo, "hi": hi, "scores": sc}
    print(f"  {ds_label:<12}  Melhor: {best['modelo']}")
    print(f"    F1 observado     = {best['f1']:.4f}")
    print(f"    IC 95% bootstrap = [{lo:.4f}, {hi:.4f}]  (largura={hi-lo:.4f})\n")

# Sobreposição BASE vs MÍDIA e BASE vs COMPLETO
for cmp_label in ["MÍDIA", "COMPLETO"]:
    if "BASE" not in boot_results or cmp_label not in boot_results:
        continue
    lo_b = boot_results["BASE"][    "lo"]; hi_b = boot_results["BASE"]["hi"]
    lo_c = boot_results[cmp_label]["lo"]; hi_c = boot_results[cmp_label]["hi"]
    sc_b = boot_results["BASE"]["scores"]; sc_c = boot_results[cmp_label]["scores"]
    p_sup = float(np.mean(sc_c > sc_b))
    sobrep = not (lo_c > hi_b or lo_b > hi_c)
    print(f"  BASE vs {cmp_label}: P(F1_{cmp_label} > F1_BASE)={p_sup:.3f}  "
          f"ICs {'sobrepostos' if sobrep else 'NÃO sobrepostos'}")


# ═════════════════════════════════════════════════════════════════════════════
# 9. SALVAR RESULTADOS
# ═════════════════════════════════════════════════════════════════════════════
section("9. SALVANDO RESULTADOS NO GCS")

df_save = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("cm", "y_pred", "best_params")}
    for r in todos_resultados
])
upload_csv(bucket, OUT_RESULT, df_save)

if not df_imp_all.empty:
    upload_csv(bucket, OUT_IMP, df_imp_all)

print(f"""
RESUMO FINAL — REPLICAÇÃO DO ARTIGO:
  Semanas treino    : {n_treino}
  Semanas teste     : {n_teste}
  Datasets avaliados: {len(DATASETS)} (BASE, MÍDIA, COMPLETO)
  Classificadores   : {len(MODELOS)} × {len(DATASETS)} datasets = {len(MODELOS)*len(DATASETS)} modelos

  Para interpretar:
  1. Compare F1 BASE → MÍDIA → COMPLETO: ganhos indicam contribuição de cada grupo
  2. Veja o ranking de feature importance no COMPLETO: geo/demo vs. polaridade
  3. Cite a coerência com o artigo: "encontramos padrão análogo ao de [Ref],
     onde variáveis [geográficas/demográficas] superaram a polaridade social
     em poder preditivo, mas a toxicidade mostrou contribuição indireta de X%"

  Arquivos gerados:
  - {OUT_RESULT}
  - {OUT_IMP}
""")
