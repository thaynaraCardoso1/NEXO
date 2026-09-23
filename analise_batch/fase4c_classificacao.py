#!/usr/bin/env python3
"""
Fase 4C — Classificação supervisionada semanal (Jimoh, 2023 adaptado).

Treina 6 classificadores × 2 datasets × {default, otimizado} + ensembles.
Avalia no holdout temporal (últimas 20% semanas).
Extrai importância de features e IC bootstrap do F1.

CONTROLE:
  PARAR_APOS_SPLIT = True   → mostra divisão treino/teste e para (confirme primeiro)
  PARAR_APOS_SPLIT = False  → executa treino completo
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
# ── verificação de dependências ───────────────────────────────────────────────
missing = []
try:
    import xgboost as xgb
except ImportError:
    missing.append("xgboost")
try:
    import shap
except ImportError:
    missing.append("shap")
if missing:
    print(f"[AVISO] Pacotes ausentes: {missing}")
    print("  Instale com: pip install " + " ".join(missing))
    if "xgboost" in missing:
        print("  XGBoost é obrigatório — encerrando.")
        sys.exit(1)
    SHAP_OK = False
else:
    SHAP_OK = True

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
from xgboost import XGBClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────
PARAR_APOS_SPLIT = False  # confirmado: 124 treino / 31 teste, balanço ok

# KEY_FILE     = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET       = "lgbtminas-dados"
BLOB_SEM_TOX = "analysis/fase4_dataset_sem_toxicidade.csv"
BLOB_COM_TOX = "analysis/fase4_dataset_com_toxicidade.csv"
OUT_RESULTADOS = "analysis/fase4_resultados_classificacao.csv"

HOLDOUT_FRAC = 0.20   # últimas 20% das semanas = teste
N_SPLITS_CV  = 5      # TimeSeriesSplit folds dentro do treino
N_ITER       = 30     # iterações do RandomizedSearchCV
SEED         = 42
N_BOOTSTRAP  = 1000   # reamostragens para IC do F1

FEAT_SEM = ["crimes_t1", "crimes_t2", "crimes_mm3", "mes", "trimestre", "sem_ano"]
FEAT_COM = FEAT_SEM + [
    "toxicidade_t1", "toxicidade_mm3",
    "tybyria_t1",    "tybyria_mm3",
    "n_posts_t1",
]
TOX_FEATS = ["toxicidade_t1", "toxicidade_mm3", "tybyria_t1", "tybyria_mm3", "n_posts_t1"]

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
# 1. CARREGAR DATASETS
# ═════════════════════════════════════════════════════════════════════════════
section("1. CARREGANDO DATASETS DO GCS")

df_sem = read_blob_csv(bucket, BLOB_SEM_TOX)
df_com = read_blob_csv(bucket, BLOB_COM_TOX)
print(f"  sem_toxicidade : {df_sem.shape}")
print(f"  com_toxicidade : {df_com.shape}")
assert list(df_sem["sem"]) == list(df_com["sem"]), "Datasets com ordem temporal diferente!"


# ═════════════════════════════════════════════════════════════════════════════
# 2. DIVISÃO TEMPORAL TREINO / TESTE
# ═════════════════════════════════════════════════════════════════════════════
section("2. DIVISÃO TEMPORAL TREINO/TESTE (80% / 20%)")

n_total   = len(df_sem)
n_teste   = max(1, round(n_total * HOLDOUT_FRAC))
n_treino  = n_total - n_teste

print(f"""
  Total de semanas      : {n_total}
  Treino + validação    : {n_treino}  (semanas 1–{n_treino})
  Teste holdout         : {n_teste}   (semanas {n_treino+1}–{n_total})

  Treino: {df_sem['sem'].iloc[0]}  →  {df_sem['sem'].iloc[n_treino-1]}
  Teste : {df_sem['sem'].iloc[n_treino]} →  {df_sem['sem'].iloc[-1]}
""")

# Distribuição do alvo em cada partição
y_all    = df_sem["alvo"].values
y_treino = y_all[:n_treino]
y_teste  = y_all[n_treino:]

print(f"  {'Partição':<22} {'n':>5}  {'Classe 0':>10}  {'Classe 1':>10}  {'Razão 0/1':>10}")
print("  " + "-" * 62)
for nome, y in [("Treino+val", y_treino), ("Teste holdout", y_teste), ("Total", y_all)]:
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    razao = n0 / n1 if n1 > 0 else float("inf")
    print(f"  {nome:<22} {len(y):>5}  {n0:>6} ({n0/len(y)*100:.1f}%)  "
          f"{n1:>6} ({n1/len(y)*100:.1f}%)  {razao:>9.3f}")

print(f"""
  TimeSeriesSplit(n_splits={N_SPLITS_CV}) dentro do treino:
    Cada fold: treino cresce de ~{n_treino//(N_SPLITS_CV+1)} a ~{n_treino*N_SPLITS_CV//(N_SPLITS_CV+1)} semanas
    Validação  : ~{n_treino//(N_SPLITS_CV+1)} semanas por fold
""")

if PARAR_APOS_SPLIT:
    print(f"""
{SEP}
PAUSA — PARAR_APOS_SPLIT = True
{SEP}
Confirme:
  1. A divisão treino/teste está correta?
  2. O balanço de classes no teste holdout é aceitável?
  3. O período de teste cobre o intervalo temporal esperado?

Para prosseguir: mude PARAR_APOS_SPLIT = False e execute novamente.
""")
    sys.exit(0)


# ═════════════════════════════════════════════════════════════════════════════
# 3. SETUP — MODELOS E GRADES DE HIPERPARÂMETROS
# ═════════════════════════════════════════════════════════════════════════════
section("3. DEFINIÇÃO DOS MODELOS")

tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])

MODELOS_DEF = {
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
        "clf__n_estimators":   [100, 200, 300, 500],
        "clf__max_depth":      [None, 5, 10, 15, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":   ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "clf__n_estimators":      [50, 100, 200, 300],
        "clf__max_depth":         [3, 4, 5, 6, 8],
        "clf__learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":         [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight":  [1, 3, 5],
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

print("  Modelos: " + ", ".join(MODELOS_DEF.keys()))
print(f"  RandomizedSearchCV: n_iter={N_ITER}, cv=TimeSeriesSplit({N_SPLITS_CV}), scoring='f1'")


# ═════════════════════════════════════════════════════════════════════════════
# 4. TREINO E AVALIAÇÃO
# ═════════════════════════════════════════════════════════════════════════════
section("4. TREINO — DEFAULT + OTIMIZADO + ENSEMBLES")

def avaliar(nome, pipe, X_test, y_test):
    # Todas as métricas usam average='binary', pos_label=1 (classe "alto: >11 crimes")
    # f1_macro é reportado como referência para equilíbrio entre classes
    y_pred = pipe.predict(X_test)
    return {
        "modelo":    nome,
        "acuracia":  round(accuracy_score(y_test, y_pred), 4),
        "precisao":  round(precision_score(y_test, y_pred, average="binary",
                                           pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="binary",
                                        pos_label=1, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="binary",
                                    pos_label=1, zero_division=0), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average="macro",
                                    zero_division=0), 4),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
        "y_pred":    y_pred.tolist(),
    }

# [EXPLÍCITO] average='binary', pos_label=1 → métricas focadas na classe ">11 crimes"
# f1_macro = média simples entre classes 0 e 1 (sem ponderação por suporte)

todos_resultados = []

for ds_label, df_ds, feat_cols in [
    ("sem_toxicidade", df_sem, FEAT_SEM),
    ("com_toxicidade", df_com, FEAT_COM),
]:
    sub(f"Dataset: {ds_label}  ({len(feat_cols)} features)")

    X = df_ds[feat_cols].values
    y = df_ds["alvo"].values

    X_train, X_test = X[:n_treino], X[n_treino:]
    y_train, y_test = y[:n_treino], y[n_treino:]

    tuned_pipes = {}   # guarda modelos otimizados para ensembles

    # ── Baselines Dummy ──────────────────────────────────────────────────────
    for strat, kw in [
        ("most_frequent", {}),
        ("stratified",    {"random_state": SEED}),
        ("constant",      {"constant": 1}),          # sempre prevê classe 1 (>11 crimes)
    ]:
        dummy = DummyClassifier(strategy=strat, **kw)
        dummy.fit(X_train, y_train)
        res_d = avaliar(f"Dummy_{strat}", dummy, X_test, y_test)
        res_d["dataset"] = ds_label
        todos_resultados.append(res_d)
        print(f"  Dummy ({strat:<14}) → F1={res_d['f1']:.4f}  Acc={res_d['acuracia']:.4f}")

    # ── Modelos individuais ──────────────────────────────────────────────────
    for nome, pipe_def in MODELOS_DEF.items():
        # ── a) Default
        from sklearn.base import clone
        pipe_d = clone(pipe_def)
        pipe_d.fit(X_train, y_train)
        res = avaliar(f"{nome}_default", pipe_d, X_test, y_test)
        res["dataset"] = ds_label
        todos_resultados.append(res)
        print(f"  {nome:<15} default  → F1={res['f1']:.4f}  Acc={res['acuracia']:.4f}")

        # ── b) Otimizado (RandomizedSearchCV)
        pipe_opt = clone(pipe_def)
        rscv = RandomizedSearchCV(
            pipe_opt, PARAM_GRIDS[nome],
            n_iter=N_ITER, cv=tscv,
            scoring="f1", refit=True,
            random_state=SEED, n_jobs=-1,
            error_score=0,
        )
        rscv.fit(X_train, y_train)
        best_pipe = rscv.best_estimator_
        tuned_pipes[nome] = best_pipe

        res_opt = avaliar(f"{nome}_otimizado", best_pipe, X_test, y_test)
        res_opt["dataset"]  = ds_label
        res_opt["best_params"] = str(rscv.best_params_)
        res_opt["cv_best"]  = round(rscv.best_score_, 4)   # F1 binário na CV (treino+val)
        todos_resultados.append(res_opt)
        print(f"  {nome:<15} otimizado→ F1={res_opt['f1']:.4f}  "
              f"Acc={res_opt['acuracia']:.4f}  "
              f"(cv_best={rscv.best_score_:.4f})")

    # ── Ensemble: VotingClassifier (soft) ───────────────────────────────────
    print(f"\n  Treinando VotingClassifier (soft)...")
    voting = VotingClassifier(
        estimators=[(n, p) for n, p in tuned_pipes.items()],
        voting="soft",
    )
    voting.fit(X_train, y_train)
    res_v = avaliar("VotingClassifier", voting, X_test, y_test)
    res_v["dataset"] = ds_label
    todos_resultados.append(res_v)
    print(f"  VotingClassifier  → F1={res_v['f1']:.4f}  Acc={res_v['acuracia']:.4f}")

    # ── Ensemble: StackingClassifier (meta: LogisticRegression) ─────────────
    print(f"  Treinando StackingClassifier (meta: LogisticRegression)...")
    stacking = StackingClassifier(
        estimators=[(n, p) for n, p in tuned_pipes.items()],
        final_estimator=LogisticRegression(random_state=SEED, max_iter=500),
        cv=KFold(n_splits=5, shuffle=False),  # sem shuffle preserva ordem temporal
        passthrough=False,
    )
    stacking.fit(X_train, y_train)
    res_s = avaliar("StackingClassifier", stacking, X_test, y_test)
    res_s["dataset"] = ds_label
    todos_resultados.append(res_s)
    print(f"  StackingClassifier→ F1={res_s['f1']:.4f}  Acc={res_s['acuracia']:.4f}")

    # ── Guardar referências para seção 6 ─────────────────────────────────────
    if ds_label == "sem_toxicidade":
        train_sem = (X_train, y_train, X_test, y_test, feat_cols, tuned_pipes)
    else:
        train_com = (X_train, y_train, X_test, y_test, feat_cols, tuned_pipes)


# ═════════════════════════════════════════════════════════════════════════════
# 5. TABELA COMPARATIVA — TODOS OS MODELOS
# ═════════════════════════════════════════════════════════════════════════════
section("5. TABELA COMPARATIVA (holdout)")

df_res = pd.DataFrame(todos_resultados)
df_res = df_res.drop(columns=["cm", "y_pred", "best_params"], errors="ignore")

print("  precision/recall/f1 = average='binary', pos_label=1  (classe '>11 crimes')")
print("  f1_macro = média simples entre classes 0 e 1\n")
for ds in ["sem_toxicidade", "com_toxicidade"]:
    sub(f"Dataset: {ds}")
    sub_df = df_res[df_res["dataset"] == ds].sort_values("f1", ascending=False)
    cols = ["modelo", "acuracia", "precisao", "recall", "f1", "f1_macro"]
    cols = [c for c in cols if c in sub_df.columns]
    print(sub_df[cols].to_string(index=False))

# Matrizes de confusão dos melhores por dataset
for ds in ["sem_toxicidade", "com_toxicidade"]:
    sub(f"Matriz de confusão — melhor modelo ({ds})")
    best_row = next(r for r in sorted(todos_resultados,
                                       key=lambda x: x["f1"], reverse=True)
                    if r["dataset"] == ds)
    cm = np.array(best_row["cm"])
    print(f"\n  Modelo: {best_row['modelo']}")
    print(f"  {'':15}  Pred 0   Pred 1")
    print(f"  {'Real 0':15}  {cm[0,0]:6}   {cm[0,1]:6}")
    print(f"  {'Real 1':15}  {cm[1,0]:6}   {cm[1,1]:6}")
    print(classification_report(
        train_sem[3] if ds == "sem_toxicidade" else train_com[3],
        best_row["y_pred"],
        target_names=["baixo (≤11)", "alto  (>11)"],
    ))


# ═════════════════════════════════════════════════════════════════════════════
# 6. IMPORTÂNCIA DE FEATURES — MELHOR MODELO DE CADA DATASET
# ═════════════════════════════════════════════════════════════════════════════
section("6. IMPORTÂNCIA DE FEATURES")

def extrair_importancia(pipe, feat_names, X_test, y_test, label):
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    X_test_sc = scaler.transform(X_test)

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        metodo = "feature_importances_"
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
        metodo = "|coef_| (SVC linear)"
    else:
        # Permutation importance — n_repeats=100 para IC estável com n_teste=31
        perm = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=100, scoring="f1", random_state=SEED
        )
        imp     = perm.importances_mean
        imp_std = perm.importances_std
        metodo  = "permutation_importance (n_repeats=100)"

    if "imp_std" not in dir():
        imp_std = np.zeros_like(imp)   # não disponível para feature_importances_

    fi = pd.DataFrame({"feature": feat_names, "importance": imp, "std": imp_std})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    # razão sinal/ruído: importance / std (NaN se std=0)
    fi["snr"] = (fi["importance"] / fi["std"].replace(0, np.nan)).round(2)

    sub(f"Feature importance — {label}  [{metodo}]")
    print(fi[["rank", "feature", "importance", "std", "snr"]].to_string(index=False))
    if "permutation" in metodo:
        print("\n  [NOTA] std = desvio-padrão entre as 100 permutações de cada feature.")
        print("  snr = importance/std: valores < 1 indicam que a estimativa não se")
        print("  distingue de ruído — interpretar ranking com cautela.")

    # Posição das features de toxicidade
    if any(f in feat_names for f in TOX_FEATS):
        print("\n  Ranking das features de toxicidade:")
        for tf in TOX_FEATS:
            if tf in fi["feature"].values:
                row = fi[fi["feature"] == tf].iloc[0]
                print(f"    {tf:<22} rank={int(row['rank']):2d}  "
                      f"importance={row['importance']:.4f}")

    # SHAP (apenas para modelos baseados em árvore)
    if SHAP_OK and hasattr(clf, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_test_sc)
            # sv pode ser lista [class0, class1] ou array 3D
            if isinstance(sv, list):
                sv_class1 = sv[1]
            elif sv.ndim == 3:
                sv_class1 = sv[:, :, 1]
            else:
                sv_class1 = sv
            shap_mean = np.abs(sv_class1).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": shap_mean})
            shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_df["rank"] = shap_df.index + 1
            sub(f"SHAP mean |value| — {label}")
            print(shap_df[["rank", "feature", "mean_abs_shap"]].to_string(index=False))
            if any(f in feat_names for f in TOX_FEATS):
                print("\n  Ranking SHAP das features de toxicidade:")
                for tf in TOX_FEATS:
                    if tf in shap_df["feature"].values:
                        row = shap_df[shap_df["feature"] == tf].iloc[0]
                        print(f"    {tf:<22} rank={int(row['rank']):2d}  "
                              f"shap={row['mean_abs_shap']:.4f}")
        except Exception as e:
            print(f"  [SHAP erro] {e}")

    return fi

fi_com_tox = None   # guardar para seção de colinearidade

for ds_label, train_tuple in [("sem_toxicidade", train_sem), ("com_toxicidade", train_com)]:
    X_tr, y_tr, X_te, y_te, feats, tuned = train_tuple

    # CRITÉRIO DEFINITIVO: cv_best = F1 binário médio na TimeSeriesSplit (124 sem. treino+val)
    # O holdout não é acessado para selecionar o modelo — apenas para avaliá-lo uma única vez.
    # Critério anterior (f1_macro no holdout) era vazamento: escolha e avaliação no mesmo dado.
    candidatos_otimizados = [r for r in todos_resultados
                             if r["dataset"] == ds_label
                             and r.get("cv_best") is not None]
    best_row  = max(candidatos_otimizados, key=lambda x: x["cv_best"])
    nome_best = best_row["modelo"].replace("_otimizado", "")
    print(f"\n  [SELEÇÃO — cv_best, sem acesso ao holdout] {ds_label}:")
    print(f"    modelo selecionado : {best_row['modelo']}")
    print(f"    cv_best (treino)   : {best_row['cv_best']:.4f}")
    print(f"    f1_macro holdout   : {best_row['f1_macro']:.4f}  (avaliado APÓS seleção)")
    print(f"    f1_binario holdout : {best_row['f1']:.4f}")
    if nome_best in ("VotingClassifier", "StackingClassifier"):
        nome_best = "RandomForest"
        print(f"  [NOTA] Melhor modelo é ensemble — usando RandomForest otimizado "
              f"para importância de features.")
    best_pipe = tuned.get(nome_best, tuned["RandomForest"])
    fi = extrair_importancia(best_pipe, feats, X_te, y_te,
                             f"{ds_label} ({best_row['modelo']})")
    if ds_label == "com_toxicidade":
        fi_com_tox   = fi
        pipe_com_tox = best_pipe
        nome_com_tox = best_row["modelo"]

# ── 6.5 Verificação de colinearidade: mes vs. trimestre ─────────────────────
sub("6.5 Colinearidade mes × trimestre — com vs. sem trimestre no conjunto de features")
print("""
  Se mes e trimestre capturam o mesmo efeito sazonal (colinearidade),
  a importância de mes deve aumentar quando trimestre é removida.
  Se não aumenta (ou cai), são efeitos distintos.

  Procedimento: retreinar o melhor modelo com_toxicidade SEM trimestre,
  rodar permutation_importance novamente, comparar importância de mes.
""")

from sklearn.base import clone

FEAT_SEM_TRI = [f for f in FEAT_COM if f != "trimestre"]
X_tr_nt = df_com[FEAT_SEM_TRI].values[:n_treino]
X_te_nt = df_com[FEAT_SEM_TRI].values[n_treino:]
y_tr_nt = df_com["alvo"].values[:n_treino]
y_te_nt = df_com["alvo"].values[n_treino:]

pipe_nt = clone(pipe_com_tox)
pipe_nt.fit(X_tr_nt, y_tr_nt)

perm_nt = permutation_importance(
    pipe_nt, X_te_nt, y_te_nt,
    n_repeats=100, scoring="f1", random_state=SEED
)
fi_nt = pd.DataFrame({
    "feature":    FEAT_SEM_TRI,
    "imp_sem_tri": perm_nt.importances_mean,
    "std_sem_tri": perm_nt.importances_std,
})
fi_nt = fi_nt.sort_values("imp_sem_tri", ascending=False).reset_index(drop=True)
fi_nt["rank_sem_tri"] = fi_nt.index + 1

# Montar tabela comparativa para mes e features de toxicidade
feats_de_interesse = ["mes", "toxicidade_t1", "toxicidade_mm3",
                       "tybyria_t1", "tybyria_mm3", "n_posts_t1", "crimes_t1"]

print(f"  {'Feature':<22} {'imp_COM_tri':>12} {'rank':>5}  │  "
      f"{'imp_SEM_tri':>12} {'rank':>5}  {'Δ imp':>9}")
print("  " + "-" * 75)

for feat in feats_de_interesse:
    # valor com trimestre (fi_com_tox)
    row_c = fi_com_tox[fi_com_tox["feature"] == feat]
    imp_c  = row_c["importance"].values[0] if len(row_c) else float("nan")
    rank_c = int(row_c["rank"].values[0])  if len(row_c) else "-"
    # valor sem trimestre (fi_nt)
    row_n  = fi_nt[fi_nt["feature"] == feat]
    imp_n  = row_n["imp_sem_tri"].values[0] if len(row_n) else float("nan")
    rank_n = int(row_n["rank_sem_tri"].values[0]) if len(row_n) else "-"
    delta  = imp_n - imp_c if not (np.isnan(imp_n) or np.isnan(imp_c)) else float("nan")
    delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "n/a"
    print(f"  {feat:<22} {imp_c:>12.4f} {str(rank_c):>5}  │  "
          f"{imp_n:>12.4f} {str(rank_n):>5}  {delta_str:>9}")

# Importância de trimestre no modelo completo
row_tri = fi_com_tox[fi_com_tox["feature"] == "trimestre"]
if len(row_tri):
    print(f"\n  trimestre (excluída)        imp={row_tri['importance'].values[0]:.4f}  "
          f"rank={int(row_tri['rank'].values[0])}")

print(f"""
  INTERPRETAÇÃO:
    Δ imp(mes) positivo e grande → mes dependia de trimestre para "dividir" o sinal
      (colinearidade confirmada: as duas capturam a mesma variância)
    Δ imp(mes) próximo de zero   → mes e trimestre são efeitos distintos
      (sazonalidade mensal e sazonal são complementares)
""")


# ═════════════════════════════════════════════════════════════════════════════
# 7. BOOTSTRAP CI DO F1 — COMPARAÇÃO SEM vs. COM TOXICIDADE
# ═════════════════════════════════════════════════════════════════════════════
section("7. BOOTSTRAP CI DO F1 (1000 reamostragens, α=0.05)")

print(f"""
  n_teste = {n_teste} semanas — muito pequeno para teste paramétrico confiável.
  Usamos bootstrap com reposição sobre as previsões no conjunto de teste.
  ATENÇÃO: bootstrap sobre n={n_teste} ainda produz ICs largos; não confundir
  largura do IC com ausência de efeito — apenas reflte o n pequeno.
""")

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

y_test_arr = df_sem["alvo"].values[n_treino:]

for ds_label in ["sem_toxicidade", "com_toxicidade"]:
    best_row = next(r for r in sorted(todos_resultados,
                                       key=lambda x: x["f1"], reverse=True)
                    if r["dataset"] == ds_label)
    scores = bootstrap_f1(y_test_arr, best_row["y_pred"])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    print(f"  {ds_label:<22}  Melhor: {best_row['modelo']}")
    print(f"    F1 observado     = {best_row['f1']:.4f}")
    print(f"    IC 95% bootstrap = [{lo:.4f}, {hi:.4f}]")
    print(f"    (largura IC = {hi-lo:.4f})\n")

# Sobreposição dos ICs
rows_sem = [r for r in todos_resultados if r["dataset"] == "sem_toxicidade"]
rows_com = [r for r in todos_resultados if r["dataset"] == "com_toxicidade"]
best_sem = max(rows_sem, key=lambda x: x["f1"])
best_com = max(rows_com, key=lambda x: x["f1"])
f1_sem = best_sem["f1"]
f1_com = best_com["f1"]

sc_sem = bootstrap_f1(y_test_arr, best_sem["y_pred"])
sc_com = bootstrap_f1(y_test_arr, best_com["y_pred"])
lo_sem, hi_sem = np.percentile(sc_sem, [2.5, 97.5])
lo_com, hi_com = np.percentile(sc_com, [2.5, 97.5])

delta = float(np.mean(sc_com) - np.mean(sc_sem))
p_superior = float(np.mean(sc_com > sc_sem))

sub("Comparação direta: com_toxicidade vs. sem_toxicidade")
print(f"""
  F1 sem_toxicidade : {f1_sem:.4f}  IC=[{lo_sem:.4f}, {hi_sem:.4f}]
  F1 com_toxicidade : {f1_com:.4f}  IC=[{lo_com:.4f}, {hi_com:.4f}]

  Δ F1 médio (bootstrap) : {delta:+.4f}
  P(F1_com > F1_sem)     : {p_superior:.3f}

  {'Os ICs SE SOBREPÕEM — NÃO é possível afirmar diferença significativa com'
   if not (lo_com > hi_sem or lo_sem > hi_com)
   else 'Os ICs NÃO se sobrepõem — há evidência de diferença com'}
  n_teste={n_teste} semanas. Qualquer conclusão deve ser qualificada pelo n pequeno.
""")

# ── 7.2 Bootstrap f1_macro — modelo selecionado por cv_best vs. Dummy_stratified
sub("7.2  Bootstrap f1_macro — modelo pré-selecionado por cv_best vs. Dummy_stratified")

def bootstrap_f1macro(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = np.array(y_true)[idx], np.array(y_pred)[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(f1_score(yt, yp, average="macro", zero_division=0))
    return np.array(scores)

# Pré-selecionar por cv_best (nenhum acesso ao holdout nesta etapa)
modelos_cv = {}
for _ds in ["sem_toxicidade", "com_toxicidade"]:
    _cands = [r for r in todos_resultados
              if r["dataset"] == _ds and r.get("cv_best") is not None]
    modelos_cv[_ds] = max(_cands, key=lambda x: x["cv_best"])

# ── DIAGNÓSTICO: confirmar que y_pred são vetores distintos ──────────────────
print("  [DIAGNÓSTICO] y_pred dos modelos selecionados por cv_best (holdout, n=31):")
print(f"  {'Dataset':<22}  {'Modelo':<28}  {'y_pred[:10]'}")
ypred_sem_cv = modelos_cv["sem_toxicidade"]["y_pred"]
ypred_com_cv = modelos_cv["com_toxicidade"]["y_pred"]
for _ds, _m in modelos_cv.items():
    print(f"  {_ds:<22}  {_m['modelo']:<28}  {_m['y_pred'][:10]}")
print(f"  y_true[:10]                                          "
      f"  {list(y_test_arr[:10])}")
n_dif = sum(a != b for a, b in zip(ypred_sem_cv, ypred_com_cv))
print(f"\n  Posições com previsões distintas entre os dois modelos: {n_dif}/{len(ypred_sem_cv)}")
print(f"  Vetores idênticos? {'SIM — PROBLEMA DE REUSO' if n_dif == 0 else 'NÃO — bootstrap calculado sobre arrays independentes'}\n")

# Nota explicativa sobre ICs que podem coincidir numericamente com n=31:
print(f"""  [NOTA sobre ICs idênticos — rodada anterior]
  Na rodada anterior MLP_otimizado (sem) e KNN_otimizado (com) produziram
  IC=[0.3870, 0.7395] idênticos. Ambos tinham acurácia=18/31 (0.5806), ou seja,
  o mesmo número de acertos — apenas em posições diferentes. Com n=31 e apenas
  ~1000 valores discretos possíveis de f1_macro via bootstrap, os percentis
  [2.5%, 97.5%] coincidiram até a 4ª casa decimal. Não havia reuso de array;
  os y_pred eram diferentes (18 acertos em posições distintas). Com os modelos
  atuais (KNN e DecisionTree) e desempenhos bem diferentes, isso não deve repetir.
""")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"  Métrica: f1_score(average='macro').  Baseline: Dummy_stratified.")
print(f"  n_bootstrap={N_BOOTSTRAP}, IC=[percentil 2.5%, 97.5%]\n")

dummy_strat_row = next(r for r in todos_resultados
                       if r["modelo"] == "Dummy_stratified")
sc_dummy_mac = bootstrap_f1macro(y_test_arr, dummy_strat_row["y_pred"])
lo_d, hi_d   = np.percentile(sc_dummy_mac, [2.5, 97.5])
print(f"  Dummy_stratified   f1_macro={dummy_strat_row['f1_macro']:.4f}  "
      f"IC=[{lo_d:.4f}, {hi_d:.4f}]\n")

results_macro = {}
for ds_label in ["sem_toxicidade", "com_toxicidade"]:
    best_real  = modelos_cv[ds_label]
    sc_real    = bootstrap_f1macro(y_test_arr, best_real["y_pred"])
    lo_r, hi_r = np.percentile(sc_real, [2.5, 97.5])
    p_vs_dummy = float(np.mean(sc_real > sc_dummy_mac))
    sobrep     = not (lo_r > hi_d or lo_d > hi_r)
    results_macro[ds_label] = {"row": best_real, "scores": sc_real,
                                "lo": lo_r, "hi": hi_r}
    print(f"  {ds_label}  →  {best_real['modelo']}  (cv_best={best_real['cv_best']:.4f})")
    print(f"    f1_macro holdout       = {best_real['f1_macro']:.4f}")
    print(f"    IC 95% bootstrap       = [{lo_r:.4f}, {hi_r:.4f}]")
    print(f"    P(real > Dummy_strat)  = {p_vs_dummy:.3f}")
    print(f"    ICs sobrepostos?       = {'SIM' if sobrep else 'NÃO'}\n")

# Verificar se ICs ainda coincidem após correção
if (results_macro["sem_toxicidade"]["lo"] == results_macro["com_toxicidade"]["lo"] and
        results_macro["sem_toxicidade"]["hi"] == results_macro["com_toxicidade"]["hi"]):
    print("  [ATENÇÃO] ICs ainda idênticos — verificar y_pred no diagnóstico acima.")

# ── 7.3 Bootstrap f1_macro — com_toxicidade vs. sem_toxicidade ───────────────
sub("7.3  Bootstrap f1_macro — com_toxicidade vs. sem_toxicidade")

sc_sem_mac  = results_macro["sem_toxicidade"]["scores"]
sc_com_mac  = results_macro["com_toxicidade"]["scores"]
lo_sem_m    = results_macro["sem_toxicidade"]["lo"]
hi_sem_m    = results_macro["sem_toxicidade"]["hi"]
lo_com_m    = results_macro["com_toxicidade"]["lo"]
hi_com_m    = results_macro["com_toxicidade"]["hi"]
best_sem_m  = results_macro["sem_toxicidade"]["row"]
best_com_m  = results_macro["com_toxicidade"]["row"]

delta_mac   = float(np.mean(sc_com_mac) - np.mean(sc_sem_mac))
p_com_mac   = float(np.mean(sc_com_mac > sc_sem_mac))
sobrep_mac  = not (lo_com_m > hi_sem_m or lo_sem_m > hi_com_m)

print(f"""
  f1_macro sem_toxicidade ({best_sem_m['modelo']:<24}) : {best_sem_m['f1_macro']:.4f}  IC=[{lo_sem_m:.4f}, {hi_sem_m:.4f}]
  f1_macro com_toxicidade ({best_com_m['modelo']:<24}) : {best_com_m['f1_macro']:.4f}  IC=[{lo_com_m:.4f}, {hi_com_m:.4f}]

  Δ f1_macro médio (bootstrap)   : {delta_mac:+.4f}
  P(f1_macro_com > f1_macro_sem) : {p_com_mac:.3f}

  {'Os ICs SE SOBREPÕEM — diferença não detectável com'
   if sobrep_mac
   else 'Os ICs NÃO se sobrepõem — evidência de diferença com'}
  n_teste={n_teste} semanas.
""")

# ── 7.4 Comparação controlada: 6 arquiteturas sem_tox vs. com_tox ────────────
sub("7.4  Comparação controlada — 6 arquiteturas: sem_toxicidade vs. com_toxicidade")

from scipy.stats import binomtest

ARQUITETURAS_74 = ["RandomForest", "XGBoost", "DecisionTree", "SVC", "KNN", "MLP"]
N_COMP_74       = len(ARQUITETURAS_74)
# Nota: bootstrap usa mesmo seed=SEED nas duas chamadas de cada par
# → mesmos índices de reamostragrem → comparação pareada válida para sc_delta

print(f"""
  Versão _otimizado de cada arquitetura (já treinadas na seção 4, sem retreino).
  Única diferença intra-par: features sem toxicidade (6) vs. com toxicidade (11).
  Bootstrap pareado: mesmos índices de reamostragem nos dois lados do par
    (ambas as chamadas usam seed={SEED} → sc_delta = sc_com - sc_sem é pareado).
  p_bruto  = proporção bootstrap two-tailed: 2 × min(P(com>sem), 1−P(com>sem))
  Bonferroni: p_bonf = min(p_bruto × {N_COMP_74}, 1.0)   →  α_adj = {0.05/N_COMP_74:.4f}
""")

rows_74 = []
for arq in ARQUITETURAS_74:
    nome = f"{arq}_otimizado"
    r_sem = next(r for r in todos_resultados
                 if r["modelo"] == nome and r["dataset"] == "sem_toxicidade")
    r_com = next(r for r in todos_resultados
                 if r["modelo"] == nome and r["dataset"] == "com_toxicidade")

    sc_sem  = bootstrap_f1macro(y_test_arr, r_sem["y_pred"])   # seed=SEED → índices A
    sc_com  = bootstrap_f1macro(y_test_arr, r_com["y_pred"])   # seed=SEED → índices A (pareado)
    sc_delt = sc_com - sc_sem

    lo_s, hi_s   = np.percentile(sc_sem,  [2.5, 97.5])
    lo_c, hi_c   = np.percentile(sc_com,  [2.5, 97.5])
    lo_d, hi_d   = np.percentile(sc_delt, [2.5, 97.5])

    delta_obs = r_com["f1_macro"] - r_sem["f1_macro"]
    p_one     = float(np.mean(sc_com > sc_sem))
    p_two     = round(2 * min(p_one, 1 - p_one), 4)
    p_bonf    = round(min(p_two * N_COMP_74, 1.0), 4)

    rows_74.append({
        "arq":         arq,
        "f1m_sem":     r_sem["f1_macro"],
        "ic_sem":      f"[{lo_s:.3f},{hi_s:.3f}]",
        "f1m_com":     r_com["f1_macro"],
        "ic_com":      f"[{lo_c:.3f},{hi_c:.3f}]",
        "delta_obs":   round(delta_obs, 4),
        "ic_delta":    f"[{lo_d:.3f},{hi_d:.3f}]",
        "delta_0":     "SIM" if lo_d <= 0 <= hi_d else "NÃO",
        "p_two":       p_two,
        "p_bonf":      p_bonf,
        "dir":         "+" if delta_obs > 0 else ("−" if delta_obs < 0 else "0"),
    })

rows_74.sort(key=lambda x: x["delta_obs"], reverse=True)

# Tabela principal
hdr = (f"  {'Arq.':<14} {'f1m_sem':>7} {'f1m_com':>7} {'Δ f1m':>8} "
       f"{'IC 95%(Δ)':>16} {'Δ⊃0?':>5} {'dir':>4} {'p_bruto':>8} {'p_bonf':>8}")
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for r in rows_74:
    print(f"  {r['arq']:<14} {r['f1m_sem']:>7.4f} {r['f1m_com']:>7.4f} "
          f"  {r['delta_obs']:>+7.4f}  {r['ic_delta']:>16} "
          f"  {r['delta_0']:>5}  {r['dir']:>3}  {r['p_two']:>7.4f}  {r['p_bonf']:>7.4f}")

# Tabela de detalhe: IC por lado
print()
hdr2 = f"  {'Arq.':<14} {'f1m_sem':>7}  {'IC 95% sem':>16}   {'f1m_com':>7}  {'IC 95% com':>16}"
print(hdr2)
print("  " + "─" * (len(hdr2) - 2))
for r in rows_74:
    print(f"  {r['arq']:<14} {r['f1m_sem']:>7.4f}  {r['ic_sem']:>16}   "
          f"{r['f1m_com']:>7.4f}  {r['ic_com']:>16}")

# Resumo de consistência direcional
n_pos = sum(1 for r in rows_74 if r["dir"] == "+")
n_neg = sum(1 for r in rows_74 if r["dir"] == "−")
n_nul = sum(1 for r in rows_74 if r["dir"] == "0")
binom_res = binomtest(n_pos, N_COMP_74, p=0.5, alternative="two-sided")

_arqs_pos  = ', '.join(r['arq'] for r in rows_74 if r['dir'] == '+')
_arqs_neg  = ', '.join(r['arq'] for r in rows_74 if r['dir'] == '−')
_nenhum_sig = all(r['p_bonf'] >= 0.05 for r in rows_74)
_sig_str    = ("SIM — nenhuma comparação significativa após correção"
               if _nenhum_sig else
               "NÃO — há comparação(ões) com p_bonf < α_adj")

sub(f"Consistência direcional — {N_COMP_74} arquiteturas")
print(f"""
  Δ positivo (com_tox > sem_tox) : {n_pos} / {N_COMP_74}  [{_arqs_pos}]
  Δ negativo (com_tox < sem_tox) : {n_neg} / {N_COMP_74}  [{_arqs_neg}]
  Δ = 0 exato                    : {n_nul} / {N_COMP_74}

  Teste binomial two-sided (H0: P(Δ>0)=0.5, n={N_COMP_74}, k={n_pos}):
    p-valor = {binom_res.pvalue:.4f}
    [baixo poder com n={N_COMP_74} — reportado formalmente, não para conclusão forte]

  Nenhum p_bonf < α_adj ({0.05/N_COMP_74:.4f})?  {_sig_str}
""")


# ═════════════════════════════════════════════════════════════════════════════
# 8. SALVAR RESULTADOS NO GCS
# ═════════════════════════════════════════════════════════════════════════════
section("8. SALVANDO RESULTADOS NO GCS")

df_save = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("cm", "y_pred", "best_params")}
    for r in todos_resultados
])
upload_csv(bucket, OUT_RESULTADOS, df_save)

print(f"""
RESUMO FINAL:
  Semanas de treino  : {n_treino}
  Semanas de teste   : {n_teste}
  Melhor sem_tox     : {best_sem['modelo']}  F1={f1_sem:.4f}
  Melhor com_tox     : {best_com['modelo']}  F1={f1_com:.4f}
  Δ F1 (com - sem)   : {f1_com - f1_sem:+.4f}
  P(com > sem) boot  : {p_superior:.3f}
  ICs sobrepostos?   : {'SIM' if not (lo_com > hi_sem or lo_sem > hi_com) else 'NÃO'}
""")
