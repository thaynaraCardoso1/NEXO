#!/usr/bin/env python3
"""
Fase 1 — Inspeção inicial dos dados: Instagram (GCS) + Registros Criminais SEJUSP/MG.

Objetivo: carregar, concatenar e diagnosticar os datasets ANTES de qualquer agregação.
NÃO realiza análise estatística — apenas diagnóstico de qualidade e estrutura.

Execute:
    python fase1_inspecao_dados.py
"""

# ── ADAPTADO PARA EXECUCAO LOCAL ──────────────────────────────────────────────
# Este script foi adaptado para rodar sem Google Cloud Storage.
# Coloque os arquivos CSV em:   analise_batch/dados/
# Os resultados sao salvos em:  analise_batch/saida/
# Mais detalhes:                analise_batch/README_analise.md
# ──────────────────────────────────────────────────────────────────────────────


import io
import sys
from pathlib import Path

import pandas as pd
from gcs_local import gcs_client, storage
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ── Configuração ──────────────────────────────────────────────────────────────
# KEY_FILE      = "lgbtminas-22e9f3503589.json"  # removido no modo local
BUCKET        = "lgbtminas-dados"
INSTA_PREFIX  = "rede social/instagram/processed/"
INSTA_SUFFIX  = "_comments_2025_processed.csv"
CRIME_BLOB    = (
    "criminal/processed/"
    "criminal_processed_DIS - Registros - Eventos de LGBTQIAfobia - Jan 2025_geocoded.csv"
)
OUTPUT_BLOB   = "rede social/instagram/processed/compilado_comments_2025.csv"

EXPECTED_COLS = {
    "comment_id", "post_id", "comment_text", "comment_created_at",
    "comment_author_username", "parent_comment_id", "is_reply",
    "tybyria_score", "tybyria_label", "vader_compound",
    "vader_pos", "vader_neu", "vader_neg",
}

SEP  = "=" * 70
SEP2 = "-" * 70


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def subsection(title: str) -> None:
    print(f"\n{SEP2}\n{title}\n{SEP2}")


# gcs_client() importada de gcs_local


def read_blob_csv(bucket: storage.Bucket, blob_name: str, **kwargs) -> pd.DataFrame:
    data = bucket.blob(blob_name).download_as_bytes()
    return pd.read_csv(io.BytesIO(data), **kwargs)


def inspect_df(df: pd.DataFrame, label: str) -> None:
    print(f"\nShape : {df.shape}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")
    print(f"\nNulos por coluna:\n{df.isnull().sum().to_string()}")
    print(f"\nPrimeiras 5 linhas:\n{df.head().to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LISTAR ARQUIVOS DE INSTAGRAM
# ─────────────────────────────────────────────────────────────────────────────
section("1. LISTANDO ARQUIVOS DE INSTAGRAM NO GCS")

client = gcs_client()
bucket = client.bucket(BUCKET)

all_blobs = list(bucket.list_blobs(prefix=INSTA_PREFIX))
comment_blobs = [
    b for b in all_blobs
    if b.name.endswith(INSTA_SUFFIX) and "compilado" not in b.name
]

if not comment_blobs:
    print(f"[FATAL] Nenhum arquivo encontrado com sufixo '{INSTA_SUFFIX}' em {INSTA_PREFIX}")
    sys.exit(1)

print(f"\nArquivos encontrados: {len(comment_blobs)}")
for b in comment_blobs:
    size_kb = b.size / 1024
    print(f"  {b.name}  ({size_kb:,.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARREGAR E CONCATENAR
# ─────────────────────────────────────────────────────────────────────────────
section("2. CARREGANDO E CONCATENANDO PERFIS")

frames = []
per_profile_report = {}   # perfil -> lista de observações

for blob in comment_blobs:
    stem   = Path(blob.name).stem                          # ex: belohorizontemg_comments_2025_processed
    perfil = stem.replace("_comments_2025_processed", "")  # ex: belohorizontemg

    try:
        df = read_blob_csv(bucket, blob.name, low_memory=False)
        df["perfil"] = perfil
        frames.append(df)

        notes = []

        # Colunas ausentes em relação ao schema esperado
        missing_cols = EXPECTED_COLS - set(df.columns)
        if missing_cols:
            notes.append(f"[AVISO] Colunas ausentes: {sorted(missing_cols)}")

        # Colunas extras não esperadas
        extra_cols = set(df.columns) - EXPECTED_COLS - {"perfil"}
        if extra_cols:
            notes.append(f"[INFO] Colunas extras: {sorted(extra_cols)}")

        # Amostras de data para detecção de formato
        if "comment_created_at" in df.columns:
            samples = df["comment_created_at"].dropna().head(3).tolist()
            notes.append(f"[DATA] comment_created_at dtype={df['comment_created_at'].dtype} | amostras={samples}")
        else:
            notes.append("[AVISO] Coluna comment_created_at AUSENTE")

        # Nulos em colunas críticas de NLP
        for col in ["tybyria_score", "tybyria_label", "vader_compound"]:
            if col in df.columns:
                n_null = int(df[col].isnull().sum())
                pct    = n_null / len(df) * 100
                tag    = "[AVISO]" if pct > 5 else "[OK]"
                notes.append(f"{tag} {col}: {n_null} nulos ({pct:.2f}%)")

        per_profile_report[perfil] = {"status": "OK", "rows": len(df), "cols": len(df.columns), "notes": notes}
        print(f"\n  [OK] {perfil}: {len(df):,} linhas | {len(df.columns)} colunas")

    except Exception as exc:
        per_profile_report[perfil] = {"status": "ERRO", "rows": 0, "cols": 0, "notes": [str(exc)]}
        print(f"\n  [ERRO] {perfil}: {exc}")

if not frames:
    print("\n[FATAL] Nenhum DataFrame carregado com sucesso. Abortando.")
    sys.exit(1)

df_insta = pd.concat(frames, ignore_index=True)
print(f"\nDataFrame consolidado: {df_insta.shape}")
print(f"Perfis: {sorted(df_insta['perfil'].unique())}")
print(f"\nRegistros por perfil:\n{df_insta['perfil'].value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARREGAR REGISTROS CRIMINAIS
# ─────────────────────────────────────────────────────────────────────────────
section("3. CARREGANDO REGISTROS CRIMINAIS (SEJUSP/MG)")
print(f"\nBlob: gs://{BUCKET}/{CRIME_BLOB}")

df_crime = pd.DataFrame()
try:
    df_crime = read_blob_csv(
        bucket, CRIME_BLOB,
        encoding="iso-8859-1", sep=";", on_bad_lines="skip", low_memory=False,
    )
    # Remove soft-hyphens e espaços extras dos nomes de coluna (problema comum em exports do REDS)
    df_crime.columns = [c.replace("\xad", "").strip() for c in df_crime.columns]
    print(f"\n[OK] Registros criminais: {df_crime.shape}")
except Exception as exc:
    print(f"\n[ERRO] Não foi possível carregar o CSV criminal: {exc}")
    print("       Verifique o caminho exato no GCS e tente novamente.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. INSPEÇÃO DETALHADA
# ─────────────────────────────────────────────────────────────────────────────
section("4A. INSPEÇÃO — INSTAGRAM CONSOLIDADO")
inspect_df(df_insta, "instagram")

if not df_crime.empty:
    section("4B. INSPEÇÃO — REGISTROS CRIMINAIS")
    inspect_df(df_crime, "crime")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DIAGNÓSTICO DE INCONSISTÊNCIAS
# ─────────────────────────────────────────────────────────────────────────────
section("5. DIAGNÓSTICO DE INCONSISTÊNCIAS")

# 5.1 Relatório por perfil
subsection("5.1 Relatório por perfil")
for perfil, report in per_profile_report.items():
    status = report["status"]
    print(f"\n  [{status}] {perfil}  ({report['rows']:,} linhas, {report['cols']} colunas)")
    for note in report["notes"]:
        print(f"        {note}")

# 5.2 Duplicatas de comment_id
subsection("5.2 Duplicatas de comment_id")
if "comment_id" in df_insta.columns:
    n_total    = len(df_insta)
    n_dup      = int(df_insta.duplicated(subset="comment_id").sum())
    n_dup_keep = int(df_insta.duplicated(subset="comment_id", keep=False).sum())
    print(f"\n  Total de linhas: {n_total:,}")
    print(f"  Duplicatas (excluindo 1ª ocorrência): {n_dup}")
    print(f"  Linhas envolvidas em duplicação: {n_dup_keep}")

    if n_dup > 0:
        dup_df = df_insta[df_insta.duplicated(subset="comment_id", keep=False)]

        # Duplicatas dentro do mesmo perfil
        dup_by_perfil = dup_df.groupby("perfil")["comment_id"].count()
        print(f"\n  Duplicatas por perfil:\n{dup_by_perfil.to_string()}")

        # Duplicatas cruzadas entre perfis (comment_id aparece em mais de 1 perfil)
        cross = dup_df.groupby("comment_id")["perfil"].nunique()
        cross_multi = cross[cross > 1]
        if len(cross_multi) > 0:
            print(f"\n  [AVISO] {len(cross_multi)} comment_id(s) aparecem em múltiplos perfis.")
            print(f"  Exemplos:\n{cross_multi.head(10).to_string()}")
        else:
            print("\n  [OK] Nenhuma duplicata cruzada entre perfis.")
else:
    print("\n  [AVISO] Coluna comment_id não encontrada — não foi possível checar duplicatas.")

# 5.3 Análise de formato de datas — Instagram
subsection("5.3 Formatos de data — comment_created_at (por perfil)")
if "comment_created_at" in df_insta.columns:
    print(f"\n  dtype consolidado: {df_insta['comment_created_at'].dtype}")
    print(f"\n  Amostra de 10 valores não-nulos:")
    print(f"  {df_insta['comment_created_at'].dropna().head(10).tolist()}")

    print(f"\n  Amostras por perfil:")
    for perfil in sorted(df_insta["perfil"].unique()):
        mask    = df_insta["perfil"] == perfil
        samples = df_insta.loc[mask, "comment_created_at"].dropna().head(3).tolist()
        n_null  = int(df_insta.loc[mask, "comment_created_at"].isnull().sum())
        print(f"    [{perfil}]  nulos={n_null}  amostras={samples}")

    # Tenta inferir se já é datetime ou string
    try:
        parsed = pd.to_datetime(df_insta["comment_created_at"], errors="coerce", utc=True)
        n_ok   = int(parsed.notna().sum())
        n_fail = int(parsed.isna().sum())
        print(f"\n  Parsing automático (pd.to_datetime, utc=True):")
        print(f"    Parseados OK: {n_ok:,} | Falhou: {n_fail:,}")
        if n_ok > 0:
            print(f"    Mínimo: {parsed.min()} | Máximo: {parsed.max()}")
    except Exception as exc:
        print(f"\n  [AVISO] pd.to_datetime falhou: {exc}")
else:
    print("\n  [AVISO] Coluna comment_created_at não encontrada no consolidado.")

# 5.4 Análise de formato de datas — Registros criminais
subsection("5.4 Formatos de data — Registros Criminais")
if not df_crime.empty:
    date_cols = [
        c for c in df_crime.columns
        if any(kw in c.lower() for kw in ("data", "hora", "ocorr", "data/hora"))
    ]
    print(f"\n  Colunas com 'data'/'hora'/'ocorr' no nome: {date_cols}")
    for col in date_cols[:4]:
        dtype   = df_crime[col].dtype
        samples = df_crime[col].dropna().head(5).tolist()
        n_null  = int(df_crime[col].isnull().sum())
        print(f"\n  [{col}]")
        print(f"    dtype  : {dtype}")
        print(f"    nulos  : {n_null}")
        print(f"    amostras: {samples}")
        # Tenta parsear com o formato conhecido do REDS
        try:
            parsed = pd.to_datetime(df_crime[col], format="%d/%m/%Y %H:%M", errors="coerce")
            n_ok   = int(parsed.notna().sum())
            n_fail = int(parsed.isna().sum())
            print(f"    parse '%d/%m/%Y %H:%M' → OK: {n_ok:,} | Falhou: {n_fail:,}")
            if n_ok > 0:
                print(f"    Intervalo: {parsed.min()} → {parsed.max()}")
        except Exception as exc:
            print(f"    parse falhou: {exc}")
else:
    print("\n  [AVISO] df_crime vazio — verifique erros na seção 3.")

# 5.5 Nulos críticos — NLP
subsection("5.5 Nulos em colunas de NLP (Instagram consolidado)")
nlp_cols = ["tybyria_score", "tybyria_label", "vader_compound", "vader_pos", "vader_neu", "vader_neg"]
for col in nlp_cols:
    if col in df_insta.columns:
        n_null = int(df_insta[col].isnull().sum())
        pct    = n_null / len(df_insta) * 100
        tag    = "[AVISO]" if pct > 5 else "[OK]"
        print(f"  {tag} {col}: {n_null:,} nulos ({pct:.2f}%)")
    else:
        print(f"  [AVISO] {col}: coluna não encontrada")

# 5.6 Estatísticas descritivas básicas das colunas NLP (só range/distribuição, sem análise)
subsection("5.6 Range dos scores NLP (verificação de sanidade)")
for col in ["tybyria_score", "vader_compound"]:
    if col in df_insta.columns:
        s = df_insta[col].dropna()
        print(f"\n  {col}:")
        print(f"    min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}  median={s.median():.4f}")
        # Verifica se valores estão dentro dos ranges esperados
        if col == "tybyria_score":
            out_of_range = int(((s < 0) | (s > 1)).sum())
            print(f"    Fora do range [0,1]: {out_of_range}")
        elif col == "vader_compound":
            out_of_range = int(((s < -1) | (s > 1)).sum())
            print(f"    Fora do range [-1,1]: {out_of_range}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SALVAR CONSOLIDADO NO GCS
# ─────────────────────────────────────────────────────────────────────────────
section("6. SALVANDO CONSOLIDADO NO GCS")

csv_bytes = df_insta.to_csv(index=False).encode("utf-8")
out_blob  = bucket.blob(OUTPUT_BLOB)
out_blob.upload_from_string(csv_bytes, content_type="text/csv")
size_mb = len(csv_bytes) / 1_048_576
print(f"\n[OK] gs://{BUCKET}/{OUTPUT_BLOB}")
print(f"     {size_mb:.2f} MB | {len(df_insta):,} linhas | {df_insta.shape[1]} colunas")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMÁRIO FINAL
# ─────────────────────────────────────────────────────────────────────────────
section("7. SUMÁRIO — AGUARDANDO CONFIRMAÇÃO ANTES DA FASE 2")

n_profiles  = df_insta["perfil"].nunique()
n_comments  = len(df_insta)
n_crime     = len(df_crime) if not df_crime.empty else "N/A (erro de carga)"
n_dup_total = int(df_insta.duplicated(subset="comment_id").sum()) if "comment_id" in df_insta.columns else "N/A"
nlp_null    = int(df_insta["tybyria_score"].isnull().sum()) if "tybyria_score" in df_insta.columns else "N/A"

print(f"""
  Perfis carregados       : {n_profiles}
  Total de comentários    : {n_comments:,}
  Total de registros crime: {n_crime}
  Duplicatas comment_id   : {n_dup_total}
  Nulos em tybyria_score  : {nlp_null}
  Consolidado salvo em    : gs://{BUCKET}/{OUTPUT_BLOB}

ANTES DE PROSSEGUIR PARA A FASE 2, confirme:
  1. O formato de comment_created_at está correto? (ISO 8601 / Unix timestamp / outro?)
  2. A coluna de data nos registros criminais está correta? (esperado: 'Data/Hora Ocorrência')
  3. As duplicatas de comment_id devem ser removidas ou são registros legítimos?
  4. Os nulos em tybyria_score devem ser excluídos ou imputados?
""")
