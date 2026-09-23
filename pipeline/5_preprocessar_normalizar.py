"""
5_preprocessar_normalizar.py — Limpeza e normalizacao de texto para NLP.

Operacoes aplicadas:
    1. Remove URLs
    2. Remove mencoes (@usuario) e hashtags opcionalmente
    3. Remove emojis
    4. Remove pontuacao excessiva
    5. Converte para minusculas
    6. Remove espacos extras
    7. (Opcional) Normaliza score de toxicidade para escala 0-1 por z-score

Entrada: CSV com coluna de texto bruto
Saida  : mesmo CSV com coluna text_clean adicionada
         (colunas de score existentes sao preservadas)

Uso:
    python 5_preprocessar_normalizar.py --input dados.csv --output dados_limpos.csv
    python 5_preprocessar_normalizar.py --input dados.csv --col-texto body --manter-hashtags
    python 5_preprocessar_normalizar.py --input dados.csv --normalizar-score tybyria_score
"""

import argparse
import sys
import os
import re
import unicodedata

import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Funcoes de limpeza ────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENCAO_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_PONTUACAO_RE = re.compile(r"[^\w\s]", re.UNICODE)
_ESPACO_RE = re.compile(r"\s+")

def _remover_emojis(texto: str) -> str:
    return "".join(
        c for c in texto
        if not unicodedata.category(c).startswith("So")
    )

def limpar_texto(
    texto: str,
    manter_hashtags: bool = False,
    manter_mencoes: bool = False,
) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return ""
    t = _URL_RE.sub(" ", texto)
    if not manter_mencoes:
        t = _MENCAO_RE.sub(" ", t)
    if not manter_hashtags:
        t = _HASHTAG_RE.sub(" ", t)
    t = _remover_emojis(t)
    t = _PONTUACAO_RE.sub(" ", t)
    t = t.lower()
    t = _ESPACO_RE.sub(" ", t).strip()
    return t


def normalizar_score(series: pd.Series) -> pd.Series:
    """Normaliza coluna numerica para [0,1] usando min-max."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series.clip(0, 1)
    return ((series - mn) / (mx - mn)).clip(0, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Limpeza e normalizacao de texto para NLP"
    )
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", required=True, help="CSV de saida")
    parser.add_argument("--col-texto", default="text_original",
                        help="Coluna com texto bruto (padrao: text_original)")
    parser.add_argument("--col-saida", default="text_clean",
                        help="Nome da coluna de saida (padrao: text_clean)")
    parser.add_argument("--manter-hashtags", action="store_true",
                        help="Manter hashtags no texto limpo")
    parser.add_argument("--manter-mencoes", action="store_true",
                        help="Manter mencoes (@usuario) no texto limpo")
    parser.add_argument("--normalizar-score", metavar="COLUNA",
                        help="Coluna numerica de score para normalizar em [0,1]")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERRO: arquivo nao encontrado: {args.input}")
        sys.exit(1)

    print(f"Carregando {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} registros | colunas: {list(df.columns)}")

    if args.col_texto not in df.columns:
        print(f"ERRO: coluna '{args.col_texto}' nao encontrada.")
        print(f"Colunas disponíveis: {list(df.columns)}")
        sys.exit(1)

    tqdm.pandas(desc="Limpando textos")
    df[args.col_saida] = df[args.col_texto].progress_apply(
        lambda t: limpar_texto(t, args.manter_hashtags, args.manter_mencoes)
    )

    if args.normalizar_score:
        if args.normalizar_score not in df.columns:
            print(f"AVISO: coluna '{args.normalizar_score}' nao encontrada, pulando normalizacao.")
        else:
            col_norm = args.normalizar_score + "_norm"
            df[col_norm] = normalizar_score(pd.to_numeric(df[args.normalizar_score], errors="coerce"))
            print(f"  Coluna normalizada criada: {col_norm}")

    df.to_csv(args.output, index=False)
    vazios = (df[args.col_saida] == "").sum()
    print(f"Salvo em: {args.output}  ({len(df):,} registros)")
    print(f"  Textos limpos vazios: {vazios:,}  ({vazios/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
