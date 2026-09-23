"""
3_aplicar_vader.py — Aplica analise de sentimento VADER a um CSV de textos.

Entrada : CSV com coluna de texto (padrao: text_original)
Saida   : mesmo CSV com colunas adicionadas:
            vader_compound  (-1.0 a +1.0)
            vader_pos       (0.0 a 1.0)
            vader_neg       (0.0 a 1.0)
            vader_neu       (0.0 a 1.0)
            vader_label     (positive / negative / neutral)

Uso:
    python 3_aplicar_vader.py --input dados.csv --output dados_vader.csv
    python 3_aplicar_vader.py --input dados.csv --col-texto text_clean
"""

import argparse
import sys
import os
import pandas as pd
from tqdm import tqdm

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("ERRO: vaderSentiment nao instalado.")
    print("Execute: python -m pip install vaderSentiment")
    sys.exit(1)


def classificar_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def processar(df: pd.DataFrame, col_texto: str, batch_size: int = 512) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()

    compostos, pos_list, neg_list, neu_list, labels = [], [], [], [], []

    texts = df[col_texto].fillna("").astype(str).tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="VADER"):
        for text in texts[i : i + batch_size]:
            sc = analyzer.polarity_scores(text)
            compostos.append(sc["compound"])
            pos_list.append(sc["pos"])
            neg_list.append(sc["neg"])
            neu_list.append(sc["neu"])
            labels.append(classificar_label(sc["compound"]))

    df["vader_compound"] = compostos
    df["vader_pos"] = pos_list
    df["vader_neg"] = neg_list
    df["vader_neu"] = neu_list
    df["vader_label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description="Aplica VADER a um CSV de textos")
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", required=True, help="CSV de saida com colunas VADER")
    parser.add_argument("--col-texto", default="text_original",
                        help="Nome da coluna de texto (padrao: text_original)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Tamanho do lote de processamento (padrao: 512)")
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

    df_out = processar(df, args.col_texto, args.batch_size)

    df_out.to_csv(args.output, index=False)
    print(f"Salvo em: {args.output}  ({len(df_out):,} registros)")
    print(f"  Positivo: {(df_out['vader_label'] == 'positive').sum():,}")
    print(f"  Negativo: {(df_out['vader_label'] == 'negative').sum():,}")
    print(f"  Neutro  : {(df_out['vader_label'] == 'neutral').sum():,}")


if __name__ == "__main__":
    main()
