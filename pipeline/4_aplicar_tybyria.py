"""
4_aplicar_tybyria.py — Aplica o modelo TybyrIA (deteccao de toxicidade) a um CSV de textos.

Modelo : Veronyka/tybyria-v2.1 (HuggingFace)
Entrada: CSV com coluna de texto (padrao: text_original)
Saida  : mesmo CSV com colunas adicionadas:
            tybyria_score  (0.0 a 1.0 — probabilidade de conteudo toxico)
            tybyria_label  (0 = nao-toxico, 1 = toxico, threshold padrao 0.30)

Uso:
    python 4_aplicar_tybyria.py --input dados.csv --output dados_tybyria.csv
    python 4_aplicar_tybyria.py --input dados.csv --col-texto text_clean --threshold 0.40
    python 4_aplicar_tybyria.py --input dados.csv --output saida.csv --device cpu

Notas:
    - O modelo sera baixado automaticamente do HuggingFace (~500 MB na 1a execucao).
    - Use --device cuda se tiver GPU NVIDIA. Caso contrario usa CPU (mais lento).
    - Use --batch-size menor (ex: 8) se tiver pouca RAM/VRAM.
"""

import argparse
import sys
import os
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("ERRO: torch ou transformers nao instalados.")
    print("Execute: python -m pip install torch transformers")
    sys.exit(1)

MODEL_NAME = "Veronyka/tybyria-v2.1"
DEFAULT_THRESHOLD = 0.30
DEFAULT_BATCH = 32
MAX_LENGTH = 64


def carregar_modelo(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Dispositivo: {device}")
    print(f"Carregando modelo {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def processar(df: pd.DataFrame, col_texto: str, tokenizer, model, device,
              threshold: float, batch_size: int) -> pd.DataFrame:
    df = df.copy()
    texts = df[col_texto].fillna("").astype(str).tolist()
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="TybyrIA"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            batch_scores = probs[:, 1].cpu().numpy()
        scores.extend(batch_scores.tolist())

    df["tybyria_score"] = scores
    df["tybyria_label"] = [1 if s >= threshold else 0 for s in scores]
    return df


def main():
    parser = argparse.ArgumentParser(description="Aplica TybyrIA a um CSV de textos")
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", required=True, help="CSV de saida com colunas TybyrIA")
    parser.add_argument("--col-texto", default="text_original",
                        help="Nome da coluna de texto (padrao: text_original)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Limiar de toxicidade (padrao: {DEFAULT_THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Tamanho do lote (padrao: {DEFAULT_BATCH})")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Dispositivo de computacao (padrao: auto)")
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

    tokenizer, model, device = carregar_modelo(args.device)
    df_out = processar(df, args.col_texto, tokenizer, model, device,
                       args.threshold, args.batch_size)

    df_out.to_csv(args.output, index=False)
    n_toxicos = (df_out["tybyria_label"] == 1).sum()
    print(f"\nSalvo em: {args.output}  ({len(df_out):,} registros)")
    print(f"  Toxicos (>= {args.threshold}): {n_toxicos:,}  ({n_toxicos/len(df_out)*100:.1f}%)")
    print(f"  Score medio: {df_out['tybyria_score'].mean():.4f}")


if __name__ == "__main__":
    main()
