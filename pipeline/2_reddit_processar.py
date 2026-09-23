"""
2_reddit_processar.py — Processa dumps Reddit (.zst) e extrai posts/comentarios
relevantes para o Brasil com filtro opcional por termos LGBTQIA+.

Formato dos dumps: arquivos NDJSON comprimidos em .zst (Academic Torrents).

Saida: CSV com colunas:
    id, author, created_utc, subreddit, text_original, text_clean,
    has_lgbt_term, has_hate_term

Uso:
    python 2_reddit_processar.py --input RC_2023-01.zst --output saida.csv
    python 2_reddit_processar.py --input raw/ --output processado/ --subreddits brasil,brasilivre
    python 2_reddit_processar.py --input arquivo.zst --output saida.csv --sem-filtro-lgbt

Subreddits brasileiros incluidos por padrao:
    brasil, BrasildoB, brasilivre, BeloHorizonte, MinasGerais,
    Uberlandia, juizdefora, OuroPreto, saopaulo, curitiba, riodejaneiro
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    print("ERRO: zstandard nao instalado.")
    print("Execute: python -m pip install zstandard")
    sys.exit(1)


SUBREDDITS_BR_DEFAULT = {
    "brasil", "brasildob", "brasilivre", "belohorizonte", "minasgerais",
    "uberlandia", "juizdefora", "ouropreto", "saopaulo", "curitiba", "riodejaneiro",
    "brdev", "investimentos", "futebol",
}

TERMOS_LGBT = [
    "lgbtq", "lgbt", "gay", "lesbica", "bissexual", "transgenero", "travesti",
    "queer", "homossexual", "trans ", "trans,", "trans.", "pride", "diversidade",
    "lgbtfobia", "homofobia", "transfobia", "identidade de genero", "orientacao sexual",
]

TERMOS_ODIO = [
    "viadinho", "sapatao", "bicha", "traveco", "anti-lgbt", "antilgbt",
    "normal e homem e mulher", "ideologia de genero", "kit gay",
]

CAMPOS_CSV = [
    "id", "author", "created_utc", "subreddit",
    "text_original", "text_clean", "has_lgbt_term", "has_hate_term",
]

# ── Limpeza de texto ──────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PONTUACAO_RE = re.compile(r"[^\w\s]", re.UNICODE)
_ESPACO_RE = re.compile(r"\s+")

def _remover_emojis(t: str) -> str:
    return "".join(c for c in t if not unicodedata.category(c).startswith("So"))

def limpar(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    t = _URL_RE.sub(" ", texto)
    t = _remover_emojis(t)
    t = _PONTUACAO_RE.sub(" ", t)
    t = t.lower()
    return _ESPACO_RE.sub(" ", t).strip()

# ── Processamento do arquivo .zst ─────────────────────────────────────────────

def iter_zst(filepath: str, skip_to: int = 0):
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = ""
            total = 0
            while True:
                chunk = reader.read(2 ** 20)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                linhas = buffer.split("\n")
                buffer = linhas[-1]
                for linha in linhas[:-1]:
                    total += 1
                    if total <= skip_to:
                        continue
                    linha = linha.strip()
                    if not linha:
                        continue
                    try:
                        yield json.loads(linha), total
                    except json.JSONDecodeError:
                        continue


def extrair_texto(obj: dict) -> str:
    if "body" in obj:
        return obj.get("body") or ""
    titulo = obj.get("title") or ""
    corpo = obj.get("selftext") or ""
    return f"{titulo} {corpo}".strip()


def tem_termo(texto_lower: str, termos: list) -> int:
    return int(any(t in texto_lower for t in termos))


def processar_arquivo(caminho_zst: str, caminho_csv: str,
                      subreddits: set, sem_filtro_lgbt: bool,
                      checkpoint_cada: int = 100_000):
    nome = os.path.basename(caminho_zst)
    checkpoint_path = caminho_zst + ".checkpoint"

    skip_to = 0
    modo = "w"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            skip_to = int(f.read().strip())
        modo = "a"
        print(f"  Retomando {nome} da linha {skip_to:,}")
    else:
        print(f"  Iniciando {nome}")

    encontrados = 0
    os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)

    with open(caminho_csv, modo, newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=CAMPOS_CSV)
        if modo == "w":
            writer.writeheader()

        for obj, num_linha in iter_zst(caminho_zst, skip_to):
            subreddit = (obj.get("subreddit") or "").lower()
            if subreddit not in subreddits:
                continue

            texto_orig = extrair_texto(obj)
            texto_limpo = limpar(texto_orig)

            has_lgbt = tem_termo(texto_limpo, TERMOS_LGBT)
            has_odio = tem_termo(texto_limpo, TERMOS_ODIO)

            if not sem_filtro_lgbt and not has_lgbt and not has_odio:
                continue

            encontrados += 1
            writer.writerow({
                "id": obj.get("id"),
                "author": obj.get("author"),
                "created_utc": obj.get("created_utc"),
                "subreddit": obj.get("subreddit"),
                "text_original": texto_orig[:2000],
                "text_clean": texto_limpo[:2000],
                "has_lgbt_term": has_lgbt,
                "has_hate_term": has_odio,
            })

            if num_linha % checkpoint_cada == 0:
                with open(checkpoint_path, "w") as f:
                    f.write(str(num_linha))
                print(f"    Linha {num_linha:,} | encontrados ate agora: {encontrados:,}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"  Concluido: {encontrados:,} registros -> {caminho_csv}")
    return encontrados


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Processa dumps Reddit .zst")
    parser.add_argument("--input", required=True,
                        help="Arquivo .zst ou pasta com varios .zst")
    parser.add_argument("--output", required=True,
                        help="Arquivo .csv de saida ou pasta de saida")
    parser.add_argument("--subreddits", default="",
                        help="Subreddits separados por virgula (substitui lista padrao)")
    parser.add_argument("--sem-filtro-lgbt", action="store_true",
                        help="Extrair todos os posts BR sem filtro de termos LGBTQIA+")
    args = parser.parse_args()

    subreddits = SUBREDDITS_BR_DEFAULT
    if args.subreddits:
        subreddits = set(s.strip().lower() for s in args.subreddits.split(","))
    print(f"Subreddits filtrados: {sorted(subreddits)}")

    entrada = Path(args.input)
    saida = Path(args.output)

    if entrada.is_dir():
        arquivos = sorted(entrada.glob("*.zst"))
        if not arquivos:
            print(f"Nenhum arquivo .zst encontrado em {entrada}")
            sys.exit(1)
        saida.mkdir(parents=True, exist_ok=True)
        for arq in arquivos:
            csv_out = saida / arq.stem.replace(".zst", "") + ".csv"
            processar_arquivo(str(arq), str(csv_out), subreddits, args.sem_filtro_lgbt)
    elif entrada.suffix == ".zst":
        if saida.is_dir():
            csv_out = saida / (entrada.stem + ".csv")
        else:
            csv_out = saida
        processar_arquivo(str(entrada), str(csv_out), subreddits, args.sem_filtro_lgbt)
    else:
        print(f"ERRO: --input deve ser um arquivo .zst ou uma pasta.")
        sys.exit(1)


if __name__ == "__main__":
    main()
