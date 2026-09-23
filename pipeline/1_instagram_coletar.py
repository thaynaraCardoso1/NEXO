"""
1_instagram_coletar.py — Coleta comentarios do Instagram via GraphQL API.

Requer cookies exportados do navegador (formato JSON Netscape/EditThisCookie).

Como exportar os cookies:
    1. Instale a extensao "EditThisCookie" no Chrome/Firefox
    2. Faca login no Instagram
    3. Exporte os cookies para um arquivo (ex: cookies_instagram.json)
    4. Passe o arquivo com --cookies

Saida: CSV com colunas:
    post_id, post_url, post_timestamp, post_caption,
    comment_id, comment_text, comment_timestamp, comment_author, comment_likes

Uso:
    python 1_instagram_coletar.py --perfil belohorizontemg --cookies cookies.json --saida comentarios.csv
    python 1_instagram_coletar.py --perfil minasgerais --cookies cookies.json --saida saida.csv --max-posts 50

IMPORTANTE:
    - Respeite os Termos de Servico do Instagram.
    - Nao use em producao sem avaliar os limites de rate da API.
    - Este script e apenas para fins academicos e de pesquisa.
    - Nao colete dados pessoais alem do necessario (LGPD).
"""

import argparse
import csv
import json
import os
import sys
import time
import random
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERRO: requests nao instalado.")
    print("Execute: python -m pip install requests")
    sys.exit(1)

GRAPHQL_URL = "https://www.instagram.com/graphql/query/"

QUERY_HASH_POSTS = "58b6785bea111c67129decbe6a448951"
QUERY_HASH_COMMENTS = "bc3296d1ce80a24b1b6e40b1e72903f5"

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "X-IG-App-ID": "936619743392459",
}

CAMPOS_CSV = [
    "post_id", "post_url", "post_timestamp", "post_caption",
    "comment_id", "comment_text", "comment_timestamp",
    "comment_author", "comment_likes",
]


# ── Carregamento de cookies ───────────────────────────────────────────────────

def carregar_cookies(caminho: str) -> dict:
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de cookies nao encontrado: {caminho}")
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)

    cookies = {}
    # Suporta formato lista de objetos (EditThisCookie / cookies.txt export)
    if isinstance(dados, list):
        for item in dados:
            name = item.get("name") or item.get("Name")
            value = item.get("value") or item.get("Value")
            if name and value:
                cookies[name] = value
    elif isinstance(dados, dict):
        cookies = dados

    if "sessionid" not in cookies:
        print("AVISO: 'sessionid' nao encontrado nos cookies.")
        print("  Certifique-se de estar logado no Instagram ao exportar.")
    return cookies


# ── Requisicoes GraphQL ───────────────────────────────────────────────────────

def get_user_id(username: str, session: requests.Session) -> str:
    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
    resp = session.get(url, headers={**HEADERS_BASE, "X-Requested-With": "XMLHttpRequest"})
    resp.raise_for_status()
    data = resp.json()
    user_id = data["data"]["user"]["id"]
    print(f"  User ID de @{username}: {user_id}")
    return user_id


def buscar_posts(user_id: str, session: requests.Session, max_posts: int = 50):
    posts = []
    cursor = None

    while len(posts) < max_posts:
        variaveis = {"id": user_id, "first": min(12, max_posts - len(posts))}
        if cursor:
            variaveis["after"] = cursor

        params = {
            "query_hash": QUERY_HASH_POSTS,
            "variables": json.dumps(variaveis),
        }
        resp = session.get(GRAPHQL_URL, params=params, headers=HEADERS_BASE)
        if resp.status_code != 200:
            print(f"  AVISO: status {resp.status_code} ao buscar posts. Encerrando.")
            break

        data = resp.json()
        timeline = data["data"]["user"]["edge_owner_to_timeline_media"]
        edges = timeline.get("edges", [])

        for edge in edges:
            node = edge["node"]
            posts.append({
                "id": node["id"],
                "shortcode": node["shortcode"],
                "timestamp": node.get("taken_at_timestamp"),
                "caption": (node.get("edge_media_to_caption", {})
                             .get("edges", [{}])[0]
                             .get("node", {})
                             .get("text", "")),
            })

        page_info = timeline.get("page_info", {})
        if not page_info.get("has_next_page"):
            break
        cursor = page_info.get("end_cursor")
        _aguardar()

    return posts[:max_posts]


def buscar_comentarios(shortcode: str, session: requests.Session, max_comentarios: int = 200):
    comentarios = []
    cursor = None

    while len(comentarios) < max_comentarios:
        variaveis = {"shortcode": shortcode, "first": min(20, max_comentarios - len(comentarios))}
        if cursor:
            variaveis["after"] = cursor

        params = {
            "query_hash": QUERY_HASH_COMMENTS,
            "variables": json.dumps(variaveis),
        }
        resp = session.get(GRAPHQL_URL, params=params, headers=HEADERS_BASE)
        if resp.status_code != 200:
            print(f"    AVISO: status {resp.status_code} ao buscar comentarios.")
            break

        data = resp.json()
        media = data["data"]["shortcode_media"]
        edges = media.get("edge_media_to_parent_comment", {}).get("edges", [])

        for edge in edges:
            node = edge["node"]
            comentarios.append({
                "comment_id": node["id"],
                "comment_text": node.get("text", ""),
                "comment_timestamp": node.get("created_at"),
                "comment_author": node.get("owner", {}).get("username", ""),
                "comment_likes": node.get("edge_liked_by", {}).get("count", 0),
            })

        page_info = media.get("edge_media_to_parent_comment", {}).get("page_info", {})
        if not page_info.get("has_next_page"):
            break
        cursor = page_info.get("end_cursor")
        _aguardar()

    return comentarios[:max_comentarios]


def _aguardar():
    tempo = random.uniform(2.0, 4.5)
    time.sleep(tempo)


def ts_para_iso(ts) -> str:
    if ts is None:
        return ""
    try:
        return datetime.utcfromtimestamp(int(ts)).isoformat()
    except Exception:
        return str(ts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Coleta comentarios do Instagram")
    parser.add_argument("--perfil", required=True, help="Username do perfil (sem @)")
    parser.add_argument("--cookies", required=True, help="Arquivo JSON de cookies")
    parser.add_argument("--saida", required=True, help="CSV de saida")
    parser.add_argument("--max-posts", type=int, default=30,
                        help="Maximo de posts a coletar (padrao: 30)")
    parser.add_argument("--max-comentarios", type=int, default=100,
                        help="Maximo de comentarios por post (padrao: 100)")
    args = parser.parse_args()

    print(f"Carregando cookies de {args.cookies} ...")
    cookies = carregar_cookies(args.cookies)

    session = requests.Session()
    session.cookies.update(cookies)
    session.headers.update(HEADERS_BASE)

    print(f"Buscando perfil @{args.perfil} ...")
    try:
        user_id = get_user_id(args.perfil, session)
    except Exception as e:
        print(f"ERRO ao buscar perfil: {e}")
        print("Verifique se os cookies sao validos e se o perfil existe.")
        sys.exit(1)

    print(f"Coletando ate {args.max_posts} posts ...")
    posts = buscar_posts(user_id, session, args.max_posts)
    print(f"  {len(posts)} posts encontrados.")

    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    total_comentarios = 0

    with open(args.saida, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=CAMPOS_CSV)
        writer.writeheader()

        for i, post in enumerate(posts, 1):
            shortcode = post["shortcode"]
            print(f"  [{i}/{len(posts)}] Post {shortcode} ...")
            try:
                comentarios = buscar_comentarios(shortcode, session, args.max_comentarios)
            except Exception as e:
                print(f"    AVISO: erro ao buscar comentarios: {e}")
                comentarios = []

            for c in comentarios:
                writer.writerow({
                    "post_id": post["id"],
                    "post_url": f"https://www.instagram.com/p/{shortcode}/",
                    "post_timestamp": ts_para_iso(post.get("timestamp")),
                    "post_caption": (post.get("caption") or "")[:500],
                    **c,
                    "comment_timestamp": ts_para_iso(c.get("comment_timestamp")),
                })
            total_comentarios += len(comentarios)
            print(f"    {len(comentarios)} comentarios coletados.")
            _aguardar()

    print(f"\nConcluido!")
    print(f"  Posts   : {len(posts)}")
    print(f"  Total   : {total_comentarios} comentarios")
    print(f"  Arquivo : {args.saida}")


if __name__ == "__main__":
    main()
