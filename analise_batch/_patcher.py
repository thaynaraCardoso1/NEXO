"""
_patcher.py — Copia os scripts fase*.py da raiz do projeto para analise_batch/,
substituindo as dependencias do google.cloud.storage pelo adaptador local.

Execute uma vez na raiz do projeto:
    python NEXO/analise_batch/_patcher.py

Ou dentro da pasta analise_batch:
    python _patcher.py
"""

import re
import sys
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent         # analise_batch/
_RAIZ      = _HERE.parent.parent           # raiz do projeto

SCRIPTS = [
    "fase1_inspecao_dados.py",
    "fase2_agregacao_analise.py",
    "fase3_agregacao_analise.py",
    "fase4_engenharia_atributos.py",
    "fase4b_features_geo_demograficas.py",
    "fase4c_classificacao.py",
    "fase4d_classificacao_artigo.py",
    "fase5_correlacao_semanal.py",
]

# ── Substituicoes de texto ────────────────────────────────────────────────────

def _patch(src: str) -> str:
    """Aplica todas as substituicoes necessarias ao conteudo do script."""

    # 1. Troca o shebang / encoding para garantir UTF-8 no Windows
    src = src.replace(
        'sys.stdout.reconfigure(encoding="utf-8", errors="replace")',
        'try:\n    sys.stdout.reconfigure(encoding="utf-8", errors="replace")\nexcept AttributeError:\n    pass',
    )

    # 2. Remove a importacao do google.cloud.storage e adiciona o adaptador local
    #    (pode aparecer com espacos variados)
    src = re.sub(
        r"^from google\.cloud import storage\s*$",
        "from gcs_local import gcs_client, storage",
        src,
        flags=re.MULTILINE,
    )

    # 3. Remove import io (desnecessario no modo local — bytes vem dos arquivos)
    #    Mas MANTEM se usado em outro contexto — apenas remove se for orfao
    # (deixamos import io pois o BytesIO ainda e usado em read_blob_csv)

    # 4. Comenta KEY_FILE (seguranca — nunca expor credenciais)
    src = re.sub(
        r'^(KEY_FILE\s*=\s*"[^"]*")',
        r"# \1  # removido no modo local",
        src,
        flags=re.MULTILINE,
    )

    # 5. Substitui a funcao gcs_client() definida dentro de cada script
    #    pela versao do adaptador (que ja foi importada na linha 2 acima).
    #    Isso evita que a definicao local sobrescreva a importacao.
    src = re.sub(
        r"^def gcs_client\(\):\n    return storage\.Client\.from_service_account_json\(KEY_FILE\)\n",
        "# gcs_client() importada de gcs_local\n",
        src,
        flags=re.MULTILINE,
    )

    # 6. Comenta linhas que referenciam KEY_FILE apos a remocao da funcao
    src = re.sub(
        r"^(\s*return storage\.Client\.from_service_account_json\(KEY_FILE\))",
        r"    # \1  # substituido por gcs_local",
        src,
        flags=re.MULTILINE,
    )

    # 7. Adiciona nota de adaptacao no docstring (logo apos o triple-quote de abertura)
    nota = (
        "\n# ── ADAPTADO PARA EXECUCAO LOCAL ──────────────────────────────────────────────\n"
        "# Este script foi adaptado para rodar sem Google Cloud Storage.\n"
        "# Coloque os arquivos CSV em:   analise_batch/dados/\n"
        "# Os resultados sao salvos em:  analise_batch/saida/\n"
        "# Mais detalhes:                analise_batch/README_analise.md\n"
        "# ──────────────────────────────────────────────────────────────────────────────\n"
    )
    # Insere apos a primeira linha (shebang ou docstring de abertura)
    lines = src.split("\n")
    insert_at = 1
    for i, line in enumerate(lines[:10]):
        if line.strip().startswith('"""') and i > 0:
            # Encontra o fim do docstring
            end = src.find('"""', src.find('"""') + 3)
            insert_at = src[:end + 3].count("\n") + 1
            break
    lines.insert(insert_at, nota)
    src = "\n".join(lines)

    return src


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ok = 0
    erros = []

    for nome in SCRIPTS:
        origem = _RAIZ / nome
        destino = _HERE / nome

        if not origem.exists():
            print(f"  [AVISO] Nao encontrado: {origem}")
            erros.append(nome)
            continue

        src = origem.read_text(encoding="utf-8", errors="replace")
        src_patched = _patch(src)
        destino.write_text(src_patched, encoding="utf-8")
        print(f"  [OK] {nome}  ({len(src_patched):,} bytes)")
        ok += 1

    print(f"\n  Scripts copiados: {ok}/{len(SCRIPTS)}")
    if erros:
        print(f"  Nao encontrados: {erros}")
        print(f"  Certifique-se de executar a partir da raiz do projeto.")
    else:
        print("  Concluido! Configure analise_batch/dados/ e execute os scripts.")


if __name__ == "__main__":
    main()
