"""
gcs_local.py — Adaptador local: substitui google.cloud.storage para
execucao dos scripts de analise sem conexao com o GCS.

Como usar:
    1. Coloque seus arquivos CSV na pasta  analise_batch/dados/
    2. Execute qualquer script fase*.py normalmente:
           python fase2_agregacao_analise.py
    3. Os resultados sao salvos em  analise_batch/saida/

Arquivos esperados em dados/ (nomes exatos):
    instagram_comments_2023_2025_limpo_normalizado.csv
    DIS - Envolvidos - Eventos de LGBTQIAfobia - Jan 2023 a Dez 2025.csv
    DIS - Registros - Eventos de LGBTQIAfobia - Jan 2023 a Jan 2025_geocoded.csv
    reddit_tybyria_RC_total.csv
    reddit_vader_RC_total.csv
    fase2_mensal_instagram.csv          (produzido pela fase2, necessario na fase3)
    fase3_mensal_reddit_mg.csv          (produzido pela fase3, necessario na fase5)
    fase3_mensal_criminal_2023_2025.csv (produzido pela fase3, necessario na fase5)
    fase4_dataset_sem_toxicidade.csv    (produzido pela fase4)
    fase4_dataset_com_toxicidade.csv    (produzido pela fase4)
    fase4_artigo_base.csv               (produzido pela fase4b)
    fase4_artigo_midia.csv              (produzido pela fase4b)
    fase4_artigo_completo.csv           (produzido pela fase4b)
"""

from pathlib import Path

# ── Diretorios ────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent
DADOS_DIR  = _HERE / "dados"
SAIDA_DIR  = _HERE / "saida"

DADOS_DIR.mkdir(exist_ok=True)
SAIDA_DIR.mkdir(exist_ok=True)

# ── Classes que simulam google.cloud.storage ──────────────────────────────────

class _Blob:
    """Simula storage.Blob."""

    def __init__(self, name: str):
        self.name = name
        fname = Path(name).name
        # Tenta em DADOS_DIR primeiro; depois em SAIDA_DIR (arquivos intermediarios)
        if (DADOS_DIR / fname).exists():
            self._path = DADOS_DIR / fname
        elif (SAIDA_DIR / fname).exists():
            self._path = SAIDA_DIR / fname
        else:
            self._path = DADOS_DIR / fname  # path padrao para mensagem de erro
        self.size = self._path.stat().st_size if self._path.exists() else 0

    def download_as_bytes(self) -> bytes:
        if not self._path.exists():
            raise FileNotFoundError(
                f"\n[ERRO LOCAL] Arquivo nao encontrado: {self._path}\n"
                f"  Coloque o arquivo em: {DADOS_DIR.resolve()}\n"
                f"  Nome esperado: {self._path.name}\n"
            )
        return self._path.read_bytes()

    def upload_from_string(self, data, content_type: str = None):
        out = SAIDA_DIR / Path(self.name).name
        if isinstance(data, str):
            data = data.encode("utf-8")
        out.write_bytes(data)
        print(f"  [SALVO LOCAL] {out}  ({out.stat().st_size / 1024:.1f} KB)")


class _Bucket:
    """Simula storage.Bucket."""

    def blob(self, name: str) -> _Blob:
        return _Blob(name)

    def list_blobs(self, prefix: str = ""):
        """Lista arquivos em DADOS_DIR (ignora prefix — retorna todos os CSVs)."""
        return [_Blob(p.name) for p in sorted(DADOS_DIR.glob("*.csv"))]


class _Client:
    """Simula storage.Client."""

    def bucket(self, name: str) -> _Bucket:
        return _Bucket()

    @classmethod
    def from_service_account_json(cls, key_file: str) -> "_Client":
        print(f"  [MODO LOCAL] Credenciais ignoradas — lendo de {DADOS_DIR.resolve()}")
        return cls()


# ── API publica (usada pelos scripts) ─────────────────────────────────────────

class storage:
    """Namespace que imita google.cloud.storage."""
    Client = _Client


def gcs_client() -> _Client:
    """Substituto de storage.Client.from_service_account_json(KEY_FILE)."""
    print(f"  [MODO LOCAL] Lendo dados de: {DADOS_DIR.resolve()}")
    print(f"  [MODO LOCAL] Saida em:       {SAIDA_DIR.resolve()}")
    return _Client()
