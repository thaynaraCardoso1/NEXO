import re
import unicodedata
import pandas

def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""

    # Remove URLs
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)

    # Remove emojis e símbolos
    texto = "".join(
        c for c in texto
        if unicodedata.category(c)[0] != "So"
    )

    # Remove pontuação
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Remove números
    texto = re.sub(r"\d+", " ", texto)

    # Lowercase
    texto = texto.lower()

    # Remove espaços duplicados
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def limpar_dataframe_resultados(df, coluna_texto='text_original'):
    """
    Remove linhas onde o texto está vazio ou é apenas espaço,
    e remove duplicados para limpar a base de análise.
    """
    # 1. Remove linhas onde a coluna de texto é NaN
    df = df.dropna(subset=[coluna_texto])
    
    # 2. Remove linhas onde o texto está vazio "" ou tem apenas espaços "  "
    df = df[df[coluna_texto].str.strip() != ""]
    
    # 3. (Opcional) Remove duplicados exatos para não enviesar a estatística
    df = df.drop_duplicates(subset=[coluna_texto])
    
    return df