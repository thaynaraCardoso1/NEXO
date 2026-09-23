# Pipeline de Coleta e NLP — NEXO

Este diretorio contem os scripts do pipeline de dados que produz os CSVs
utilizados pela ferramenta `app_correlacao_academica.py`.

---

## Ordem de execucao

```
1_instagram_coletar.py        ← coleta comentarios do Instagram
2_reddit_processar.py         ← extrai e filtra dumps .zst do Reddit
       |
       v
5_preprocessar_normalizar.py  ← limpeza e normalizacao de texto
       |
       v
3_aplicar_vader.py            ← sentimento VADER (vader_compound)
4_aplicar_tybyria.py          ← toxicidade TybyrIA (tybyria_score)
       |
       v
app_correlacao_academica.py   ← analise de correlacao temporal
```

---

## 1. Coletar comentarios do Instagram

```bash
python 1_instagram_coletar.py \
  --perfil belohorizontemg \
  --cookies cookies_instagram.json \
  --saida instagram_raw.csv \
  --max-posts 50 \
  --max-comentarios 200
```

**Prerequisitos:**
- Estar logado no Instagram no navegador
- Exportar cookies com a extensao EditThisCookie (formato JSON)
- `python -m pip install requests`

**Saida:** `instagram_raw.csv` com colunas:
`post_id, post_url, post_timestamp, post_caption, comment_id, comment_text, comment_timestamp, comment_author, comment_likes`

**AVISO:** Respeite os Termos de Servico do Instagram e a LGPD.

---

## 2. Processar dumps Reddit (.zst)

```bash
# Um arquivo:
python 2_reddit_processar.py \
  --input RC_2023-01.zst \
  --output reddit_jan2023.csv

# Pasta com varios arquivos:
python 2_reddit_processar.py \
  --input pasta_dumps/ \
  --output pasta_saida/ \
  --subreddits brasil,brasilivre,BeloHorizonte

# Sem filtro LGBTQIA+ (todos os posts BR):
python 2_reddit_processar.py \
  --input RC_2023-01.zst \
  --output saida.csv \
  --sem-filtro-lgbt
```

**Prerequisitos:**
- Dumps .zst obtidos no Academic Torrents
- `python -m pip install zstandard`

**Saida:** CSV com colunas:
`id, author, created_utc, subreddit, text_original, text_clean, has_lgbt_term, has_hate_term`

---

## 3. Preprocessar e normalizar texto

Execute **antes** dos modelos NLP para melhorar a qualidade:

```bash
python 5_preprocessar_normalizar.py \
  --input instagram_raw.csv \
  --output instagram_limpo.csv \
  --col-texto comment_text

# Com normalizacao de score existente:
python 5_preprocessar_normalizar.py \
  --input dados_vader.csv \
  --output dados_norm.csv \
  --normalizar-score vader_compound
```

**Opcoes:**
- `--manter-hashtags` — preserva hashtags (#tag)
- `--manter-mencoes` — preserva @mencoes
- `--normalizar-score COLUNA` — normaliza coluna numerica para [0,1]

---

## 4. Aplicar VADER (sentimento)

```bash
python 3_aplicar_vader.py \
  --input instagram_limpo.csv \
  --output instagram_vader.csv \
  --col-texto text_clean
```

**Prerequisitos:** `python -m pip install vaderSentiment`

**Saida:** adiciona colunas:
`vader_compound (-1 a +1), vader_pos, vader_neg, vader_neu, vader_label`

**Nota:** VADER foi originalmente desenvolvido para ingles. Funciona parcialmente
em portugues mas com menor precisao. Prefira TybyrIA para textos em portugues.

---

## 5. Aplicar TybyrIA (toxicidade em portugues)

```bash
python 4_aplicar_tybyria.py \
  --input instagram_limpo.csv \
  --output instagram_tybyria.csv \
  --col-texto text_clean \
  --threshold 0.30

# Com GPU NVIDIA:
python 4_aplicar_tybyria.py --input dados.csv --output saida.csv --device cuda

# Batch menor se tiver pouca memoria:
python 4_aplicar_tybyria.py --input dados.csv --output saida.csv --batch-size 8
```

**Prerequisitos:** `python -m pip install torch transformers tqdm`
- O modelo (~500 MB) e baixado automaticamente do HuggingFace na 1a execucao.
- Requer conexao com a internet na primeira vez.

**Saida:** adiciona colunas:
`tybyria_score (0.0 a 1.0), tybyria_label (0/1)`

---

## Arquivo final para a ferramenta NEXO

O CSV que voce vai carregar na ferramenta deve ter **no minimo**:

| Coluna               | Tipo      | Descricao                               |
|----------------------|-----------|-----------------------------------------|
| `created_at` ou `data` | data     | Data da publicacao                      |
| `text_clean`         | texto     | Texto preprocessado (opcional, mas recomendado) |
| `tybyria_score` ou `vader_compound` | float | Score de polaridade/toxicidade |

Para ativar o mapa no arquivo de crimes, inclua tambem `latitude` e `longitude`.

---

## Dicas

- Para datasets grandes (>100k registros), TybyrIA pode ser lento sem GPU.
  Considere processar em lotes menores ou usar uma VM com GPU.
- Sempre aplique `5_preprocessar_normalizar.py` antes dos modelos NLP.
- O checkpoint automatico do Reddit permite retomar processamentos interrompidos.
- Para Instagram, respeite os limites de taxa — o script inclui delays aleatorios.
