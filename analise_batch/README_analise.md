# Pipeline de Analise — NEXO / analise_batch

Scripts de analise estatistica adaptados para execucao local (sem Google Cloud Storage).

Metodologia baseada em CRISP-DM, replicando a abordagem de
"Real-time crime prediction using social media" (Jimoh, 2023).

---

## Dependencias

```bash
python -m pip install pandas numpy scipy statsmodels scikit-learn xgboost shap tqdm
```

---

## Estrutura de pastas

```
analise_batch/
├── gcs_local.py                   ← adaptador local (substitui google.cloud.storage)
├── dados/                         ← coloque aqui os arquivos CSV de entrada
├── saida/                         ← resultados sao salvos aqui automaticamente
├── fase1_inspecao_dados.py
├── fase2_agregacao_analise.py
├── fase3_agregacao_analise.py
├── fase4_engenharia_atributos.py
├── fase4b_features_geo_demograficas.py
├── fase4c_classificacao.py
├── fase4d_classificacao_artigo.py
└── fase5_correlacao_semanal.py
```

---

## Arquivos necessarios em dados/

Coloque os arquivos com os nomes EXATOS abaixo:

| Arquivo                                                                    | Usado em        |
|----------------------------------------------------------------------------|-----------------|
| `instagram_comments_2023_2025_limpo_normalizado.csv`                       | fase1, fase2    |
| `DIS - Envolvidos - Eventos de LGBTQIAfobia - Jan 2023 a Dez 2025.csv`    | fase2, fase3, fase4, fase4b |
| `DIS - Registros - Eventos de LGBTQIAfobia - Jan 2023 a Jan 2025_geocoded.csv` | fase4b     |
| `reddit_tybyria_RC_total.csv`                                              | fase3, fase4, fase4b |
| `reddit_vader_RC_total.csv`                                                | fase3 (opcional)|

---

## Ordem de execucao

```
fase1  →  fase2  →  fase3  →  fase4  →  fase4b  →  fase4c  →  fase4d  →  fase5
```

Cada fase le os CSVs de `dados/` e grava resultados em `saida/`.
Os scripts seguintes leem de `dados/` OU `saida/` (o adaptador verifica os dois).

```bash
# Execute cada script na pasta analise_batch/
cd NEXO/analise_batch

python fase1_inspecao_dados.py
python fase2_agregacao_analise.py
python fase3_agregacao_analise.py
python fase4_engenharia_atributos.py
python fase4b_features_geo_demograficas.py
python fase4c_classificacao.py
python fase4d_classificacao_artigo.py
python fase5_correlacao_semanal.py
```

---

## O que cada fase faz

### Fase 1 — Inspecao inicial
- Carrega e concatena os dados brutos do Instagram e dos registros criminais SEJUSP/MG
- Diagnostica qualidade: nulos, duplicatas, formato de datas, range dos scores NLP
- **Nao realiza analise estatistica** — apenas diagnostico

### Fase 2 — Agregacao mensal + analise (Instagram × Crimes)
- Agrega comentarios do Instagram e crimes por mes (2023-2025)
- Calcula: Pearson, Spearman, IC 95% (Fisher), ADF + Durbin-Watson, CCF com p-valores
- n = 12 meses (Instagram cobre 2025); base para comparacao

### Fase 3 — Reddit MG × Crimes (n=36)
- Expande analise para Reddit (2023-2025) com n=36 meses
- Adiciona KPSS ao diagnostico de estacionariedade
- **Teste de Granger** (max_lag=4) com correcao de Bonferroni
- Sensibilidade: BeloHorizonte isolado vs. MG agregado
- Comparacao plataformas: Reddit vs. Instagram

### Fase 4 — Engenharia de atributos (ML)
- Produz datasets semanais com features autoregressivas + toxicidade
- Alvo: alto/baixo relativo a mediana de crimes (classificacao binaria)
- Sem leakage: features de toxicidade sempre em T-1

### Fase 4B — Features geograficas + demograficas + Instagram
- Replica metodologia do artigo (Jimoh, 2023):
  - lat/lon dos crimes (centroide e dispersao espacial) em T-1
  - etnia/genero dos envolvidos (REDS) em T-1
  - toxicidade Instagram (TybyrIA + VADER) em T-1
- Gera 3 datasets: BASE, MIDIA, COMPLETO

### Fase 4C — Classificacao supervisionada
- 6 classificadores: RandomForest, XGBoost, DecisionTree, SVC, KNN, MLP
- Validacao temporal (TimeSeriesSplit 5-folds + holdout 20%)
- Feature importance (permutation + SHAP)

### Fase 4D — Replicacao do artigo
- Mesmos 6 classificadores em 3 datasets (BASE, MIDIA, COMPLETO)
- Tabela comparativa: contribuicao da polarity vs. variaveis geograficas e demograficas

### Fase 5 — Correlacao semanal (analise completa)
- Series semanais (n ~ 155 semanas)
- Pearson e Spearman em lags 0–8 semanas com Bonferroni
- **CCF** com IC 95% (statsmodels)
- **Causalidade de Granger** (max_lag=6) com Bonferroni
- **Correlacao parcial** via Frisch-Waugh (OLS residuais)
- Tabela resumo executiva para escrita academica

---

## Como o adaptador local funciona

O arquivo `gcs_local.py` substitui `google.cloud.storage` transparentemente:
- `read_blob_csv(bucket, nome_blob, ...)` le o arquivo de `dados/<nome_do_arquivo>`
- `upload_csv(bucket, nome_blob, df)` salva o CSV em `saida/<nome_do_arquivo>`
- O adaptador procura em `dados/` primeiro, depois em `saida/` (para arquivos intermediarios)

Voce **nao precisa modificar nenhuma linha** dos scripts — o adaptador e carregado automaticamente.

---

## Notas metodologicas

- Correlacao nao implica causalidade — todos os resultados sao exploratórios
- ADF + KPSS devem concordar antes de diferenciar a serie
- Bonferroni e aplicado em todas as baterias de multiplos testes
- TybyrIA (Veronyka/tybyria-v2.1) foi treinado em portugues generico — validar para dominio LGBTQIA+
- VADER foi originalmente desenvolvido para ingles — resultados parciais em portugues
- Dados criminais SEJUSP/MG podem ter subnotificacao desconhecida (vies de captura)
