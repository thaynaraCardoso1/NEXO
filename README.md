# NEXO — Nucleo de Exploracao de Relacoes Temporais

Ferramenta academica para analise de correlacao temporal entre
registros criminais e dados de redes sociais.

Desenvolvida no contexto da pesquisa sobre discurso de odio online
e violencia LGBTfobica no Brasil.

---

## Estrutura da pasta

```
NEXO/
├── app_correlacao_academica.py   ← ferramenta principal (Streamlit)
├── gerador_dados_sinteticos.py   ← gera CSVs de teste controlados
├── requirements.txt              ← dependencias Python
├── README.md                     ← este arquivo
├── exemplos/                     ← dados de demonstracao prontos para uso
│   ├── crimes_exemplo.csv        ← registros criminais de exemplo
│   ├── midia_alta_correlacao.csv ← Cenario A: correlacao forte (r≈0.97)
│   ├── midia_sem_correlacao.csv  ← Cenario B: sem correlacao  (r≈0.20)
│   └── midia_correlacao_parcial.csv ← Cenario C: correlacao parcial
└── pipeline/                     ← scripts para coletar e processar dados reais
    ├── README_pipeline.md        ← guia de uso do pipeline
    ├── 1_instagram_coletar.py    ← coleta comentarios do Instagram
    ├── 2_reddit_processar.py     ← processa dumps .zst do Reddit
    ├── 3_aplicar_vader.py        ← analise de sentimento VADER
    ├── 4_aplicar_tybyria.py      ← deteccao de toxicidade (portugues)
    ├── 5_preprocessar_normalizar.py ← limpeza e normalizacao de texto
    └── utils/
        └── limpeza.py            ← funcoes utilitarias de limpeza de texto
```

---

## Instalacao rapida

```bash
# 1. Instale as dependencias:
python -m pip install -r requirements.txt

# 2. Abra a ferramenta:
python -m streamlit run app_correlacao_academica.py
```

A ferramenta abre automaticamente no navegador em `http://localhost:8501`.

---

## Como usar a ferramenta

### Passo 1 — Carregar os dados

Na barra lateral esquerda:

1. **Arquivo Criminal** — carregue um CSV com registros de crimes.
   - Colunas minimas: uma coluna de **data** + uma coluna numerica qualquer.
   - Opcional: colunas de **latitude** e **longitude** (para o mapa).

2. **Arquivo de Redes Sociais** — carregue um CSV com publicacoes ou comentarios.
   - Colunas minimas: uma coluna de **data** + uma coluna de **score de polaridade**.
   - Opcional: coluna de **texto** (para os snippets coloridos).

### Passo 2 — Mapear colunas

Selecione quais colunas correspondem a cada campo. A ferramenta detecta
automaticamente as colunas mais provaveis.

### Passo 3 — Configurar e analisar

Ajuste a granularidade temporal (semanal/mensal), o lag maximo e clique em
**Rodar Analise**. Os resultados aparecem nas abas:

| Aba | Conteudo |
|-----|----------|
| Serie Temporal | grafico sobreposto crimes vs. polaridade |
| Textos & Picos | snippets mais toxicos e mais neutros |
| Calendario | heatmap de toxicidade por dia/semana |
| Mapa | distribuicao geografica dos crimes |
| Estacionariedade | testes ADF + KPSS |
| Correlacao por Lag | Pearson e Spearman com Bonferroni |
| CCF | funcao de correlacao cruzada com IC 95% |
| Granger | teste de causalidade de Granger |
| Correlacao Parcial | correlacao residual (Frisch-Waugh) |
| Interpretacao | resumo em linguagem simples |
| Manual | guia de uso |

---

## Teste rapido com dados de exemplo

1. Abra a ferramenta com `python -m streamlit run app_correlacao_academica.py`
2. Carregue `exemplos/crimes_exemplo.csv` como arquivo criminal
3. Carregue `exemplos/midia_alta_correlacao.csv` como redes sociais
4. Clique em **Rodar Analise**
5. Explore os resultados — voce deve ver correlacao forte (r ≈ 0.97)
6. Troque para `midia_sem_correlacao.csv` e compare os resultados

---

## Gerar novos dados sinteticos

```bash
# Precisa de crimes_exemplo.csv ou crimes_.csv na mesma pasta:
python gerador_dados_sinteticos.py
```

Gera os 3 arquivos de midia na pasta atual.

---

## Pipeline de dados reais

Para coletar e processar dados reais, veja o guia completo em:
`pipeline/README_pipeline.md`

Fluxo resumido:
```
Instagram  ──► 1_instagram_coletar.py
Reddit     ──► 2_reddit_processar.py
                        |
                5_preprocessar_normalizar.py
                        |
              3_aplicar_vader.py   4_aplicar_tybyria.py
                        |
              app_correlacao_academica.py
```

---

## Requisitos do sistema

- Python 3.9 ou superior
- 4 GB de RAM (8 GB recomendado para TybyrIA)
- Conexao com internet (para baixar o modelo TybyrIA na primeira vez)
- GPU opcional (acelera o processamento com TybyrIA)

---

## Notas eticas

- Esta ferramenta e destinada exclusivamente a pesquisa academica.
- Nao colete dados pessoais alem do necessario (LGPD).
- Anonimize textos antes de compartilhar capturas de tela da aba "Textos & Picos".
- Correlacao nao implica causalidade — interprete os resultados com cautela.

---

## Suporte

Ferramenta desenvolvida para defesa da Dissertação no Programa de pós-graduação em Informática na Universidade Federal do Estado do Rio de Janeiro (UNIRIO).
Caso queira utilizar, favor dar os devidos créditos