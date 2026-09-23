#!/usr/bin/env python3
"""
Gerador de Dados Sinteticos v2 -- NEXO Validation Suite
========================================================
Le crimes_.csv (dados reais SEJUSP/MG) para construir a serie temporal
de referencia e gera 3 bases de midias sociais para validacao da
ferramenta NEXO.

Cenarios:
  A -- Forte correlacao temporal  -> midia_alta_correlacao.csv
  B -- Sem correlacao             -> midia_sem_correlacao.csv
  C -- Correlacao parcial         -> midia_correlacao_parcial.csv

Seed: 42
Formato de saida: post_id, data_publicacao, texto, toxicity_score
"""

import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_SOCIAL   = 10_000
BASE_DIR   = Path(__file__).parent
CRIMES_CSV = BASE_DIR / "crimes_.csv"

RNG = np.random.default_rng(SEED)
random.seed(SEED)

# ── Pools de texto ─────────────────────────────────────────────────────────────
#
# Conteudo sintetico gerado por templates combinatorios para validacao
# academica da ferramenta NEXO (deteccao de toxicidade).
# Dados inteiramente artificiais, sem referencia a pessoas reais.
#
# Cenario A (alta toxicidade):  xenofobico -- 14 x 16 x 12 = 2.688 combinacoes
# Cenario B (baixa toxicidade): cotidiano neutro
# Cenario C (media toxicidade): irritacao/conflito generico
# ─────────────────────────────────────────────────────────────────────────────

# --- Cenario A: alta toxicidade -- xenofobico (score >= 0.62, media ~0.83) ---
_X_INI = [
    "Esse povo estrangeiro que invade o pais",
    "Imigrante ilegal que chega sem documentos",
    "Esse gringo que aparece aqui sem ser chamado",
    "Estrangeiro sem visto regularizado",
    "Gente de fora que nao foi convidada",
    "Forasteiro que vem de longe explorar",
    "Imigrante que entra ilegalmente",
    "Esse pessoal de outro pais",
    "Quem nao e daqui e quer mandar",
    "Nao aguento mais esse povo de fora",
    "Chega de invasao de",
    "Estrangeiro que nao tem nem documento",
    "Esse refugiado que chega",
    "Gente que vem la de fora sem respeito",
]

_X_MEI = [
    "tomando emprego de brasileiro honesto",
    "que nao fala portugues e nao tem interesse em aprender",
    "chegando sem documentos validos",
    "querendo se beneficiar do sistema publico sem contribuir",
    "que nao respeita nossa cultura e nossos costumes",
    "trazendo problemas para a nossa comunidade",
    "ocupando vaga em hospital e escola que deveria ser do brasileiro",
    "que nao se integra mas exige direitos iguais",
    "explorando nosso sistema de saude e previdencia",
    "que foi ao pais sem ser convidado",
    "invadindo o territorio nacional como se fosse terra sem dono",
    "sem qualquer respaldo legal querendo disputar espaco",
    "que tira recursos que sao dos cidadaos daqui",
    "usando servico publico sem ter contribuido nada",
    "recebendo auxilio que deveria ir para o brasileiro",
    "ignorando as leis e vivendo como bem entende",
]

_X_FIM = [
    " Esse pais e dos brasileiros.",
    " Vai embora pra sua terra.",
    " Nao e preconceito, e realidade.",
    " Chega dessa invasao silenciosa.",
    " O brasileiro sofre enquanto isso acontece.",
    " Nao ha mais espaco.",
    " Cansei desse descaso com quem nasceu aqui.",
    " Cada um no seu pais, simples assim.",
    " Fora.",
    " Nao vou ficar calado sobre isso.",
    " Ate quando isso vai continuar?",
    " Alguem precisa falar a verdade.",
]

# --- Cenario B: baixa toxicidade -- cotidiano neutro (score <= 0.20) ---
_NEUTRO = [
    # Clima
    "Que calor hoje, tive que ligar o ar condicionado logo cedo.",
    "Choveu a tarde toda, dia perfeito pra ficar em casa lendo.",
    "Vento agradavel hoje, aproveitei para caminhar no parque.",
    "O outono chegou cedo esse ano, que frio gostoso de manha.",
    "Sol lindo hoje, cafe na varanda foi otimo.",
    "Dia nublado mas tranquilo, bom para estudar sem distracao.",
    "Que manha fresca e agradavel para uma caminhada.",
    "Chuva passageira a tarde, depois o tempo abriu lindo.",
    # Comida
    "Almoco incrivel hoje, fiz lasanha caseira do zero.",
    "Fui num restaurante novo, a pizza estava espetacular.",
    "Cafe da manha caprichado: tapioca com ovo e suco de laranja.",
    "Tentei receita nova de bolo de cenoura, ficou fofinho.",
    "Churrasco com a familia no final de semana, delicioso.",
    "Acai gelado essa tarde, refrescante demais.",
    "Minha mae fez feijao com arroz e linguica, classico perfeito.",
    "Pipoca e serie boa na sexta, programa perfeito.",
    "Sopa de legumes no frio, reconfortante e gostosa.",
    "Brigadeiro de panela saiu incrivel hoje.",
    "Padaria da esquina lancou sabor novo, recomendo.",
    "Jantar simples mas gostoso com a familia.",
    # Transporte
    "Onibus chegou no horario hoje, que surpresa agradavel.",
    "Consegui estacionamento logo de cara, dia de sorte mesmo.",
    "Metro bem tranquilo essa manha, cheguei bem.",
    "Fui de bike hoje, que exercicio agradavel de manha.",
    "Estrada livre, cheguei no trabalho rapido hoje.",
    "Aplicativo de carona funcionou perfeitamente.",
    "Caminhada ate a estacao foi otima hoje.",
    # Estudos
    "Estudando para a prova de amanha, bem concentrado.",
    "Terminei o trabalho da faculdade com antecedencia.",
    "Aula incrivel sobre estatistica aplicada hoje.",
    "Livro excelente sobre ciencia de dados, recomendo muito.",
    "Passei na certificacao, muito feliz com o resultado.",
    "Finalmente entendi aquele conceito que me travava.",
    "Grupo de estudos foi muito produtivo hoje.",
    "Biblioteca da universidade, ambiente otimo para estudar.",
    # Trabalho
    "Reuniao produtiva hoje, saimos com otimas ideias.",
    "Home office com cafe fresquinho do lado, otimo.",
    "Fechei projeto importante essa semana, que alegria.",
    "Pausa pro cafe com os colegas, bate-papo otimo.",
    "Semana intensa mas muito satisfatoria no trabalho.",
    "Apresentacao correu muito bem, todos gostaram.",
    "Tarefa dificil concluida com sucesso, que satisfacao.",
    # Entretenimento
    "Serie nova incrivel, amei o primeiro episodio.",
    "Filme otimo ontem, fui ao cinema com a familia.",
    "Musica nova do meu artista favorito, de outro nivel.",
    "Show ao vivo incrivel no fim de semana.",
    "Podcast sobre tecnologia muito interessante.",
    "Maratonei a temporada completa, valeu cada minuto.",
    "Jogo novo me viciou de um jeito bom.",
    "Teatro incrivel ontem, peca muito bem dirigida.",
    # Esportes
    "Corri 5km de manha, semana otima de treino.",
    "Academia hoje, treino puxado mas valeu muito.",
    "Meu time ganhou hoje, que jogo bonito foi.",
    "Volei com os amigos, muita risada e diversao.",
    "Natacao de manha, melhor inicio de dia possivel.",
    "Pedalada de tarde com vista incrivel do parque.",
    "Yoga pela manha, comecei o dia muito bem.",
    # Tecnologia
    "Novo celular chegou, a camera e impressionante.",
    "Atualizacao deixou o sistema muito mais rapido.",
    "App de produtividade novo que encontrei e otimo.",
    "Testei IA para ajudar com codigo, funcionou bem.",
    "Notebook novo chegou, que desempenho incrivel.",
    "Configurei o home office com setup novo, otimo.",
    # Viagem
    "Passeio no interior no fim de semana, natureza linda.",
    "Praia tranquila, sol perfeito, descanso total.",
    "Museu da cidade muito interessante, aprendi bastante.",
    "Feira de artesanato gostosa de passear.",
    "Trilha ecologica incrivel, natureza pura.",
    "Viagem de fim de semana para cidade historica, valeu.",
    # Cotidiano
    "Tarde produtiva organizando e limpando a casa.",
    "Plantei temperos na sacada, que satisfacao.",
    "Domingo em familia, melhor dia da semana.",
    "Reencontrei um amigo antigo por acaso, que alegria.",
    "Meditacao de manha, comecando o dia muito bem.",
    "Que tarde relaxante lendo na rede.",
    "Cuidando das plantas em casa, muito terapeutico.",
    "Aprendi receita nova no youtube, ficou otimo.",
    "Passeio com o cachorro no parque, ele adorou.",
    "Leitura no jardim com tempo agradavel.",
]

# --- Cenario C: media toxicidade -- irritacao/conflito generico (score 0.40-0.70) ---
_MEDIO = [
    "Que dia difícil, nada saiu como eu havia planejado.",
    "Cansado desse sistema publico que nao funciona.",
    "Mais um dia de transito horrivel, situacao absurda.",
    "Governo aumentando imposto de novo, inacreditavel.",
    "Nao consigo entender como essa situacao ainda nao mudou.",
    "Preco de tudo subindo e salario parado, e muita coisa.",
    "Servico publico uma vergonha, esperei horas na fila.",
    "Vizinho fazendo barulho ate tarde da noite, falta de respeito.",
    "Internet caindo toda hora, impossivel trabalhar assim.",
    "Atendimento pessimo no banco, perdi a tarde toda.",
    "Mais um acidente por causa de imprudencia no transito.",
    "Politico corrupto aparecendo na midia de novo.",
    "Saturado desse emprego, preciso mudar de vida.",
    "Que frustracao, meu time perdeu mais uma vez.",
    "Fila do hospital andando devagar, sistema falido.",
    "Tudo caro e qualidade cada vez pior.",
    "Barulho da rua me deixou irritado o dia todo.",
    "Nao aguento mais reclamacao sem solucao pratica.",
    "Calor absurdo com falta de agua, descaso total.",
    "Aplicativo travou bem na hora da reuniao importante.",
    "Chega de promessa vazia, quero resultado de verdade.",
    "Nao acredito mais nessa politica que nao resolve.",
    "Multa de transito injusta, vou recorrer.",
    "Pacote atrasado pelos correios pela terceira vez.",
    "Dia completamente perdido, muita raiva acumulada.",
    "Que bagunca, nada funcionando direito.",
    "Nao aguento mais essas politicas que nao funcionam.",
    "Servico contratado nao foi entregue no prazo, absurdo.",
    "Burocracia excessiva impossibilitando qualquer coisa.",
    "Que semana frustrante, precisava que as coisas dessem certo.",
]


# ── Funcoes utilitarias ────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_score(v, lo=0.0, hi=1.0):
    return float(np.clip(v, lo, hi))


def xeno_text():
    return (random.choice(_X_INI) + " " +
            random.choice(_X_MEI) +
            random.choice(_X_FIM))


def neutral_text():
    return random.choice(_NEUTRO)


def medium_text():
    return random.choice(_MEDIO)


def timestamps_in_week(week: pd.Period, n: int) -> list:
    """Gera n timestamps aleatorios dentro da semana."""
    start_ts = week.start_time.timestamp()
    end_ts   = week.end_time.timestamp()
    offsets  = RNG.uniform(start_ts, end_ts, size=n)
    return [pd.Timestamp(o, unit="s").strftime("%Y-%m-%d %H:%M:%S") for o in offsets]


def finalize(records: list) -> pd.DataFrame:
    """Converte lista de tuplas em DataFrame ordenado, renumerando post_id."""
    df = pd.DataFrame(records, columns=["post_id", "data_publicacao", "texto", "toxicity_score"])
    df = df.sort_values("data_publicacao").reset_index(drop=True)
    df["post_id"] = [f"PA{i+1:06d}" for i in range(len(df))]
    return df


def print_stats(label: str, df: pd.DataFrame, r: float, p: float):
    sc = df["toxicity_score"]
    ts = pd.to_datetime(df["data_publicacao"])
    sig = " *" if p < 0.05 else ""
    print(f"\n  {label}")
    print(f"    Registros  : {len(df):,}")
    print(f"    Periodo    : {ts.min().date()} a {ts.max().date()}")
    print(f"    Score      : media={sc.mean():.3f}  std={sc.std():.3f}  "
          f"min={sc.min():.3f}  max={sc.max():.3f}")
    print(f"    >= 0.50    : {(sc >= 0.5).mean()*100:.1f}%")
    print(f"    Pearson r  : {r:.3f} (p={p:.4f}){sig}")


def calc_pearson(df_soc: pd.DataFrame, weeks, c_vals: np.ndarray):
    tmp = df_soc.copy()
    tmp["data"] = pd.to_datetime(tmp["data_publicacao"])
    tmp["semana"] = tmp["data"].dt.to_period("W")
    wk_sc = tmp.groupby("semana")["toxicity_score"].mean().reindex(weeks).fillna(0)
    s_c = pd.Series(c_vals, index=weeks)
    if HAS_SCIPY:
        r, p = pearsonr(wk_sc.values, s_c.values)
    else:
        r = float(wk_sc.corr(s_c))
        p = float("nan")
    return float(r), float(p)


# ── Carregar e analisar crimes _.csv ─────────────────────────────────────────

print("=" * 58)
print("  Gerador de Dados Sinteticos v2 -- NEXO")
print("=" * 58)
print(f"  Seed: {SEED}  |  Registros/cenario: {N_SOCIAL:,}\n")

if not CRIMES_CSV.exists():
    print(f"ERRO: arquivo nao encontrado: {CRIMES_CSV}")
    sys.exit(1)

print("Carregando crimes_.csv ...")
df_c = pd.read_csv(CRIMES_CSV)
df_c["data"] = pd.to_datetime(df_c["data_ocorrencia"], errors="coerce")
df_c = df_c.dropna(subset=["data"])
df_c["semana"] = df_c["data"].dt.to_period("W")

weekly_c = df_c.groupby("semana").size().rename("n")
weeks    = weekly_c.index
n_weeks  = len(weeks)
c_vals   = weekly_c.values.astype(float)
c_z      = (c_vals - c_vals.mean()) / c_vals.std()

print(f"  Periodo : {weeks[0]}  a  {weeks[-1]}")
print(f"  Semanas : {n_weeks}")
print(f"  Crimes/semana : min={c_vals.min():.0f}  max={c_vals.max():.0f}  "
      f"media={c_vals.mean():.1f}  total={int(c_vals.sum())}")


# ══════════════════════════════════════════════════════════════════════════════
# CENARIO A — Forte correlacao
# Volume de posts proporcional ao volume de crimes
# P(toxico) = sigmoid(1.8 * crime_z + 0.25) ∈ [~0.27, ~0.87]
# Toxico  : score ~ N(0.83, 0.09) ∩ [0.62, 1.00]  texto xenofobico
# Neutro  : score ~ N(0.07, 0.05) ∩ [0.00, 0.20]  texto cotidiano
# ══════════════════════════════════════════════════════════════════════════════
print("\nGerando Cenario A (forte correlacao) ...")

w_A     = (c_vals / c_vals.sum()).astype(float)
n_wk_A  = RNG.multinomial(N_SOCIAL, w_A)
p_tox_A = sigmoid(1.8 * c_z + 0.25)

rec_A = []
for wk, n_wk, p_t in zip(weeks, n_wk_A, p_tox_A):
    if n_wk == 0:
        continue
    tss  = timestamps_in_week(wk, int(n_wk))
    mask = RNG.random(int(n_wk)) < p_t
    for ts, is_tox in zip(tss, mask):
        if is_tox:
            s = clip_score(float(RNG.normal(0.83, 0.09)), 0.62, 1.00)
            t = xeno_text()
        else:
            s = clip_score(float(RNG.normal(0.07, 0.05)), 0.00, 0.20)
            t = neutral_text()
        rec_A.append(("", ts, t, round(s, 4)))

df_A = finalize(rec_A)
r_A, p_A = calc_pearson(df_A, weeks, c_vals)
print_stats("Cenario A", df_A, r_A, p_A)


# ══════════════════════════════════════════════════════════════════════════════
# CENARIO B — Sem correlacao
# Volume uniforme entre semanas (independente dos crimes)
# Score ~ N(0.08, 0.05) ∩ [0.00, 0.20]  + 2% ruido
# Textos: cotidiano neutro
# ══════════════════════════════════════════════════════════════════════════════
print("\nGerando Cenario B (sem correlacao) ...")

n_wk_B = np.full(n_weeks, N_SOCIAL // n_weeks, dtype=int)
n_wk_B[: N_SOCIAL % n_weeks] += 1

rec_B = []
for wk, n_wk in zip(weeks, n_wk_B):
    tss = timestamps_in_week(wk, int(n_wk))
    for ts in tss:
        if RNG.random() < 0.02:
            s = clip_score(float(RNG.uniform(0.0, 1.0)))
            t = medium_text() if s >= 0.4 else neutral_text()
        else:
            s = clip_score(float(RNG.normal(0.08, 0.05)), 0.00, 0.20)
            t = neutral_text()
        rec_B.append(("", ts, t, round(s, 4)))

df_B = finalize(rec_B)
r_B, p_B = calc_pearson(df_B, weeks, c_vals)
print_stats("Cenario B", df_B, r_B, p_B)


# ══════════════════════════════════════════════════════════════════════════════
# CENARIO C — Correlacao parcial (~1/8 dos picos de crime)
# Semanas acima do P75 identificadas; 1/8 sao "correlacionadas"
# Correlacionadas  : score ~ N(0.54, 0.10) ∩ [0.40, 0.70]  texto medio
# Outras semanas   : score ~ N(0.08, 0.05) ∩ [0.00, 0.22]  texto neutro
# 5% de ruido puro em todos os registros
# ══════════════════════════════════════════════════════════════════════════════
print("\nGerando Cenario C (correlacao parcial) ...")

thr75    = float(np.percentile(c_vals, 75))
high_idx = np.where(c_vals >= thr75)[0]
n_corr   = max(1, len(high_idx) // 8)
corr_set = set(int(i) for i in RNG.choice(high_idx, n_corr, replace=False))

print(f"    Semanas acima P75 ({thr75:.0f} crimes/sem): {len(high_idx)}")
print(f"    Semanas correlacionadas selecionadas     : {n_corr}")

n_wk_C = np.full(n_weeks, N_SOCIAL // n_weeks, dtype=int)
n_wk_C[: N_SOCIAL % n_weeks] += 1

rec_C = []
for i, (wk, n_wk) in enumerate(zip(weeks, n_wk_C)):
    tss     = timestamps_in_week(wk, int(n_wk))
    is_corr = (i in corr_set)
    for ts in tss:
        if RNG.random() < 0.05:
            s = clip_score(float(RNG.uniform(0.0, 1.0)))
            t = medium_text()
        elif is_corr:
            s = clip_score(float(RNG.normal(0.54, 0.10)), 0.40, 0.70)
            t = medium_text()
        else:
            s = clip_score(float(RNG.normal(0.08, 0.05)), 0.00, 0.22)
            t = neutral_text()
        rec_C.append(("", ts, t, round(s, 4)))

df_C = finalize(rec_C)
r_C, p_C = calc_pearson(df_C, weeks, c_vals)
print_stats("Cenario C", df_C, r_C, p_C)


# ── Salvar arquivos ───────────────────────────────────────────────────────────
print("\nSalvando arquivos ...")
out = {
    "midia_alta_correlacao.csv"   : df_A,
    "midia_sem_correlacao.csv"    : df_B,
    "midia_correlacao_parcial.csv": df_C,
}
for fname, df in out.items():
    fp = BASE_DIR / fname
    df.to_csv(fp, index=False, encoding="utf-8")
    print(f"  OK  {fname}  ({len(df):,} registros)")


# ── Resumo final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  RESUMO FINAL")
print("=" * 58)
print(f"  Seed utilizada        : {SEED}")
print(f"  Base de crimes        : {CRIMES_CSV.name}  ({len(df_c):,} registros)")
print(f"  Periodo coberto       : {weeks[0]}  a  {weeks[-1]}")
print()
print(f"  Cenario A (forte)     : {len(df_A):,} registros  |  Pearson r = {r_A:+.3f}")
print(f"  Cenario B (zero)      : {len(df_B):,} registros  |  Pearson r = {r_B:+.3f}")
print(f"  Cenario C (parcial)   : {len(df_C):,} registros  |  Pearson r = {r_C:+.3f}")
print()
print("  Arquivos prontos para uso em app_correlacao_academica.py")
print("=" * 58)
