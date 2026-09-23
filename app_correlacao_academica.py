"""
Correlação Temporal: Crime × Discurso Online
============================================
Ferramenta interativa para análise de associação temporal entre registros
criminais e scores de polaridade/toxicidade em redes sociais.

Dependências:
    pip install streamlit pandas numpy scipy statsmodels plotly

Execução:
    streamlit run app_correlacao_academica.py
"""

import io
import re
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

try:
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_WORDCLOUD = True
except ImportError:
    try:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "wordcloud", "matplotlib", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _HAS_WORDCLOUD = True
    except Exception:
        _HAS_WORDCLOUD = False

warnings.filterwarnings("ignore")

# ─── Configuração da página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="NEXO — Exploração de Relações Temporais",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicializa o cronômetro da sessão — persiste enquanto a página não for recarregada
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()

st.title("NEXO")
st.caption("Núcleo de Exploração de Relações Temporais entre Fenômenos Online e Offline")
st.markdown(
    """
Ferramenta para análise de associação temporal entre registros criminais e polaridade/toxicidade
em redes sociais. Carregue seus dados, mapeie as colunas e execute a análise completa.

**Análises incluídas:** série temporal · snippets de texto coloridos · calendário diário ·
estacionariedade (ADF+KPSS) · Pearson/Spearman por lag ·
CCF com IC 95% · Causalidade de Granger · Correlação parcial (Frisch-Waugh) · Mapa geocodificado
"""
)

# ─── Funções utilitárias ──────────────────────────────────────────────────────


def load_csv_flexible(uploaded_file) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "iso-8859-1", "latin-1", "cp1252"):
        for sep in (",", ";", "\t", "|"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file, sep=sep, encoding=enc,
                    low_memory=False, on_bad_lines="warn",
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    raise ValueError("Não foi possível ler o arquivo. Verifique se é um CSV válido.")


def parse_date_series(series: pd.Series) -> pd.Series:
    for fmt in (
        None,
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise", dayfirst=True)
        except Exception:
            continue
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def aggregate_to_periods(
    df_crime: pd.DataFrame,
    df_social: pd.DataFrame,
    crime_date_col: str,
    crime_id_col: str,
    social_date_col: str,
    social_score_cols: list[str],
    granularity: str,
) -> pd.DataFrame:
    freq = "W" if granularity == "Semanal" else "M"

    dc = df_crime.copy()
    dc["__date__"] = parse_date_series(dc[crime_date_col])
    dc = dc.dropna(subset=["__date__"])
    if dc.empty:
        raise ValueError(f"Nenhuma data válida na coluna '{crime_date_col}'.")
    dc["__per__"] = dc["__date__"].dt.to_period(freq)

    if crime_id_col and crime_id_col != "(contar linhas)":
        crime_agg = (
            dc.drop_duplicates(subset=[crime_id_col])
            .groupby("__per__").size().rename("n_crimes")
        )
    else:
        crime_agg = dc.groupby("__per__").size().rename("n_crimes")

    ds = df_social.copy()
    ds["__date__"] = parse_date_series(ds[social_date_col])
    ds = ds.dropna(subset=["__date__"])
    if ds.empty:
        raise ValueError(f"Nenhuma data válida na coluna '{social_date_col}'.")
    ds["__per__"] = ds["__date__"].dt.to_period(freq)

    social_parts = {}
    for col in social_score_cols:
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
        social_parts[col] = ds.groupby("__per__")[col].mean()
    social_agg = pd.DataFrame(social_parts)

    all_periods = crime_agg.index.union(social_agg.index)
    df = pd.DataFrame(index=all_periods)
    df = df.join(crime_agg).join(social_agg)
    df = df.sort_index()
    df["n_crimes"] = df["n_crimes"].fillna(0).astype(int)
    df.index.name = "periodo"
    return df


def run_stationarity(series: pd.Series, name: str) -> dict:
    s = series.dropna()
    if len(s) < 10:
        return {"Série": name, "ADF p": "—", "KPSS p": "—", "Estacionária": "dados insuficientes"}
    try:
        _, adf_p, *_ = adfuller(s, autolag="AIC")
    except Exception:
        adf_p = np.nan
    try:
        _, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_p = np.nan
    estacionaria = (not np.isnan(adf_p) and adf_p < 0.05) and (
        not np.isnan(kpss_p) and kpss_p > 0.05
    )
    return {
        "Série": name,
        "ADF p": f"{adf_p:.4f}" if not np.isnan(adf_p) else "—",
        "KPSS p": f"{kpss_p:.4f}" if not np.isnan(kpss_p) else "—",
        "Estacionária": "SIM" if estacionaria else "NÃO",
    }


def compute_lag_correlations(
    crimes: pd.Series, polarity: pd.Series, n_lags: int
) -> pd.DataFrame:
    rows = []
    for k in range(n_lags + 1):
        crimes_shifted = crimes.shift(-k)
        mask = crimes_shifted.notna() & polarity.notna()
        n = int(mask.sum())
        if n < 8:
            continue
        s_c = crimes_shifted[mask].values
        s_p = polarity[mask].values
        r_p, p_p = stats.pearsonr(s_p, s_c)
        r_s, p_s = stats.spearmanr(s_p, s_c)
        rows.append(dict(Lag=k, Pearson_r=round(r_p, 4), Pearson_p=round(p_p, 4),
                         Spearman_r=round(r_s, 4), Spearman_p=round(p_s, 4), n=n))
    return pd.DataFrame(rows)


def compute_ccf(
    crimes: pd.Series, polarity: pd.Series, n_lags: int, diff: bool
) -> tuple[list, list, float]:
    if diff:
        s_c = crimes.diff().dropna()
        s_p = polarity.diff().dropna()
        idx = s_c.index.intersection(s_p.index)
        s_c = s_c.loc[idx].values
        s_p = s_p.loc[idx].values
    else:
        mask = crimes.notna() & polarity.notna()
        s_c = crimes[mask].values
        s_p = polarity[mask].values
    n = len(s_c)
    ic95 = 1.96 / np.sqrt(n) if n > 0 else 0.2
    lags = list(range(-n_lags, n_lags + 1))
    ccf_vals = []
    for k in lags:
        if k == 0:
            r, _ = stats.pearsonr(s_p, s_c)
        elif k > 0:
            r = stats.pearsonr(s_p[:-k], s_c[k:])[0] if k < n else 0.0
        else:
            k_abs = abs(k)
            r = stats.pearsonr(s_p[k_abs:], s_c[:-k_abs])[0] if k_abs < n else 0.0
        ccf_vals.append(round(r, 4))
    return lags, ccf_vals, ic95


def run_granger(
    crimes: pd.Series, polarity: pd.Series, max_lag: int, diff: bool
) -> pd.DataFrame | None:
    if diff:
        s_c = crimes.diff().dropna()
        s_p = polarity.diff().dropna()
        idx = s_c.index.intersection(s_p.index)
        s_c = s_c.loc[idx]
        s_p = s_p.loc[idx]
    else:
        mask = crimes.notna() & polarity.notna()
        s_c = crimes[mask]
        s_p = polarity[mask]
    df_g = pd.DataFrame({"crimes": s_c.values, "polarity": s_p.values})
    if len(df_g) < max_lag * 3 + 5:
        return None
    try:
        res = grangercausalitytests(df_g[["crimes", "polarity"]], maxlag=max_lag, verbose=False)
        rows = []
        for lag, data in res.items():
            f_stat = data[0]["ssr_ftest"][0]
            p_val = data[0]["ssr_ftest"][1]
            rows.append(dict(Lag=lag, F_stat=round(f_stat, 4), p_valor=round(p_val, 4)))
        return pd.DataFrame(rows)
    except Exception:
        return None


def run_partial_correlation(
    crimes: pd.Series, polarity: pd.Series
) -> tuple[float | None, float | None, int]:
    df_pc = pd.DataFrame({
        "crimes": crimes.values,
        "polarity": polarity.values,
        "crimes_t1": crimes.shift(1).values,
        "crimes_mm3": crimes.rolling(3, min_periods=2).mean().values,
    }).dropna()
    if len(df_pc) < 10:
        return None, None, len(df_pc)
    X_ctrl = add_constant(df_pc[["crimes_t1", "crimes_mm3"]])
    res_crimes = OLS(df_pc["crimes"], X_ctrl).fit().resid
    res_pol = OLS(df_pc["polarity"], X_ctrl).fit().resid
    r_p, p_p = stats.pearsonr(res_pol.values, res_crimes.values)
    return round(r_p, 4), round(p_p, 4), len(df_pc)


def sig_label(p_val: float, alpha_raw: float, alpha_adj: float) -> str:
    if p_val < alpha_adj:
        return "★ Bonferroni"
    if p_val < alpha_raw:
        return "· α bruto"
    return ""


def _ratio_to_rgb(ratio: float) -> str:
    """Interpola verde(0) → amarelo(0.5) → vermelho(1)."""
    ratio = max(0.0, min(1.0, ratio))
    if ratio < 0.5:
        r = int(255 * ratio * 2)
        g = 200
    else:
        r = 255
        g = int(200 * (1 - (ratio - 0.5) * 2))
    return f"rgb({r},{g},50)"


def score_color(score: float, min_v: float, max_v: float, inverse: bool = False) -> str:
    ratio = (score - min_v) / (max_v - min_v) if max_v != min_v else 0.5
    return _ratio_to_rgb(1 - ratio if inverse else ratio)


def make_wordcloud(texts: list[str], colormap: str, title: str) -> "plt.Figure | None":
    """Gera nuvem de palavras a partir de uma lista de textos. Retorna figura matplotlib."""
    if not _HAS_WORDCLOUD or not texts:
        return None
    stopwords_pt = set(STOPWORDS) | {
        "de", "da", "do", "das", "dos", "e", "em", "um", "uma", "o", "a", "os", "as",
        "que", "se", "é", "não", "com", "para", "por", "isso", "isso", "mais",
        "mas", "ou", "já", "esse", "essa", "esse", "esses", "essa", "seu", "sua",
        "meu", "minha", "me", "te", "nos", "vou", "vai", "ser", "ter", "ao", "na",
        "no", "nem", "só", "tá", "tô", "muito", "bem", "ainda", "todo", "tudo",
    }
    corpus = " ".join(texts)
    wc = WordCloud(
        width=700, height=360,
        background_color="white",
        colormap=colormap,
        stopwords=stopwords_pt,
        max_words=80,
        collocations=False,
        prefer_horizontal=0.85,
    ).generate(corpus)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout(pad=0.5)
    return fig


def fig_to_streamlit(fig) -> None:
    """Renderiza figura matplotlib no Streamlit via buffer PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def snippet_card(text: str, score: float, meta: str, score_label: str, color: str) -> str:
    safe_text = str(text)[:450].replace("<", "&lt;").replace(">", "&gt;")
    ellipsis = "…" if len(str(text)) > 450 else ""
    # Fundo branco fixo garante legibilidade em temas claros e escuros
    return (
        f'<div style="background:#ffffff;border-left:5px solid {color};'
        f'padding:10px 14px;margin-bottom:8px;border-radius:6px;font-size:0.88rem;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.15);">'
        f'<span style="color:#555555;font-size:0.75rem;">{meta} &nbsp;|&nbsp; '
        f'{score_label}: <b style="color:{color}">{score:.3f}</b></span><br>'
        f'<span style="color:#111111;line-height:1.55;">{safe_text}{ellipsis}</span></div>'
    )


_MONTHS_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
              "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def make_calendar_heatmap(
    daily_series: pd.Series, title: str, colorscale: str
) -> "go.Figure | None":
    """
    Heatmap estilo GitHub contributions: cada quadradinho = um dia.
    Colunas = semanas (da esquerda para direita).
    Linhas = dias da semana (Dom no topo, Sáb na base).
    """
    if daily_series.empty:
        return None

    idx = pd.to_datetime(daily_series.index)
    start, end = idx.min(), idx.max()

    # Recuar até o domingo imediatamente anterior (ou o próprio, se já for domingo)
    # pandas dayofweek: Mon=0 … Sun=6  →  offset desejado: Sun→0, Mon→1 … Sat→6
    offset = (start.dayofweek + 1) % 7
    start_aligned = start - pd.Timedelta(days=offset)

    full_range = pd.date_range(start=start_aligned, end=end, freq="D")
    n_weeks = (len(full_range) + 6) // 7

    z = np.full((7, n_weeks), np.nan)
    hover = np.empty((7, n_weeks), dtype=object)
    hover[:] = ""

    daily_dict = {
        pd.Timestamp(k).normalize(): float(v)
        for k, v in daily_series.items()
    }

    for i, d in enumerate(full_range):
        row = (d.dayofweek + 1) % 7   # 0 = Domingo
        col = i // 7
        val = daily_dict.get(d.normalize(), 0.0)
        z[row, col] = val
        hover[row, col] = (
            f"{d.strftime('%d/%m/%Y')} "
            f"({d.strftime('%a')})<br>Valor: {val:.1f}"
        )

    # Marca a semana onde cada mês começa (para o eixo X)
    month_cols: dict[tuple, int] = {}
    for i, d in enumerate(full_range):
        key = (d.year, d.month)
        if key not in month_cols:
            month_cols[key] = i // 7

    x_tickvals = list(month_cols.values())
    x_ticktext = [
        f"{_MONTHS_PT[m - 1]}<br><b>{y}</b>" if m == 1 else _MONTHS_PT[m - 1]
        for (y, m) in month_cols.keys()
    ]

    day_abbr = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=colorscale,
        showscale=True,
        xgap=3,
        ygap=3,
        zmin=0,
    ))
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=day_abbr,
        autorange="reversed",
        tickfont=dict(size=10),
        fixedrange=True,
    )
    fig.update_xaxes(
        tickvals=x_tickvals,
        ticktext=x_ticktext,
        tickfont=dict(size=9),
        fixedrange=True,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=230,
        margin=dict(t=45, b=35, l=50, r=20),
    )
    return fig


# ─── Cronômetro de sessão ────────────────────────────────────────────────────

@st.fragment(run_every=1)
def show_timer():
    elapsed = int(time.time() - st.session_state.session_start)
    hh = elapsed // 3600
    mm = (elapsed % 3600) // 60
    ss = elapsed % 60
    time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

    # Pulso visual: alterna borda a cada segundo par/ímpar
    border_color = "#e63946" if ss % 2 == 0 else "#ff6b6b"

    st.sidebar.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            border: 2.5px solid {border_color};
            border-radius: 14px;
            padding: 14px 10px 10px 10px;
            text-align: center;
            margin-bottom: 4px;
            box-shadow: 0 0 12px {border_color}55;
        ">
            <div style="color:#aaa;font-size:0.65rem;letter-spacing:3px;
                        text-transform:uppercase;margin-bottom:4px;">
                &#9203; Tempo de sessão
            </div>
            <div style="color:#ff4b5c;font-size:2.1rem;font-weight:900;
                        font-family:'Courier New',monospace;letter-spacing:6px;
                        text-shadow: 0 0 10px #e6394688;">
                {time_str}
            </div>
        </div>
        <div style="
            background:#fff3cd;border-left:4px solid #f0a500;
            border-radius:6px;padding:6px 10px;font-size:0.72rem;
            color:#7a5800;margin-top:4px;
        ">
            &#9888; O cronometro zera apenas ao recarregar a pagina (F5).
            Troca de arquivos e analises nao o afetam.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    show_timer()
    st.divider()
    st.markdown("### 📖 Como usar")
    st.markdown(
        "1. Carregue o arquivo de **crimes** (CSV)\n"
        "2. Carregue o arquivo de **redes sociais** (CSV)\n"
        "3. Mapeie as colunas — texto e lat/lon são opcionais\n"
        "4. Clique em **▶ Executar Análise**"
    )
    with st.expander("Formato mínimo dos arquivos"):
        st.markdown(
            "**Crimes:** coluna de data + ID do evento (opcional para deduplicar)\n\n"
            "**Redes sociais:** coluna de data + coluna numérica de score (toxicidade, VADER etc.)"
        )
    st.divider()

    st.markdown("### ⚙️ Configurações")

    granularity = st.selectbox(
        "Granularidade temporal",
        ["Semanal", "Mensal"],
        help=(
            "Define o tamanho do período de agrupamento.\n\n"
            "**Semanal** → mais pontos, mais detalhes, mas mais ruído.\n"
            "**Mensal** → série mais suave, melhor para dados escassos.\n\n"
            "Recomendado: Semanal quando há >2 anos de dados."
        ),
    )
    n_lags = st.slider(
        "Lags máximos — Correlação & CCF",
        2, 20, 8,
        help=(
            "Quantas defasagens temporais testar.\n\n"
            "Lag k = correlação entre a polaridade hoje e os crimes k períodos no futuro.\n"
            "Ex.: lag 4 semanal = discurso hoje vs crimes em ~1 mês.\n\n"
            "Valores maiores exploram relações mais distantes no tempo, "
            "mas reduzem o poder estatístico."
        ),
    )
    max_lag_granger = st.slider(
        "Lags máximos — Granger",
        2, 12, 6,
        help=(
            "Número máximo de defasagens para o teste de Granger.\n\n"
            "O Granger testa se a polaridade melhora a previsão dos crimes "
            "além do que o próprio histórico de crimes já prevê.\n\n"
            "Mantenha menor que os lags de correlação: cada lag adicional "
            "consome graus de liberdade e pode tornar o teste instável."
        ),
    )
    alpha_raw = st.number_input(
        "Nível de significância (α)",
        0.01, 0.20, 0.05, 0.01,
        help=(
            "Limiar de significância estatística.\n\n"
            "Resultados com p < α são considerados 'significativos'.\n"
            "Padrão 0,05 (5%) é o convencional em ciências sociais.\n\n"
            "Valores menores (ex.: 0,01) tornam o critério mais rigoroso."
        ),
    )
    aplicar_bonferroni = st.checkbox(
        "Correção de Bonferroni",
        value=True,
        help=(
            "Ajusta o α para múltiplos testes simultâneos.\n\n"
            "Ao testar vários lags e várias variáveis, a chance de "
            "encontrar um falso positivo por acaso cresce. "
            "Bonferroni divide o α pelo número total de testes, "
            "tornando o critério mais conservador.\n\n"
            "Recomendado: manter ativo para análise exploratória."
        ),
    )
    diferenciacao = st.checkbox(
        "Diferenciar séries (D1) — Granger & CCF",
        value=True,
        help=(
            "Aplica diferenciação de 1ª ordem nas séries antes do Granger e CCF.\n\n"
            "O teste de Granger exige séries **estacionárias** "
            "(sem tendência temporal crescente ou decrescente). "
            "A diferenciação subtrai cada valor do anterior, "
            "removendo a tendência.\n\n"
            "A aba Estacionariedade (ADF + KPSS) indica se isso é necessário."
        ),
    )
    n_snippets = st.slider(
        "Nº de snippets por categoria",
        5, 20, 10,
        help=(
            "Quantos cards de texto exibir em cada coluna "
            "na aba Textos & Picos.\n\n"
            "Afeta os blocos 'maior toxicidade' e 'menor toxicidade', "
            "tanto na seção geral quanto nos períodos de pico de crimes."
        ),
    )

# ─── Upload ───────────────────────────────────────────────────────────────────

st.subheader("1 · Carregar arquivos")
col_u1, col_u2 = st.columns(2)

with col_u1:
    st.markdown("**Dados Criminais**")
    crime_file = st.file_uploader(
        "Arquivo de crimes (CSV)", type=["csv"], key="crime",
        help="Cada linha pode representar um crime ou um envolvido."
    )

with col_u2:
    st.markdown("**Dados de Redes Sociais**")
    social_file = st.file_uploader(
        "Arquivo de posts/comentários (CSV)", type=["csv"], key="social",
        help="Cada linha é um post ou comentário com score numérico de polaridade."
    )

if not crime_file or not social_file:
    st.info("Carregue os dois arquivos para continuar.")
    with st.expander("Exemplos de formato aceito"):
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("**Crimes** — exemplo:")
            st.code(
                "data_ocorrencia;id_ocorrencia;municipio;latitude;longitude\n"
                "15/01/2023 14:32;OC0001;Belo Horizonte;-19.917;-43.934\n"
                "15/01/2023 14:32;OC0001;Belo Horizonte;-19.917;-43.934\n"
                "16/01/2023 09:15;OC0002;Uberlândia;-18.918;-48.276",
                language="text",
            )
        with ex2:
            st.markdown("**Redes Sociais** — exemplo:")
            st.code(
                "data_post,texto,toxicidade,vader_compound\n"
                "2023-01-14,esse grupo é um absurdo,0.85,-0.65\n"
                "2023-01-15,que dia lindo parabéns a todos,0.04,+0.91\n"
                "2023-01-15,mais um escândalo hoje,0.61,-0.32",
                language="text",
            )
    st.stop()

try:
    df_crime = load_csv_flexible(crime_file)
    st.success(
        f"Crimes: {len(df_crime):,} linhas · {df_crime.shape[1]} colunas · "
        f"primeiras: {', '.join(df_crime.columns[:5].tolist())}"
    )
except Exception as e:
    st.error(f"Erro ao carregar crimes: {e}")
    st.stop()

try:
    df_social = load_csv_flexible(social_file)
    st.success(
        f"Redes sociais: {len(df_social):,} linhas · {df_social.shape[1]} colunas · "
        f"primeiras: {', '.join(df_social.columns[:5].tolist())}"
    )
except Exception as e:
    st.error(f"Erro ao carregar redes sociais: {e}")
    st.stop()

# ─── Mapeamento de colunas ────────────────────────────────────────────────────

st.divider()
st.subheader("2 · Mapear colunas")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("**Arquivo de Crimes**")

    date_hint = next(
        (c for c in df_crime.columns
         if any(k in c.lower() for k in ["data", "date", "hora", "dt_"])),
        df_crime.columns[0],
    )
    crime_date_col = st.selectbox(
        "Coluna de data/hora dos crimes",
        df_crime.columns.tolist(),
        index=df_crime.columns.tolist().index(date_hint),
    )
    try:
        sample = parse_date_series(df_crime[crime_date_col]).dropna().dt.strftime("%d/%m/%Y")
        st.caption(f"Amostra: {', '.join(sample.head(3).values)}")
    except Exception:
        pass

    id_options = ["(contar linhas)"] + df_crime.columns.tolist()
    id_hint = next(
        (c for c in df_crime.columns
         if any(k in c.lower() for k in ["id", "ocorr", "event", "registro"])),
        None,
    )
    crime_id_col = st.selectbox(
        "Coluna de ID único (para deduplicar por evento)",
        id_options,
        index=id_options.index(id_hint) if id_hint else 0,
        help="Se cada linha já é um crime único → '(contar linhas)'. "
             "Se há múltiplas linhas por evento (ex: envolvidos) → selecione o ID do evento.",
    )

    st.markdown("**Localização (opcional — habilita mapa):**")
    geo_opts = ["(não disponível)"] + df_crime.columns.tolist()
    lat_hint = next((c for c in df_crime.columns if "lat" in c.lower()), None)
    lon_hint = next(
        (c for c in df_crime.columns
         if any(k in c.lower() for k in ["lon", "lng"]) and "lat" not in c.lower()),
        None,
    )
    lat_col = st.selectbox(
        "Latitude", geo_opts,
        index=geo_opts.index(lat_hint) if lat_hint else 0, key="lat_sel",
    )
    lon_col = st.selectbox(
        "Longitude", geo_opts,
        index=geo_opts.index(lon_hint) if lon_hint else 0, key="lon_sel",
    )

with col_m2:
    st.markdown("**Arquivo de Redes Sociais**")

    date_hint_s = next(
        (c for c in df_social.columns
         if any(k in c.lower() for k in ["data", "date", "created", "hora", "dt_"])),
        df_social.columns[0],
    )
    social_date_col = st.selectbox(
        "Coluna de data/hora dos posts/comentários",
        df_social.columns.tolist(),
        index=df_social.columns.tolist().index(date_hint_s),
    )
    try:
        sample_s = parse_date_series(df_social[social_date_col]).dropna().dt.strftime("%d/%m/%Y")
        st.caption(f"Amostra: {', '.join(sample_s.head(3).values)}")
    except Exception:
        pass

    numeric_cols = df_social.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("Nenhuma coluna numérica detectada. Selecione manualmente.")
        numeric_cols = df_social.columns.tolist()

    score_hint = [
        c for c in numeric_cols
        if any(k in c.lower() for k in [
            "score", "toxicidade", "vader", "compound", "sentiment",
            "polarity", "tybyria", "hate", "negat", "posit",
        ])
    ]
    social_score_cols = st.multiselect(
        "Colunas de score de polaridade/toxicidade",
        numeric_cols,
        default=score_hint[:4] if score_hint else numeric_cols[:2],
        help="Selecione uma ou mais colunas numéricas de polaridade ou toxicidade.",
    )
    if social_score_cols:
        st.caption(
            f"Amostra '{social_score_cols[0]}': "
            f"{df_social[social_score_cols[0]].dropna().round(3).head(3).tolist()}"
        )

    st.markdown("**Texto (opcional — habilita snippets):**")
    text_opts = ["(não disponível)"] + df_social.columns.tolist()
    _txt_exact = {"text", "texto", "body", "conteudo", "conteúdo", "mensagem", "sentence", "content"}
    _txt_partial = ["text", "texto", "body", "conteudo", "conteúdo", "mensagem", "sentence", "content"]
    text_hint = (
        # 1) correspondência exata (sem "id" no nome)
        next((c for c in df_social.columns if c.lower() in _txt_exact and "id" not in c.lower()), None)
        # 2) parcial mas exclui colunas com "id"
        or next((c for c in df_social.columns
                 if any(k in c.lower() for k in _txt_partial) and "id" not in c.lower()), None)
    )
    social_text_col = st.selectbox(
        "Coluna de texto do post/comentário",
        text_opts,
        index=text_opts.index(text_hint) if text_hint else 0,
        key="text_sel",
    )

# ─── Botão de execução ────────────────────────────────────────────────────────

st.divider()
run_btn = st.button("▶ Executar Análise", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

if not social_score_cols:
    st.error("Selecione pelo menos uma coluna de score de polaridade para continuar.")
    st.stop()

# ─── Agregação temporal ───────────────────────────────────────────────────────

with st.spinner("Agregando dados temporalmente..."):
    try:
        df_agg = aggregate_to_periods(
            df_crime, df_social,
            crime_date_col, crime_id_col,
            social_date_col, social_score_cols,
            granularity,
        )
    except Exception as e:
        st.error(f"Erro na agregação: {e}")
        st.code(traceback.format_exc())
        st.stop()

n_periodos = len(df_agg)
crimes_series = df_agg["n_crimes"].astype(float)
idx_str = df_agg.index.astype(str)
n_scores = len(social_score_cols)
n_tests_total = n_scores * (n_lags + 1)
alpha_adj = alpha_raw / n_tests_total if aplicar_bonferroni else alpha_raw
alpha_g = alpha_raw / (n_scores * max_lag_granger) if aplicar_bonferroni else alpha_raw

st.success(
    f"Agregação concluída: **{n_periodos} períodos {granularity.lower()}s** | "
    f"crimes: {int(crimes_series.sum()):,} total | "
    f"mediana: {crimes_series.median():.1f}/período | "
    f"α Bonferroni: {alpha_adj:.5f} ({n_tests_total} testes)"
)

# ─── Pré-computar todos os resultados ────────────────────────────────────────

with st.spinner("Executando análises estatísticas..."):
    stat_rows = [run_stationarity(crimes_series, "n_crimes")]
    for _col in social_score_cols:
        stat_rows.append(run_stationarity(df_agg[_col].astype(float), _col))
    df_stat = pd.DataFrame(stat_rows)

    lag_results: dict[str, pd.DataFrame] = {}
    ccf_results: dict[str, tuple] = {}
    granger_results: dict[str, pd.DataFrame | None] = {}
    pc_results: dict[str, tuple] = {}

    for _col in social_score_cols:
        _pol = df_agg[_col].astype(float)
        lag_results[_col] = compute_lag_correlations(crimes_series, _pol, n_lags)
        ccf_results[_col] = compute_ccf(crimes_series, _pol, n_lags, diferenciacao)
        granger_results[_col] = run_granger(crimes_series, _pol, max_lag_granger, diferenciacao)
        pc_results[_col] = run_partial_correlation(crimes_series, _pol)

# ─── Abas ─────────────────────────────────────────────────────────────────────

COLORS = px.colors.qualitative.Safe

(
    tab_ts, tab_snip, tab_cal, tab_map,
    tab_stat_tab, tab_lag, tab_ccf, tab_gr, tab_pc,
    tab_interp, tab_manual,
) = st.tabs([
    "📈 Série Temporal",
    "📝 Textos & Picos",
    "📅 Calendário",
    "🗺️ Mapa",
    "🧪 Estacionariedade",
    "📊 Correlação por Lag",
    "〰️ CCF",
    "⚡ Granger",
    "🔗 Correlação Parcial",
    "🔎 Interpretação",
    "📖 Manual",
])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — Série Temporal
# ══════════════════════════════════════════════════════════════════════════════
with tab_ts:
    st.subheader("Séries Temporais Agregadas")
    st.markdown(
        "Barras vermelhas = volume de crimes (eixo esquerdo). "
        "Linhas coloridas = scores de polaridade (eixo direito)."
    )

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Bar(
        x=idx_str, y=crimes_series.values,
        name="Crimes (n)", marker_color="#e63946", opacity=0.65, yaxis="y1",
    ))
    for i, col in enumerate(social_score_cols):
        fig_ts.add_trace(go.Scatter(
            x=idx_str, y=df_agg[col].values,
            name=col, mode="lines+markers",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            yaxis="y2",
        ))
    fig_ts.update_layout(
        yaxis=dict(title="N° de crimes", side="left", showgrid=False),
        yaxis2=dict(title="Score de polaridade/toxicidade", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.28),
        height=480, hovermode="x unified", margin=dict(t=20),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("Ver tabela de dados agregados"):
        st.dataframe(df_agg.reset_index())
    st.download_button(
        "⬇ Download dados agregados (CSV)",
        df_agg.reset_index().to_csv(index=False).encode("utf-8"),
        "dados_agregados.csv", "text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — Textos & Picos
# ══════════════════════════════════════════════════════════════════════════════
with tab_snip:
    st.subheader("Textos & Períodos de Pico")

    # ── Snippets de texto ──────────────────────────────────────────────────
    if social_text_col == "(não disponível)":
        st.info(
            "Selecione a coluna de texto no mapeamento (seção **Texto — opcional**) "
            "para ver os snippets coloridos por polaridade."
        )
    else:
        st.caption(
            "⚠️ Os textos abaixo vêm diretamente dos dados carregados. "
            "Certifique-se de que foram devidamente anonimizados antes de compartilhar capturas de tela."
        )

        for score_col in social_score_cols:
            st.markdown(f"---\n**Score analisado: `{score_col}`**")

            ds_t = df_social[[social_text_col, score_col, social_date_col]].copy()
            ds_t[score_col] = pd.to_numeric(ds_t[score_col], errors="coerce")
            ds_t["__date__"] = parse_date_series(ds_t[social_date_col])
            ds_t = ds_t.dropna(subset=[score_col])

            if ds_t.empty:
                st.warning("Sem dados para esta coluna de score.")
                continue

            min_v, max_v = ds_t[score_col].min(), ds_t[score_col].max()
            top_df  = ds_t.nlargest(n_snippets, score_col)
            bot_df  = ds_t.nsmallest(n_snippets, score_col)
            top20_df = ds_t.nlargest(20, score_col)
            bot20_df = ds_t.nsmallest(20, score_col)

            # ── Nuvens de palavras (top 20 mais / menos tóxicos) ──────────────
            if _HAS_WORDCLOUD:
                st.markdown("**Nuvem de palavras — top 20 por toxicidade**")
                wc_col1, wc_col2 = st.columns(2)
                with wc_col1:
                    fig_wc = make_wordcloud(
                        top20_df[social_text_col].dropna().tolist(),
                        colormap="Reds",
                        title=f"20 mais tóxicos ({score_col})",
                    )
                    if fig_wc:
                        fig_to_streamlit(fig_wc)
                with wc_col2:
                    fig_wc2 = make_wordcloud(
                        bot20_df[social_text_col].dropna().tolist(),
                        colormap="Greens",
                        title=f"20 menos tóxicos ({score_col})",
                    )
                    if fig_wc2:
                        fig_to_streamlit(fig_wc2)
            else:
                st.caption(
                    "Instale `wordcloud` para ver as nuvens de palavras: "
                    "`pip install wordcloud`"
                )

            # ── Cards de texto ────────────────────────────────────────────────
            col_top, col_bot = st.columns(2)

            with col_top:
                st.markdown(f"🔴 **Top {n_snippets} — maior polaridade/toxicidade**")
                html = ""
                for _, row in top_df.iterrows():
                    color = score_color(row[score_col], min_v, max_v)
                    date_s = row["__date__"].strftime("%d/%m/%Y") if pd.notna(row["__date__"]) else "—"
                    html += snippet_card(row[social_text_col], row[score_col], date_s, score_col, color)
                st.markdown(html, unsafe_allow_html=True)

            with col_bot:
                st.markdown(f"🟢 **Top {n_snippets} — menor polaridade/toxicidade**")
                html = ""
                for _, row in bot_df.iterrows():
                    color = score_color(row[score_col], min_v, max_v, inverse=True)
                    date_s = row["__date__"].strftime("%d/%m/%Y") if pd.notna(row["__date__"]) else "—"
                    html += snippet_card(row[social_text_col], row[score_col], date_s, score_col, color)
                st.markdown(html, unsafe_allow_html=True)

    # ── Picos de crimes ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Top 10 períodos com maior volume de crimes")

    top_crime_df = (
        df_agg[["n_crimes"] + social_score_cols]
        .nlargest(10, "n_crimes")
        .reset_index()
    )
    period_labels = top_crime_df["periodo"].astype(str).tolist()
    crimes_vals = top_crime_df["n_crimes"].tolist()

    fig_peaks = go.Figure()
    fig_peaks.add_trace(go.Bar(
        x=period_labels, y=crimes_vals,
        marker=dict(
            color=crimes_vals,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Crimes"),
        ),
        text=[str(v) for v in crimes_vals],
        textposition="outside",
        name="Crimes",
    ))
    for i, col in enumerate(social_score_cols[:2]):
        fig_peaks.add_trace(go.Scatter(
            x=period_labels, y=top_crime_df[col].tolist(),
            name=col, mode="markers+lines", yaxis="y2",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig_peaks.update_layout(
        title="Períodos de pico — crimes e scores de polaridade",
        yaxis=dict(title="N° de crimes", side="left"),
        yaxis2=dict(title="Score de polaridade", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.3),
        height=400,
    )
    st.plotly_chart(fig_peaks, use_container_width=True)

    # Textos dos períodos de pico
    if social_text_col != "(não disponível)" and social_score_cols:
        st.markdown(f"**Textos publicados nos 5 períodos de maior crime:**")
        freq_map = "W" if granularity == "Semanal" else "M"
        sc = social_score_cols[0]
        ds_peak = df_social[[social_text_col, sc, social_date_col]].copy()
        ds_peak["__date__"] = parse_date_series(ds_peak[social_date_col])
        ds_peak = ds_peak.dropna(subset=["__date__"])
        ds_peak["__per__"] = ds_peak["__date__"].dt.to_period(freq_map).astype(str)

        top5_periods = top_crime_df["periodo"].astype(str).head(5).tolist()
        ds_peak = ds_peak[ds_peak["__per__"].isin(top5_periods)]
        ds_peak[sc] = pd.to_numeric(ds_peak[sc], errors="coerce")
        ds_peak = ds_peak.dropna(subset=[sc])

        if not ds_peak.empty:
            min_v_pk = ds_peak[sc].min()
            max_v_pk = ds_peak[sc].max()
            n_pk = min(n_snippets, len(ds_peak))
            top_pk = ds_peak.nlargest(n_pk, sc)
            bot_pk = ds_peak.nsmallest(n_pk, sc)

            pk_col1, pk_col2 = st.columns(2)
            with pk_col1:
                st.markdown(f"🔴 **Top {n_pk} — maior toxicidade**")
                html_pk = ""
                for _, row in top_pk.iterrows():
                    color = score_color(row[sc], min_v_pk, max_v_pk)
                    meta = f"Período {row['__per__']} | {row['__date__'].strftime('%d/%m/%Y')}"
                    html_pk += snippet_card(row[social_text_col], row[sc], meta, sc, color)
                st.markdown(html_pk, unsafe_allow_html=True)
            with pk_col2:
                st.markdown(f"🟢 **Top {n_pk} — menor toxicidade**")
                html_pk2 = ""
                for _, row in bot_pk.iterrows():
                    color = score_color(row[sc], min_v_pk, max_v_pk, inverse=True)
                    meta = f"Período {row['__per__']} | {row['__date__'].strftime('%d/%m/%Y')}"
                    html_pk2 += snippet_card(row[social_text_col], row[sc], meta, sc, color)
                st.markdown(html_pk2, unsafe_allow_html=True)
        else:
            st.info("Nenhum post encontrado nos períodos de pico.")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — Calendário / Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab_cal:
    st.subheader("Heatmap Calendário — Intensidade ao Longo do Tempo")

    cal_modo = st.radio(
        "Modo de visualização",
        ["📅 Calendário diário", "📊 Heatmap por período"],
        horizontal=True,
    )

    if cal_modo == "📅 Calendário diário":
        st.caption(
            "Cada quadradinho = um dia. Semanas da esquerda pra direita, "
            "Dom no topo → Sáb na base. Passe o mouse para ver data e valor."
        )
        try:
            # Agregação diária de crimes
            dc_day = df_crime.copy()
            dc_day["__date__"] = parse_date_series(dc_day[crime_date_col])
            dc_day = dc_day.dropna(subset=["__date__"])
            if crime_id_col and crime_id_col != "(contar linhas)":
                dc_day = dc_day.drop_duplicates(subset=[crime_id_col])
            daily_crimes_s = (
                dc_day.groupby(dc_day["__date__"].dt.normalize())
                .size()
                .rename("n_crimes")
            )
            fig_cal_c = make_calendar_heatmap(
                daily_crimes_s, "Crimes por dia", "Reds"
            )
            if fig_cal_c:
                st.plotly_chart(fig_cal_c, use_container_width=True)

            # Agregação diária dos scores sociais
            ds_day = df_social.copy()
            ds_day["__date__"] = parse_date_series(ds_day[social_date_col])
            ds_day = ds_day.dropna(subset=["__date__"])
            ds_day["__day__"] = ds_day["__date__"].dt.normalize()
            for sc in social_score_cols[:3]:
                ds_day[sc] = pd.to_numeric(ds_day[sc], errors="coerce")
                daily_sc = ds_day.groupby("__day__")[sc].mean()
                fig_cal_sc = make_calendar_heatmap(
                    daily_sc, f"{sc} — média por dia", "RdYlGn_r"
                )
                if fig_cal_sc:
                    st.plotly_chart(fig_cal_sc, use_container_width=True)

        except Exception as e:
            st.warning(f"Não foi possível construir o calendário diário: {e}")

    else:
        st.markdown(
            "Tons mais escuros = valores mais altos. "
            "Passe o mouse sobre cada célula para ver os valores exatos."
        )
        df_cal = df_agg.reset_index()
        try:
            df_cal["ano"] = df_cal["periodo"].apply(lambda p: str(p.year))
            if granularity == "Semanal":
                df_cal["eixo_x"] = df_cal["periodo"].apply(lambda p: p.week)
                x_label = "Semana do Ano"
            else:
                df_cal["eixo_x"] = df_cal["periodo"].apply(lambda p: p.month)
                x_label = "Mês"

            # Crimes
            pivot_c = df_cal.pivot_table(index="ano", columns="eixo_x", values="n_crimes", aggfunc="sum")
            fig_hm_c = px.imshow(
                pivot_c,
                color_continuous_scale="Reds",
                labels=dict(x=x_label, y="Ano", color="N° Crimes"),
                title="Volume de crimes por período",
                aspect="auto",
            )
            fig_hm_c.update_layout(height=280, margin=dict(t=40))
            st.plotly_chart(fig_hm_c, use_container_width=True)

            # Scores de polaridade
            for score_col in social_score_cols[:3]:
                if score_col not in df_cal.columns:
                    continue
                pivot_p = df_cal.pivot_table(index="ano", columns="eixo_x", values=score_col, aggfunc="mean")
                fig_hm_p = px.imshow(
                    pivot_p,
                    color_continuous_scale="RdYlGn_r",
                    labels=dict(x=x_label, y="Ano", color=score_col),
                    title=f"{score_col} — média por período",
                    aspect="auto",
                )
                fig_hm_p.update_layout(height=280, margin=dict(t=40))
                st.plotly_chart(fig_hm_p, use_container_width=True)

        except Exception as e:
            st.warning(f"Não foi possível construir o calendário: {e}")

    # Séries normalizadas sobrepostas
    st.divider()
    st.markdown("**Crimes e polaridade normalizados (0–1) — comparação visual direta**")
    st.caption(
        "Ambas as séries são escalonadas para 0–1 para comparação visual. "
        "A escala original é diferente entre elas."
    )

    df_norm = df_agg.copy()
    for col in ["n_crimes"] + social_score_cols:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[f"{col}_norm"] = (df_norm[col] - mn) / (mx - mn) if mx > mn else 0.5

    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(
        x=idx_str, y=df_norm["n_crimes_norm"].values,
        fill="tozeroy", fillcolor="rgba(230,57,70,0.12)",
        line=dict(color="#e63946", width=2.5),
        name="Crimes (norm.)",
    ))
    _pal = ["#4361ee", "#f77f00", "#06d6a0", "#9b5de5"]
    for i, col in enumerate(social_score_cols[:4]):
        fig_norm.add_trace(go.Scatter(
            x=idx_str, y=df_norm[f"{col}_norm"].values,
            line=dict(color=_pal[i % 4], width=1.8, dash="dot"),
            name=f"{col} (norm.)",
        ))
    fig_norm.update_layout(
        yaxis_title="Valor normalizado (0–1)",
        legend=dict(orientation="h", y=-0.3),
        height=350, hovermode="x unified", margin=dict(t=20),
    )
    st.plotly_chart(fig_norm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 — Mapa
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Mapa de Distribuição Geográfica dos Crimes")
    has_geo = lat_col != "(não disponível)" and lon_col != "(não disponível)"

    if not has_geo:
        st.info(
            "Selecione as colunas de **Latitude** e **Longitude** no mapeamento "
            "(seção 'Localização — opcional') para visualizar o mapa."
        )
    else:
        df_map = df_crime.copy()
        df_map["__lat__"] = pd.to_numeric(df_map[lat_col], errors="coerce")
        df_map["__lon__"] = pd.to_numeric(df_map[lon_col], errors="coerce")
        df_map["__date__"] = parse_date_series(df_map[crime_date_col])
        df_map = df_map.dropna(subset=["__lat__", "__lon__", "__date__"])

        if df_map.empty:
            st.warning("Nenhuma coordenada válida encontrada. Verifique as colunas selecionadas.")
        else:
            df_map["__ano__"] = df_map["__date__"].dt.year.astype(str)
            df_map["__mes__"] = df_map["__date__"].dt.to_period("M").astype(str)
            df_map["data_fmt"] = df_map["__date__"].dt.strftime("%d/%m/%Y")

            st.markdown(
                f"**{len(df_map):,} registros com coordenadas válidas** "
                f"(de {len(df_crime):,} totais)"
            )

            filtro_mes = st.selectbox(
                "Filtrar por período",
                ["Todos"] + sorted(df_map["__mes__"].unique().tolist()),
                key="map_filter",
            )
            df_mf = df_map if filtro_mes == "Todos" else df_map[df_map["__mes__"] == filtro_mes]

            # Scatter — pontos coloridos por ano
            # px.scatter_map (Plotly ≥6) substitui px.scatter_mapbox
            _scatter_fn = getattr(px, "scatter_map", None) or getattr(px, "scatter_mapbox")
            fig_scatter = _scatter_fn(
                df_mf,
                lat="__lat__", lon="__lon__",
                color="__ano__",
                hover_data={
                    "data_fmt": True, "__lat__": False,
                    "__lon__": False, "__ano__": False,
                },
                labels={"__ano__": "Ano", "data_fmt": "Data"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                zoom=6, height=520,
                title=f"Distribuição de crimes — {filtro_mes}",
            )
            _map_style = dict(map_style="open-street-map") if hasattr(px, "scatter_map") \
                         else dict(mapbox_style="open-street-map")
            fig_scatter.update_layout(**_map_style, margin=dict(t=40))
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Densidade
            st.markdown("**Mapa de Densidade — concentração geográfica**")
            _density_fn = getattr(px, "density_map", None) or getattr(px, "density_mapbox")
            fig_density = _density_fn(
                df_mf,
                lat="__lat__", lon="__lon__",
                radius=18, zoom=6, height=450,
                color_continuous_scale="Reds",
                title="Concentração de crimes (mais escuro = mais crimes)",
            )
            fig_density.update_layout(**_map_style, margin=dict(t=40))
            st.plotly_chart(fig_density, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 5 — Estacionariedade
# ══════════════════════════════════════════════════════════════════════════════
with tab_stat_tab:
    st.subheader("Testes de Estacionariedade")
    st.markdown("""
| Teste | H₀ | Rejeita H₀ se | Interpretação |
|---|---|---|---|
| **ADF** | Série é não-estacionária | p < 0,05 | Série estacionária |
| **KPSS** | Série é estacionária | p < 0,05 | Série não-estacionária |

A série é **estacionária** quando ADF rejeita H₀ **E** KPSS não rejeita H₀.
""")

    def _color_stat(val):
        if val == "SIM":
            return "background-color:#d4edda;color:#155724;font-weight:bold"
        if val == "NÃO":
            return "background-color:#f8d7da;color:#721c24;font-weight:bold"
        return ""

    st.dataframe(
        df_stat.style.map(_color_stat, subset=["Estacionária"]),
        use_container_width=True,
    )
    nao_estac = [r["Série"] for r in stat_rows if r["Estacionária"] == "NÃO"]
    if nao_estac and diferenciacao:
        st.info(
            f"Séries não-estacionárias detectadas: **{', '.join(nao_estac)}**.  \n"
            "Diferenciação (D1) aplicada no Granger e CCF."
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 6 — Correlação por Lag
# ══════════════════════════════════════════════════════════════════════════════
with tab_lag:
    st.subheader("Correlação por Defasagem Temporal")
    st.markdown(f"""
**Lag k** = correlação entre polaridade no período T e crimes em **T + k**.
Correlação positiva em lag k > 0 indica que maior polaridade precede mais crimes k períodos depois.

- **★** = significativo após Bonferroni (α_adj = {alpha_adj:.5f})
- **·** = significativo ao α bruto ({alpha_raw:.2f}) — menos confiável
- **Barra vermelha** = lag de correlação mais forte
""")

    all_lag_dfs = []
    for col in social_score_cols:
        df_lc = lag_results[col]
        if df_lc.empty:
            st.warning(f"Dados insuficientes para `{col}`.")
            continue

        st.markdown(f"---\n**`{col}`**")
        df_lc = df_lc.copy()
        df_lc["Sig"] = df_lc["Pearson_p"].apply(
            lambda p: sig_label(p, alpha_raw, alpha_adj)
        )
        df_lc["nome_serie"] = col
        all_lag_dfs.append(df_lc)

        best_idx = df_lc["Pearson_r"].abs().idxmax()
        bar_colors = ["#e63946" if i == best_idx else "#457b9d" for i in df_lc.index]
        ic_band = 1.96 / np.sqrt(df_lc["n"].max()) if not df_lc.empty else 0.15

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_lc["Lag"], y=df_lc["Pearson_r"],
            marker_color=bar_colors,
            text=df_lc["Sig"], textposition="outside",
            hovertemplate="Lag=%{x}<br>r=%{y:.4f}<extra></extra>",
        ))
        fig_bar.add_hline(y=ic_band, line_dash="dot", line_color="#aaa")
        fig_bar.add_hline(
            y=-ic_band, line_dash="dot", line_color="#aaa",
            annotation_text=f"±IC95%≈{ic_band:.3f}",
        )
        fig_bar.update_layout(
            xaxis_title=f"Lag ({granularity.lower()}s)",
            yaxis_title="Pearson r",
            height=280, margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        best = df_lc.loc[best_idx]
        st.caption(
            f"Melhor lag: {int(best.Lag)} | r = {best.Pearson_r:.4f} | "
            f"p = {best.Pearson_p:.4f} {best.Sig}"
        )
        st.dataframe(
            df_lc[["Lag", "Pearson_r", "Pearson_p", "Sig", "Spearman_r", "Spearman_p", "n"]]
            .rename(columns={"Pearson_r": "Pearson r", "Pearson_p": "Pearson p",
                             "Spearman_r": "Spearman r", "Spearman_p": "Spearman p",
                             "Sig": "Sig."}),
            use_container_width=True,
        )

    if all_lag_dfs:
        st.download_button(
            "⬇ Download correlações por lag (CSV)",
            pd.concat(all_lag_dfs, ignore_index=True).to_csv(index=False).encode("utf-8"),
            "correlacoes_lag.csv", "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 7 — CCF
# ══════════════════════════════════════════════════════════════════════════════
with tab_ccf:
    st.subheader("Função de Correlação Cruzada (CCF)")
    suffix_d = " [série diferenciada D1]" if diferenciacao else ""
    st.markdown(f"""
Lags **negativos** = polaridade vem *depois* dos crimes.
Lags **positivos** = polaridade vem *antes* dos crimes → indício de antecipação.
Barras **vermelhas** ultrapassam o IC 95% = ±1,96/√n.
Análise realizada{suffix_d}.
""")

    for col in social_score_cols:
        lags_range, ccf_vals, ic95 = ccf_results[col]
        n_used = int(
            pd.DataFrame({"c": crimes_series, "p": df_agg[col]}).dropna().shape[0]
        ) - (1 if diferenciacao else 0)

        bar_colors_ccf = ["#e63946" if abs(v) > ic95 else "#adb5bd" for v in ccf_vals]

        fig_ccf_plot = go.Figure()
        fig_ccf_plot.add_trace(go.Bar(
            x=lags_range, y=ccf_vals,
            marker_color=bar_colors_ccf,
            name=f"CCF {col}",
            hovertemplate="Lag=%{x}<br>r=%{y:.4f}<extra></extra>",
        ))
        fig_ccf_plot.add_hline(
            y=ic95, line_dash="dash", line_color="#2d6a4f",
            annotation_text=f"+IC95% ({ic95:.3f})",
        )
        fig_ccf_plot.add_hline(
            y=-ic95, line_dash="dash", line_color="#2d6a4f",
            annotation_text=f"-IC95% ({-ic95:.3f})",
        )
        fig_ccf_plot.update_layout(
            title=f"CCF: {col}{suffix_d} × n_crimes  (n={n_used})",
            xaxis_title=f"Lag ({granularity.lower()}s)",
            yaxis_title="Correlação",
            height=380, margin=dict(t=40),
        )
        st.plotly_chart(fig_ccf_plot, use_container_width=True)

        n_sig_ccf = sum(1 for v in ccf_vals if abs(v) > ic95)
        lag_max_ccf = lags_range[int(np.argmax(np.abs(ccf_vals)))]
        val_max_ccf = ccf_vals[int(np.argmax(np.abs(ccf_vals)))]
        st.caption(
            f"Lags fora do IC 95%: **{n_sig_ccf}** | "
            f"Lag mais alto: {lag_max_ccf} (r = {val_max_ccf:.4f})"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 8 — Granger
# ══════════════════════════════════════════════════════════════════════════════
with tab_gr:
    st.subheader("Causalidade de Granger")
    suffix_g = " [séries diferenciadas D1]" if diferenciacao else ""
    st.markdown(f"""
**H₀:** a polaridade **não** melhora a previsão dos crimes além do histórico autorregressivo.
p < {alpha_raw:.2f} → rejeita H₀ → polaridade tem poder preditivo incremental{suffix_g}.

- **★** = significativo após Bonferroni (α_adj = {alpha_g:.5f}, {n_scores * max_lag_granger} testes)
- **·** = significativo ao α bruto
""")

    all_gr_dfs = []
    for col in social_score_cols:
        df_g = granger_results[col]
        if df_g is None:
            st.warning(f"`{col}`: dados insuficientes para o Granger (aumente série ou reduza lags).")
            continue

        df_g = df_g.copy()
        df_g["Sig"] = df_g["p_valor"].apply(lambda p: sig_label(p, alpha_raw, alpha_g))
        df_g["Série"] = col
        all_gr_dfs.append(df_g)

        st.markdown(f"**`{col}` → crimes**")

        bar_g = ["#e63946" if p < alpha_raw else "#adb5bd" for p in df_g["p_valor"]]
        fig_g = go.Figure(go.Bar(
            x=df_g["Lag"], y=df_g["p_valor"],
            marker_color=bar_g,
            text=df_g["Sig"], textposition="outside",
            hovertemplate="Lag=%{x}<br>p=%{y:.4f}<extra></extra>",
        ))
        fig_g.add_hline(y=alpha_raw, line_dash="dot", line_color="#e63946",
                        annotation_text=f"α={alpha_raw:.2f}")
        if aplicar_bonferroni:
            fig_g.add_hline(y=alpha_g, line_dash="dot", line_color="#9b2226",
                            annotation_text=f"α Bonf={alpha_g:.4f}")
        fig_g.update_layout(
            xaxis_title="Lag", yaxis_title="p-valor (Granger F-test)",
            height=280, margin=dict(t=20),
        )
        st.plotly_chart(fig_g, use_container_width=True)

        min_p_row = df_g.loc[df_g["p_valor"].idxmin()]
        st.caption(
            f"Menor p-valor: Lag={int(min_p_row.Lag)}, "
            f"F={min_p_row.F_stat:.4f}, p={min_p_row.p_valor:.4f} {min_p_row.Sig}"
        )
        st.dataframe(
            df_g[["Lag", "F_stat", "p_valor", "Sig", "Série"]]
            .rename(columns={"F_stat": "F-stat", "p_valor": "p-valor", "Sig": "Sig."}),
            use_container_width=True,
        )

    if all_gr_dfs:
        st.download_button(
            "⬇ Download resultados Granger (CSV)",
            pd.concat(all_gr_dfs, ignore_index=True).to_csv(index=False).encode("utf-8"),
            "granger.csv", "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 9 — Correlação Parcial
# ══════════════════════════════════════════════════════════════════════════════
with tab_pc:
    st.subheader("Correlação Parcial (Frisch-Waugh)")
    st.markdown(f"""
Mede a associação entre polaridade e crimes **após descontar o efeito autorregressivo dos próprios crimes**
(lag-1 e média móvel de 3 períodos). Método: resíduos de OLS.

- **r parcial ≈ 0** → a relação é explicada pelo padrão temporal dos crimes
- **r parcial positivo e significativo** → a polaridade tem associação independente com os crimes
- **★** = p < {alpha_raw:.2f}
""")

    pc_rows = []
    for col in social_score_cols:
        r_pc, p_pc, n_pc = pc_results[col]
        if r_pc is None:
            st.warning(f"`{col}`: dados insuficientes (n={n_pc}).")
            continue
        pc_rows.append({
            "Série": col, "r parcial": r_pc, "p-valor": p_pc, "n": n_pc,
            "Sig.": "★" if p_pc < alpha_raw else "",
        })

    if pc_rows:
        df_pc_out = pd.DataFrame(pc_rows)
        fig_pc = go.Figure(go.Bar(
            y=df_pc_out["Série"], x=df_pc_out["r parcial"],
            orientation="h",
            marker_color=[
                "#e63946" if p < alpha_raw else "#adb5bd"
                for p in df_pc_out["p-valor"]
            ],
            text=[f"r={r:.4f}, p={p:.4f}" for r, p in
                  zip(df_pc_out["r parcial"], df_pc_out["p-valor"])],
            textposition="outside",
        ))
        fig_pc.add_vline(x=0, line_color="black", line_width=1)
        fig_pc.update_layout(
            xaxis_title="r parcial",
            height=max(250, 70 * len(pc_rows)),
            margin=dict(t=20),
        )
        st.plotly_chart(fig_pc, use_container_width=True)
        st.dataframe(df_pc_out, use_container_width=True)
        st.download_button(
            "⬇ Download correlações parciais (CSV)",
            df_pc_out.to_csv(index=False).encode("utf-8"),
            "correlacao_parcial.csv", "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 10 — Interpretação
# ══════════════════════════════════════════════════════════════════════════════
with tab_interp:
    st.subheader("Interpretação dos Resultados")
    st.markdown("Síntese automática dos achados em linguagem acessível para não especialistas.")

    # Métricas gerais
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Períodos analisados", f"{n_periodos}")
    c2.metric("Granularidade", granularity)
    c3.metric("Total de crimes", f"{int(crimes_series.sum()):,}")
    c4.metric("Média/período", f"{crimes_series.mean():.1f}")

    st.divider()
    st.markdown("### O que os dados dizem — por variável de polaridade")

    for col in social_score_cols:
        df_lc = lag_results.get(col, pd.DataFrame())
        lags_r, ccf_vals, ic95 = ccf_results.get(col, ([], [], 0.2))
        df_g = granger_results.get(col)
        r_pc, p_pc, n_pc = pc_results.get(col, (None, None, 0))

        with st.expander(f"🔍 {col}", expanded=True):
            if df_lc.empty:
                st.warning("Dados insuficientes para análise.")
                continue

            best_row = df_lc.loc[df_lc["Pearson_r"].abs().idxmax()]
            best_lag = int(best_row["Lag"])
            best_r = float(best_row["Pearson_r"])
            best_p = float(best_row["Pearson_p"])

            direction = "positiva" if best_r > 0 else "negativa"
            strength = (
                "fraca" if abs(best_r) < 0.2
                else ("moderada" if abs(best_r) < 0.4 else "forte")
            )
            lag_unit = "semanas" if granularity == "Semanal" else "meses"
            lag_text = (
                "no mesmo período (sem defasagem)"
                if best_lag == 0
                else f"{best_lag} {lag_unit} depois"
            )

            # Resultado Pearson
            if best_p < alpha_adj:
                sig_icon = "🔴"
                sig_msg = f"**significativa** (p={best_p:.4f} < Bonferroni {alpha_adj:.4f})"
            elif best_p < alpha_raw:
                sig_icon = "🟡"
                sig_msg = f"**marginalmente significativa** ao α bruto (p={best_p:.4f}), mas não após Bonferroni"
            else:
                sig_icon = "⚪"
                sig_msg = f"**não significativa** (p={best_p:.4f})"

            st.markdown(
                f"{sig_icon} **Correlação {direction} {strength}** entre `{col}` e crimes {lag_text}  \n"
                f"&nbsp;&nbsp;&nbsp;r = {best_r:.3f} — {sig_msg}"
            )

            if best_lag > 0:
                dir_text = "maior discurso tóxico/negativo" if best_r > 0 else "menor discurso tóxico"
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;→ Períodos com **{dir_text}** precedem períodos com "
                    f"**{'mais' if best_r > 0 else 'menos'} crimes** em ~{best_lag} {lag_unit}."
                )

            # Resultado Granger
            if df_g is not None and not df_g.empty:
                min_p_row = df_g.loc[df_g["p_valor"].idxmin()]
                gr_p = float(min_p_row["p_valor"])
                gr_lag = int(min_p_row["Lag"])
                if gr_p < alpha_adj:
                    st.markdown(
                        f"⚡ **Granger significativo** no lag {gr_lag} (p={gr_p:.4f} < {alpha_adj:.4f})  \n"
                        f"&nbsp;&nbsp;&nbsp;→ A polaridade **melhora a previsão** de crimes "
                        f"além do que o histórico dos próprios crimes já prevê."
                    )
                elif gr_p < alpha_raw:
                    st.markdown(
                        f"🟡 **Granger marginal** no lag {gr_lag} (p={gr_p:.4f})  \n"
                        f"&nbsp;&nbsp;&nbsp;→ Indício fraco de poder preditivo — não passa por Bonferroni."
                    )
                else:
                    st.markdown(
                        f"⚪ **Granger não significativo** (p mín={gr_p:.4f})  \n"
                        f"&nbsp;&nbsp;&nbsp;→ A polaridade não melhora a previsão além do histórico."
                    )
            else:
                st.markdown("⚪ Granger não computado (dados insuficientes para os lags configurados).")

            # Resultado correlação parcial
            if r_pc is not None:
                if p_pc < alpha_raw:
                    st.markdown(
                        f"🔗 **Correlação parcial significativa** (r = {r_pc:.3f}, p = {p_pc:.4f})  \n"
                        f"&nbsp;&nbsp;&nbsp;→ A associação **persiste mesmo controlando** a autocorrelação dos crimes."
                    )
                else:
                    st.markdown(
                        f"⚪ **Correlação parcial não significativa** (r = {r_pc:.3f}, p = {p_pc:.4f})  \n"
                        f"&nbsp;&nbsp;&nbsp;→ A associação pode ser explicada pelo padrão temporal dos crimes."
                    )

            # CCF lags positivos significativos
            if lags_r and ccf_vals:
                sig_pos = [lags_r[i] for i, v in enumerate(ccf_vals)
                           if abs(v) > ic95 and lags_r[i] > 0]
                sig_neg = [lags_r[i] for i, v in enumerate(ccf_vals)
                           if abs(v) > ic95 and lags_r[i] < 0]
                if sig_pos:
                    st.markdown(
                        f"〰️ **CCF: lags {sig_pos} acima do IC 95%**  \n"
                        f"&nbsp;&nbsp;&nbsp;→ A polaridade em T está correlacionada com crimes em T+{sig_pos[0]}."
                    )
                if sig_neg:
                    st.markdown(
                        f"〰️ **CCF: lags negativos {sig_neg} acima do IC 95%**  \n"
                        f"&nbsp;&nbsp;&nbsp;→ Os crimes em T estão correlacionados com polaridade em T+{abs(sig_neg[0])}: "
                        f"possível relação bidirecional."
                    )

    # Conclusão geral
    st.divider()
    st.markdown("### Conclusão geral")

    n_sig_bonf = sum(
        1 for col in social_score_cols
        if not lag_results.get(col, pd.DataFrame()).empty
        and lag_results[col]["Pearson_p"].min() < alpha_adj
    )
    n_sig_raw = sum(
        1 for col in social_score_cols
        if not lag_results.get(col, pd.DataFrame()).empty
        and lag_results[col]["Pearson_p"].min() < alpha_raw
    )
    n_sig_gr = sum(
        1 for col in social_score_cols
        if granger_results.get(col) is not None
        and not granger_results[col].empty
        and granger_results[col]["p_valor"].min() < alpha_raw
    )

    if n_sig_bonf > 0:
        st.success(
            f"**Evidência robusta:** {n_sig_bonf} de {n_scores} séries de polaridade apresentam "
            f"correlação temporal significativa após correção de Bonferroni. "
            f"Os dados são consistentes com uma associação entre o discurso online e os crimes registrados."
        )
    elif n_sig_raw > 0:
        st.warning(
            f"**Evidência sugestiva (fraca):** {n_sig_raw} de {n_scores} séries são significativas "
            f"ao α bruto, mas não após correção para múltiplos testes. "
            f"Os resultados são indicativos, mas devem ser interpretados com cautela — "
            f"podem ser falsos positivos."
        )
    else:
        st.info(
            f"**Sem evidência estatística clara:** nenhuma série apresentou correlação significativa "
            f"(α = {alpha_raw:.2f}) após {n_tests_total} testes. "
            f"Isso pode refletir ausência de associação ou limitação de tamanho amostral "
            f"(n = {n_periodos} períodos)."
        )

    if n_sig_gr > 0:
        st.success(
            f"**Granger:** {n_sig_gr} série(s) apresentam poder preditivo incremental ao α bruto."
        )

    st.markdown("""
---
**Lembrete metodológico importante:** Esta análise é **exploratória e correlacional**.
Resultados significativos indicam *associação temporal*, não *causalidade*.
Interprete sempre à luz do contexto, das limitações dos dados e de possíveis
variáveis de confusão (eventos externos, sazonalidade, subregistro criminal,
viés de cobertura das redes sociais).
""")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 11 — Manual
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Manual de uso")
    st.markdown("""
Esta ferramenta analisa se existe **associação temporal** entre indicadores de polaridade
ou toxicidade em redes sociais e o volume de crimes registrados num mesmo período.
Não exige conhecimento de programação — basta carregar dois arquivos CSV e mapear as colunas.
""")

    with st.expander("**Passo 1 — Prepare seus arquivos**", expanded=True):
        st.markdown("""
### Arquivo de crimes
Cada linha = um registro criminal ou uma pessoa envolvida. Necessário:
- **Coluna de data** da ocorrência (formatos aceitos: `15/01/2023`, `2023-01-15`, `15/01/2023 14:32`)
- Opcionalmente: **ID único** do evento (para deduplicar quando há múltiplas linhas por crime)
- Opcionalmente: **Latitude e Longitude** (para o mapa geográfico)

### Arquivo de redes sociais
Cada linha = um post ou comentário. Necessário:
- **Coluna de data** do conteúdo
- **Uma ou mais colunas numéricas** de score de polaridade, toxicidade ou sentimento
- Opcionalmente: **Coluna de texto** (para exibir os snippets coloridos)

> Separadores vírgula `,` e ponto-e-vírgula `;` são detectados automaticamente.
> Encodings UTF-8 e ISO-8859-1 (Windows) também são detectados automaticamente.
""")

    with st.expander("**Passo 2 — Mapeie as colunas**"):
        st.markdown("""
| Campo | O que selecionar |
|---|---|
| **Data dos crimes** | Coluna com data/hora de cada crime |
| **ID único do crime** | `(contar linhas)` se cada linha é um crime único; ID do evento se há múltiplas linhas por crime |
| **Latitude / Longitude** | Opcional — habilita o mapa geográfico |
| **Data dos posts** | Coluna com data/hora de cada post ou comentário |
| **Scores de polaridade** | Uma ou mais colunas numéricas (toxicidade, VADER, etc.) |
| **Texto do post** | Opcional — habilita os snippets coloridos por intensidade |
""")

    with st.expander("**O que cada aba mostra**"):
        st.markdown("""
| Aba | O que é |
|---|---|
| 📈 Série Temporal | Crimes (barras) e polaridade (linhas) ao longo do tempo |
| 📝 Textos & Picos | Textos mais e menos tóxicos + textos nos períodos de pico de crimes |
| 📅 Calendário | Heatmap de intensidade por período — tons mais escuros = valores mais altos |
| 🗺️ Mapa | Distribuição geográfica dos crimes (requer lat/lon) |
| 🧪 Estacionariedade | Se as séries têm comportamento estável (pré-requisito do Granger) |
| 📊 Correlação por Lag | Pearson/Spearman em múltiplos lags — polaridade precede crimes? |
| 〰️ CCF | Função de Correlação Cruzada com bandas IC 95% |
| ⚡ Granger | A polaridade melhora a previsão além do histórico dos crimes? |
| 🔗 Correlação Parcial | Associação após controlar autocorrelação dos crimes |
| 🔎 Interpretação | Resumo automático em linguagem acessível |
""")

    with st.expander("**Configurações da barra lateral**"):
        st.markdown("""
| Configuração | Recomendação |
|---|---|
| **Granularidade** | Semanal para > 2 anos de dados; Mensal para séries curtas |
| **Lags máximos** | 8 a 12 para séries semanais |
| **Lags Granger** | 4 a 6 para n < 100 períodos |
| **Correção de Bonferroni** | Manter ativada para evitar falsos positivos |
| **Diferenciar (D1)** | Manter ativado para séries não-estacionárias |
| **Nº de snippets** | 10 a 15 por categoria |
""")

    st.divider()
    st.warning("""
**Limitações importantes:**
- Correlação ≠ causalidade. Associações temporais podem ter causas comuns não medidas.
- Com menos de 50 períodos, os intervalos de confiança são largos e os resultados pouco confiáveis.
- Dados criminais oficiais representam apenas crimes registrados (subregistro é real).
- Scores de polaridade são médias agregadas — mascaram variações de conteúdo específico.
""")

# ─── Rodapé ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Ferramenta de pesquisa para análise exploratória de séries temporais. "
    "Resultados devem ser interpretados com cautela — associações estatísticas "
    "não implicam relação causal. Adequado para estudos exploratórios e qualificações acadêmicas."
)
