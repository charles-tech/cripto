import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from ta.momentum import RSIIndicator
from datetime import timedelta
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.size': 10})

st.set_page_config(page_title="Análise Cripto: Solana (SOL)", layout="centered")
st.title("🔮 Análise e Previsão de Solana (SOL)")

with st.sidebar:
    st.header("Configuração")
    symbol = st.text_input("Símbolo do ativo (ex: SOL-USD)", value="SOL-USD")
    periodo = st.selectbox(
        "Período",
        options=["1y", "6mo", "3mo", "1mo"],
        index=0,
        format_func=lambda x: {"1y": "1 ano", "6mo": "6 meses", "3mo": "3 meses", "1mo": "1 mês"}[x]
    )
    n_dias_previsao = st.number_input(
        "Dias para previsão",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
        help="Quantidade de dias a prever para frente."
    )

data = yf.download(symbol, period=periodo)

if data.empty:
    st.error("Não foi possível carregar os dados.")
    st.stop()

# --- AJUSTE PARA COLUNAS E TIPOS ---

# Desfaz MultiIndex caso exista
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Checagem e tratamento de colunas essenciais
ohlc_cols = ["Open", "High", "Low", "Close", "Volume"]
missing = [col for col in ohlc_cols if col not in data.columns]
if missing:
    st.error(f"As colunas {missing} não estão presentes no dataset baixado. Não é possível prosseguir.")
    st.stop()

for col in ohlc_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=ohlc_cols).copy()

# --- FIM DO BLOCO DE LIMPEZA ---

close_series = pd.Series(data["Close"].values.flatten(), index=data.index)

# Médias móveis do Didi Index
data["SMA8"] = data["Close"].rolling(window=8).mean()
data["SMA34"] = data["Close"].rolling(window=34).mean()
data["SMA144"] = data["Close"].rolling(window=144).mean()

# Médias móveis tradicionais (opcional, para adicionar no gráfico principal)
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()

# Gráfico CANDLE com painel auxiliar do Didi Index
with st.expander("📊 Candlestick + Didi Index Auxiliar", expanded=True):
    apds = [
        mpf.make_addplot(data["SMA20"], color='g', width=1, panel=0, linestyle='dashed', ylabel="SMA20"),
        mpf.make_addplot(data["SMA50"], color='orange', width=1, panel=0, linestyle='dotted', ylabel="SMA50"),
        mpf.make_addplot(data["SMA8"], color='gold', width=1.2, panel=1, ylabel="SMA8"),
        mpf.make_addplot(data["SMA34"], color='red', width=1.2, panel=1, ylabel="SMA34"),
        mpf.make_addplot(data["SMA144"], color='blue', width=1.2, panel=1, ylabel="SMA144"),
    ]

    fig, axes = mpf.plot(
        data,
        type="candle",
        style="yahoo",
        addplot=apds,
        volume=False,
        returnfig=True,
        figsize=(9,6),
        title=f"{symbol} - Candlestick e Didi Index (8/34/144)",
        panel_ratios=(2,1)
    )
    st.pyplot(fig)
    st.markdown(
        "Painel superior: candles e médias móveis tradicionais (20, 50).\n"
        "Painel inferior: médias do Didi Index (8, 34, 144).\n\n"
        "Observe cruzamentos e aproximações entre as linhas do painel auxiliar!"
    )

# Se quiser manter o RSI ainda disponível…
data["RSI"] = RSIIndicator(close=close_series, window=14).rsi()
with st.expander("📊 Visualizar RSI", expanded=False):
    fig2, ax2 = plt.subplots(figsize=(7, 1.8))
    ax2.plot(data["RSI"], label="RSI", color="purple")
    ax2.axhline(70, linestyle="--", color="red", alpha=0.7, lw=0.8)
    ax2.axhline(30, linestyle="--", color="green", alpha=0.7, lw=0.8)
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 30, 70, 100])
    ax2.set_xlabel("Data")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

# Previsão futura baseada nos últimos retornos médios
st.subheader(f"🔮 Previsão simplificada para os próximos {n_dias_previsao} dias")
try:
    last_close = data["Close"].iloc[-1]
    mean_return = data["Close"].pct_change().tail(10).mean()
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, int(n_dias_previsao)+1)]
    future_prices = [float(last_close) * (1 + float(mean_return)) ** i for i in range(1, int(n_dias_previsao)+1)]

    future_df = pd.DataFrame({
        "Data": future_dates,
        "Preço Previsto": future_prices
    })

    with st.expander("Ver tabela de previsão ±", expanded=False):
        df_fmt = future_df.copy()
        df_fmt["Preço Previsto"] = pd.to_numeric(df_fmt["Preço Previsto"], errors="coerce")
        df_fmt["Preço Previsto (USD)"] = df_fmt["Preço Previsto"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "-"
        )
        st.table(df_fmt[["Data", "Preço Previsto (USD)"]])

    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(data.index, data["Close"], label="Fechamento Histórico", color="royalblue", lw=1.4)
    ax3.plot(future_df["Data"], future_df["Preço Previsto"], label=f"Previsto ({n_dias_previsao} dias)", color="orange", linestyle="--", marker="o")
    ax3.set_ylabel("Preço (USD)")
    ax3.set_xlabel("Data")
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.legend(ncol=2, fontsize=8, loc="upper left", frameon=True)
    plt.tight_layout()
    st.pyplot(fig3)

    st.info("A previsão acima é PURAMENTE INDICATIVA, baseada em retorno médio recente. Não utilize como conselho de investimento.")
except Exception as e:
    st.warning(f"Não foi possível gerar previsão: {e}")
