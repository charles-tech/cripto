import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ta.momentum import RSIIndicator
from datetime import timedelta
import numpy as np

# Configuração visual
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

data = yf.download(symbol, period=periodo)

if data.empty:
    st.error("Não foi possível carregar os dados.")
    st.stop()

close_series = pd.Series(data["Close"].values.flatten(), index=data.index)

# Indicadores técnicos
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()
rsi = RSIIndicator(close=close_series, window=14)
data["RSI"] = rsi.rsi()

# Didi Index
data["SMA3"] = data["Close"].rolling(window=3).mean()
data["SMA8"] = data["Close"].rolling(window=8).mean()
data["SMA20_DI"] = data["Close"].rolling(window=20).mean()

# Estratégia de sinais do Didi Index
data["Didi_buy"] = (
    (data["SMA3"] > data["SMA8"]) & (data["SMA8"] < data["SMA20_DI"]) & 
    (data["SMA3"].shift(1) <= data["SMA8"].shift(1))
)
data["Didi_sell"] = (
    (data["SMA3"] < data["SMA8"]) & (data["SMA8"] > data["SMA20_DI"]) &
    (data["SMA3"].shift(1) >= data["SMA8"].shift(1))
)

# Gráfico de preço + médias móveis + sinais Didi
with st.expander("📈 Visualizar gráfico de preço, médias móveis e estratégia Didi", expanded=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["Close"], label="Fechamento", color="royalblue", lw=1.4)
    ax.plot(data["SMA20"], label="Média Móvel 20d", color="green", ls="--", lw=1)
    ax.plot(data["SMA50"], label="Média Móvel 50d", color="orange", ls=":", lw=1)

    # Sinais de compra e venda
    buy_signals = data.loc[data["Didi_buy"]]
    sell_signals = data.loc[data["Didi_sell"]]
    ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="lime", s=80, label="Didi BUY", zorder=5)
    ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", s=80, label="Didi SELL", zorder=5)

    ax.set_ylabel("Preço (USD)")
    ax.set_xlabel("Data")
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(ncol=4, fontsize=8, loc="upper left", frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "**Como funciona:** Setas verdes (^) mostram sinal de compra (Didi Index). Setas vermelhas (v) mostram sinal de venda."
    )

# Gráfico RSI
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

# Gráfico Didi Index (curvas das médias)
with st.expander("🪡 Visualizar Didi Index (agulhada do Didi)", expanded=False):
    fig_didi, axdidi = plt.subplots(figsize=(7, 2.5))
    axdidi.plot(data["SMA3"], label="SMA 3", color="red", linewidth=1.2)
    axdidi.plot(data["SMA8"], label="SMA 8", color="blue", linewidth=1.2)
    axdidi.plot(data["SMA20_DI"], label="SMA 20", color="green", linewidth=1.2)
    axdidi.set_title("Didi Index - Médias Móveis (3, 8, 20)")
    axdidi.set_ylabel("Valor")
    axdidi.set_xlabel("Data")
    axdidi.spines[['top', 'right']].set_visible(False)
    axdidi.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_didi)
    st.markdown(
        "Quando as curvas se cruzam ou ficam muito próximas, pode indicar reversão ('agulhada').\n"
        "SMA 3 cruzando as demais para cima = sinal de compra; cruzando para baixo = venda."
    )

# Previsão futura
st.subheader("🔮 Previsão simplificada para os próximos 10 dias")
last_close = data["Close"].iloc[-1]
mean_return = data["Close"].pct_change().tail(10).mean()
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 11)]
future_prices = [float(last_close) * (1 + float(mean_return)) ** i for i in range(1, 11)]

future_df = pd.DataFrame({
    "Data": future_dates,
    "Preço Previsto": future_prices
})

# ----- Tabela de previsão final -----
with st.expander("Ver tabela de previsão ±", expanded=False):
    df_fmt = future_df.copy()
    df_fmt["Preço Previsto"] = pd.to_numeric(df_fmt["Preço Previsto"], errors="coerce")
    df_fmt["Preço Previsto (USD)"] = df_fmt["Preço Previsto"].apply(
        lambda x: f"{x:.2f}" if pd.notnull(x) else "-"
    )
    st.table(df_fmt[["Data", "Preço Previsto (USD)"]])
# ----- fim do bloco tabela formatada -----

# Gráfico previsão junto com histórico
fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.plot(data.index, data["Close"], label="Fechamento Histórico", color="royalblue", lw=1.4)
ax3.plot(future_df["Data"], future_df["Preço Previsto"], label="Previsto (10 dias)", color="orange", linestyle="--", marker="o")
ax3.set_ylabel("Preço (USD)")
ax3.set_xlabel("Data")
ax3.spines[['top', 'right']].set_visible(False)
ax3.legend(ncol=2, fontsize=8, loc="upper left", frameon=True)
plt.tight_layout()
st.pyplot(fig3)

st.info("A previsão acima é PURAMENTE INDICATIVA, baseada em retorno médio recente. Não utilize como conselho de investimento.")
