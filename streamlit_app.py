
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY 50 Trading Signals",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 TRADING SIGNALS")
st.caption("Technical analysis dashboard | 5-minute timeframe")

st.warning(
    "Educational tool only. Signals are not guaranteed. "
    "Verify market data and risk before trading."
)

@st.cache_data(ttl=60)
def get_data():
    df = yf.download(
        "^NSEI",
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    return df.dropna()

if st.button("🔄 Refresh Market Data"):
    st.cache_data.clear()

try:
    df = get_data()

    if df.empty or len(df) < 30:
        st.error("Market data unavailable. Please try again.")
        st.stop()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    last = df.iloc[-1]

    price = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi = float(last["RSI"])
    atr = float(last["ATR"])

    st.subheader("📊 Market Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("NIFTY 50", f"{price:,.2f}")
    c2.metric("RSI (14)", f"{rsi:.2f}")
    c3.metric("ATR (14)", f"{atr:.2f}")

    st.divider()

    st.subheader("🎯 Trading Signal")

    if ema9 > ema21 and 50 < rsi < 70:
        signal = "BUY"
        sl = price - (1.2 * atr)
        risk = price - sl
        t1 = price + (1.5 * risk)
        t2 = price + (2 * risk)

    elif ema9 < ema21 and 30 < rsi < 50:
        signal = "SELL"
        sl = price + (1.2 * atr)
        risk = sl - price
        t1 = price - (1.5 * risk)
        t2 = price - (2 * risk)

    else:
        signal = "WAIT"
        sl = t1 = t2 = None

    if signal == "BUY":
        st.success("🟢 BUY SIGNAL")
    elif signal == "SELL":
        st.error("🔴 SELL SIGNAL")
    else:
        st.info("🟡 WAIT - No clear setup")

    if signal != "WAIT":
        a, b = st.columns(2)

        a.metric("Entry Price", f"{price:,.2f}")
        b.metric("Stop Loss", f"{sl:,.2f}")

        a.metric("Target 1", f"{t1:,.2f}")
        b.metric("Target 2", f"{t2:,.2f}")

    st.divider()

    st.subheader("📉 Technical Indicators")

    x, y = st.columns(2)

    x.metric("EMA 9", f"{ema9:,.2f}")
    y.metric("EMA 21", f"{ema21:,.2f}")

    st.line_chart(df[["Close", "EMA9", "EMA21"]])

    st.subheader("📋 Latest Market Data")
    st.dataframe(df.tail(10), use_container_width=True)

    st.caption(
        "Data source: Yahoo Finance | "
        "5-minute data may be delayed or unavailable."
    )

except Exception as e:
    st.error(f"Data error: {e}")
    st.info("Please refresh and try again.")
    
