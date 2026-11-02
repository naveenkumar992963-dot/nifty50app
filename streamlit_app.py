
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY 50 Advanced Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 ADVANCED TRADING DASHBOARD")
st.caption("5-minute technical analysis | Paper trading only")

st.warning(
    "Educational tool. Signals are not guaranteed. "
    "Verify data and option prices before trading."
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

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()

if st.button("🔄 Refresh Market Data"):
    st.cache_data.clear()

try:
    df = get_data()

    if df.empty or len(df) < 50:
        st.error("Insufficient market data. Try again later.")
        st.stop()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    last = df.iloc[-1]

    price = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi = float(last["RSI"])
    atr = float(last["ATR"])
    macd = float(last["MACD"])
    macd_signal = float(last["MACD_Signal"])

    if not np.isfinite([price, ema9, ema21, rsi, atr, macd, macd_signal]).all() or atr <= 0:
        st.error("Indicator data is incomplete. Refresh later.")
        st.stop()

    bullish = (
        ema9 > ema21
        and 50 < rsi < 70
        and macd > macd_signal
    )

    bearish = (
        ema9 < ema21
        and 30 < rsi < 50
        and macd < macd_signal
    )

    st.subheader("📊 Market Overview")

    a, b, c, d = st.columns(4)

    a.metric("NIFTY 50", f"{price:,.2f}")
    b.metric("RSI", f"{rsi:.2f}")
    c.metric("ATR", f"{atr:.2f}")
    d.metric("MACD", f"{macd:.2f}")

    st.divider()

    st.subheader("🎯 Trading Setup")

    if bullish:
        signal = "BULLISH"
        direction = "CALL BIAS"
        entry = price
        sl = price - atr * 1.2
        risk = entry - sl

    elif bearish:
        signal = "BEARISH"
        direction = "PUT BIAS"
        entry = price
        sl = price + atr * 1.2
        risk = sl - entry

    else:
        signal = "WAIT"
        direction = "NO TRADE"
        entry = sl = risk = None

    if signal == "BULLISH":
        st.success("🟢 BULLISH - CALL BIAS")
    elif signal == "BEARISH":
        st.error("🔴 BEARISH - PUT BIAS")
    else:
        st.info("🟡 WAIT - No confirmed setup")

    if signal != "WAIT":
        t1 = entry + risk * 1.5 if signal == "BULLISH" else entry - risk * 1.5
        t2 = entry + risk * 2 if signal == "BULLISH" else entry - risk * 2
        t3 = entry + risk * 3 if signal == "BULLISH" else entry - risk * 3

        x, y, z = st.columns(3)
        x.metric("Possible Entry", f"{entry:,.2f}")
        y.metric("NIFTY Spot SL", f"{sl:,.2f}")
        z.metric("Direction", direction)

        x, y, z = st.columns(3)
        x.metric("Target 1", f"{t1:,.2f}")
        y.metric("Target 2", f"{t2:,.2f}")
        z.metric("Target 3", f"{t3:,.2f}")

        st.caption(
            "Targets and SL are calculated on NIFTY spot, "
            "not option premium."
        )

    st.divider()

    st.subheader("📉 Technical Indicators")

    x, y, z = st.columns(3)
    x.metric("EMA 9", f"{ema9:,.2f}")
    y.metric("EMA 21", f"{ema21:,.2f}")
    z.metric("MACD Signal", f"{macd_signal:.2f}")

    st.line_chart(df[["Close", "EMA9", "EMA21"]])

    st.subheader("📋 Latest Market Data")
    st.dataframe(df.tail(10), use_container_width=True)

    st.caption(
        f"Last candle timestamp: {df.index[-1]} | "
        "Source: Yahoo Finance | "
        "Data may be delayed or incomplete."
    )

except Exception as e:
    st.error(f"Data error: {e}")
    st.info("Please refresh and try again.")
    
