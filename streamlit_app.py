import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(
    page_title="NIFTY 50 Trading Levels",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 Trading Dashboard")
st.caption("Entry Price | Stop Loss | Target 1 | Target 2 | Target 3")

# Fetch NIFTY 50 data
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
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["High", "Low", "Close"]).copy()

    # ATR calculation
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    df["TR"] = pd.concat(
        [tr1, tr2, tr3], axis=1
    ).max(axis=1)

    df["ATR"] = df["TR"].rolling(14).mean()

    # EMA indicators
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()

    return df.dropna()


try:
    df = get_data()

    if df is None or df.empty:
        st.error("Market data nahi mil raha. Thodi der baad try karein.")
        st.stop()

    price = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])

    ema9 = float(df["EMA9"].iloc[-1])
    ema21 = float(df["EMA21"].iloc[-1])

    if atr <= 0:
        st.error("ATR calculate nahi ho pa raha.")
        st.stop()

    st.success("NIFTY 50 Data Loaded")

    # Market information
    c1, c2, c3 = st.columns(3)

    c1.metric("NIFTY 50 Price", f"{price:,.2f}")
    c2.metric("ATR (14)", f"{atr:.2f}")
    c3.metric("EMA 9 / EMA 21",
              "Bullish" if ema9 > ema21 else "Bearish")

    st.divider()

    st.subheader("🎯 Trading Setup")

    direction = st.selectbox(
        "Trade Direction",
        ["BUY / LONG", "SELL / SHORT"]
    )

    entry = st.number_input(
        "Entry Price",
        min_value=0.01,
        value=round(price, 2),
        step=0.05
    )

    multiplier = st.slider(
        "Stop Loss ATR Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5
    )

    risk = atr * multiplier

    if direction == "BUY / LONG":

        sl = entry - risk
        target1 = entry + risk
        target2 = entry + (risk * 2)
        target3 = entry + (risk * 3)

    else:

        sl = entry + risk
        target1 = entry - risk
        target2 = entry - (risk * 2)
        target3 = entry - (risk * 3)

    st.divider()

    st.subheader("📊 Your Trading Levels")

    a, b = st.columns(2)

    a.metric("Entry Price", f"{entry:,.2f}")
    b.metric("Stop Loss", f"{sl:,.2f}")

    st.divider()

    t1, t2, t3 = st.columns(3)

    t1.metric("🎯 TARGET 1", f"{target1:,.2f}")
    t2.metric("🎯 TARGET 2", f"{target2:,.2f}")
    t3.metric("🎯 TARGET 3", f"{target3:,.2f}")

    st.divider()

    st.subheader("📈 Market Trend")

    if ema9 > ema21:
        st.success("EMA Trend: BULLISH")
    elif ema9 < ema21:
        st.error("EMA Trend: BEARISH")
    else:
        st.info("EMA Trend: NEUTRAL")

    st.subheader("Recent NIFTY 50 Candles")

    st.line_chart(df["Close"].tail(50))

    st.caption(
        "Data source: Yahoo Finance. "
        "5-minute data may be delayed or unavailable. "
        "Levels are ATR-based estimates, not guaranteed support/resistance."
    )

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please refresh the app and try again.")
