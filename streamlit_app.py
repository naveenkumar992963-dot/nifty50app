
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY Options Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 OPTIONS SCANNER")
st.caption("5M + 15M | CALL / PUT | Paper Trading")

st.warning(
    "Educational tool only. CALL/PUT bias is not an option "
    "premium signal. Verify live prices before trading."
)

@st.cache_data(ttl=60)
def get_data(interval):
    return yf.download(
        "^NSEI",
        period="5d",
        interval=interval,
        progress=False,
        auto_adjust=False
    )

def prepare(df):
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    if len(df) < 50:
        return None

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
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    return df

def analyze(df):
    if df is None:
        return None

    row = df.iloc[-1]

    values = [
        row["Close"], row["EMA9"], row["EMA21"],
        row["RSI"], row["MACD"], row["MACD_SIGNAL"],
        row["ATR"]
    ]

    if not np.isfinite(values).all() or row["ATR"] <= 0:
        return None

    bullish = (
        row["EMA9"] > row["EMA21"]
        and row["RSI"] > 50
        and row["MACD"] > row["MACD_SIGNAL"]
    )

    bearish = (
        row["EMA9"] < row["EMA21"]
        and row["RSI"] < 50
        and row["MACD"] < row["MACD_SIGNAL"]
    )

    signal = "CALL BIAS" if bullish else "PUT BIAS" if bearish else "WAIT"

    price = float(row["Close"])
    atr = float(row["ATR"])

    if signal == "CALL BIAS":
        sl = price - 1.2 * atr
        risk = price - sl
        targets = [price + risk * n for n in (1.5, 2, 3)]
    elif signal == "PUT BIAS":
        sl = price + 1.2 * atr
        risk = sl - price
        targets = [price - risk * n for n in (1.5, 2, 3)]
    else:
        sl = None
        targets = [None, None, None]

    return {
        "signal": signal,
        "price": price,
        "rsi": float(row["RSI"]),
        "ema9": float(row["EMA9"]),
        "ema21": float(row["EMA21"]),
        "macd": float(row["MACD"]),
        "atr": atr,
        "sl": sl,
        "targets": targets,
        "time": df.index[-1]
    }

if st.button("🔄 Refresh Both Timeframes"):
    st.cache_data.clear()

try:
    with st.spinner("Loading NIFTY market data..."):
        df5 = prepare(get_data("5m"))
        df15 = prepare(get_data("15m"))

        result5 = analyze(df5)
        result15 = analyze(df15)

    if result5 is None or result15 is None:
        st.error("Data unavailable or insufficient. Try again later.")
        st.stop()

    st.subheader("📊 Timeframe Comparison")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 5 MINUTE")
        st.metric("NIFTY Spot", f'{result5["price"]:,.2f}')
        st.metric("Signal", result5["signal"])
        st.metric("RSI", f'{result5["rsi"]:.2f}')

    with c2:
        st.markdown("### 15 MINUTE")
        st.metric("NIFTY Spot", f'{result15["price"]:,.2f}')
        st.metric("Signal", result15["signal"])
        st.metric("RSI", f'{result15["rsi"]:.2f}')

    st.divider()

    st.subheader("🎯 FINAL TRADE FILTER")

    if result5["signal"] == result15["signal"] and result5["signal"] != "WAIT":
        final_signal = result5["signal"]
        st.success(f"CONFIRMED DIRECTION: {final_signal}")
        st.caption("Both timeframes agree. This is a paper-trading setup, not a trade instruction.")

        st.metric("Possible NIFTY Spot Entry", f'{result5["price"]:,.2f}')
        st.metric("NIFTY Spot Stop Loss", f'{result5["sl"]:,.2f}')

        a, b, c = st.columns(3)

        a.metric("Target 1", f'{result5["targets"][0]:,.2f}')
        b.metric("Target 2", f'{result5["targets"][1]:,.2f}')
        c.metric("Target 3", f'{result5["targets"][2]:,.2f}')

        st.info("These levels refer to NIFTY spot, NOT option premium.")
    else:
        st.warning("WAIT - No confirmed setup. Timeframes do not agree.")

    st.divider()

    st.subheader("📈 Technical Indicators")

    st.write("5-minute chart")
    st.line_chart(df5[["Close", "EMA9", "EMA21"]])

    st.write("15-minute chart")
    st.line_chart(df15[["Close", "EMA9", "EMA21"]])

    st.subheader("📋 Latest Data")
    st.write("5-minute candle:", result5["time"])
    st.write("15-minute candle:", result15["time"])

    st.caption(
        "Source: Yahoo Finance. Data can be delayed or unavailable. "
        "No live option-chain or option-premium data is included."
    )

except Exception as e:
    st.error(f"Data error: {e}")
    st.info("Refresh and try again later.")
    

    
