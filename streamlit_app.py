
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY Advanced Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 ADVANCED SCANNER")
st.caption("5M + 15M | Strength Meter | Paper Trading")

st.warning(
    "Educational tool only. Scores are rule-based indicators, "
    "not probabilities of profit or guaranteed trading signals."
)

@st.cache_data(ttl=60)
def get_data(interval):
    df = yf.download(
        "^NSEI",
        period="5d",
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna().copy()


def analyze(interval):
    df = get_data(interval)

    if df.empty or len(df) < 50:
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

    row = df.iloc[-1]

    vals = [
        row["Close"], row["EMA9"], row["EMA21"],
        row["RSI"], row["MACD"], row["MACD_SIGNAL"],
        row["ATR"]
    ]

    if not np.isfinite(vals).all() or row["ATR"] <= 0:
        return None

    bull = 0
    bear = 0

    if row["EMA9"] > row["EMA21"]:
        bull += 40
    elif row["EMA9"] < row["EMA21"]:
        bear += 40

    if row["RSI"] > 55:
        bull += 30
    elif row["RSI"] < 45:
        bear += 30
    else:
        bull += 10
        bear += 10

    if row["MACD"] > row["MACD_SIGNAL"]:
        bull += 30
    elif row["MACD"] < row["MACD_SIGNAL"]:
        bear += 30

    signal = (
        "CALL BIAS" if bull >= 70
        else "PUT BIAS" if bear >= 70
        else "WAIT"
    )

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
        "df": df,
        "price": price,
        "bull": bull,
        "bear": bear,
        "signal": signal,
        "rsi": float(row["RSI"]),
        "atr": atr,
        "sl": sl,
        "targets": targets,
        "time": df.index[-1]
    }


if st.button("🔄 Refresh Scanner"):
    st.cache_data.clear()

try:
    with st.spinner("Analyzing 5M and 15M market data..."):
        r5 = analyze("5m")
        r15 = analyze("15m")

    if r5 is None or r15 is None:
        st.error("Market data unavailable or insufficient.")
        st.stop()

    st.subheader("📊 Timeframe Signals")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 5 MINUTE")
        st.metric("NIFTY Spot", f'{r5["price"]:,.2f}')
        st.metric("Signal", r5["signal"])
        st.metric("Bullish Score", f'{r5["bull"]}%')
        st.progress(r5["bull"] / 100)
        st.metric("Bearish Score", f'{r5["bear"]}%')
        st.progress(r5["bear"] / 100)
        st.metric("RSI", f'{r5["rsi"]:.2f}')

    with c2:
        st.markdown("### 15 MINUTE")
        st.metric("NIFTY Spot", f'{r15["price"]:,.2f}')
        st.metric("Signal", r15["signal"])
        st.metric("Bullish Score", f'{r15["bull"]}%')
        st.progress(r15["bull"] / 100)
        st.metric("Bearish Score", f'{r15["bear"]}%')
        st.progress(r15["bear"] / 100)
        st.metric("RSI", f'{r15["rsi"]:.2f}')

    st.divider()
    st.subheader("🎯 FINAL CONFIRMATION")

    if r5["signal"] == r15["signal"] and r5["signal"] != "WAIT":
        final = r5["signal"]
        st.success(f"CONFIRMED DIRECTION: {final}")
        st.metric("Confirmation Meter", "2 / 2")
        
        st.metric("NIFTY Spot Entry Reference", f'{r5["price"]:,.2f}')
        st.metric("NIFTY Spot Stop Loss", f'{r5["sl"]:,.2f}')

        a, b, c = st.columns(3)
        a.metric("Target 1", f'{r5["targets"][0]:,.2f}')
        b.metric("Target 2", f'{r5["targets"][1]:,.2f}')
        c.metric("Target 3", f'{r5["targets"][2]:,.2f}')

    else:
        st.warning("WAIT - No confirmed setup")
        st.metric(
            "Confirmation Meter",
            f'{int(r5["signal"] != "WAIT") + int(r15["signal"] != "WAIT")} / 2'
        )

    st.caption(
        "Confirmation means both rule-based signals agree. "
        "It does not mean a 100% probability of success."
    )

    st.info(
        "Entry, SL and targets are NIFTY SPOT reference levels, "
        "not CALL/PUT option premium levels."
    )

    st.divider()

    st.subheader("📈 5M Technical Chart")
    st.line_chart(r5["df"][["Close", "EMA9", "EMA21"]])

    st.subheader("📈 15M Technical Chart")
    st.line_chart(r15["df"][["Close", "EMA9", "EMA21"]])

    st.subheader("🕒 Data Timestamps")
    st.write("5M:", r5["time"])
    st.write("15M:", r15["time"])

    st.caption(
        "Source: Yahoo Finance. Data may be delayed. "
        "Use paper trading and verify completed candles."
    )

except Exception as e:
    st.error(f"Scanner error: {e}")
        
