
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="NIFTY 50 AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 AI Trading Dashboard")
st.caption("EMA + RSI + MACD + Supertrend | 5m / 15m")

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Dashboard Controls")

    timeframe = st.selectbox(
        "Select Timeframe",
        ["5m", "15m"]
    )

    capital = st.number_input(
        "Your Trading Capital (₹)",
        min_value=1000,
        value=10000,
        step=1000
    )

    risk_percent = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.5
    )

    refresh = st.button(
        "🔄 REFRESH DATA",
        use_container_width=True
    )

    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Data source: Yahoo Finance")
    st.caption("Data may be delayed.")

# ---------------- DATA ----------------

@st.cache_data(ttl=60)
def get_data(interval):
    period = "5d" if interval == "5m" else "1mo"

    df = yf.download(
        "^NSEI",
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df

# ---------------- INDICATORS ----------------

def calculate_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # Supertrend
    hl2 = (high + low) / 2
    upper = hl2 + (3 * df["ATR"])
    lower = hl2 - (3 * df["ATR"])

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
            continue

        previous_st = supertrend.iloc[i - 1]
        previous_direction = direction.iloc[i - 1]

        if close.iloc[i] > previous_st:
            direction.iloc[i] = 1
            supertrend.iloc[i] = max(
                lower.iloc[i],
                previous_st
            )
        elif close.iloc[i] < previous_st:
            direction.iloc[i] = -1
            supertrend.iloc[i] = min(
                upper.iloc[i],
                previous_st
            )
        else:
            direction.iloc[i] = previous_direction
            supertrend.iloc[i] = previous_st

    df["Supertrend"] = supertrend
    df["ST_Direction"] = direction

    return df.dropna()

# ---------------- STRATEGY ----------------

def make_signal(df):
    last = df.iloc[-1]

    price = float(last["Close"])
    atr = float(last["ATR"])

    bullish = (
        last["EMA9"] > last["EMA21"]
        and last["EMA21"] > last["EMA50"]
        and last["RSI"] > 50
        and last["RSI"] < 70
        and last["MACD"] > last["MACD_Signal"]
        and last["ST_Direction"] == 1
    )

    bearish = (
        last["EMA9"] < last["EMA21"]
        and last["EMA21"] < last["EMA50"]
        and last["RSI"] < 50
        and last["RSI"] > 30
        and last["MACD"] < last["MACD_Signal"]
        and last["ST_Direction"] == -1
    )

    if atr <= 0 or not np.isfinite(atr):
        return {
            "signal": "NO TRADE",
            "entry": price,
            "sl": None,
            "t1": None,
            "t2": None,
            "t3": None,
            "reasons": ["ATR data unavailable"]
        }

    if bullish:
        sl = price - (1.5 * atr)
        risk = price - sl

        return {
            "signal": "BUY",
            "entry": price,
            "sl": sl,
            "t1": price + risk,
            "t2": price + (2 * risk),
            "t3": price + (3 * risk),
            "reasons": [
                "EMA 9 above EMA 21 and EMA 50",
                "RSI confirms bullish momentum",
                "MACD bullish crossover alignment",
                "Supertrend bullish"
            ]
        }

    if bearish:
        sl = price + (1.5 * atr)
        risk = sl - price

        return {
            "signal": "SELL",
            "entry": price,
            "sl": sl,
            "t1": price - risk,
            "t2": price - (2 * risk),
            "t3": price - (3 * risk),
            "reasons": [
                "EMA 9 below EMA 21 and EMA 50",
                "RSI confirms bearish momentum",
                "MACD bearish alignment",
                "Supertrend bearish"
            ]
        }

    return {
        "signal": "NO TRADE",
        "entry": price,
        "sl": None,
        "t1": None,
        "t2": None,
        "t3": None,
        "reasons": [
            "All strategy conditions are not aligned",
            "Wait for trend confirmation"
        ]
    }

# ---------------- LOAD DATA ----------------

try:
    raw = get_data(timeframe)

    if raw.empty or len(raw) < 60:
        st.error(
            "Market data unavailable or insufficient. "
            "Please refresh after some time."
        )
        st.stop()

    df = calculate_indicators(raw)

    if df.empty:
        st.error("Indicators could not be calculated.")
        st.stop()

except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

result = make_signal(df)
last = df.iloc[-1]

# ---------------- MARKET STATUS ----------------

st.subheader("Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "NIFTY Price",
    f"{result['entry']:,.2f}"
)

c2.metric(
    "RSI",
    f"{last['RSI']:.2f}"
)

c3.metric(
    "ATR",
    f"{last['ATR']:.2f}"
)

c4.metric(
    "Timeframe",
    timeframe
)

st.caption(
    f"Last candle timestamp: {df.index[-1]}"
)

# ---------------- SIGNAL ----------------

st.divider()
st.subheader("🎯 Trading Signal")

signal = result["signal"]

if signal == "BUY":
    st.success("BUY SIGNAL — Bullish conditions aligned")
elif signal == "SELL":
    st.error("SELL SIGNAL — Bearish conditions aligned")
else:
    st.warning("NO TRADE — Wait for confirmation")

a, b, c = st.columns(3)

a.metric("Entry Price", f"{result['entry']:,.2f}")

if result["sl"] is not None:
    b.metric("Stop Loss", f"{result['sl']:,.2f}")
    c.metric("Target 1", f"{result['t1']:,.2f}")

    x, y = st.columns(2)
    x.metric("Target 2", f"{result['t2']:,.2f}")
    y.metric("Target 3", f"{result['t3']:,.2f}")
else:
    b.metric("Stop Loss", "Not generated")
    c.metric("Targets", "Not generated")

st.markdown("**Signal Reasons**")
for reason in result["reasons"]:
    st.write("•", reason)

# ---------------- RISK MANAGEMENT ----------------

st.divider()
st.subheader("💰 Risk Management")

if result["sl"] is not None:
    risk_points = abs(result["entry"] - result["sl"])
    risk_amount = capital * risk_percent / 100

    if risk_points > 0:
        estimated_quantity = int(risk_amount / risk_points)

        st.write(
            f"Maximum planned risk: ₹{risk_amount:,.2f}"
        )
        st.write(
            f"Risk distance: {risk_points:.2f} index points"
        )
        st.write(
            f"Illustrative quantity by risk: {estimated_quantity}"
        )

        st.caption(
            "This is an index-point calculation, not an option lot "
            "recommendation. For options, premium, lot size, spread, "
            "slippage and contract specifications must be checked separately."
        )
else:
    st.info("No trade: risk sizing is not available.")

# ---------------- INDICATOR TABLE ----------------

st.divider()
st.subheader("📊 Indicator Details")

indicator_data = pd.DataFrame({
    "Indicator": [
        "EMA 9",
        "EMA 21",
        "EMA 50",
        "RSI",
        "MACD",
        "MACD Signal",
        "Supertrend",
        "ATR"
    ],
    "Value": [
        round(float(last["EMA9"]), 2),
        round(float(last["EMA21"]), 2),
        round(float(last["EMA50"]), 2),
        round(float(last["RSI"]), 2),
        round(float(last["MACD"]), 2),
        round(float(last["MACD_Signal"]), 2),
        round(float(last["Supertrend"]), 2),
        round(float(last["ATR"]), 2)
    ]
})

st.dataframe(
    indicator_data,
    use_container_width=True,
    hide_index=True
)

# ---------------- CHART ----------------

st.divider()
st.subheader("📈 NIFTY Chart")

chart = go.Figure()

chart.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="NIFTY"
))

chart.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA9"],
    name="EMA 9"
))

chart.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA21"],
    name="EMA 21"
))

chart.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA50"],
    name="EMA 50"
))

chart.add_trace(go.Scatter(
    x=df.index,
    y=df["Supertrend"],
    name="Supertrend"
))

chart.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(
    chart,
    use_container_width=True
)

# ---------------- RSI CHART ----------------

st.subheader("RSI Chart")

rsi_chart = go.Figure()

rsi_chart.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    name="RSI"
))

rsi_chart.add_hline(y=70, line_dash="dash")
rsi_chart.add_hline(y=30, line_dash="dash")
rsi_chart.add_hline(y=50, line_dash="dot")

rsi_chart.update_layout(
    height=250,
    template="plotly_dark"
)

st.plotly_chart(
    rsi_chart,
    use_container_width=True
)

# ---------------- DISCLAIMER ----------------

st.divider()

st.warning(
    "IMPORTANT: This dashboard is an educational technical-analysis "
    "tool, not a guaranteed trading system. BUY/SELL signals can fail. "
    "Targets are calculated risk/reward levels, not price forecasts. "
    "Yahoo Finance data may be delayed or incomplete. Verify prices, "
    "liquidity, brokerage costs and exchange data before trading. "
    "Paper trade and backtest before risking real money."
)

st.caption(
    f"Dashboard refreshed/rendered: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)
