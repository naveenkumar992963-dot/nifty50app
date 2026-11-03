
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="NIFTY Multi-Timeframe AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 Multi-Timeframe Trading System")
st.caption("3M + 5M + 15M | One Final Signal | ATR Risk Management")

# ---------------- SETTINGS ----------------

with st.sidebar:
    st.header("Dashboard Settings")

    capital = st.number_input(
        "Trading Capital (₹)",
        min_value=1000,
        value=10000,
        step=1000
    )

    risk_percent = st.slider(
        "Risk per trade (%)",
        0.5, 2.0, 1.0, 0.5
    )

    st.divider()

    if st.button("🔄 REFRESH DATA", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("Source: Yahoo Finance")
    st.caption("Market data may be delayed.")

# ---------------- DOWNLOAD ----------------

@st.cache_data(ttl=60)
def download_data():
    data = yf.download(
        "^NSEI",
        period="5d",
        interval="1m",
        progress=False,
        auto_adjust=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    data = data[required].copy()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    data.index = data.index.tz_convert("Asia/Kolkata")

    return data

# ---------------- RESAMPLE ----------------

def make_timeframe(data, minutes):
    result = data.resample(
        f"{minutes}min",
        origin="start_day",
        offset="15min"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    result = result.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    # Remove incomplete latest candle
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    if len(result) > 0:
        last_time = result.index[-1]
        candle_end = last_time + pd.Timedelta(minutes=minutes)

        if candle_end > now:
            result = result.iloc[:-1]

    return result

# ---------------- INDICATORS ----------------

def add_indicators(data):
    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
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
        alpha=1/14, adjust=False
    ).mean()

    # Supertrend
    hl2 = (high + low) / 2

    upper = hl2 + 3 * df["ATR"]
    lower = hl2 - 3 * df["ATR"]

    st_line = pd.Series(
        index=df.index,
        dtype=float
    )

    st_direction = pd.Series(
        index=df.index,
        dtype=float
    )

    for i in range(len(df)):
        if i == 0:
            st_line.iloc[i] = lower.iloc[i]
            st_direction.iloc[i] = 1
            continue

        previous_line = st_line.iloc[i - 1]

        if close.iloc[i] > previous_line:
            st_direction.iloc[i] = 1
            st_line.iloc[i] = max(
                lower.iloc[i], previous_line
            )

        elif close.iloc[i] < previous_line:
            st_direction.iloc[i] = -1
            st_line.iloc[i] = min(
                upper.iloc[i], previous_line
            )

        else:
            st_direction.iloc[i] = st_direction.iloc[i - 1]
            st_line.iloc[i] = previous_line

    df["Supertrend"] = st_line
    df["ST_DIRECTION"] = st_direction

    return df.dropna()

# ---------------- TIMEFRAME ANALYSIS ----------------

def analyze_timeframe(df):
    if len(df) < 55:
        return None

    last = df.iloc[-1]

    bullish = (
        last["EMA9"] > last["EMA21"]
        and last["EMA21"] > last["EMA50"]
        and last["Close"] > last["EMA21"]
        and last["ST_DIRECTION"] == 1
    )

    bearish = (
        last["EMA9"] < last["EMA21"]
        and last["EMA21"] < last["EMA50"]
        and last["Close"] < last["EMA21"]
        and last["ST_DIRECTION"] == -1
    )

    macd_bullish = last["MACD"] > last["MACD_SIGNAL"]
    macd_bearish = last["MACD"] < last["MACD_SIGNAL"]

    return {
        "price": float(last["Close"]),
        "rsi": float(last["RSI"]),
        "atr": float(last["ATR"]),
        "ema9": float(last["EMA9"]),
        "ema21": float(last["EMA21"]),
        "ema50": float(last["EMA50"]),
        "macd_bullish": bool(macd_bullish),
        "macd_bearish": bool(macd_bearish),
        "bullish": bool(bullish),
        "bearish": bool(bearish),
        "supertrend": float(last["Supertrend"]),
        "time": str(df.index[-1])
    }

# ---------------- FINAL SIGNAL ----------------

def final_signal(a3, a5, a15):
    if not all([a3, a5, a15]):
        return "NO TRADE", ["Insufficient timeframe data"]

    reasons = []

    bullish_count = sum([
        a3["bullish"],
        a5["bullish"],
        a15["bullish"]
    ])

    bearish_count = sum([
        a3["bearish"],
        a5["bearish"],
        a15["bearish"]
    ])

    # 15m main trend and 5m confirmation
    # 3m entry timing

    if a15["bullish"] and a5["bullish"] and a3["bullish"]:
        # Avoid chasing an excessively extended candle
        extension = (
            a3["price"] - a3["ema21"]
        ) / max(a3["atr"], 0.01)

        if extension > 2.5:
            reasons.append(
                "Bullish trend, but price is too extended. "
                "Wait for pullback."
            )
            return "NO TRADE", reasons

        reasons.extend([
            "15-minute bullish trend confirmed",
            "5-minute bullish momentum confirmed",
            "3-minute entry confirmation received"
        ])

        if a5["macd_bullish"]:
            reasons.append("5-minute MACD bullish confirmation")
        else:
            reasons.append(
                "MACD is lagging; EMA and trend confirmation are aligned"
            )

        return "BUY", reasons

    if a15["bearish"] and a5["bearish"] and a3["bearish"]:
        extension = (
            a3["ema21"] - a3["price"]
        ) / max(a3["atr"], 0.01)

        if extension > 2.5:
            reasons.append(
                "Bearish trend, but price is too extended. "
                "Wait for pullback."
            )
            return "NO TRADE", reasons

        reasons.extend([
            "15-minute bearish trend confirmed",
            "5-minute bearish momentum confirmed",
            "3-minute entry confirmation received"
        ])

        if a5["macd_bearish"]:
            reasons.append("5-minute MACD bearish confirmation")
        else:
            reasons.append(
                "MACD is lagging; EMA and trend confirmation are aligned"
            )

        return "SELL", reasons

    reasons.append(
        f"Bullish timeframes: {bullish_count}/3"
    )
    reasons.append(
        f"Bearish timeframes: {bearish_count}/3"
    )
    reasons.append(
        "All three timeframes are not aligned"
    )

    return "NO TRADE", reasons

# ---------------- MAIN APP ----------------

try:
    raw = download_data()

    if raw.empty:
        st.error(
            "Yahoo Finance data unavailable. "
            "Refresh after some time."
        )
        st.stop()

    df3 = add_indicators(make_timeframe(raw, 3))
    df5 = add_indicators(make_timeframe(raw, 5))
    df15 = add_indicators(make_timeframe(raw, 15))

    a3 = analyze_timeframe(df3)
    a5 = analyze_timeframe(df5)
    a15 = analyze_timeframe(df15)

    if not all([a3, a5, a15]):
        st.error(
            "Not enough completed candles for all timeframes."
        )
        st.stop()

    # ==========================================
# FINAL SIGNAL WITH MOMENTUM CONFIRMATION
# ==========================================

signal, reasons = final_signal(a3, a5, a15)

# RSI confirmation
rsi_values = [
    a3["rsi"],
    a5["rsi"],
    a15["rsi"]
]

rsi_bull_count = sum(x > 50 for x in rsi_values)
rsi_bear_count = sum(x < 50 for x in rsi_values)

# MACD confirmation
macd_bull_count = sum([
    a3["macd_bullish"],
    a5["macd_bullish"],
    a15["macd_bullish"]
])

macd_bear_count = sum([
    a3["macd_bearish"],
    a5["macd_bearish"],
    a15["macd_bearish"]
])

# VWAP calculation (only if valid volume exists)
vwap_available = False
vwap_bull = False
vwap_bear = False

volume_available = False
volume_confirmed = False

vwap_data = df3.copy()

vwap_data["Volume"] = pd.to_numeric(
    vwap_data["Volume"], errors="coerce"
).fillna(0)

if (vwap_data["Volume"] > 0).any():

    typical_price = (
        vwap_data["High"] +
        vwap_data["Low"] +
        vwap_data["Close"]
    ) / 3

    session_date = pd.Series(
        vwap_data.index.date,
        index=vwap_data.index
    )

    cumulative_pv = (
        typical_price * vwap_data["Volume"]
    ).groupby(session_date).cumsum()

    cumulative_volume = (
        vwap_data["Volume"]
    ).groupby(session_date).cumsum()

    vwap_data["VWAP"] = (
        cumulative_pv / cumulative_volume.replace(0, np.nan)
    )

    current_vwap = vwap_data["VWAP"].iloc[-1]

    if pd.notna(current_vwap):
        vwap_available = True
        vwap_bull = a3["price"] > current_vwap
        vwap_bear = a3["price"] < current_vwap

    vwap_data["Volume_AVG"] = (
        vwap_data["Volume"].rolling(20).mean()
    )

    avg_volume = vwap_data["Volume_AVG"].iloc[-1]
    current_volume = vwap_data["Volume"].iloc[-1]

    if pd.notna(avg_volume) and avg_volume > 0:
        volume_available = True
        volume_confirmed = current_volume > avg_volume

# Confirmation score
if signal == "BUY":

    score = (
        (rsi_bull_count >= 2) +
        (macd_bull_count >= 2) +
        (vwap_available and vwap_bull) +
        (volume_available and volume_confirmed)
    )

    if score < 2:
        signal = "NO TRADE"
        reasons.append(
            "BUY rejected: momentum confirmation insufficient."
        )
    else:
        reasons.append(
            f"Bullish confirmation score: {score}/4"
        )

elif signal == "SELL":

    score = (
        (rsi_bear_count >= 2) +
        (macd_bear_count >= 2) +
        (vwap_available and vwap_bear) +
        (volume_available and volume_confirmed)
    )

    if score < 2:
        signal = "NO TRADE"
        reasons.append(
            "SELL rejected: momentum confirmation insufficient."
        )
    else:
        reasons.append(
            f"Bearish confirmation score: {score}/4"
        )

else:
    score = 0

# Data availability
if not vwap_available:
    reasons.append("VWAP unavailable: volume data missing.")

if not volume_available:
    reasons.append("Volume confirmation unavailable.")

except Exception as error:
    st.error(f"Dashboard error: {error}")
    st.stop()

# ---------------- MARKET SUMMARY ----------------

st.subheader("Market Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric("NIFTY Price", f"{a3['price']:,.2f}")
m2.metric("3M RSI", f"{a3['rsi']:.2f}")
m3.metric("5M RSI", f"{a5['rsi']:.2f}")
m4.metric("15M RSI", f"{a15['rsi']:.2f}")

st.caption(
    f"Last completed 3m candle: {a3['time']}"
)

# ---------------- TIMEFRAME TABLE ----------------

st.divider()
st.subheader("Multi-Timeframe Confirmation")

timeframe_table = pd.DataFrame([
    {
        "Timeframe": "3 Minutes",
        "Trend": "BULLISH" if a3["bullish"] else (
            "BEARISH" if a3["bearish"] else "NEUTRAL"
        ),
        "RSI": round(a3["rsi"], 2),
        "MACD": "Bullish" if a3["macd_bullish"] else "Bearish",
        "Price": round(a3["price"], 2)
    },
    {
        "Timeframe": "5 Minutes",
        "Trend": "BULLISH" if a5["bullish"] else (
            "BEARISH" if a5["bearish"] else "NEUTRAL"
        ),
        "RSI": round(a5["rsi"], 2),
        "MACD": "Bullish" if a5["macd_bullish"] else "Bearish",
        "Price": round(a5["price"], 2)
    },
    {
        "Timeframe": "15 Minutes",
        "Trend": "BULLISH" if a15["bullish"] else (
            "BEARISH" if a15["bearish"] else "NEUTRAL"
        ),
        "RSI": round(a15["rsi"], 2),
        "MACD": "Bullish" if a15["macd_bullish"] else "Bearish",
        "Price": round(a15["price"], 2)
    }
])

st.dataframe(
    timeframe_table,
    use_container_width=True,
    hide_index=True
)

# ---------------- FINAL TRADE SIGNAL ----------------

st.divider()
st.subheader("🎯 FINAL TRADING SIGNAL")

if signal == "BUY":
    st.success("🟢 FINAL SIGNAL: BUY")
elif signal == "SELL":
    st.error("🔴 FINAL SIGNAL: SELL")
else:
    st.warning("🟡 FINAL SIGNAL: NO TRADE")

for reason in reasons:
    st.write("•", reason)

# ---------------- ENTRY SL TARGETS ----------------

st.divider()
st.subheader("Entry, Stop Loss & Targets")

if signal in ["BUY", "SELL"]:

    entry = a3["price"]

    # Use 3m ATR, with 5m ATR as a volatility reference
    atr = max(a3["atr"], a5["atr"] * 0.5)

    risk_points = 1.5 * atr

    if signal == "BUY":
        sl = entry - risk_points
        t1 = entry + risk_points
        t2 = entry + (2 * risk_points)
        t3 = entry + (3 * risk_points)
    else:
        sl = entry + risk_points
        t1 = entry - risk_points
        t2 = entry - (2 * risk_points)
        t3 = entry - (3 * risk_points)

    p1, p2, p3 = st.columns(3)

    p1.metric("Entry", f"{entry:,.2f}")
    p2.metric("Stop Loss", f"{sl:,.2f}")
    p3.metric("Target 1", f"{t1:,.2f}")

    p4, p5 = st.columns(2)

    p4.metric("Target 2", f"{t2:,.2f}")
    p5.metric("Target 3", f"{t3:,.2f}")

    # Index-point risk illustration only
    risk_amount = capital * risk_percent / 100
    index_quantity = int(risk_amount / risk_points)

    st.write(f"Risk budget: ₹{risk_amount:,.2f}")
    st.write(f"Risk distance: {risk_points:.2f} index points")
    st.write(
        f"Illustrative index quantity: {index_quantity}"
    )

    st.caption(
        "This is not an options position size. "
        "NIFTY index points are not option premium points. "
        "Verify actual CE/PE premium, spread, lot size and slippage separately."
    )

else:
    st.info(
        "No entry, stop loss or targets generated because "
        "the three timeframes are not aligned."
    )

# ---------------- CHART ----------------

st.divider()
st.subheader("NIFTY 3-Minute Chart")

chart = go.Figure()

chart.add_trace(go.Candlestick(
    x=df3.index,
    open=df3["Open"],
    high=df3["High"],
    low=df3["Low"],
    close=df3["Close"],
    name="NIFTY"
))

chart.add_trace(go.Scatter(
    x=df3.index,
    y=df3["EMA9"],
    name="EMA 9"
))

chart.add_trace(go.Scatter(
    x=df3.index,
    y=df3["EMA21"],
    name="EMA 21"
))

chart.add_trace(go.Scatter(
    x=df3.index,
    y=df3["EMA50"],
    name="EMA 50"
))

chart.add_trace(go.Scatter(
    x=df3.index,
    y=df3["Supertrend"],
    name="Supertrend"
))

chart.update_layout(
    height=600,
    template="plotly_dark",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    chart,
    use_container_width=True
)

# ---------------- RSI ----------------

st.subheader("3-Minute RSI")

rsi_chart = go.Figure()

rsi_chart.add_trace(go.Scatter(
    x=df3.index,
    y=df3["RSI"],
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

# ---------------- STATUS ----------------

st.divider()
st.subheader("System Status")

s1, s2, s3 = st.columns(3)

s1.success("3-minute data calculated")
s2.success("5-minute data calculated")
s3.success("15-minute data calculated")

st.caption(
    "Data source: Yahoo Finance. Data may be delayed, incomplete "
    "or unavailable. Resampled candles are calculated from 1-minute data."
)

st.warning(
    "Trading risk: Signals are technical conditions, not guaranteed "
    "predictions. SL and targets are ATR-based reference levels, "
    "not guaranteed execution prices. Backtest and paper trade before "
    "using real capital. This dashboard does not provide personalized "
    "financial advice."
)

st.caption(
    "Rendered: " +
    datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
        "%d-%m-%Y %H:%M:%S"
    )
    )
# ==============================
# OPTION BUYING RECOMMENDATION
# ==============================

st.subheader("🎯 Option Buying Recommendation")

if "option_chain" in locals() and "final_signal" in locals():

    if final_signal in ["BUY", "BULLISH"]:
        option_type = "CE"
        premium_col = "CE_LTP"

    elif final_signal in ["SELL", "BEARISH"]:
        option_type = "PE"
        premium_col = "PE_LTP"

    else:
        option_type = None

    if option_type:
        chain = option_chain.dropna(subset=[premium_col, "strike"])

        chain = chain[chain[premium_col] > 0]

        if not chain.empty:
            selected = chain.iloc[
                (chain[premium_col] - 120).abs().argsort()[:1]
            ].iloc[0]

            strike = selected["strike"]
            premium = float(selected[premium_col])

            sl = round(premium * 0.80, 2)
            target1 = round(premium * 1.30, 2)
            target2 = round(premium * 1.50, 2)

            st.success(f"BUY {option_type}")

            st.write(f"**Option:** NIFTY {strike} {option_type}")
            st.write(f"**Strike Price:** {strike}")
            st.write(f"**Current Premium:** ₹{premium:.2f}")
            st.write(f"**Entry Premium:** ₹{premium:.2f}")
            st.write(f"**Stop Loss:** ₹{sl:.2f}")
            st.write(f"**Target 1:** ₹{target1:.2f}")
            st.write(f"**Target 2:** ₹{target2:.2f}")

        else:
            st.warning("Live option premium unavailable.")

    else:
        st.info("NO TRADE - Wait for confirmation.")

else:
    st.warning("Option chain data or signal variable not found.")
# ===== SAFE ADVANCED CONFIRMATION =====

st.divider()
st.subheader("📊 Advanced Market Confirmation")

vwap_data = df3.copy()

vwap_data["Volume"] = pd.to_numeric(
    vwap_data["Volume"], errors="coerce"
).fillna(0)

vwap_data["Volume_AVG"] = (
    vwap_data["Volume"].rolling(20, min_periods=1).mean()
)

typical_price = (
    vwap_data["High"] +
    vwap_data["Low"] +
    vwap_data["Close"]
) / 3

cum_vol = vwap_data["Volume"].cumsum()

vwap_data["VWAP"] = (
    (typical_price * vwap_data["Volume"]).cumsum()
    / cum_vol.replace(0, np.nan)
)

last = vwap_data.iloc[-1]

st.write("RSI:", round(float(a3["rsi"]), 2))
st.write("MACD:", "Bullish" if a3["macd_bullish"] else "Bearish")

if pd.notna(last["VWAP"]):
    st.write("VWAP:", round(float(last["VWAP"]), 2))
else:
    st.write("VWAP: Unavailable")

st.write(
    "Volume:",
    "Confirmed" if last["Volume"] > last["Volume_AVG"]
    else "Weak"
)

st.caption("Additional analysis only. Original signal unchanged.")
