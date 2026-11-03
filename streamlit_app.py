
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="NIFTY Multi-Timeframe AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 Multi-Timeframe Trading System")
st.caption(
    "3M + 5M + 15M | EMA + RSI + MACD + Supertrend | ATR Risk Management"
)

# ==========================================
# SIDEBAR SETTINGS
# ==========================================

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

# ==========================================
# DOWNLOAD DATA
# ==========================================

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

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    data = data[required].copy()

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    data.index = data.index.tz_convert("Asia/Kolkata")

    return data

# ==========================================
# TIMEFRAME RESAMPLING
# ==========================================

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

    now = pd.Timestamp.now(tz="Asia/Kolkata")

    if len(result) > 0:

        last_time = result.index[-1]

        candle_end = (
            last_time + pd.Timedelta(minutes=minutes)
        )

        if candle_end > now:
            result = result.iloc[:-1]

    return result

# ==========================================
# INDICATORS
# ==========================================

def add_indicators(data):

    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA
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
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

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

    # STANDARD SUPERTREND
    multiplier = 3.0

    hl2 = (high + low) / 2

    basic_upper = hl2 + multiplier * df["ATR"]
    basic_lower = hl2 - multiplier * df["ATR"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):

        if (
            basic_upper.iloc[i] < final_upper.iloc[i-1]
            or close.iloc[i-1] > final_upper.iloc[i-1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i-1]
            or close.iloc[i-1] < final_lower.iloc[i-1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    st_line = pd.Series(index=df.index, dtype=float)
    st_direction = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):

        if i == 0:
            st_direction.iloc[i] = 1
            st_line.iloc[i] = final_lower.iloc[i]
            continue

        if st_direction.iloc[i-1] == 1:

            if close.iloc[i] < final_lower.iloc[i]:
                st_direction.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_direction.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]

        else:

            if close.iloc[i] > final_upper.iloc[i]:
                st_direction.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
            else:
                st_direction.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]

    df["Supertrend"] = st_line
    df["ST_DIRECTION"] = st_direction

    return df.dropna()

# ==========================================
# VWAP AND VOLUME ANALYSIS
# ==========================================

def add_vwap_analysis(data):

    df = data.copy()

    df["Volume"] = pd.to_numeric(
        df["Volume"], errors="coerce"
    ).fillna(0)

    df["VWAP"] = np.nan
    df["Volume_AVG"] = np.nan

    # VWAP only when usable volume exists
    if (df["Volume"] > 0).any():

        typical_price = (
            df["High"] + df["Low"] + df["Close"]
        ) / 3

        session_date = pd.Series(
            df.index.date,
            index=df.index
        )

        cumulative_pv = (
            typical_price * df["Volume"]
        ).groupby(session_date).cumsum()

        cumulative_volume = (
            df["Volume"]
        ).groupby(session_date).cumsum()

        df["VWAP"] = (
            cumulative_pv /
            cumulative_volume.replace(0, np.nan)
        )

        df["Volume_AVG"] = (
            df["Volume"].rolling(
                20, min_periods=20
            ).mean()
        )

    return df

# ==========================================
# TIMEFRAME ANALYSIS
# ==========================================

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

    return {
        "price": float(last["Close"]),
        "rsi": float(last["RSI"]),
        "atr": float(last["ATR"]),
        "ema9": float(last["EMA9"]),
        "ema21": float(last["EMA21"]),
        "ema50": float(last["EMA50"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"]),
        "macd_bullish": bool(
            last["MACD"] > last["MACD_SIGNAL"]
        ),
        "macd_bearish": bool(
            last["MACD"] < last["MACD_SIGNAL"]
        ),
        "bullish": bool(bullish),
        "bearish": bool(bearish),
        "supertrend": float(last["Supertrend"]),
        "time": str(df.index[-1])
    }

# ==========================================
# ORIGINAL TREND SIGNAL
# ==========================================

def final_signal(a3, a5, a15):

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

    # BUY
    if (
        a15["bullish"]
        and a5["bullish"]
        and a3["bullish"]
    ):

        extension = (
            a3["price"] - a3["ema21"]
        ) / max(a3["atr"], 0.01)

        if extension > 2.5:

            reasons.append(
                "Bullish trend, but price is too extended."
            )

            return "NO TRADE", reasons

        reasons.extend([
            "15-minute bullish trend confirmed",
            "5-minute bullish trend confirmed",
            "3-minute entry trend confirmed"
        ])

        return "BUY", reasons

    # SELL
    if (
        a15["bearish"]
        and a5["bearish"]
        and a3["bearish"]
    ):

        extension = (
            a3["ema21"] - a3["price"]
        ) / max(a3["atr"], 0.01)

        if extension > 2.5:

            reasons.append(
                "Bearish trend, but price is too extended."
            )

            return "NO TRADE", reasons

        reasons.extend([
            "15-minute bearish trend confirmed",
            "5-minute bearish trend confirmed",
            "3-minute entry trend confirmed"
        ])

        return "SELL", reasons

    reasons.append(
        f"Bullish timeframes: {bullish_count}/3"
    )

    reasons.append(
        f"Bearish timeframes: {bearish_count}/3"
    )

    reasons.append(
        "All three timeframes are not aligned."
    )

    return "NO TRADE", reasons

# ==========================================
# MAIN APP
# ==========================================

try:

    raw = download_data()

    if raw.empty:
        st.error("Yahoo Finance data unavailable.")
        st.stop()

    df3 = add_indicators(
        make_timeframe(raw, 3)
    )

    df5 = add_indicators(
        make_timeframe(raw, 5)
    )

    df15 = add_indicators(
        make_timeframe(raw, 15)
    )

    if min(len(df3), len(df5), len(df15)) < 55:
        st.error("Not enough completed candles.")
        st.stop()

    a3 = analyze_timeframe(df3)
    a5 = analyze_timeframe(df5)
    a15 = analyze_timeframe(df15)

    if not all([a3, a5, a15]):
        st.error("Timeframe analysis unavailable.")
        st.stop()

    # Original signal
    signal, reasons = final_signal(
        a3, a5, a15
    )

    # RSI confirmation across 3 timeframes
    rsi_values = [
        a3["rsi"],
        a5["rsi"],
        a15["rsi"]
    ]

    rsi_bull_count = sum(
        x > 50 for x in rsi_values
    )

    rsi_bear_count = sum(
        x < 50 for x in rsi_values
    )

    # MACD confirmation across 3 timeframes
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

    # RSI + MACD affect final signal
    score = 0

    if signal == "BUY":

        score = (
            int(rsi_bull_count >= 2)
            + int(macd_bull_count >= 2)
        )

        if score < 2:

            signal = "NO TRADE"

            reasons.append(
                "BUY rejected: RSI/MACD confirmation weak."
            )

        else:

            reasons.append(
                "RSI bullish confirmation received."
            )

            reasons.append(
                "MACD bullish confirmation received."
            )

    elif signal == "SELL":

        score = (
            int(rsi_bear_count >= 2)
            + int(macd_bear_count >= 2)
        )

        if score < 2:

            signal = "NO TRADE"

            reasons.append(
                "SELL rejected: RSI/MACD confirmation weak."
            )

        else:

            reasons.append(
                "RSI bearish confirmation received."
            )

            reasons.append(
                "MACD bearish confirmation received."
            )

    # VWAP data
    vwap_data = add_vwap_analysis(df3)

    last_vwap = vwap_data.iloc[-1]

    vwap_available = pd.notna(
        last_vwap["VWAP"]
    )

    volume_available = pd.notna(
        last_vwap["Volume_AVG"]
    )

    vwap_bull = False
    vwap_bear = False
    volume_confirmed = False

    if vwap_available:

        vwap_bull = (
            a3["price"] > float(last_vwap["VWAP"])
        )

        vwap_bear = (
            a3["price"] < float(last_vwap["VWAP"])
        )

    if volume_available:

        volume_confirmed = (
            float(last_vwap["Volume"])
            > float(last_vwap["Volume_AVG"])
        )

except Exception as error:

    st.error(f"Dashboard error: {error}")
    st.stop()

# ==========================================
# MARKET SUMMARY
# ==========================================

st.divider()
st.subheader("📊 Market Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "NIFTY Price",
    f"{a3['price']:,.2f}"
)

m2.metric(
    "3M RSI",
    f"{a3['rsi']:.2f}"
)

m3.metric(
    "5M RSI",
    f"{a5['rsi']:.2f}"
)

m4.metric(
    "15M RSI",
    f"{a15['rsi']:.2f}"
)

st.caption(
    f"Last completed 3m candle: {a3['time']}"
)

# ==========================================
# TIMEFRAME TABLE
# ==========================================

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

# ==========================================
# FINAL SIGNAL
# ==========================================

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

# ==========================================
# ADVANCED CONFIRMATION
# ==========================================

st.divider()
st.subheader("📊 Advanced Market Confirmation")

confirmation_table = pd.DataFrame([

    {
        "Indicator": "RSI",
        "3M": round(a3["rsi"], 2),
        "5M": round(a5["rsi"], 2),
        "15M": round(a15["rsi"], 2)
    },

    {
        "Indicator": "MACD",
        "3M": "Bullish" if a3["macd_bullish"] else "Bearish",
        "5M": "Bullish" if a5["macd_bullish"] else "Bearish",
        "15M": "Bullish" if a15["macd_bullish"] else "Bearish"
    }

])

st.dataframe(
    confirmation_table,
    use_container_width=True,
    hide_index=True
)

c1, c2 = st.columns(2)

c1.metric(
    "RSI Confirmation",
    f"{max(rsi_bull_count, rsi_bear_count)}/3"
)

c2.metric(
    "MACD Confirmation",
    f"{max(macd_bull_count, macd_bear_count)}/3"
)

if vwap_available:

    st.write(
        f"VWAP: ₹{float(last_vwap['VWAP']):,.2f}"
    )

else:

    st.info(
        "VWAP unavailable: usable volume data not provided."
    )

if volume_available:

    st.write(
        "Volume:",
        "Confirmed" if volume_confirmed else "Weak"
    )

else:

    st.info(
        "Volume confirmation unavailable."
    )

st.caption(
    "RSI and MACD affect the final signal. "
    "VWAP and Volume are displayed only when usable data exists."
)

# ==========================================
# ENTRY, STOP LOSS AND TARGETS
# ==========================================

st.divider()
st.subheader("Entry, Stop Loss & Targets")

if signal in ["BUY", "SELL"]:

    entry = a3["price"]

    atr = max(
        a3["atr"],
        a5["atr"] * 0.5
    )

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

    risk_amount = capital * risk_percent / 100

    index_quantity = int(
        risk_amount / risk_points
    )

    st.write(
        f"Risk budget: ₹{risk_amount:,.2f}"
    )

    st.write(
        f"Risk distance: {risk_points:.2f} index points"
    )

    st.write(
        f"Illustrative index quantity: {index_quantity}"
    )

    st.caption(
        "Index quantity is illustrative only, not an options lot quantity."
    )

else:

    st.info(
        "No entry, stop loss or targets generated because "
        "the final conditions are not aligned."
    )

# ==========================================
# OPTION BUYING RECOMMENDATION
# ==========================================

st.divider()
st.subheader("🎯 Option Buying Recommendation")

if signal == "BUY":

    st.success("Potential Direction: BUY CE")

elif signal == "SELL":

    st.error("Potential Direction: BUY PE")

else:

    st.warning("NO TRADE - Wait for confirmation.")

st.info(
    "Live option chain is not connected. "
    "Actual strike price, CE/PE premium, entry premium, "
    "option stop loss and targets cannot be calculated "
    "from the NIFTY index feed alone."
)

st.caption(
    "Connect an authorized options market data feed "
    "before using actual option premium recommendations."
)

# ==========================================
# NIFTY CHART
# ==========================================

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

if vwap_available:

    chart.add_trace(go.Scatter(
        x=vwap_data.index,
        y=vwap_data["VWAP"],
        name="VWAP"
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

# ==========================================
# RSI CHART
# ==========================================

st.divider()
st.subheader("3-Minute RSI")

rsi_chart = go.Figure()

rsi_chart.add_trace(go.Scatter(
    x=df3.index,
    y=df3["RSI"],
    name="RSI"
))

rsi_chart.add_hline(
    y=70,
    line_dash="dash"
)

rsi_chart.add_hline(
    y=30,
    line_dash="dash"
)

rsi_chart.add_hline(
    y=50,
    line_dash="dot"
)

rsi_chart.update_layout(
    height=250,
    template="plotly_dark"
)

st.plotly_chart(
    rsi_chart,
    use_container_width=True
)
# ==========================================
# SYSTEM STATUS
# ==========================================

st.divider()
st.subheader("System Status")

s1, s2, s3 = st.columns(3)

s1.success("3-minute data calculated")
s2.success("5-minute data calculated")
s3.success("15-minute data calculated")

st.caption(
    "Data source: Yahoo Finance. Data may be delayed, "
    "incomplete or unavailable."
)

st.warning(
    "Trading risk: Signals are technical conditions, not guaranteed "
    "predictions. Backtest and paper trade before using real capital. "
    "Index point levels are not option premium levels."
)

st.caption(
    "Rendered: "
    + datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d-%m-%Y %H:%M:%S")
)
