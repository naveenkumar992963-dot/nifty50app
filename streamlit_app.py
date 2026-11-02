
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="NIFTY Options AI Engine",
    page_icon="📈",
    layout="wide"
)

SYMBOLS = {
    "NIFTY 50": ("^NSEI", 50),
    "BANK NIFTY": ("^NSEBANK", 100),
    "FIN NIFTY": ("^CNXFIN", 50),
    "SENSEX": ("^BSESN", 100)
}

TIMEFRAMES = {
    "5 Minute": ("5m", "5d"),
    "15 Minute": ("15m", "1mo"),
    "30 Minute": ("30m", "1mo"),
    "1 Hour": ("60m", "3mo")
}

st.title("🚀 NIFTY OPTIONS BUYING AI ENGINE")
st.caption("Multi-strategy CE / PE analysis | Educational tool")

# ---------------- DATA ----------------

@st.cache_data(ttl=45)
def get_data(symbol, interval, period):
    data = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(x in data.columns for x in required):
        return pd.DataFrame()

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["Open", "High", "Low", "Close"])


# ---------------- INDICATORS ----------------

def indicators(df):
    d = df.copy()

    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]

    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()

    delta = c.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14, adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    d["RSI"] = (100 - 100/(1+rs)).fillna(50)

    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()

    d["MACD"] = fast - slow
    d["MACD_SIGNAL"] = d["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    previous = c.shift(1)

    tr = pd.concat([
        h-l,
        (h-previous).abs(),
        (l-previous).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # ADX

    up = h.diff()
    down = -l.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=d.index
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=d.index
    )

    atr = d["ATR"].replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(
        alpha=1/14, adjust=False
    ).mean() / atr

    minus_di = 100 * minus_dm.ewm(
        alpha=1/14, adjust=False
    ).mean() / atr

    dx = 100 * (plus_di-minus_di).abs() / (
        plus_di+minus_di
    ).replace(0, np.nan)

    d["ADX"] = dx.ewm(
        alpha=1/14, adjust=False
    ).mean().fillna(0)

    # VWAP

    typical = (h+l+c)/3
    volume = v.replace(0, np.nan)

    d["VWAP"] = (
        (typical*volume).groupby(d.index.date).cumsum()
        / volume.groupby(d.index.date).cumsum()
    ).fillna(c)

    # Volume confirmation

    d["VOL_AVG"] = v.rolling(20).mean()

    # Previous candle breakout

    d["PREV_HIGH"] = h.shift(1).rolling(5).max()
    d["PREV_LOW"] = l.shift(1).rolling(5).min()

    return d.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


# ---------------- STRATEGY ENGINE ----------------

def analyze(d):

    x = d.iloc[-1]
    previous = d.iloc[-2]

    bullish = []
    bearish = []

    if x["EMA9"] > x["EMA21"]:
        bullish.append("EMA 9/21 bullish")
    else:
        bearish.append("EMA 9/21 bearish")

    if x["Close"] > x["EMA50"]:
        bullish.append("Above EMA 50")
    else:
        bearish.append("Below EMA 50")

    if x["MACD"] > x["MACD_SIGNAL"]:
        bullish.append("MACD bullish")
    else:
        bearish.append("MACD bearish")

    if x["Close"] > x["VWAP"]:
        bullish.append("Above VWAP")
    else:
        bearish.append("Below VWAP")

    if 50 <= x["RSI"] <= 68:
        bullish.append("RSI bullish zone")

    if 32 <= x["RSI"] <= 50:
        bearish.append("RSI bearish zone")

    if x["ADX"] >= 20:
        if x["EMA9"] > x["EMA21"]:
            bullish.append("Strong trend ADX")
        elif x["EMA9"] < x["EMA21"]:
            bearish.append("Strong trend ADX")

    if (
        x["Volume"] > x["VOL_AVG"]
        and x["Close"] > x["Open"]
    ):
        bullish.append("Buying volume confirmation")

    if (
        x["Volume"] > x["VOL_AVG"]
        and x["Close"] < x["Open"]
    ):
        bearish.append("Selling volume confirmation")

    if x["Close"] > x["PREV_HIGH"]:
        bullish.append("Previous high breakout")

    if x["Close"] < x["PREV_LOW"]:
        bearish.append("Previous low breakdown")

    bull_score = len(bullish)
    bear_score = len(bearish)

    if bull_score >= 5 and bull_score >= bear_score + 2:
        trend = "BULLISH"
    elif bear_score >= 5 and bear_score >= bull_score + 2:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    return {
        "trend": trend,
        "bull": bull_score,
        "bear": bear_score,
        "price": float(x["Close"]),
        "rsi": float(x["RSI"]),
        "adx": float(x["ADX"]),
        "atr": float(x["ATR"]),
        "bull_reasons": bullish,
        "bear_reasons": bearish
    }


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("⚙️ SETTINGS")

    index_name = st.selectbox(
        "Select Index",
        list(SYMBOLS.keys())
    )

    timeframe = st.selectbox(
        "Entry Timeframe",
        list(TIMEFRAMES.keys()),
        index=1
    )

    capital = st.number_input(
        "Trading Capital ₹",
        min_value=1000,
        value=20000,
        step=1000
    )

    risk_percent = st.slider(
        "Risk per trade %",
        0.5, 5.0, 1.0, 0.5
    )

    lot_size = st.number_input(
        "Current Lot Size",
        min_value=1,
        value=65
    )

symbol, strike_step = SYMBOLS[index_name]

# ---------------- MULTI TIMEFRAME ----------------

st.subheader("🧭 MULTI TIMEFRAME ANALYSIS")

results = {}
rows = []

for tf_name, (interval, period) in TIMEFRAMES.items():

    raw = get_data(symbol, interval, period)

    if raw.empty or len(raw) < 70:
        rows.append({
            "Timeframe": tf_name,
            "Trend": "Insufficient data"
        })
        continue

    calculated = indicators(raw)

    if len(calculated) < 3:
        continue

    result = analyze(calculated)

    results[tf_name] = result

    rows.append({
        "Timeframe": tf_name,
        "Trend": result["trend"],
        "Bull Score": result["bull"],
        "Bear Score": result["bear"],
        "RSI": round(result["rsi"], 1),
        "ADX": round(result["adx"], 1)
    })

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True
)

if timeframe not in results:
    st.error("Selected timeframe data unavailable.")
    st.stop()

main = results[timeframe]

bull_count = sum(
    1 for x in results.values()
    if x["trend"] == "BULLISH"
)

bear_count = sum(
    1 for x in results.values()
    if x["trend"] == "BEARISH"
)

valid_tf_count = len(results)

# ---------------- FINAL SIGNAL ----------------

st.divider()
st.subheader("🎯 FINAL OPTIONS SIGNAL")

ce_confirmation = (
    main["trend"] == "BULLISH"
    and main["rsi"] >= 50
    and main["rsi"] < 70
    and main["adx"] >= 20
    and bull_count >= 3
)

pe_confirmation = (
    main["trend"] == "BEARISH"
    and main["rsi"] <= 50
    and main["rsi"] > 30
    and main["adx"] >= 20
    and bear_count >= 3
)

if valid_tf_count < 3:
    signal = "NO TRADE"
    reason = "Not enough timeframe confirmations."

elif ce_confirmation:
    signal = "BUY CE"
    reason = "Bullish multi-timeframe confirmation."

elif pe_confirmation:
    signal = "BUY PE"
    reason = "Bearish multi-timeframe confirmation."

else:
    signal = "NO TRADE"
    reason = "Strategy conditions are not fully confirmed."

c1, c2, c3, c4 = st.columns(4)

c1.metric("Spot Price", f"{main['price']:,.2f}")
c2.metric("Market Trend", main["trend"])
c3.metric("RSI", f"{main['rsi']:.2f}")
c4.metric("ADX", f"{main['adx']:.2f}")

if signal == "BUY CE":
    st.success("🟢 BUY CE SETUP")
elif signal == "BUY PE":
    st.error("🔴 BUY PE SETUP")
else:
    st.warning("🟡 NO TRADE")

st.write(reason)

# ---------------- AUTO STRIKE SELECTION ----------------

st.divider()
st.subheader("🏆 AUTOMATIC STRIKE SELECTION")

spot = main["price"]

atm = int(round(spot / strike_step) * strike_step)

if signal == "BUY CE":
    option_type = "CE"
    selected_strike = atm - strike_step
elif signal == "BUY PE":
    option_type = "PE"
    selected_strike = atm + strike_step
else:
    option_type = None
    selected_strike = atm

if option_type:

    st.success(
        f"Suggested Contract: {selected_strike} {option_type}"
    )

    st.write(f"ATM Strike: {atm}")
    st.write(f"ITM Strike: {selected_strike}")

    st.caption(
        "Strike is calculated from index spot and standard strike interval. "
        "Verify the actual listed contract and expiry with your broker."
    )

else:
    st.info("No CE/PE contract recommended until confirmation.")

# ---------------- OPTION PREMIUM CALCULATOR ----------------

st.divider()
st.subheader("💰 OPTION ENTRY / SL / TARGET")

st.info(
    "Live option chain is not connected. Enter actual premium from broker."
)

premium = st.number_input(
    "Selected option premium ₹",
    min_value=0.0,
    value=100.0,
    step=0.5
)

if premium > 0:

    entry = premium
    sl = entry * 0.80
    target1 = entry * 1.30
    target2 = entry * 1.60

    risk_budget = capital * risk_percent / 100

    risk_per_unit = entry - sl

    quantity = int(risk_budget // risk_per_unit)

    quantity = (
        quantity // int(lot_size)
    ) * int(lot_size)

    lots = quantity // int(lot_size)

    a, b, c, d = st.columns(4)

    a.metric("Entry Premium", f"₹{entry:.2f}")
    b.metric("Stop Loss", f"₹{sl:.2f}")
    c.metric("Target 1", f"₹{target1:.2f}")
    d.metric("Target 2", f"₹{target2:.2f}")

    x, y, z = st.columns(3)

    x.metric("Quantity", quantity)
    y.metric("Lots", lots)
    z.metric("Capital Required", f"₹{quantity*entry:,.2f}")

    if quantity == 0:
        st.warning("Risk budget insufficient for one full lot.")

st.caption(
    "Premium SL/targets are illustrative percentages, not live option technical levels."
)

# ---------------- CHARTS ----------------

st.divider()
st.subheader("📊 PRICE CHART")

st.line_chart(
    indicators(
        get_data(
            symbol,
            TIMEFRAMES[timeframe][0],
            TIMEFRAMES[timeframe][1]
        )
    )[["Close", "EMA9", "EMA21", "EMA50", "VWAP"]].tail(200),
    use_container_width=True
)

# ---------------- DISCLAIMER ----------------

st.warning(
    "Options buying involves substantial risk and may result in "
    "loss of the entire premium. This tool does not access live option "
    "chain, Greeks, implied volatility, bid-ask spread or actual execution. "
    "Signals are not guaranteed. Verify every trade independently."
)

st.caption(
    "Yahoo Finance data | Multi-indicator strategy | "
    "No trades are automatically executed."
)
