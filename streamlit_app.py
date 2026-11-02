
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY AI Advanced Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 ADVANCED PREDICTION ENGINE")
st.caption("Multi-Strategy | 30,000 Scenarios | 4 Timeframes")

st.warning(
    "Educational and paper-trading tool. Scenario shares are simulations, "
    "not guaranteed probabilities of market direction or profit."
)

SCENARIOS = 30000
HORIZON = 6

@st.cache_data(ttl=60)
def get_data(interval, period):
    try:
        df = yf.download(
            "^NSEI",
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            for level in range(df.columns.nlevels):
                names = df.columns.get_level_values(level)
                if "Close" in names:
                    df.columns = names
                    break

        needed = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()

        return df.dropna(subset=needed).copy()

    except Exception:
        return pd.DataFrame()


def add_indicators(df):
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    volume = (
        df["Volume"].astype(float)
        if "Volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    # Moving averages
    for n in [9, 21, 50, 200]:
        df[f"EMA{n}"] = close.ewm(
            span=n, adjust=False
        ).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/14, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # True Range and ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # ADX
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=df.index
    )

    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(
        alpha=1/14, adjust=False
    ).mean() / atr14.replace(0, np.nan)

    minus_di = 100 * minus_dm.ewm(
        alpha=1/14, adjust=False
    ).mean() / atr14.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX"] = dx.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # Bollinger Bands
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["BB_UPPER"] = middle + 2 * std
    df["BB_LOWER"] = middle - 2 * std

    # VWAP (intraday)
    typical = (high + low + close) / 3
    dates = pd.Series(df.index.date, index=df.index)

    pv = typical * volume
    cumulative_pv = pv.groupby(dates).cumsum()
    cumulative_volume = volume.groupby(dates).cumsum()

    df["VWAP"] = (
        cumulative_pv / cumulative_volume.replace(0, np.nan)
    )

    # Volume ratio
    avg_volume = volume.rolling(20).mean()
    df["VOLUME_RATIO"] = (
        volume / avg_volume.replace(0, np.nan)
    )

    # Supertrend
    basic_upper = ((high + low) / 2) + 3 * df["ATR"]
    basic_lower = ((high + low) / 2) - 3 * df["ATR"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if (
            basic_upper.iloc[i] < final_upper.iloc[i-1]
            or close.iloc[i-1] > final_upper.iloc[i-1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i-1]
            or close.iloc[i-1] < final_lower.iloc[i-1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]

        if direction.iloc[i-1] == 1:
            direction.iloc[i] = (
                -1 if close.iloc[i] < final_lower.iloc[i]
                else 1
            )
        else:
            direction.iloc[i] = (
                1 if close.iloc[i] > final_upper.iloc[i]
                else -1
            )

        supertrend.iloc[i] = (
            final_lower.iloc[i]
            if direction.iloc[i] == 1
            else final_upper.iloc[i]
        )

    df["ST_DIRECTION"] = direction
    df["SUPERTREND"] = supertrend

    # Recent breakout levels
    df["PREV_HIGH"] = high.shift(1).rolling(20).max()
    df["PREV_LOW"] = low.shift(1).rolling(20).min()

    return df


def score_market(df):
    if len(df) < 50:
        return None

    row = df.iloc[-1]

    required = [
        "Close", "EMA9", "EMA21", "EMA50",
        "RSI", "MACD", "MACD_SIGNAL",
        "ATR", "ADX", "PLUS_DI", "MINUS_DI"
    ]

    if not np.isfinite(
        [row[x] for x in required]
    ).all():
        return None

    bull = 0
    bear = 0
    reasons = []

    # EMA trend: 20 points
    if row["EMA9"] > row["EMA21"] > row["EMA50"]:
        bull += 20
        reasons.append("Bullish EMA alignment")
    elif row["EMA9"] < row["EMA21"] < row["EMA50"]:
        bear += 20
        reasons.append("Bearish EMA alignment")

    # MACD: 15 points
    if row["MACD"] > row["MACD_SIGNAL"]:
        bull += 15
        reasons.append("MACD bullish")
    else:
        bear += 15
        reasons.append("MACD bearish")

    # RSI: 15 points
    if 55 < row["RSI"] < 75:
        bull += 15
        reasons.append("RSI bullish")
    elif 25 < row["RSI"] < 45:
        bear += 15
        reasons.append("RSI bearish")

    # VWAP: 15 points
    if np.isfinite(row["VWAP"]):
        if row["Close"] > row["VWAP"]:
            bull += 15
            reasons.append("Price above VWAP")
        else:
            bear += 15
            reasons.append("Price below VWAP")

    # ADX directional strength: 10 points
    if row["ADX"] > 20:
        if row["PLUS_DI"] > row["MINUS_DI"]:
            bull += 10
            reasons.append("ADX bullish trend")
        else:
            bear += 10
            reasons.append("ADX bearish trend")

    # Supertrend: 10 points
    if row["ST_DIRECTION"] == 1:
        bull += 10
        reasons.append("Supertrend bullish")
    else:
        bear += 10
        reasons.append("Supertrend bearish")

    # Breakout: 10 points
    if row["Close"] > row["PREV_HIGH"]:
        bull += 10
        reasons.append("20 candle high breakout")
    elif row["Close"] < row["PREV_LOW"]:
        bear += 10
        reasons.append("20 candle low breakdown")

    # Volume: 5 points
    if (
        np.isfinite(row["VOLUME_RATIO"])
        and row["VOLUME_RATIO"] > 1.5
    ):
        if row["Close"] > row["Open"]:
            bull += 5
            reasons.append("High bullish volume")
        elif row["Close"] < row["Open"]:
            bear += 5
            reasons.append("High bearish volume")

    if bull >= 65 and bull - bear >= 20:
        signal = "CALL BIAS"
    elif bear >= 65 and bear - bull >= 20:
        signal = "PUT BIAS"
    else:
        signal = "WAIT"

    return {
        "df": df,
        "price": float(row["Close"]),
        "bull": bull,
        "bear": bear,
        "signal": signal,
        "rsi": float(row["RSI"]),
        "atr": float(row["ATR"]),
        "adx": float(row["ADX"]),
        "vwap": float(row["VWAP"])
            if np.isfinite(row["VWAP"]) else None,
        "volume_ratio": float(row["VOLUME_RATIO"])
            if np.isfinite(row["VOLUME_RATIO"]) else None,
        "time": df.index[-1],
        "reasons": reasons
    }


def simulate_scenarios(df):
    close = df["Close"].astype(float)
    returns = np.log(close / close.shift()).dropna()

    if len(returns) < 50:
        return None

    recent = returns.tail(96)
    if len(recent) < 30:
        return None

    # Historical residual bootstrap with recent trend drift
    drift = (
        0.35 * returns.tail(12).mean()
        + 0.15 * returns.tail(48).mean()
    )

    volatility = recent.std()

    if not np.isfinite(volatility) or volatility <= 0:
        return None

    residuals = (
        recent - recent.mean()
    ).dropna().to_numpy()

    rng = np.random.default_rng()

    samples = rng.choice(
        residuals,
        size=(SCENARIOS, HORIZON),
        replace=True
    )

    simulated_returns = samples + drift

    final_returns = np.exp(
        simulated_returns.sum(axis=1)
    ) - 1

    atr = float(df["ATR"].iloc[-1])
    price = float(close.iloc[-1])

    threshold = max(
        0.20 * atr / price,
        volatility * np.sqrt(HORIZON) * 0.15
    )

    bullish = int(np.sum(final_returns > threshold))
    bearish = int(np.sum(final_returns < -threshold))
    sideways = SCENARIOS - bullish - bearish

    return {
        "total": SCENARIOS,
        "bullish": bullish,
        "bearish": bearish,
        "sideways": sideways,
        "bull_pct": bullish / SCENARIOS * 100,
        "bear_pct": bearish / SCENARIOS * 100,
        "side_pct": sideways / SCENARIOS * 100
    }


def make_levels(result):
    price = result["price"]
    atr = result["atr"]
    signal = result["signal"]

    if signal == "CALL BIAS":
        sl = price - 1.2 * atr
        risk = price - sl
        targets = [
            price + risk * 1.5,
            price + risk * 2,
            price + risk * 3
        ]
    elif signal == "PUT BIAS":
        sl = price + 1.2 * atr
        risk = sl - price
        targets = [
            price - risk * 1.5,
            price - risk * 2,
            price - risk * 3
        ]
    else:
        return None

    return {"sl": sl, "targets": targets}


if st.button("🔄 Refresh Scanner"):
    st.cache_data.clear()

periods = {
    "5 MIN": ("5m", "5d"),
    "15 MIN": ("15m", "60d"),
    "30 MIN": ("30m", "60d"),
    "1 HOUR": ("60m", "730d")
}

results = {}

with st.spinner("Running indicators and market analysis..."):
    for name, (interval, period) in periods.items():
        raw = get_data(interval, period)
        if not raw.empty:
            prepared = add_indicators(raw)
            results[name] = score_market(prepared)

if "5 MIN" not in results or "15 MIN" not in results:
    st.error(
        "Required market data unavailable. "
        "Yahoo Finance may be delayed or temporarily unavailable."
    )
    st.stop()

st.subheader("📊 MULTI-TIMEFRAME ANALYSIS")

cols = st.columns(4)

for col, name in zip(cols, periods):
    with col:
        result = results.get(name)

        st.markdown(f"### {name}")

        if result is None:
            st.warning("Data unavailable")
            continue

        st.metric("NIFTY Spot", f'{result["price"]:,.2f}')
        st.metric("Signal", result["signal"])
        st.metric("Bullish Score", f'{result["bull"]}/100')
        st.metric("Bearish Score", f'{result["bear"]}/100')
        st.metric("RSI", f'{result["rsi"]:.1f}')
        st.metric("ADX", f'{result["adx"]:.1f}')

st.divider()
st.subheader("🧠 30,000 SCENARIO ANALYSIS")

simulation = simulate_scenarios(
    results["15 MIN"]["df"]
)

if simulation:
    a, b, c = st.columns(3)

    a.metric(
        "Bullish Scenarios",
        f'{simulation["bullish"]:,}',
        f'{simulation["bull_pct"]:.1f}%'
    )
    b.metric(
        "Bearish Scenarios",
        f'{simulation["bearish"]:,}',
        f'{simulation["bear_pct"]:.1f}%'
    )
    c.metric(
        "Sideways Scenarios",
        f'{simulation["sideways"]:,}',
        f'{simulation["side_pct"]:.1f}%'
    )

    st.caption(
        "30,000 bootstrap simulations based on historical 15M returns. "
        "Scenario shares are model outputs, not calibrated probabilities."
    )
else:
    st.warning("Not enough historical data for scenario simulation.")

st.divider()
st.subheader("🎯 FINAL DEEP ANALYSIS")

available = [
    r for r in results.values()
    if r is not None
]

bull_count = sum(r["signal"] == "CALL BIAS" for r in available)
bear_count = sum(r["signal"] == "PUT BIAS" for r in available)

scenario_call = (
    simulation is not None
    and simulation["bull_pct"] >= 55
)
scenario_put = (
    simulation is not None
    and simulation["bear_pct"] >= 55
)

# Require 3 of 4 timeframe confirmations,
# including 15-minute and at least one higher timeframe.
higher = [
    results.get("30 MIN"),
    results.get("1 HOUR")
]

call_confirm = (
    bull_count >= 3
    and results["15 MIN"]["signal"] == "CALL BIAS"
    and any(x is not None and x["signal"] == "CALL BIAS" for x in higher)
    and scenario_call
)

put_confirm = (
    bear_count >= 3
    and results["15 MIN"]["signal"] == "PUT BIAS"
    and any(x is not None and x["signal"] == "PUT BIAS" for x in higher)
    and scenario_put
)

if call_confirm:
    final = "CALL BIAS"
elif put_confirm:
    final = "PUT BIAS"
else:
    final = "NO TRADE"

if final == "CALL BIAS":
    st.success("FINAL RESULT: STRONG CALL BIAS")
elif final == "PUT BIAS":
    st.error("FINAL RESULT: STRONG PUT BIAS")
else:
    st.warning("FINAL RESULT: NO TRADE — CONDITIONS NOT CONFIRMED")

st.metric(
    "Timeframe Bullish Confirmations",
    f"{bull_count}/{len(available)}"
)
st.metric(
    "Timeframe Bearish Confirmations",
    f"{bear_count}/{len(available)}"
)

reference = results["15 MIN"]

st.write("**15M Analysis Reasons:**")
for reason in reference["reasons"]:
    st.write("•", reason)

levels = make_levels({
    **reference,
    "signal": final
})

if levels:
    st.divider()
    st.subheader("📍 NIFTY SPOT REFERENCE LEVELS")

    st.metric("Entry Reference", f'{reference["price"]:,.2f}')
    st.metric("Stop Loss", f'{levels["sl"]:,.2f}')

    x, y, z = st.columns(3)
    x.metric("Target 1", f'{levels["targets"][0]:,.2f}')
    y.metric("Target 2", f'{levels["targets"][1]:,.2f}')
    z.metric("Target 3", f'{levels["targets"][2]:,.2f}')

st.divider()
st.subheader("📈 15 MINUTE TECHNICAL CHART")

chart = reference["df"][
    ["Close", "EMA9", "EMA21", "EMA50", "EMA200", "VWAP"]
].tail(150)

st.line_chart(chart)

st.subheader("🕒 Data Timestamps")

for name, result in results.items():
    if result is not None:
        st.write(name, result["time"])

st.info(
    "All levels are NIFTY SPOT references, not option premiums. "
    "Yahoo Finance data may be delayed. This version does not place orders."
            )
        
