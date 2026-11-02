
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY Advanced Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 ADVANCED PREDICTION ENGINE")
st.caption("Multi-Strategy | 30,000 Scenarios | Backtesting | Paper Analysis")

st.warning(
    "Educational tool only. Scenario shares are simulations, "
    "not guaranteed probabilities or trading success."
)

SCENARIOS = 30000


# =====================================================
# 1. MARKET DATA
# =====================================================

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

        required = ["Open", "High", "Low", "Close"]

        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        return df.dropna(subset=required).copy()

    except Exception:
        return pd.DataFrame()


# =====================================================
# 2. TECHNICAL INDICATORS
# =====================================================

def add_indicators(df):
    x = df.copy()

    close = x["Close"].astype(float)
    high = x["High"].astype(float)
    low = x["Low"].astype(float)

    if "Volume" in x.columns:
        volume = x["Volume"].astype(float)
    else:
        volume = pd.Series(0.0, index=x.index)

    for n in [9, 21, 50, 200]:
        x[f"EMA{n}"] = close.ewm(
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
    x["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    x["ATR"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # ADX
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=x.index
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=x.index
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

    x["PLUS_DI"] = plus_di
    x["MINUS_DI"] = minus_di
    x["ADX"] = dx.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # Bollinger Bands
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    x["BB_UPPER"] = middle + 2 * std
    x["BB_LOWER"] = middle - 2 * std

    # Intraday VWAP
    typical = (high + low + close) / 3
    dates = pd.Series(x.index.date, index=x.index)

    cumulative_pv = (typical * volume).groupby(dates).cumsum()
    cumulative_volume = volume.groupby(dates).cumsum()

    x["VWAP"] = (
        cumulative_pv / cumulative_volume.replace(0, np.nan)
    )

    # Volume ratio
    avg_volume = volume.rolling(20).mean()
    x["VOLUME_RATIO"] = (
        volume / avg_volume.replace(0, np.nan)
    )

    # Supertrend
    midpoint = (high + low) / 2
    basic_upper = midpoint + 3 * x["ATR"]
    basic_lower = midpoint - 3 * x["ATR"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    direction = pd.Series(1, index=x.index)
    supertrend = pd.Series(np.nan, index=x.index)

    for i in range(1, len(x)):
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

    x["ST_DIRECTION"] = direction
    x["SUPERTREND"] = supertrend

    # Previous 20 candle breakout levels
    x["PREV_HIGH"] = high.shift(1).rolling(20).max()
    x["PREV_LOW"] = low.shift(1).rolling(20).min()

    return x


# =====================================================
# 3. MULTI-STRATEGY SCANNER
# =====================================================

def analyze_market(df):
    if len(df) < 50:
        return None

    row = df.iloc[-1]

    required = [
        "Close", "EMA9", "EMA21", "EMA50",
        "RSI", "MACD", "MACD_SIGNAL",
        "ATR", "ADX", "PLUS_DI", "MINUS_DI"
    ]

    if not np.isfinite([row[c] for c in required]).all():
        return None

    bull = 0
    bear = 0
    reasons = []

    # EMA alignment
    if row["EMA9"] > row["EMA21"] > row["EMA50"]:
        bull += 20
        reasons.append("Bullish EMA alignment")
    elif row["EMA9"] < row["EMA21"] < row["EMA50"]:
        bear += 20
        reasons.append("Bearish EMA alignment")

    # MACD
    if row["MACD"] > row["MACD_SIGNAL"]:
        bull += 15
        reasons.append("MACD bullish")
    else:
        bear += 15
        reasons.append("MACD bearish")

    # RSI
    if 55 < row["RSI"] < 75:
        bull += 15
        reasons.append("RSI bullish")
    elif 25 < row["RSI"] < 45:
        bear += 15
        reasons.append("RSI bearish")

    # VWAP
    if np.isfinite(row["VWAP"]):
        if row["Close"] > row["VWAP"]:
            bull += 15
            reasons.append("Price above VWAP")
        else:
            bear += 15
            reasons.append("Price below VWAP")

    # ADX
    if row["ADX"] > 20:
        if row["PLUS_DI"] > row["MINUS_DI"]:
            bull += 10
            reasons.append("ADX bullish trend")
        else:
            bear += 10
            reasons.append("ADX bearish trend")

    # Supertrend
    if row["ST_DIRECTION"] == 1:
        bull += 10
        reasons.append("Supertrend bullish")
    else:
        bear += 10
        reasons.append("Supertrend bearish")

    # Breakout
    if np.isfinite(row["PREV_HIGH"]) and row["Close"] > row["PREV_HIGH"]:
        bull += 10
        reasons.append("20 candle high breakout")

    elif np.isfinite(row["PREV_LOW"]) and row["Close"] < row["PREV_LOW"]:
        bear += 10
        reasons.append("20 candle low breakdown")

    # Volume
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
        "time": df.index[-1],
        "reasons": reasons
    }


# =====================================================
# 4. 30,000 MONTE CARLO SCENARIOS
# =====================================================

def simulate_scenarios(df):
    close = df["Close"].astype(float)
    returns = np.log(close / close.shift()).dropna()

    if len(returns) < 50:
        return None

    recent = returns.tail(96)

    if len(recent) < 30:
        return None

    drift = (
        0.35 * returns.tail(12).mean()
        + 0.15 * returns.tail(48).mean()
    )

    residuals = (
        recent - recent.mean()
    ).dropna().to_numpy()

    volatility = recent.std()

    if len(residuals) < 20 or not np.isfinite(volatility) or volatility <= 0:
        return None

    rng = np.random.default_rng()

    samples = rng.choice(
        residuals,
        size=(SCENARIOS, 6),
        replace=True
    )

    simulated_returns = samples + drift

    final_returns = np.exp(
        simulated_returns.sum(axis=1)
    ) - 1

    price = float(close.iloc[-1])
    atr = float(df["ATR"].iloc[-1])

    threshold = max(
        0.20 * atr / price,
        volatility * np.sqrt(6) * 0.15
    )

    bullish = int(np.sum(final_returns > threshold))
    bearish = int(np.sum(final_returns < -threshold))
    sideways = SCENARIOS - bullish - bearish

    return {
        "bullish": bullish,
        "bearish": bearish,
        "sideways": sideways,
        "bull_pct": bullish / SCENARIOS * 100,
        "bear_pct": bearish / SCENARIOS * 100,
        "side_pct": sideways / SCENARIOS * 100
    }


# =====================================================
# 5. BACKTEST STRATEGY SIGNALS
# =====================================================

def build_strategy_signals(df):
    x = df
    close = x["Close"]

    ema = pd.Series(0, index=x.index)
    ema.loc[
        (x["EMA9"] > x["EMA21"]) &
        (x["EMA9"].shift(1) <= x["EMA21"].shift(1))
    ] = 1
    ema.loc[
        (x["EMA9"] < x["EMA21"]) &
        (x["EMA9"].shift(1) >= x["EMA21"].shift(1))
    ] = -1

    rsi = pd.Series(0, index=x.index)
    rsi.loc[
        (x["RSI"] > 30) & (x["RSI"].shift(1) <= 30)
    ] = 1
    rsi.loc[
        (x["RSI"] < 70) & (x["RSI"].shift(1) >= 70)
    ] = -1

    macd = pd.Series(0, index=x.index)
    macd.loc[
        (x["MACD"] > x["MACD_SIGNAL"]) &
        (x["MACD"].shift(1) <= x["MACD_SIGNAL"].shift(1))
    ] = 1
    macd.loc[
        (x["MACD"] < x["MACD_SIGNAL"]) &
        (x["MACD"].shift(1) >= x["MACD_SIGNAL"].shift(1))
    ] = -1

    vwap = pd.Series(0, index=x.index)
    vwap.loc[
        (close > x["VWAP"]) &
        (close.shift(1) <= x["VWAP"].shift(1))
    ] = 1
    vwap.loc[
        (close < x["VWAP"]) &
        (close.shift(1) >= x["VWAP"].shift(1))
    ] = -1

    supertrend = pd.Series(0, index=x.index)
    supertrend.loc[
        (x["ST_DIRECTION"] == 1) &
        (x["ST_DIRECTION"].shift(1) == -1)
    ] = 1
    supertrend.loc[
        (x["ST_DIRECTION"] == -1) &
        (x["ST_DIRECTION"].shift(1) == 1)
    ] = -1

    bull = (
        (x["EMA9"] > x["EMA21"]).astype(int)
        + (x["MACD"] > x["MACD_SIGNAL"]).astype(int)
        + ((x["RSI"] > 50) & (x["RSI"] < 75)).astype(int)
        + (close > x["VWAP"]).astype(int)
        + (x["ST_DIRECTION"] == 1).astype(int)
    )

    bear = (
        (x["EMA9"] < x["EMA21"]).astype(int)
        + (x["MACD"] < x["MACD_SIGNAL"]).astype(int)
        + ((x["RSI"] < 50) & (x["RSI"] > 25)).astype(int)
        + (close < x["VWAP"]).astype(int)
        + (x["ST_DIRECTION"] == -1).astype(int)
    )

    combined = pd.Series(0, index=x.index)
    combined.loc[(bull >= 4) & (bull > bear)] = 1
    combined.loc[(bear >= 4) & (bear > bull)] = -1

    return {
        "EMA Crossover": ema,
        "RSI Reversal": rsi,
        "MACD Momentum": macd,
        "VWAP Crossover": vwap,
        "Supertrend": supertrend,
        "Combined Strategy": combined
    }


# =====================================================
# 6. BACKTEST ENGINE
# =====================================================

def run_backtest(
    df,
    signals,
    sl_multiplier,
    rr,
    holding,
    fee_bps,
    slip_bps
):
    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        signal = int(signals.iloc[i-1])

        if position is None:
            if signal == 0:
                continue

            entry = float(row["Open"])
            atr = float(df["ATR"].iloc[i-1])

            if not np.isfinite(entry) or not np.isfinite(atr):
                continue

            if entry <= 0 or atr <= 0:
                continue

            direction = signal
            distance = sl_multiplier * atr

            position = {
                "direction": direction,
                "entry": entry,
                "sl": entry - direction * distance,
                "target": entry + direction * distance * rr,
                "entry_index": i,
                "entry_time": df.index[i]
            }
            continue

        direction = position["direction"]

        stop_hit = (
            float(row["Low"]) <= position["sl"]
            if direction == 1
            else float(row["High"]) >= position["sl"]
        )

        target_hit = (
            float(row["High"]) >= position["target"]
            if direction == 1
            else float(row["Low"]) <= position["target"]
        )

        exit_price = None
        exit_reason = None

        # Conservative: SL wins if SL and target hit in same candle.
        if stop_hit:
            exit_price = position["sl"]
            exit_reason = "Stop Loss"

        elif target_hit:
            exit_price = position["target"]
            exit_reason = "Target"

        elif i - position["entry_index"] >= holding:
            exit_price = float(row["Close"])
            exit_reason = "Time Exit"

        if exit_price is not None:
            entry = position["entry"]

            gross = direction * (exit_price - entry) / entry
            costs = 2 * (fee_bps + slip_bps) / 10000
            net = gross - costs

            trades.append({
                "Entry Time": position["entry_time"],
                "Exit Time": df.index[i],
                "Direction": "LONG" if direction == 1 else "SHORT",
                "Entry": entry,
                "Exit": exit_price,
                "Gross Return %": gross * 100,
                "Net Return %": net * 100,
                "Exit Reason": exit_reason,
                "Result": "WIN" if net > 0 else "LOSS"
            })

            position = None

    if position is not None:
        row = df.iloc[-1]
        exit_price = float(row["Close"])
        entry = position["entry"]
        direction = position["direction"]

        gross = direction * (exit_price - entry) / entry
        costs = 2 * (fee_bps + slip_bps) / 10000
        net = gross - costs

        trades.append({
            "Entry Time": position["entry_time"],
            "Exit Time": df.index[-1],
            "Direction": "LONG" if direction == 1 else "SHORT",
            "Entry": entry,
            "Exit": exit_price,
            "Gross Return %": gross * 100,
            "Net Return %": net * 100,
            "Exit Reason": "End of Data",
            "Result": "WIN" if net > 0 else "LOSS"
        })

    if not trades:
        return None

    trade_df = pd.DataFrame(trades)
    net = trade_df["Net Return %"]

    wins = net[net > 0]
    losses = net[net <= 0]

    equity = (1 + net / 100).cumprod()
    drawdown = (equity / equity.cummax() - 1) * 100

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.inf
    )

    return {
        "trades": trade_df,
        "count": len(trade_df),
        "win_rate": (net > 0).mean() * 100,
        "net_return": (equity.iloc[-1] - 1) * 100,
        "profit_factor": profit_factor,
        "max_drawdown": abs(drawdown.min()),
        "wins": int((net > 0).sum()),
        "losses": int((net <= 0).sum())
    }


# =====================================================
# 7. MAIN LIVE SCANNER
# =====================================================

if st.button("🔄 REFRESH SCANNER"):
    st.cache_data.clear()

periods = {
    "5 MIN": ("5m", "5d"),
    "15 MIN": ("15m", "60d"),
    "30 MIN": ("30m", "60d"),
    "1 HOUR": ("60m", "730d")
}

results = {}

with st.spinner("Analyzing Indian market data..."):
    for name, (interval, period) in periods.items():
        raw = get_data(interval, period)

        if not raw.empty:
            prepared = add_indicators(raw)
            results[name] = analyze_market(prepared)

if "5 MIN" not in results or "15 MIN" not in results:
    st.error("Market data unavailable or insufficient.")
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

# =====================================================
# 8. SCENARIO ANALYSIS
# =====================================================

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
        "Historical bootstrap simulation. Scenario shares are not "
        "calibrated probabilities of future market movement."
    )
else:
    st.warning("Insufficient data for scenario analysis.")

# =====================================================
# 9. FINAL CONFIRMATION
# =====================================================

st.divider()
st.subheader("🎯 FINAL DEEP ANALYSIS")

available = [
    r for r in results.values()
    if r is not None
]

bull_count = sum(
    r["signal"] == "CALL BIAS" for r in available
)

bear_count = sum(
    r["signal"] == "PUT BIAS" for r in available
)

higher = [
    results.get("30 MIN"),
    results.get("1 HOUR")
]

call_confirm = (
    bull_count >= 3
    and results["15 MIN"]["signal"] == "CALL BIAS"
    and any(
        x is not None and x["signal"] == "CALL BIAS"
        for x in higher
    )
    and simulation is not None
    and simulation["bull_pct"] >= 55
)

put_confirm = (
    bear_count >= 3
    and results["15 MIN"]["signal"] == "PUT BIAS"
    and any(
        x is not None and x["signal"] == "PUT BIAS"
        for x in higher
    )
    and simulation is not None
    and simulation["bear_pct"] >= 55
)

if call_confirm:
    final = "CALL BIAS"
    st.success("FINAL RESULT: STRONG CALL BIAS")

elif put_confirm:
    final = "PUT BIAS"
    st.error("FINAL RESULT: STRONG PUT BIAS")

else:
    final = "NO TRADE"
    st.warning("NO TRADE — CONDITIONS NOT CONFIRMED")

st.metric(
    "Bullish Timeframe Confirmations",
    f"{bull_count}/{len(available)}"
)

st.metric(
    "Bearish Timeframe 
