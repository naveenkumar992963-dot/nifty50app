
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
        
# ==========================================================
# BACKTESTING + STRATEGY ACCURACY TRACKER V2
# Append this block at the END of streamlit_app.py
# ==========================================================

st.divider()
st.title("🧪 STRATEGY BACKTESTING CENTER")
st.caption("Historical testing | Multiple strategies | Risk analysis")

fee_bps = st.number_input(
    "Estimated brokerage and other charges (bps per side)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0
)

slippage_bps = st.number_input(
    "Estimated slippage (bps per side)",
    min_value=0.0,
    max_value=100.0,
    value=2.0,
    step=1.0
)

risk_atr = st.number_input(
    "Stop Loss ATR Multiplier",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1
)

reward_ratio = st.number_input(
    "Target Risk Reward",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

max_hold = st.number_input(
    "Maximum holding candles",
    min_value=1,
    max_value=100,
    value=12
)

bt_interval = st.selectbox(
    "Backtesting timeframe",
    ["5m", "15m", "30m", "60m"]
)

@st.cache_data(ttl=300)
def load_backtest_data(interval):
    return get_data(interval, "60d")


def build_strategy_signals(df):
    x = df.copy()

    close = x["Close"]
    ema9 = x["EMA9"]
    ema21 = x["EMA21"]

    # 1. EMA crossover
    ema_signal = pd.Series(0, index=x.index)
    ema_signal.loc[
        (ema9 > ema21) &
        (ema9.shift(1) <= ema21.shift(1))
    ] = 1
    ema_signal.loc[
        (ema9 < ema21) &
        (ema9.shift(1) >= ema21.shift(1))
    ] = -1

    # 2. RSI reversal
    rsi_signal = pd.Series(0, index=x.index)
    rsi_signal.loc[
        (x["RSI"] > 30) &
        (x["RSI"].shift(1) <= 30)
    ] = 1
    rsi_signal.loc[
        (x["RSI"] < 70) &
        (x["RSI"].shift(1) >= 70)
    ] = -1

    # 3. MACD crossover
    macd_signal = pd.Series(0, index=x.index)
    macd_signal.loc[
        (x["MACD"] > x["MACD_SIGNAL"]) &
        (x["MACD"].shift(1) <= x["MACD_SIGNAL"].shift(1))
    ] = 1
    macd_signal.loc[
        (x["MACD"] < x["MACD_SIGNAL"]) &
        (x["MACD"].shift(1) >= x["MACD_SIGNAL"].shift(1))
    ] = -1

    # 4. VWAP crossover
    vwap_signal = pd.Series(0, index=x.index)
    vwap_signal.loc[
        (close > x["VWAP"]) &
        (close.shift(1) <= x["VWAP"].shift(1))
    ] = 1
    vwap_signal.loc[
        (close < x["VWAP"]) &
        (close.shift(1) >= x["VWAP"].shift(1))
    ] = -1

    # 5. Supertrend direction change
    st_signal = pd.Series(0, index=x.index)
    st_signal.loc[
        (x["ST_DIRECTION"] == 1) &
        (x["ST_DIRECTION"].shift(1) == -1)
    ] = 1
    st_signal.loc[
        (x["ST_DIRECTION"] == -1) &
        (x["ST_DIRECTION"].shift(1) == 1)
    ] = -1

    # 6. Combined strategy
    bull = (
        (ema9 > ema21).astype(int)
        + (x["MACD"] > x["MACD_SIGNAL"]).astype(int)
        + ((x["RSI"] > 50) & (x["RSI"] < 75)).astype(int)
        + (close > x["VWAP"]).astype(int)
        + (x["ST_DIRECTION"] == 1).astype(int)
    )

    bear = (
        (ema9 < ema21).astype(int)
        + (x["MACD"] < x["MACD_SIGNAL"]).astype(int)
        + ((x["RSI"] < 50) & (x["RSI"] > 25)).astype(int)
        + (close < x["VWAP"]).astype(int)
        + (x["ST_DIRECTION"] == -1).astype(int)
    )

    combined_signal = pd.Series(0, index=x.index)
    combined_signal.loc[(bull >= 4) & (bull > bear)] = 1
    combined_signal.loc[(bear >= 4) & (bear > bull)] = -1

    return {
        "EMA Crossover": ema_signal,
        "RSI Reversal": rsi_signal,
        "MACD Momentum": macd_signal,
        "VWAP Crossover": vwap_signal,
        "Supertrend": st_signal,
        "Combined Strategy": combined_signal
    }


def run_backtest(
    df,
    signals,
    sl_multiplier,
    rr,
    holding,
    fee,
    slip
):
    trades = []
    position = None

    # Signals use completed candle i-1.
    # Entry occurs at next candle open i.
    for i in range(1, len(df)):
        row = df.iloc[i]
        previous_signal = int(signals.iloc[i-1])

        if position is None:
            if previous_signal == 0:
                continue

            entry = float(row["Open"])
            atr = float(df["ATR"].iloc[i-1])

            if not np.isfinite(entry) or not np.isfinite(atr):
                continue

            if entry <= 0 or atr <= 0:
                continue

            direction = previous_signal
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

        # Conservative assumption:
        # if SL and target hit in same candle, count SL first.
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
            gross_return = (
                direction * (exit_price - entry) / entry
            )

            total_cost = 2 * (fee + slip) / 10000
            net_return = gross_return - total_cost

            trades.append({
                "Entry Time": position["entry_time"],
                "Exit Time": df.index[i],
                "Direction": "LONG" if direction == 1 else "SHORT",
                "Entry": entry,
                "Exit": exit_price,
                "Gross Return %": gross_return * 100,
                "Net Return %": net_return * 100,
                "Exit Reason": exit_reason,
                "Result": "WIN" if net_return > 0 else "LOSS"
            })

            position = None

    # Close any open position at final available candle.
    if position is not None:
        last = df.iloc[-1]
        exit_price = float(last["Close"])
        entry = position["entry"]
        direction = position["direction"]

        gross_return = direction * (exit_price - entry) / entry
        total_cost = 2 * (fee + slip) / 10000
        net_return = gross_return - total_cost

        trades.append({
            "Entry Time": position["entry_time"],
            "Exit Time": df.index[-1],
            "Direction": "LONG" if direction == 1 else "SHORT",
            "Entry": entry,
            "Exit": exit_price,
            "Gross Return %": gross_return * 100,
            "Net Return %": net_return * 100,
            "Exit Reason": "End of Data",
            "Result": "WIN" if net_return > 0 else "LOSS"
        })

    if not trades:
        return None

    trade_df = pd.DataFrame(trades)

    net = trade_df["Net Return %"]
    wins = net[net > 0]
    losses = net[net <= 0]

    equity = (1 + net / 100).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100

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


if st.button("🚀 RUN STRATEGY BACKTEST", type="primary"):
    with st.spinner("Downloading historical data and testing strategies..."):

        raw_bt = load_backtest_data(bt_interval)

        if raw_bt.empty:
            st.error("Historical data unavailable.")
        else:
            bt_df = add_indicators(raw_bt).dropna().copy()

            if len(bt_df) < 100:
                st.error("Insufficient historical candles.")
            else:
                strategy_signals = build_strategy_signals(bt_df)

                reports = {}

                for name, signal_series in strategy_signals.items():
                    report = run_backtest(
                        bt_df,
                        signal_series,
                        risk_atr,
                        reward_ratio,
                        int(max_hold),
                        fee_bps,
                        slippage_bps
                    )

                    if report is not None:
                        reports[name] = report

                if not reports:
                    st.warning(
                        "No completed trades found. "
                        "Try another timeframe or more data."
                    )
                else:
                    summary = []

                    for name, report in reports.items():
                        pf = report["profit_factor"]

                        summary.append({
                            "Strategy": name,
                            "Trades": report["count"],
                            "Win Rate %": round(report["win_rate"], 2),
                            "Net Return %": round(report["net_return"], 2),
                            "Profit Factor": (
                                round(pf, 2)
                                if np.isfinite(pf) else "∞"
                            ),
                            "Max Drawdown %": round(
                                report["max_drawdown"], 2
                            )
                        })

                    table = pd.DataFrame(summary)

                    st.subheader("🏆 Strategy Performance Ranking")

                    st.dataframe(
                        table.sort_values(
                            "Net Return %",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("📊 Strategy Comparison")

                    st.bar_chart(
                        table.set_index("Strategy")[
                            ["Win Rate %", "Net Return %"]
                        ]
                    )

                    best_name = max(
                        reports,
                        key=lambda n: reports[n]["net_return"]
                    )

                    best = reports[best_name]

                    st.success(
                        f"Highest historical net return: {best_name}"
                    )

                    a, b, c, d = st.columns(4)

                    a.metric("Total Trades", best["count"])
                    b.metric("Win Rate", f'{best["win_rate"]:.2f}%')
                    c.metric(
                        "Profit Factor",
                        (
                            f'{best["profit_factor"]:.2f}'
                            if np.isfinite(best["profit_factor"])
                            else "∞"
                        )
                    )
                    d.metric(
                        "Max Drawdown",
                        f'{best["max_drawdown"]:.2f}%'
                    )

                    st.subheader("📜 Best Strategy Trade History")

                    st.dataframe(
                        best["trades"].tail(100),
                        use_container_width=True,
                        hide_index=True
                    )

                    csv = best["trades"].to_csv(index=False)

                    st.download_button(
                        "⬇️ Download Trade History CSV",
                        csv,
                        file_name="nifty_backtest_trades.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Important: this is a historical spot-price simulation, "
                        "not live option trading. The best strategy is selected "
                        "on the same sample being tested, so it is not proof of "
                        "future performance. Test it on unseen data before use."
            )
                    
