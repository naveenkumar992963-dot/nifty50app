
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="NIFTY 50 Advanced Analysis",
    page_icon="📈",
    layout="wide"
)

SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FIN NIFTY": "^CNXFIN",
    "SENSEX": "^BSESN",
    "NIFTY MIDCAP SELECT": "^NSEMDCP50"
}

INTERVALS = {
    "5 min": ("5m", "5d"),
    "15 min": ("15m", "1mo"),
    "30 min": ("30m", "1mo"),
    "1 hour": ("60m", "3mo")
}

st.title("📈 NIFTY 50 ADVANCED PREDICTION ENGINE")
st.caption("Market analysis dashboard — signals are estimates, not guaranteed predictions.")

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def download_data(symbol, interval, period):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["Open", "High", "Low", "Close"])

def add_indicators(df):
    out = df.copy()

    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()

    out["RSI"] = rsi(close)

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()

    out["MACD"] = fast - slow
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    previous_close = close.shift(1)

    tr = pd.concat([
        high-low,
        (high-previous_close).abs(),
        (low-previous_close).abs()
    ], axis=1).max(axis=1)

    out["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    typical = (high+low+close)/3
    volume = out["Volume"].replace(0, np.nan)

    out["VWAP"] = (
        (typical*volume).groupby(out.index.date).cumsum()
        / volume.groupby(out.index.date).cumsum()
    ).fillna(close)

    midpoint = (high+low)/2
    upper = midpoint + 3*out["ATR"]
    lower = midpoint - 3*out["ATR"]

    st_line = pd.Series(index=out.index, dtype=float)
    trend_up = pd.Series(index=out.index, dtype=bool)

    for i in range(len(out)):
        if i == 0:
            st_line.iloc[i] = lower.iloc[i]
            trend_up.iloc[i] = True
            continue

        prev_line = st_line.iloc[i-1]

        if close.iloc[i] >= prev_line:
            trend_up.iloc[i] = True
            st_line.iloc[i] = (
                max(lower.iloc[i], prev_line)
                if trend_up.iloc[i-1] else lower.iloc[i]
            )
        else:
            trend_up.iloc[i] = False
            st_line.iloc[i] = (
                min(upper.iloc[i], prev_line)
                if not trend_up.iloc[i-1] else upper.iloc[i]
            )

    out["Supertrend"] = st_line.ffill()
    out["ST_Up"] = trend_up.fillna(True)

    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["EMA50", "RSI", "MACD", "ATR"]
    )

def signal_for(df):
    last = df.iloc[-1]

    score = 0
    reasons = []

    checks = [
        (last["EMA9"] > last["EMA21"], "EMA 9 / EMA 21"),
        (last["Close"] > last["EMA50"], "EMA 50"),
        (last["MACD"] > last["MACD_signal"], "MACD"),
        (last["Close"] > last["VWAP"], "VWAP"),
        (bool(last["ST_Up"]), "Supertrend")
    ]

    for condition, name in checks:
        if condition:
            score += 1
            reasons.append(name + " bullish")
        else:
            score -= 1
            reasons.append(name + " bearish")

    if score >= 3:
        signal = "BULLISH"
    elif score <= -3:
        signal = "BEARISH"
    else:
        signal = "WAIT / SIDEWAYS"

    price = float(last["Close"])
    atr = max(float(last["ATR"]), price*0.001)

    if signal == "BULLISH":
        sl = price - 1.5*atr
        target1 = price + 1.5*atr
        target2 = price + 3*atr

    elif signal == "BEARISH":
        sl = price + 1.5*atr
        target1 = price - 1.5*atr
        target2 = price - 3*atr

    else:
        sl = np.nan
        target1 = np.nan
        target2 = np.nan

    return {
        "signal": signal,
        "score": score,
        "price": price,
        "sl": sl,
        "target1": target1,
        "target2": target2,
        "rsi": float(last["RSI"]),
        "atr": atr,
        "reasons": reasons
    }

def build_strategy_signals(df):
    out = df.copy()

    long_condition = (
        (out["EMA9"] > out["EMA21"])
        & (out["MACD"] > out["MACD_signal"])
        & (out["Close"] > out["EMA50"])
    )

    short_condition = (
        (out["EMA9"] < out["EMA21"])
        & (out["MACD"] < out["MACD_signal"])
        & (out["Close"] < out["EMA50"])
    )

    out["Position"] = np.select(
        [long_condition, short_condition],
        [1, -1],
        default=0
    )

    out["Market_Return"] = out["Close"].pct_change().fillna(0)

    out["Strategy_Return"] = (
        out["Position"].shift(1).fillna(0)
        * out["Market_Return"]
    )

    out["Equity"] = (1+out["Strategy_Return"]).cumprod()
    out["Buy_Hold"] = (1+out["Market_Return"]).cumprod()

    return out

def run_backtest(df):
    bt = build_strategy_signals(df)

    active = bt.loc[
        bt["Position"].shift(1).fillna(0) != 0,
        "Strategy_Return"
    ]

    win_rate = (
        (active > 0).mean()*100
        if len(active) else 0
    )

    drawdown = bt["Equity"]/bt["Equity"].cummax()-1

    return {
        "data": bt,
        "return": (bt["Equity"].iloc[-1]-1)*100,
        "buy_hold": (bt["Buy_Hold"].iloc[-1]-1)*100,
        "win_rate": win_rate,
        "max_drawdown": drawdown.min()*100
    }

def monte_carlo_projection(df, simulations=30000, steps=20):
    returns = df["Close"].pct_change().dropna().to_numpy()

    if len(returns) < 30:
        return None

    rng = np.random.default_rng(42)

    sample = rng.choice(
        returns,
        size=(simulations, steps),
        replace=True
    )

    paths = df["Close"].iloc[-1] * np.cumprod(1+sample, axis=1)
    final_values = paths[:, -1]

    return {
        "p10": np.percentile(final_values, 10),
        "p50": np.percentile(final_values, 50),
        "p90": np.percentile(final_values, 90),
        "up_probability": (
            final_values > df["Close"].iloc[-1]
        ).mean()*100
    }

# SIDEBAR

with st.sidebar:
    st.header("⚙️ Settings")

    selected_index = st.selectbox(
        "Select Index",
        list(SYMBOLS.keys())
    )

    selected_tf = st.selectbox(
        "Select Timeframe",
        list(INTERVALS.keys())
    )

    show_simulation = st.checkbox(
        "Monte Carlo Simulation",
        value=True
    )

symbol = SYMBOLS[selected_index]
interval, period = INTERVALS[selected_tf]

# MARKET DATA

with st.spinner("Loading market data..."):
    raw_df = download_data(symbol, interval, period)

if raw_df.empty or len(raw_df) < 60:
    st.error("Market data unavailable. Please try another timeframe or index.")
    st.stop()

df = add_indicators(raw_df)

if df.empty:
    st.error("Not enough candles to calculate indicators.")
    st.stop()

result = signal_for(df)

price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
change_pct = price_change / df["Close"].iloc[-2] * 100

# MAIN DASHBOARD

st.subheader("📊 Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latest Price",
    f"{result['price']:,.2f}",
    f"{change_pct:+.2f}%"
)

c2.metric(
    "Market Signal",
    result["signal"],
    f"Score {result['score']}/5"
)

c3.metric("RSI 14", f"{result['rsi']:.2f}")

c4.metric("ATR", f"{result['atr']:.2f}")

# TRADE SETUP

st.subheader("🎯 Trading Strategy")

if result["signal"] == "WAIT / SIDEWAYS":
    st.warning("No clear entry. Wait for confirmation.")

else:
    a, b, c = st.columns(3)

    a.metric("Stop Loss", f"{result['sl']:,.2f}")
    b.metric("Target 1", f"{result['target1']:,.2f}")
    c.metric("Target 2", f"{result['target2']:,.2f}")

    st.caption("ATR-based indicative index levels, not option premiums.")

with st.expander("Signal Explanation"):
    for reason in result["reasons"]:
        st.write("• " + reason)

# CHARTS

st.subheader("📈 Price Chart")

chart_data = df[
    ["Close", "EMA9", "EMA21", "EMA50", "VWAP"]
].tail(250)

st.line_chart(chart_data, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.write("RSI Indicator")
    st.line_chart(df[["RSI"]].tail(250))

with col2:
    st.write("MACD Indicator")
    st.line_chart(
        df[["MACD", "MACD_signal"]].tail(250)
    )

# MULTI TIMEFRAME

st.subheader("🧭 Multi-Timeframe Analysis")

timeframe_rows = []

for tf_name, (tf_interval, tf_period) in INTERVALS.items():

    try:
        tf_raw = download_data(
            symbol,
            tf_interval,
            tf_period
        )

        if len(tf_raw) < 60:
            timeframe_rows.append({
                "Timeframe": tf_name,
                "Signal": "Insufficient data",
                "RSI": "—"
            })
            continue

        tf_df = add_indicators(tf_raw)
        tf_signal = signal_for(tf_df)

        timeframe_rows.append({
            "Timeframe": tf_name,
            "Signal": tf_signal["signal"],
            "RSI": round(tf_signal["rsi"], 2),
            "Price": round(tf_signal["price"], 2)
        })

    except Exception:
        timeframe_rows.append({
            "Timeframe": tf_name,
            "Signal": "Data error",
            "RSI": "—"
        })

st.dataframe(
    pd.DataFrame(timeframe_rows),
    use_container_width=True,
    hide_index=True
)

# BACKTEST

st.subheader("🧪 Historical Backtesting")

bt_result = run_backtest(df)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Strategy Return", f"{bt_result['return']:.2f}%")
m2.metric("Buy & Hold", f"{bt_result['buy_hold']:.2f}%")
m3.metric("Winning Bars", f"{bt_result['win_rate']:.2f}%")
m4.metric("Maximum Drawdown", f"{bt_result['max_drawdown']:.2f}%")

st.line_chart(
    bt_result["data"][["Equity", "Buy_Hold"]].tail(500)
)

st.caption(
    "Simplified historical simulation. Brokerage, taxes, slippage "
    "and option pricing are excluded."
)

# MONTE CARLO

if show_simulation:

    st.subheader("🎲 Monte Carlo Analysis")

    with st.spinner("Running 30,000 simulations..."):
        mc = monte_carlo_projection(
            df,
            simulations=30000,
            steps=20
        )

    if mc is None:
        st.warning("Not enough historical data.")

    else:
        x1, x2, x3, x4 = st.columns(4)

        x1.metric("10th Percentile", f"{mc['p10']:,.2f}")
        x2.metric("Median Scenario", f"{mc['p50']:,.2f}")
        x3.metric("90th Percentile", f"{mc['p90']:,.2f}")
        x4.metric("Simulated Up Probability", f"{mc['up_probability']:.2f}%")

        st.caption(
            "Historical resampling scenarios are not guaranteed forecasts."
        )

# DATA TABLE

with st.expander("View Latest Market Data"):
    st.dataframe(
        df.tail(30).sort_index(ascending=False),
        use_container_width=True
    )

st.caption(
    "Data source: Yahoo Finance via yfinance. "
    "Educational analysis only. No trades are executed."
    )
