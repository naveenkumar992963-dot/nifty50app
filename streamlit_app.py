
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date

st.set_page_config(page_title="NIFTY Options Engine", page_icon="📈", layout="wide")

SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FIN NIFTY": "^CNXFIN",
    "SENSEX": "^BSESN"
}

INTERVALS = {
    "5 min": ("5m", "5d"),
    "15 min": ("15m", "1mo"),
    "30 min": ("30m", "1mo"),
    "1 hour": ("60m", "3mo")
}

st.title("📈 NIFTY OPTIONS BUYING ENGINE")
st.caption("CE / PE Trading Analysis Dashboard")

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def download_data(symbol, interval, period):
    data = yf.download(
        symbol, period=period, interval=interval,
        progress=False, auto_adjust=False, threads=False
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
    close, high, low = out["Close"], out["High"], out["Low"]

    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["RSI"] = rsi(close)

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = fast - slow
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    prev = close.shift(1)
    tr = pd.concat([
        high-low, (high-prev).abs(), (low-prev).abs()
    ], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    typical = (high+low+close)/3
    volume = out["Volume"].replace(0, np.nan)
    out["VWAP"] = (
        (typical*volume).groupby(out.index.date).cumsum()
        / volume.groupby(out.index.date).cumsum()
    ).fillna(close)

    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["EMA50", "RSI", "MACD", "ATR"]
    )

def analyze(df):
    last = df.iloc[-1]
    checks = {
        "EMA 9 above EMA 21": last["EMA9"] > last["EMA21"],
        "Price above EMA 50": last["Close"] > last["EMA50"],
        "MACD bullish": last["MACD"] > last["MACD_signal"],
        "Price above VWAP": last["Close"] > last["VWAP"]
    }
    score = sum(1 if value else -1 for value in checks.values())

    if score >= 3:
        trend = "BULLISH"
    elif score <= -3:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    return {
        "trend": trend,
        "score": score,
        "price": float(last["Close"]),
        "rsi": float(last["RSI"]),
        "atr": float(last["ATR"]),
        "checks": checks
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
        [long_condition, short_condition], [1, -1], default=0
    )
    out["Market_Return"] = out["Close"].pct_change().fillna(0)
    out["Strategy_Return"] = (
        out["Position"].shift(1).fillna(0) * out["Market_Return"]
    )
    out["Equity"] = (1+out["Strategy_Return"]).cumprod()
    out["Buy_Hold"] = (1+out["Market_Return"]).cumprod()
    return out

def backtest(df):
    bt = build_strategy_signals(df)
    active = bt.loc[
        bt["Position"].shift(1).fillna(0) != 0, "Strategy_Return"
    ]
    wins = (active > 0).mean()*100 if len(active) else 0
    dd = bt["Equity"]/bt["Equity"].cummax()-1
    return bt, (bt["Equity"].iloc[-1]-1)*100, wins, dd.min()*100

# SETTINGS

with st.sidebar:
    st.header("⚙️ Settings")
    index_name = st.selectbox("Select Index", list(SYMBOLS.keys()))
    timeframe = st.selectbox("Timeframe", list(INTERVALS.keys()), index=1)

    st.divider()
    st.subheader("Risk Management")
    capital = st.number_input("Trading Capital ₹", min_value=1000, value=20000, step=1000)
    risk_pct = st.slider("Risk per trade %", 0.5, 5.0, 1.0, 0.5)
    lot_size = st.number_input("Current Lot Size", min_value=1, value=65)

symbol = SYMBOLS[index_name]
interval, period = INTERVALS[timeframe]

with st.spinner("Loading Market Data..."):
    raw = download_data(symbol, interval, period)

if raw.empty or len(raw) < 60:
    st.error("Market data unavailable. Try another timeframe or index.")
    st.stop()

df = add_indicators(raw)

if df.empty:
    st.error("Insufficient candles for analysis.")
    st.stop()

main = analyze(df)

# MULTI TIMEFRAME

st.subheader("🧭 Multi-Timeframe Confirmation")

tf_rows = []
tf_results = []

for name, (iv, per) in INTERVALS.items():
    try:
        temp_raw = download_data(symbol, iv, per)
        if len(temp_raw) < 60:
            tf_rows.append({"Timeframe": name, "Trend": "Insufficient data"})
            continue

        temp = add_indicators(temp_raw)
        if temp.empty:
            continue

        a = analyze(temp)
        tf_results.append(a["trend"])

        tf_rows.append({
            "Timeframe": name,
            "Trend": a["trend"],
            "Score": f"{a['score']}/4"
        })

    except Exception:
        tf_rows.append({"Timeframe": name, "Trend": "Data error"})

st.dataframe(pd.DataFrame(tf_rows), use_container_width=True, hide_index=True)

bull_count = tf_results.count("BULLISH")
bear_count = tf_results.count("BEARISH")

# OPTIONS SIGNAL

st.subheader("🎯 Options Buying Signal")

rsi_ok_ce = main["rsi"] < 70
rsi_ok_pe = main["rsi"] > 30

if main["trend"] == "BULLISH" and rsi_ok_ce and bull_count >= 2:
    option_signal = "BUY CE"
    option_type = "CE"
    signal_note = "Bullish trend and timeframe confirmation."
elif main["trend"] == "BEARISH" and rsi_ok_pe and bear_count >= 2:
    option_signal = "BUY PE"
    option_type = "PE"
    signal_note = "Bearish trend and timeframe confirmation."
else:
    option_signal = "NO TRADE"
    option_type = None
    if main["trend"] == "BULLISH" and not rsi_ok_ce:
        signal_note = "RSI overbought. Wait for pullback."
    elif main["trend"] == "BEARISH" and not rsi_ok_pe:
        signal_note = "RSI oversold. Wait for bounce."
    else:
        signal_note = "Confirmation not aligned. Avoid entry."

c1, c2, c3, c4 = st.columns(4)

c1.metric("NIFTY Spot", f"{main['price']:,.2f}")
c2.metric("Market Trend", main["trend"])
c3.metric("RSI", f"{main['rsi']:.2f}")
c4.metric("Signal Score", f"{main['score']}/4")

if option_signal == "BUY CE":
    st.success("🟢 BUY CE SETUP CANDIDATE")
elif option_signal == "BUY PE":
    st.error("🔴 BUY PE SETUP CANDIDATE")
else:
    st.warning("🟡 NO TRADE - WAIT")

st.write(signal_note)

with st.expander("Indicator Confirmation"):
    for label, passed in main["checks"].items():
        st.write(("✅ " if passed else "❌ ") + label)

# STRIKE SELECTION

st.divider()
st.subheader("🧮 Option Strike & Premium Calculator")

st.info(
    "Live option chain is not connected. Select the actual contract "
    "and enter its premium from your broker."
)

step = 100 if index_name in ["BANK NIFTY", "SENSEX"] else 50
atm = int(round(main["price"]/step)*step)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Approx ATM Strike", f"{atm:,}")
    strike_choice = st.selectbox(
        "Strike Type",
        ["ATM", "1 step ITM", "1 step OTM", "2 steps OTM"]
    )

with col2:
    expiry = st.date_input("Expiry Date", value=date.today())
    premium = st.number_input(
        "Actual Option Premium ₹",
        min_value=0.0,
        value=100.0,
        step=0.5
    )

with col3:
    actual_strike = st.number_input(
        "Actual Strike from Broker",
        min_value=0,
        value=atm,
        step=step
    )
    extra_cost = st.number_input(
        "Cost Buffer per Unit ₹",
        min_value=0.0,
        value=2.0
    )

suggested_strike = atm

if strike_choice == "1 step ITM":
    suggested_strike += -step if option_type == "CE" else step
elif strike_choice == "1 step OTM":
    suggested_strike += step if option_type == "CE" else -step
elif strike_choice == "2 steps OTM":
    suggested_strike += 2*step if option_type == "CE" else -2*step

st.write(f"**Suggested Strike Reference:** {suggested_strike:,} {option_type or '—'}")

# PREMIUM RISK CALCULATOR

if premium > 0:
    option_sl = premium * 0.80
    target1 = premium * 1.30
    target2 = premium * 1.60

    risk_per_unit = premium - option_sl + extra_cost
    risk_budget = capital * risk_pct / 100

    quantity = int(risk_budget // risk_per_unit) if risk_per_unit > 0 else 0
    quantity = (quantity // int(lot_size)) * int(lot_size)
    lots = quantity // int(lot_size)

    a, b, c, d = st.columns(4)
    a.metric("Premium Stop Loss", f"₹{option_sl:.2f}")
    b.metric("Target 1", f"₹{target1:.2f}")
    c.metric("Target 2", f"₹{target2:.2f}")
    d.metric("Risk Budget", f"₹{risk_budget:,.0f}")

    x, y, z = st.columns(3)
    x.metric("Quantity", f"{quantity:,}")
    y.metric("Lots", str(lots))
    z.metric("Premium Outlay", f"₹{quantity*premium:,.0f}")

    if quantity == 0:
        st.warning("Risk budget is insufficient for one full lot.")

    st.caption(
        "Illustrative premium levels: SL 20% below entry, targets 30% and 60% above. "
        "These are not guaranteed technical levels."
    )

else:
    st.warning("Enter a valid option premium.")

# PRICE CHART

st.divider()
st.subheader("📊 Index Price Chart")

st.line_chart(
    df[["Close", "EMA9", "EMA21", "EMA50", "VWAP"]].tail(250),
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.write("RSI Indicator")
    st.line_chart(df[["RSI"]].tail(250))

with col2:
    st.write("MACD Indicator")
    st.line_chart(df[["MACD", "MACD_signal"]].tail(250))

# BACKTEST

st.divider()
st.subheader("🧪 Historical Backtest")

bt, total_return, win_rate, max_dd = backtest(df)

m1, m2, m3 = st.columns(3)
m1.metric("Strategy Return", f"{total_return:.2f}%")
m2.metric("Winning Bars", f"{win_rate:.2f}%")
m3.metric("Maximum Drawdown", f"{max_dd:.2f}%")

st.line_chart(bt[["Equity", "Buy_Hold"]].tail(500))

with st.expander("Latest Market Data"):
    st.dataframe(
        df.tail(30).sort_index(ascending=False),
        use_container_width=True
    )

st.warning(
    "Options buying can lose the entire premium. This tool does not fetch "
    "live option chain, Greeks, implied volatility or bid-ask spreads. "
    "Verify contract, premium, expiry and lot size with your broker. "
    "No trades are executed."
)

st.caption("Yahoo Finance data • Educational decision support only")
