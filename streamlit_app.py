import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Multi-Timeframe Paper Trading", page_icon="📈", layout="wide")
st.title("📈 NIFTY 50 Multi-Timeframe Trading System")
st.caption("3M + 5M + 15M | EMA + RSI + MACD + Supertrend | Paper Trading (index-point simulation)")

st_autorefresh(interval=15000, key="upstox_auto_refresh")

# ---------------- SESSION STATE ----------------
if "paper_balance" not in st.session_state:
    st.session_state.paper_balance = 10000.0
if "paper_open_trade" not in st.session_state:
    st.session_state.paper_open_trade = None
if "paper_history" not in st.session_state:
    st.session_state.paper_history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Dashboard Settings")
    risk_percent = st.slider("Virtual risk per trade (%)", 0.5, 2.0, 1.0, 0.5)
    st.divider()
    st.subheader("Paper Trading Account")
    reset_capital = st.number_input("Reset virtual capital (₹)", min_value=1000, value=10000, step=1000)
    if st.button("♻️ RESET PAPER ACCOUNT", use_container_width=True):
        st.session_state.paper_balance = float(reset_capital)
        st.session_state.paper_open_trade = None
        st.session_state.paper_history = []
        st.success("Paper account reset.")
    if st.button("🔄 REFRESH DATA", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Source: Upstox API V3 | 1-minute candle polling")
    st.caption("No broker orders are sent.")

# ---------------- DATA ----------------
IST = ZoneInfo("Asia/Kolkata")
UPSTOX_BASE = "https://api.upstox.com"

def get_upstox_token():
    try:
        return str(st.secrets["UPSTOX_ACCESS_TOKEN"]).strip()
    except Exception:
        return ""

def get_instrument_key():
    try:
        return str(st.secrets.get("UPSTOX_INSTRUMENT_KEY", "NSE_INDEX|Nifty 50")).strip()
    except Exception:
        return "NSE_INDEX|Nifty 50"

def candles_to_frame(candles):
    """Convert Upstox V3 candle arrays to a timezone-aware OHLCV DataFrame."""
    rows = []
    for candle in candles or []:
        if len(candle) >= 6:
            rows.append([candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]])

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    frame = pd.DataFrame(
        rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame = frame.dropna(subset=["Timestamp"])

    if frame["Timestamp"].dt.tz is None:
        frame["Timestamp"] = frame["Timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        frame["Timestamp"] = frame["Timestamp"].dt.tz_convert("Asia/Kolkata")

    frame = frame.set_index("Timestamp").sort_index()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=3600, show_spinner=False)
def get_previous_sessions(token, instrument_key, from_date, to_date):
    """Fetch recent 1m history to warm up EMA50 on the 15m timeframe."""
    encoded_key = quote(instrument_key, safe="")
    url = (
        f"{UPSTOX_BASE}/v3/historical-candle/{encoded_key}/"
        f"minutes/1/{to_date}/{from_date}"
    )
    response = requests.get(
        url,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        timeout=20
    )
    if response.status_code == 401:
        raise RuntimeError("Upstox token invalid/expired. नया Analytics Token बनाकर Streamlit Secrets update करो.")
    response.raise_for_status()
    return candles_to_frame(response.json().get("data", {}).get("candles", []))

@st.cache_data(ttl=12, show_spinner=False)
def get_today_intraday(token, instrument_key, refresh_bucket):
    """Fetch today's 1m candles. refresh_bucket changes during timed app reruns."""
    encoded_key = quote(instrument_key, safe="")
    url = f"{UPSTOX_BASE}/v3/historical-candle/intraday/{encoded_key}/minutes/1"
    response = requests.get(
        url,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        timeout=20
    )
    if response.status_code == 401:
        raise RuntimeError("Upstox token invalid/expired. नया Analytics Token बनाकर Streamlit Secrets update करो.")
    response.raise_for_status()
    return candles_to_frame(response.json().get("data", {}).get("candles", []))

def download_data():
    token = ""
    try:
        token = str(st.secrets["UPSTOX_ACCESS_TOKEN"]).strip()
    except Exception:
        pass

    if not token:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN missing है. Streamlit Cloud → Manage app → Settings → Secrets में token add करो."
        )

    instrument_key = get_instrument_key()
    now = datetime.now(IST)
    from_date = (now.date() - timedelta(days=7)).isoformat()
    to_date = (now.date() - timedelta(days=1)).isoformat()

    historical = get_previous_sessions(token, instrument_key, from_date, to_date)
    today = get_today_intraday(
        token, instrument_key, int(now.timestamp() // 12)
    )

    combined = pd.concat([historical, today])
    if combined.empty:
        return combined

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined[["Open", "High", "Low", "Close", "Volume"]]


def make_timeframe(data, minutes):
    result = data.resample(f"{minutes}min", origin="start_day", offset="15min").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    if len(result):
        if result.index[-1] + pd.Timedelta(minutes=minutes) > now:
            result = result.iloc[:-1]
    return result

# ---------------- INDICATORS ----------------
def add_indicators(data):
    df = data.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Supertrend
    multiplier = 3.0
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * df["ATR"]
    basic_lower = hl2 - multiplier * df["ATR"]
    final_upper, final_lower = basic_upper.copy(), basic_lower.copy()

    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    st_line = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            st_dir.iloc[i], st_line.iloc[i] = 1, final_lower.iloc[i]
        elif st_dir.iloc[i-1] == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                st_dir.iloc[i], st_line.iloc[i] = -1, final_upper.iloc[i]
            else:
                st_dir.iloc[i], st_line.iloc[i] = 1, final_lower.iloc[i]
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                st_dir.iloc[i], st_line.iloc[i] = 1, final_lower.iloc[i]
            else:
                st_dir.iloc[i], st_line.iloc[i] = -1, final_upper.iloc[i]

    df["Supertrend"] = st_line
    df["ST_DIRECTION"] = st_dir
    return df.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------- VWAP ----------------
def add_vwap_analysis(data):
    df = data.copy()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["VWAP"] = np.nan
    df["Volume_AVG"] = np.nan
    if (df["Volume"] > 0).any():
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        dates = pd.Series(df.index.date, index=df.index)
        pv = (typical * df["Volume"]).groupby(dates).cumsum()
        vol = df["Volume"].groupby(dates).cumsum()
        df["VWAP"] = pv / vol.replace(0, np.nan)
        df["Volume_AVG"] = df["Volume"].rolling(20, min_periods=20).mean()
    return df

# ---------------- ANALYSIS ----------------
def analyze_timeframe(df):
    if len(df) < 55:
        return None
    last = df.iloc[-1]
    bull = last["EMA9"] > last["EMA21"] > last["EMA50"] and last["Close"] > last["EMA21"] and last["ST_DIRECTION"] == 1
    bear = last["EMA9"] < last["EMA21"] < last["EMA50"] and last["Close"] < last["EMA21"] and last["ST_DIRECTION"] == -1
    return {
        "price": float(last["Close"]), "rsi": float(last["RSI"]), "atr": float(last["ATR"]),
        "ema9": float(last["EMA9"]), "ema21": float(last["EMA21"]), "ema50": float(last["EMA50"]),
        "macd_bullish": bool(last["MACD"] > last["MACD_SIGNAL"]),
        "macd_bearish": bool(last["MACD"] < last["MACD_SIGNAL"]),
        "bullish": bool(bull), "bearish": bool(bear),
        "supertrend": float(last["Supertrend"]), "time": str(df.index[-1])
    }

def final_signal(a3, a5, a15):
    bull_count = sum([a3["bullish"], a5["bullish"], a15["bullish"]])
    bear_count = sum([a3["bearish"], a5["bearish"], a15["bearish"]])
    reasons = []
    if a3["bullish"] and a5["bullish"] and a15["bullish"]:
        extension = (a3["price"] - a3["ema21"]) / max(a3["atr"], 0.01)
        if extension > 2.5:
            return "NO TRADE", ["Bullish trend, but price is too extended."]
        return "BUY", ["15m, 5m and 3m bullish trend confirmed."]
    if a3["bearish"] and a5["bearish"] and a15["bearish"]:
        extension = (a3["ema21"] - a3["price"]) / max(a3["atr"], 0.01)
        if extension > 2.5:
            return "NO TRADE", ["Bearish trend, but price is too extended."]
        return "SELL", ["15m, 5m and 3m bearish trend confirmed."]
    return "NO TRADE", [f"Bullish timeframes: {bull_count}/3", f"Bearish timeframes: {bear_count}/3", "All three timeframes are not aligned."]

# ---------------- PAPER TRADE EXIT ----------------
def close_paper_trade(exit_price, reason):
    trade = st.session_state.paper_open_trade
    if trade is None:
        return
    direction = 1 if trade["side"] == "BUY" else -1
    points = (float(exit_price) - trade["entry"]) * direction
    virtual_pnl = points * trade["units"]
    st.session_state.paper_balance += virtual_pnl
    st.session_state.paper_history.insert(0, {
        "Side": trade["side"], "Entry Time": trade["entry_time"],
        "Exit Time": str(pd.Timestamp.now(tz="Asia/Kolkata")),
        "Entry": round(trade["entry"], 2), "Exit": round(float(exit_price), 2),
        "Stop Loss": round(trade["sl"], 2), "Target 1": round(trade["target"], 2),
        "Points P&L": round(points, 2), "Illustrative Virtual P&L (₹)": round(virtual_pnl, 2),
        "Exit Reason": reason
    })
    st.session_state.paper_open_trade = None

# ---------------- MAIN DATA PROCESSING ----------------
try:
    raw = download_data()
    if raw.empty:
        st.error("Upstox market data unavailable. Token, instrument key या API response जाँचें.")
        st.stop()

    df3 = add_indicators(make_timeframe(raw, 3))
    df5 = add_indicators(make_timeframe(raw, 5))
    df15 = add_indicators(make_timeframe(raw, 15))
    if min(len(df3), len(df5), len(df15)) < 55:
        st.error("Not enough completed candles.")
        st.stop()

    a3, a5, a15 = analyze_timeframe(df3), analyze_timeframe(df5), analyze_timeframe(df15)
    if not all([a3, a5, a15]):
        st.error("Timeframe analysis unavailable.")
        st.stop()

    signal, reasons = final_signal(a3, a5, a15)
    rsi_bull = sum(x > 50 for x in [a3["rsi"], a5["rsi"], a15["rsi"]])
    rsi_bear = sum(x < 50 for x in [a3["rsi"], a5["rsi"], a15["rsi"]])
    macd_bull = sum([a3["macd_bullish"], a5["macd_bullish"], a15["macd_bullish"]])
    macd_bear = sum([a3["macd_bearish"], a5["macd_bearish"], a15["macd_bearish"]])

    if signal == "BUY":
        if int(rsi_bull >= 2) + int(macd_bull >= 2) < 2:
            signal = "NO TRADE"
            reasons.append("BUY rejected: RSI/MACD confirmation weak.")
        else:
            reasons.extend(["RSI bullish confirmation received.", "MACD bullish confirmation received."])
    elif signal == "SELL":
        if int(rsi_bear >= 2) + int(macd_bear >= 2) < 2:
            signal = "NO TRADE"
            reasons.append("SELL rejected: RSI/MACD confirmation weak.")
        else:
            reasons.extend(["RSI bearish confirmation received.", "MACD bearish confirmation received."])

    # Feed freshness guard (NSE regular session, IST).
    latest_data_time = raw.index.max()
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    market_open_now = (
        now_ist.weekday() < 5
        and (now_ist.hour, now_ist.minute) >= (9, 15)
        and (now_ist.hour, now_ist.minute) < (15, 30)
    )
    feed_age_seconds = (now_ist - latest_data_time).total_seconds()
    feed_stale = market_open_now and feed_age_seconds > 180
    if feed_stale:
        signal = "NO TRADE"
        reasons.append("Upstox feed is stale (>3 minutes); new signal blocked.")

    vwap_data = add_vwap_analysis(df3)
    last_vwap = vwap_data.iloc[-1]
    vwap_available = pd.notna(last_vwap["VWAP"])
    volume_available = pd.notna(last_vwap["Volume_AVG"])
    volume_confirmed = bool(volume_available and last_vwap["Volume"] > last_vwap["Volume_AVG"])

except Exception as error:
    st.error(f"Dashboard error: {error}")
    st.stop()

# ---------------- MARKET OVERVIEW ----------------
st.divider()
st.subheader("📊 Market Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("NIFTY Price", f"{a3['price']:,.2f}")
m2.metric("3M RSI", f"{a3['rsi']:.2f}")
m3.metric("5M RSI", f"{a5['rsi']:.2f}")
m4.metric("15M RSI", f"{a15['rsi']:.2f}")
st.caption(f"Last completed 3m candle: {a3['time']}")
st.caption(f"Latest Upstox 1m candle: {raw.index.max()}")
if feed_stale:
    st.error("DATA STALE: Market open है लेकिन latest candle 3 मिनट से पुरानी है. Signal रोक दिया गया है.")
else:
    st.success("Upstox data received; candle timestamp checked.")

# ---------------- TIMEFRAME TABLE ----------------
st.divider()
st.subheader("Multi-Timeframe Confirmation")
rows = []
for label, item in [("3 Minutes", a3), ("5 Minutes", a5), ("15 Minutes", a15)]:
    rows.append({
        "Timeframe": label,
        "Trend": "BULLISH" if item["bullish"] else ("BEARISH" if item["bearish"] else "NEUTRAL"),
        "RSI": round(item["rsi"], 2),
        "MACD": "Bullish" if item["macd_bullish"] else "Bearish",
        "Price": round(item["price"], 2)
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------- FINAL SIGNAL ----------------
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

# ---------------- ADVANCED CONFIRMATION ----------------
st.divider()
st.subheader("📊 Advanced Market Confirmation")
confirm_df = pd.DataFrame([
    {"Indicator": "RSI", "3M": round(a3["rsi"], 2), "5M": round(a5["rsi"], 2), "15M": round(a15["rsi"], 2)},
    {"Indicator": "MACD", "3M": "Bullish" if a3["macd_bullish"] else "Bearish",
     "5M": "Bullish" if a5["macd_bullish"] else "Bearish",
     "15M": "Bullish" if a15["macd_bullish"] else "Bearish"}
])
st.dataframe(confirm_df, use_container_width=True, hide_index=True)
cc1, cc2 = st.columns(2)
cc1.metric("RSI Confirmation", f"{max(rsi_bull, rsi_bear)}/3")
cc2.metric("MACD Confirmation", f"{max(macd_bull, macd_bear)}/3")
if vwap_available:
    st.write(f"VWAP: {float(last_vwap['VWAP']):,.2f}")
else:
    st.info("VWAP unavailable: usable volume data not provided.")
if volume_available:
    st.write("Volume:", "Confirmed" if volume_confirmed else "Weak")
else:
    st.info("Volume confirmation unavailable.")
st.caption("RSI and MACD affect the final signal. VWAP and Volume are displayed when usable data exists.")

# ---------------- INDEX LEVELS ----------------
st.divider()
st.subheader("Index Points: Entry, Stop Loss & Targets")
entry = a3["price"]
atr = max(a3["atr"], a5["atr"] * 0.5)
risk_points = max(1.5 * atr, 0.01)
if signal in ["BUY", "SELL"]:
    if signal == "BUY":
        sl, t1, t2, t3 = entry-risk_points, entry+risk_points, entry+2*risk_points, entry+3*risk_points
    else:
        sl, t1, t2, t3 = entry+risk_points, entry-risk_points, entry-2*risk_points, entry-3*risk_points
    q1, q2, q3 = st.columns(3)
    q1.metric("Index Entry", f"{entry:,.2f}")
    q2.metric("Index Stop Loss", f"{sl:,.2f}")
    q3.metric("Target 1", f"{t1:,.2f}")
    q4, q5 = st.columns(2)
    q4.metric("Target 2", f"{t2:,.2f}")
    q5.metric("Target 3", f"{t3:,.2f}")
else:
    sl = t1 = t2 = t3 = None
    st.info("No index levels generated because final signal is NO TRADE.")

# ---------------- PAPER TRADING ----------------
st.divider()
st.header("🧪 PAPER TRADING")
st.warning(
    "Simulation only; no broker order is placed. This uses NIFTY index points, "
    "not CE/PE option premiums. Virtual ₹ P&L uses an illustrative ₹1 per index "
    "point per virtual unit and is NOT actual tradable contract P&L."
)
risk_budget = st.session_state.paper_balance * risk_percent / 100
p1, p2, p3 = st.columns(3)
p1.metric("Virtual Account Balance", f"₹{st.session_state.paper_balance:,.2f}")
p2.metric("Risk Budget / Trade", f"₹{risk_budget:,.2f}")
p3.metric("Closed Trades", str(len(st.session_state.paper_history)))

trade = st.session_state.paper_open_trade
if trade is None:
    st.write("**No open paper trade.**")
    if signal in ["BUY", "SELL"]:
        st.write(f"Current signal: **{signal}** | Entry: **{entry:,.2f}** | SL: **{sl:,.2f}** | Target 1: **{t1:,.2f}**")
        virtual_units = risk_budget / risk_points if risk_points > 0 else 0.0
        if st.button(f"▶ Take {signal} Paper Trade", type="primary", use_container_width=True):
            st.session_state.paper_open_trade = {
                "side": signal, "entry": float(entry), "sl": float(sl), "target": float(t1),
                "units": float(virtual_units), "entry_time": a3["time"]
            }
            st.rerun()
    else:
        st.info("Paper trade tabhi open hoga jab final signal BUY ya SELL ho.")
else:
    side = trade["side"]
    current = float(a3["price"])
    direction = 1 if side == "BUY" else -1
    points = (current - trade["entry"]) * direction
    unrealized = points * trade["units"]
    st.subheader(f"Open Paper Trade: {side}")
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Entry", f"{trade['entry']:,.2f}")
    z2.metric("Current NIFTY", f"{current:,.2f}")
    z3.metric("Unrealized Points", f"{points:+,.2f}")
    z4.metric("Illustrative Virtual P&L", f"₹{unrealized:+,.2f}")
    st.write(f"Stop Loss: **{trade['sl']:,.2f}** | Target 1: **{trade['target']:,.2f}** | Virtual units: **{trade['units']:.3f}**")

    hit_sl = current <= trade["sl"] if side == "BUY" else current >= trade["sl"]
    hit_target = current >= trade["target"] if side == "BUY" else current <= trade["target"]
    if hit_sl:
        close_paper_trade(current, "Stop Loss condition at completed candle close")
        st.error("Paper trade closed at SL condition.")
        st.rerun()
    elif hit_target:
        close_paper_trade(current, "Target 1 condition at completed candle close")
        st.success("Paper trade closed at Target 1 condition.")
        st.rerun()
    elif st.button("✖ Close Paper Trade at Current Index Price", use_container_width=True):
        close_paper_trade(current, "Manual exit")
        st.rerun()

st.caption("Trade monitoring occurs only when the app reruns/refreshes; it is not tick-by-tick. Session history can reset if the app session/server restarts.")

st.subheader("📜 Paper Trade History")
if st.session_state.paper_history:
    hist = pd.DataFrame(st.session_state.paper_history)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    closed_pnl = sum(row["Illustrative Virtual P&L (₹)"] for row in st.session_state.paper_history)
    st.metric("Total Closed Virtual P&L", f"₹{closed_pnl:+,.2f}")
else:
    st.info("Abhi tak koi paper trade close nahi hua.")

# ---------------- OPTION BUYING STATUS ----------------
st.divider()
st.subheader("🎯 Option Buying Recommendation")
if signal == "BUY":
    st.success("Directional bias: BUY CE (not an actual option recommendation)")
elif signal == "SELL":
    st.error("Directional bias: BUY PE (not an actual option recommendation)")
else:
    st.warning("NO TRADE - Wait for confirmation.")
st.info(
    "Live option chain connected nahi hai. Actual strike, CE/PE premium, option SL "
    "aur option targets calculate nahi kiye ja sakte. Paper trading filhaal NIFTY "
    "index points par simulate hoti hai."
)# ---------------- NIFTY CHART ----------------
st.divider()
st.subheader("NIFTY 3-Minute Chart")
chart = go.Figure()
chart.add_trace(go.Candlestick(
    x=df3.index, open=df3["Open"], high=df3["High"], low=df3["Low"], close=df3["Close"], name="NIFTY"
))
for col, label in [("EMA9", "EMA 9"), ("EMA21", "EMA 21"), ("EMA50", "EMA 50"), ("Supertrend", "Supertrend")]:
    chart.add_trace(go.Scatter(x=df3.index, y=df3[col], name=label))
if vwap_available:
    chart.add_trace(go.Scatter(x=vwap_data.index, y=vwap_data["VWAP"], name="VWAP"))
chart.update_layout(height=560, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(chart, use_container_width=True)

# ---------------- RSI CHART ----------------
st.divider()
st.subheader("3-Minute RSI")
rsi_chart = go.Figure()
rsi_chart.add_trace(go.Scatter(x=df3.index, y=df3["RSI"], name="RSI"))
rsi_chart.add_hline(y=70, line_dash="dash")
rsi_chart.add_hline(y=30, line_dash="dash")
rsi_chart.add_hline(y=50, line_dash="dot")
rsi_chart.update_layout(height=250, template="plotly_dark")
st.plotly_chart(rsi_chart, use_container_width=True)

# ---------------- SYSTEM STATUS ----------------
st.divider()
st.subheader("System Status")
s1, s2, s3 = st.columns(3)
s1.success("3-minute data calculated")
s2.success("5-minute data calculated")
s3.success("15-minute data calculated")
st.caption("Data source: Upstox API V3. Polling-based candles; not tick-by-tick WebSocket data.")
st.warning(
    "Signals are technical conditions, not guaranteed predictions. Backtest and paper "
    "trade before using real capital. Index point levels are not option premium levels."
)
