
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
import plotly.graph_objects as go
from urllib.parse import quote
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NIFTY AI Ultimate Terminal",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY AI ULTIMATE TRADING TERMINAL")
st.caption(
    "News Intelligence | FII/DII | Global Markets | "
    "NIFTY 5m/15m | ML | Option Chain | CE/PE Trade Plan"
)

NSE_HOME = "https://www.nseindia.com/"
NSE_CONTRACT_URL = (
    "https://www.nseindia.com/api/option-chain-contract-info?symbol=NIFTY"
)
NSE_CHAIN_URL = "https://www.nseindia.com/api/option-chain-v3"
NSE_FII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

# =========================================================
# SETTINGS
# =========================================================
st.sidebar.header("⚙️ AI Settings")

horizon = st.sidebar.selectbox(
    "Prediction horizon (candles)",
    [2, 3, 4, 5],
    index=1
)

risk_pct = st.sidebar.slider(
    "Option premium stop-loss (%)",
    3.0, 30.0, 10.0, 1.0
)

rr1 = st.sidebar.number_input("Target 1 R:R", 1.0, 5.0, 1.0)
rr2 = st.sidebar.number_input("Target 2 R:R", 1.0, 8.0, 2.0)
rr3 = st.sidebar.number_input("Target 3 R:R", 1.0, 10.0, 3.0)

capital = st.sidebar.number_input(
    "Trading capital (₹)",
    min_value=1000,
    value=25000,
    step=1000
)

risk_per_trade = st.sidebar.slider(
    "Capital risk per trade (%)",
    0.25, 2.0, 0.5, 0.25
)

lot_size = st.sidebar.number_input(
    "NIFTY lot size",
    min_value=1,
    value=65,
    step=1,
    help="Current exchange lot size verify karke hi set karein."
)

strike_choice = st.sidebar.selectbox(
    "Option strike selection",
    ["ATM", "1 strike ITM", "1 strike OTM", "2 strikes ITM", "2 strikes OTM"]
)

if st.sidebar.button("🔄 Refresh all data"):
    st.cache_data.clear()
    st.rerun()

# =========================================================
# COMMON HELPERS
# =========================================================
def nse_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/125.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    })
    return s


def safe_num(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


# =========================================================
# NEWS INTELLIGENCE
# =========================================================
NEWS_QUERIES = [
    "NIFTY 50 India stock market",
    "RBI monetary policy India interest rates",
    "India inflation GDP rupee",
    "FII DII India stock market flows",
    "India crude oil prices stock market",
    "GIFT Nifty global markets",
    "US Federal Reserve interest rates",
    "NIFTY options market India",
    "India geopolitical risk economy",
    "NIFTY 50 company results announcements"
]

POSITIVE = [
    "surge", "rally", "gain", "growth", "profit", "strong",
    "bullish", "upgrade", "buying", "inflow", "rate cut",
    "easing", "positive", "beats estimates", "approval",
    "recovery", "stimulus", "crude falls", "oil falls",
    "record high", "deal reached"
]

NEGATIVE = [
    "crash", "fall", "falls", "decline", "loss", "weak",
    "bearish", "downgrade", "selling", "outflow", "war",
    "attack", "crisis", "inflation rises", "rate hike",
    "recession", "fraud", "sanctions", "misses estimates",
    "selloff", "sell-off", "rupee falls", "oil jumps"
]

@st.cache_data(ttl=300, show_spinner=False)
def get_news():
    rows = []

    for q in NEWS_QUERIES:
        url = (
            "https://news.google.com/rss/search?q="
            + quote(q + " when:2d")
            + "&hl=en-IN&gl=IN&ceid=IN:en"
        )
        try:
            feed = feedparser.parse(url)
            for item in feed.entries[:10]:
                title = item.get("title", "")
                link = item.get("link", "")
                if not title or not link:
                    continue

                rows.append({
                    "Headline": title,
                    "Published": item.get("published", ""),
                    "Source": item.get("source", {}).get(
                        "title", "Google News"
                    ),
                    "Link": link
                })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(
            columns=["Headline", "Published", "Source", "Link"]
        )

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["Headline"]).head(70)


def sentiment(text):
    t = str(text).lower()
    p = sum(w in t for w in POSITIVE)
    n = sum(w in t for w in NEGATIVE)

    if p > n:
        return "Positive", p - n
    if n > p:
        return "Negative", n - p
    return "Neutral", 0


def news_analysis(news):
    if news.empty:
        return 0.0, 0, 0, 0

    scored = news["Headline"].apply(sentiment)
    p = sum(x[0] == "Positive" for x in scored)
    n = sum(x[0] == "Negative" for x in scored)
    neutral = sum(x[0] == "Neutral" for x in scored)

    # Headline sentiment only; not a verified market-impact model.
    score = (p - n) / max(len(news), 1)
    return score, p, n, neutral


# =========================================================
# MARKET DATA
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_market(interval):
    data = yf.download(
        "^NSEI",
        period="60d",
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).lower() for c in data.columns]

    required = ["open", "high", "low", "close"]
    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    for c in required:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    if "volume" not in data.columns:
        data["volume"] = 0

    data["volume"] = pd.to_numeric(
        data["volume"], errors="coerce"
    ).fillna(0)

    data = data.dropna(subset=required)

    # Latest candle may still be forming.
    if len(data) > 2:
        data = data.iloc[:-1]

    return data


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def indicators(df):
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    d = df.copy()
    c = d["close"]

    d["ema9"] = c.ewm(span=9, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(
        span=9, adjust=False
    ).mean()

    prev = c.shift(1)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev).abs(),
        (d["low"] - prev).abs()
    ], axis=1).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    typical = (d["high"] + d["low"] + c) / 3
    vol_sum = d["volume"].rolling(20).sum()

    d["vwap_proxy"] = (
        (typical * d["volume"]).rolling(20).sum()
        / vol_sum.replace(0, np.nan)
    )

    d["return1"] = c.pct_change()
    d["return3"] = c.pct_change(3)
    d["return5"] = c.pct_change(5)
    d["volatility"] = d["return1"].rolling(20).std()
    d["body"] = (d["close"] - d["open"]) / c
    d["range"] = (d["high"] - d["low"]) / c

    return d.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "ema9", "ema21", "ema50", "ema200",
    "rsi", "macd", "macd_signal", "atr",
    "return1", "return3", "return5",
    "volatility", "body", "range"
]


# =========================================================
# MACHINE LEARNING
# =========================================================
def ml_predict(df, horizon):
    out = {
        "status": "Unavailable",
        "direction": "NO TRADE",
        "up": 0.0,
        "down": 0.0,
        "sideways": 0.0,
        "accuracy": None
    }

    if df.empty or len(df) < 300:
        return out

    d = df.copy()

    future = d["close"].shift(-horizon) / d["close"] - 1
    threshold = (
        d["atr"] / d["close"] * 0.25
    ).clip(lower=0.0003)

    d["target"] = np.where(
        future > threshold, 1,
        np.where(future < -threshold, -1, 0)
    )

    d = d.dropna(subset=FEATURES + ["target"])

    if len(d) < 250:
        return out

    X = d[FEATURES]
    y = d["target"].astype(int)

    split = int(len(d) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2:
        return out

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=8,
        min_samples_split=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        out["accuracy"] = accuracy_score(
            y_test, model.predict(X_test)
        )

    probabilities = model.predict_proba(X.iloc[[-1]])[0]
    mapping = dict(zip(model.classes_, probabilities))

    out["up"] = float(mapping.get(1, 0))
    out["down"] = float(mapping.get(-1, 0))
    out["sideways"] = float(mapping.get(0, 0))

    best = max(out["up"], out["down"], out["sideways"])

    if best < 0.50:
        out["direction"] = "NO TRADE"
    elif best == out["up"]:
        out["direction"] = "UP"
    elif best == out["down"]:
        out["direction"] = "DOWN"
    else:
        out["direction"] = "SIDEWAYS"

    out["status"] = "Available"
    return out


# =========================================================
# TECHNICAL REASONING
# =========================================================
def technical_score(row):
    score = 0
    reasons = []

    if row["ema9"] > row["ema21"]:
        score += 1
        reasons.append("EMA 9 > EMA 21: bullish short-term trend.")
    else:
        score -= 1
        reasons.append("EMA 9 < EMA 21: bearish short-term trend.")

    if row["close"] > row["ema50"]:
        score += 1
        reasons.append("Price EMA 50 ke upar hai.")
    else:
        score -= 1
        reasons.append("Price EMA 50 ke neeche hai.")

    if row["macd"] > row["macd_signal"]:
        score += 1
        reasons.append("MACD bullish hai.")
    else:
        score -= 1
        reasons.append("MACD bearish hai.")

    if row["rsi"] > 55:
        score += 1
        reasons.append("RSI above 55: positive momentum.")
    elif row["rsi"] < 45:
        score -= 1
        reasons.append("RSI below 45: negative momentum.")
    else:
        reasons.append("RSI neutral zone mein hai.")

    return score, reasons


# =========================================================
# NSE OPTION CHAIN - NEW API
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_nse_option_chain():
    try:
        session = nse_session()

        # Warm up cookies
        session.get(NSE_HOME + "option-chain", timeout=15)

        info_response = session.get(
            NSE_CONTRACT_URL, timeout=15
        )
        info_response.raise_for_status()
        info = info_response.json()

        expiry_list = (
            info.get("expiryDates")
            or info.get("records", {}).get("expiryDates")
            or []
        )

        if not expiry_list:
            return pd.DataFrame(), [], "Expiry list unavailable"

        # Prefer nearest listed expiry.
        expiry = expiry_list[0]

        params = {
            "type": "Indices",
            "symbol": "NIFTY",
            "expiry": expiry
        }

        chain_response = session.get(
            NSE_CHAIN_URL, params=params, timeout=20
        )
        chain_response.raise_for_status()

        payload = chain_response.json()
        records = payload.get("records", {}).get(
            "data", payload.get("data", [])
        )

        rows = []

        for item in records:
            ce = item.get("CE", {})
            pe = item.get("PE", {})

            rows.append({
                "Strike": safe_num(item.get("strikePrice")),
                "Expiry": item.get("expiryDate", expiry),

                "CE LTP": safe_num(ce.get("lastPrice")),
                "CE Bid": safe_num(ce.get("bidprice")),
                "CE Ask": safe_num(ce.get("askPrice")),
                "CE OI": safe_num(ce.get("openInterest")),
                "CE Change OI": safe_num(
                    ce.get("changeinOpenInterest")
                ),
                "CE Volume": safe_num(
                    ce.get("totalTradedVolume")
                ),
                "CE IV": safe_num(ce.get("impliedVolatility")),

                "PE LTP": safe_num(pe.get("lastPrice")),
                "PE Bid": safe_num(pe.get("bidprice")),
                "PE Ask": safe_num(pe.get("askPrice")),
                "PE OI": safe_num(pe.get("openInterest")),
                "PE Change OI": safe_num(
                    pe.get("changeinOpenInterest")
                ),
                "PE Volume": safe_num(
                    pe.get("totalTradedVolume")
                ),
                "PE IV": safe_num(pe.get("impliedVolatility"))
            })

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["Strike"])

        if df.empty:
            return df, expiry_list, "No option records returned"

        return df.sort_values("Strike"), expiry_list, (
            "Expiry loaded: " + str(expiry)
        )

    except Exception as e:
        return pd.DataFrame(), [], str(e)


# =========================================================
# OPTION CHAIN ANALYSIS
# =========================================================
def option_analysis(options, spot):
    if options.empty:
        return {
            "available": False,
            "pcr": None,
            "support": None,
            "resistance": None
        }

    ce_oi = options["CE OI"].fillna(0).sum()
    pe_oi = options["PE OI"].fillna(0).sum()

    pcr = pe_oi / ce_oi if ce_oi > 0 else None

    ce_rows = options.dropna(subset=["CE OI"])
    pe_rows = options.dropna(subset=["PE OI"])

    resistance = (
        ce_rows.loc[ce_rows["CE OI"].idxmax(), "Strike"]
        if not ce_rows.empty else None
    )

    support = (
        pe_rows.loc[pe_rows["PE OI"].idxmax(), "Strike"]
        if not pe_rows.empty else None
    )

    return {
        "available": True,
        "pcr": pcr,
        "support": support,
        "resistance": resistance
    }


def choose_option(options, spot, direction, strike_choice):
    if options.empty or direction not in ["UP", "DOWN"]:
        return None

    strikes = sorted(
        options["Strike"].dropna().unique().tolist()
    )

    if not strikes:
        return None

    atm = min(strikes, key=lambda x: abs(x - spot))
    atm_index = strikes.index(atm)

    if strike_choice == "ATM":
        offset = 0
    elif strike_choice == "1 strike ITM":
        offset = 1
    elif strike_choice == "1 strike OTM":
        offset = -1
    elif strike_choice == "2 strikes ITM":
        offset = 2
    else:
        offset = -2

    # For calls, higher strike is OTM.
    # For puts, lower strike is OTM.
    if direction == "UP":
        idx = atm_index - offset
        side = "CE"
    else:
        idx = atm_index + offset
        side = "PE"

    idx = max(0, min(len(strikes) - 1, idx))
    selected_strike = strikes[idx]

    row = options.loc[
        options["Strike"] == selected_strike
    ].iloc[0]

    premium = safe_num(row.get(side + " LTP"))
    bid = safe_num(row.get(side + " Bid"))
    ask = safe_num(row.get(side + " Ask"))
    oi = safe_num(row.get(side + " OI"))
    volume = safe_num(row.get(side + " Volume"))
    iv = safe_num(row.get(side + " IV"))

    if not np.isfinite(premium) or premium <= 0:
        return None

    # Require a usable quote. LTP alone may be stale.
    if not np.isfinite(bid) or not np.isfinite(ask):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    return {
        "side": side,
        "strike": selected_strike,
        "premium": premium,
        "bid": bid,
        "ask": ask,
        "oi": oi,
        "volume": volume,
        "iv": iv,
        "expiry": row.get("Expiry")
    }


# =========================================================
# OPTION TRADE PLAN
# =========================================================
def premium_trade_plan(option, risk_pct, rr1, rr2, rr3):
    if option is None:
        return None

    # Conservative long option entry uses ask quote.
    entry = option["ask"]
    risk = entry * risk_pct / 100
    sl = entry - risk

    return {
        "entry": entry,
        "sl": sl,
        "t1": entry + risk * rr1,
        "t2": entry + risk * rr2,
        "t3": entry + risk * rr3,
        "risk": risk
    }


def quantity_plan(capital, risk_percent, entry, sl, lot_size):
    risk_budget = capital * risk_percent / 100
    loss_per_unit = entry - sl

    if loss_per_unit <= 0 or entry <= 0:
        return None

    risk_based_lots = int(
        risk_budget / (loss_per_unit * lot_size)
    )

    capital_based_lots = int(
        capital / (entry * lot_size)
    )

    lots = min(risk_based_lots, capital_based_lots)

    if lots < 1:
        return {
            "lots": 0,
            "quantity": 0,
            "risk_budget": risk_budget,
            "required_capital": entry * lot_size
        }

    return {
        "lots": lots,
        "quantity": lots * lot_size,
        "risk_budget": risk_budget,
        "required_capital": entry * lots * lot_size
    }


# =========================================================
# FII / DII
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_fii_dii():
    try:
        session = nse_session()
        session.get(NSE_HOME, timeout=15)

        response = session.get(NSE_FII_URL, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("data", payload.get("rows", []))
        else:
            rows = []

        if not rows:
            return pd.DataFrame(), "No rows returned"

        return pd.DataFrame(rows), "NSE response received"

    except Exception as e:
        return pd.DataFrame(), str(e)


# =========================================================
# 8. GLOBAL MARKET SNAPSHOT
# =========================================================
st.divider()
st.subheader("8. Global Market Snapshot")

@st.cache_data(ttl=60)
def get_global_markets():
    tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
        "India VIX": "^INDIAVIX",
        "USD/INR": "INR=X",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "Nasdaq Futures": "NQ=F"
    }

    rows = []

    for name, symbol in tickers.items():
        try:
            data = yf.download(
                symbol,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if data.empty or len(data) < 2:
                continue

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            current = float(close.iloc[-1])
            previous = float(close.iloc[-2])

            change = current - previous
            change_pct = (
                change / previous * 100
                if previous != 0 else 0
            )

            rows.append({
                "Market": name,
                "Price": round(current, 2),
                "Change": round(change, 2),
                "Change %": round(change_pct, 2)
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


global_df = get_global_markets()

if not global_df.empty:
    st.dataframe(
        global_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Global market data unavailable.")

st.caption(
    "Quotes may be delayed. Nasdaq Futures is not GIFT Nifty. "
    "GIFT Nifty data is not available in this dashboard section."
)

# =========================================================
# 9. NIFTY TECHNICAL CHART
# =========================================================
st.divider()
st.subheader("9. NIFTY Technical Chart — 5 Minute")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df5.index,
    open=df5["open"],
    high=df5["high"],
    low=df5["low"],
    close=df5["close"],
    name="NIFTY"
))

for col in ["ema9", "ema21", "ema50"]:
    fig.add_trace(go.Scatter(
        x=df5.index,
        y=df5[col],
        name=col.upper(),
        line=dict(width=1)
    ))

fig.update_layout(
    height=550,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    yaxis_title="NIFTY"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 10. TECHNICAL INDICATOR DASHBOARD
# =========================================================
st.divider()
st.subheader("10. Technical Indicator Dashboard")

indicator_table = pd.DataFrame({
    "Indicator": [
        "EMA 9",
        "EMA 21",
        "EMA 50",
        "EMA 200",
        "RSI",
        "MACD",
        "MACD Signal",
        "ATR",
        "VWAP Proxy"
    ],
    "5 Minute": [
        last5["ema9"],
        last5["ema21"],
        last5["ema50"],
        last5["ema200"],
        last5["rsi"],
        last5["macd"],
        last5["macd_signal"],
        last5["atr"],
        last5["vwap_proxy"]
    ],
    "15 Minute": [
        last15["ema9"],
        last15["ema21"],
        last15["ema50"],
        last15["ema200"],
        last15["rsi"],
        last15["macd"],
        last15["macd_signal"],
        last15["atr"],
        last15["vwap_proxy"]
    ]
})

st.dataframe(
    indicator_table.round(2),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 11. DATA SOURCES
# =========================================================
st.divider()
st.subheader("11. Data Sources")

sources = [
    ("NSE India", "https://www.nseindia.com/"),
    ("NSE Option Chain", "https://www.nseindia.com/option-chain"),
    ("NSE FII/DII", "https://www.nseindia.com/reports/fii-dii"),
    ("RBI", "https://www.rbi.org.in/"),
    ("SEBI", "https://www.sebi.gov.in/"),
    ("Yahoo Finance", "https://finance.yahoo.com/"),
    ("Google News", "https://news.google.com/")
]

for name, url in sources:
    st.markdown(f"- [{name}]({url})")

# =========================================================
# 12. FINAL RISK WARNING
# =========================================================
st.divider()

st.warning(
    "RISK WARNING: This dashboard is for educational and "
    "research purposes only. It does not guarantee accuracy "
    "or profits. Options trading can result in loss of the "
    "entire premium. Do paper trading and backtesting before "
    "using real money."
)

st.caption(
    "NIFTY AI Ultimate Terminal | Last refresh: "
    + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)

m1, m2 = st.columns(2)

with m1:
    st.markdown("### 5-minute model")
    st.write("Direction:", ml5["direction"])
    st.write(f"UP: {ml5['up']:.1%}")
    st.write(f"DOWN: {ml5['down']:.1%}")
    st.write(f"SIDEWAYS: {ml5['sideways']:.1%}")

    if ml5["accuracy"] is not None:
        st.metric(
            "Chronological holdout accuracy",
            f"{ml5['accuracy']:.1%}"
        )

with m2:
    st.markdown("### 15-minute model")
    st.write("Direction:", ml15["direction"])
    st.write(f"UP: {ml15['up']:.1%}")
    st.write(f"DOWN: {ml15['down']:.1%}")
    st.write(f"SIDEWAYS: {ml15['sideways']:.1%}")

    if ml15["accuracy"] is not None:
        st.metric(
            "Chronological holdout accuracy",
            f"{ml15['accuracy']:.1%}"
        )

st.caption(
    "Accuracy classification accuracy hai, options trading win rate nahi. "
    "Probabilities calibrated profit probabilities nahi hain. "
    "Walk-forward testing aur transaction-cost backtesting required hai."
)

# =========================================================
# 5. NEWS INTELLIGENCE
# =========================================================
st.divider()
st.subheader("5. News Intelligence & Sentiment")

n1, n2, n3, n4 = st.columns(4)

n1.metric("Positive", news_positive)
n2.metric("Negative", news_negative)
n3.metric("Neutral", news_neutral)
n4.metric("Sentiment score", f"{news_score:.2f}")

if news.empty:
    st.warning("News feed unavailable.")
else:
    news_display = news.copy()
    news_display["Sentiment"] = news_display["Headline"].apply(
        lambda x: sentiment(x)[0]
    )

    for _, row in news_display.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Headline']}**")
            st.caption(
                f"{row['Source']} | {row['Published']} | "
                f"{row['Sentiment']}"
            )
            st.markdown(f"[Open original news]({row['Link']})")

st.caption(
    "News sentiment keyword-based screening hai. "
    "Headline alone verified market impact nahi batata."
)

# =========================================================
# 6. OPTION CHAIN
# =========================================================
st.divider()
st.subheader("6. NIFTY Option Chain — CE / PE")

st.write("**Option chain status:**", option_status)

if oc["available"]:
    o1, o2, o3 = st.columns(3)

    o1.metric(
        "PCR",
        f"{oc['pcr']:.2f}" if oc["pcr"] is not None else "N/A"
    )
    o2.metric(
        "Highest Put OI strike",
        str(oc["support"])
    )
    o3.metric(
        "Highest Call OI strike",
        str(oc["resistance"])
    )

    st.dataframe(
        options,
        use_container_width=True,
        hide_index=True
    )
else:
    st.error(
        "Option chain unavailable. Isliye premium-based trade plan "
        "generate nahi kiya gaya."
    )

st.markdown(
    "[Open official NSE Option Chain](https://www.nseindia.com/option-chain)"
)

# =========================================================
# 7. FII / DII
# =========================================================
st.divider()
st.subheader("7. FII / DII Institutional Activity")

st.write("**FII/DII source status:**", fii_status)

if not fii_dii.empty:
    st.dataframe(
        fii_dii,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("FII/DII figures unavailable. No estimated values shown.")

st.markdown(
    "[NSE official FII/DII reports](https://www.nseindia.com/reports/fii-dii)"
)

st.caption(
    "NSE institutional activity published data hai; final figures "
    "provisional data se differ kar sakte hain. Ye private institutional "
    "positions ya guaranteed intraday live flow nahi hai."
)

# =========================================================
# 8. GLOBAL MARKET DASHBOARD
# =========================================================
st.divider()
st.subheader("8. Global Market & Macro Dashboard")

if not global_df.empty:
    st.dataframe(
        global_df.round(2),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Global market quotes unavailable.")

# =========================================================
# 9. NIFTY TECHNICAL CHART
# =========================================================
st.divider()
st.subheader("9. NIFTY 5-minute Technical Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df5.index,
    open=df5["open"],
    high=df5["high"],
    low=df5["low"],
    close=df5["close"],
    name="NIFTY"
))

for col in ["ema9", "ema21", "ema50"]:
    fig.add_trace(go.Scatter(
        x=df5.index,
        y=df5[col],
        name=col.upper(),
        line=dict(width=1)
    ))

fig.update_layout(
    height=550,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    yaxis_title="NIFTY Index"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 10. TECHNICAL INDICATOR DASHBOARD
# =========================================================
st.divider()
st.subheader("10. Technical Indicator Dashboard")

indicator_table = pd.DataFrame({
    "Indicator": [
        "EMA 9", "EMA 21", "EMA 50", "EMA 200",
        "RSI", "MACD", "MACD Signal", "ATR", "VWAP Proxy"
    ],
    "5 Minute": [
        last5["ema9"], last5["ema21"], last5["ema50"],
        last5["ema200"], last5["rsi"], last5["macd"],
        last5["macd_signal"], last5["atr"], last5["vwap_proxy"]
    ],
    "15 Minute": [
        last15["ema9"], last15["ema21"], last15["ema50"],
        last15["ema200"], last15["rsi"], last15["macd"],
        last15["macd_signal"], last15["atr"], last15["vwap_proxy"]
    ]
})

st.dataframe(
    indicator_table.round(2),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 11. DATA SOURCES
# =========================================================
st.divider()
st.subheader("11. Data Sources & Verification")

sources = [
    ("NSE India", "https://www.nseindia.com/"),
    ("NSE Option Chain", "https://www.nseindia.com/option-chain"),
    ("NSE FII/DII", "https://www.nseindia.com/reports/fii-dii"),
    ("RBI", "https://www.rbi.org.in/"),
    ("SEBI", "https://www.sebi.gov.in/"),
    ("Yahoo Finance NIFTY", "https://finance.yahoo.com/quote/%5ENSEI/"),
    ("Google News", "https://news.google.com/")
]

for name, url in sources:
    st.markdown(f"- [{name}]({url})")

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.warning(
    "RISK WARNING: Ye educational research dashboard hai, "
    "guaranteed trading system nahi. Options mein poora premium "
    "loss ho sakta hai. Paper trading aur proper backtesting ke "
    "bina real capital use na karein."
)

st.caption(
    "NIFTY AI Ultimate Terminal | Last refresh: "
    + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)

: 
