


# ============================================================
# NIFTY AI ULTIMATE DASHBOARD
# PART 1 OF 2
# Core engine, data collection, indicators, ML and news
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
import plotly.graph_objects as go
import plotly.express as px

import time
import math
import warnings
import re
import json

from datetime import datetime, timedelta
from urllib.parse import quote

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NIFTY AI Ultimate Terminal",
    page_icon="📈",
    layout="wide",
    

import streamlit as st

st.set_page_config(
    page_title="NIFTY 50 AI Dashboard",
    layout="wide"
)
# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #080d18;
}
.stApp {
    background-color: #080d18;
}
h1, h2, h3 {
    color: #ffffff;
}
.metric-card {
    background: #121c2d;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #26364d;
}
.buy-text {
    color: #00d68f;
    font-weight: bold;
}
.sell-text {
    color: #ff5252;
    font-weight: bold;
}
.neutral-text {
    color: #ffc857;
    font-weight: bold;
}
.small-note {
    color: #9aa9bd;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. APP HEADER
# ============================================================

st.title("📈 NIFTY AI ULTIMATE TERMINAL")

st.caption(
    "AI + ML | News Intelligence | Technical Analysis | "
    "Option Chain | FII/DII | Risk Management"
)

st.info(
    "Educational research dashboard. No prediction guarantees "
    "profit. Verify every quote before placing an order."
)

# ============================================================
# 4. SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙️ Dashboard Settings")

capital = st.sidebar.number_input(
    "Trading Capital (₹)",
    min_value=1000,
    max_value=10000000,
    value=50000,
    step=1000
)

risk_percent = st.sidebar.slider(
    "Risk per trade (%)",
    min_value=0.25,
    max_value=5.0,
    value=1.0,
    step=0.25
)

lot_size = st.sidebar.number_input(
    "NIFTY lot size (verify current exchange lot size)",
    min_value=1,
    max_value=1000,
    value=65,
    step=1
)

strike_mode = st.sidebar.selectbox(
    "Option strike preference",
    ["ATM", "ITM", "OTM"]
)

stop_loss_percent = st.sidebar.slider(
    "Option premium stop loss (%)",
    min_value=5,
    max_value=50,
    value=20
)

refresh_seconds = st.sidebar.selectbox(
    "Refresh interval",
    [30, 60, 120, 300],
    index=1
)

# ============================================================
# 5. CONSTANTS
# ============================================================

NIFTY_SYMBOL = "^NSEI"

NEWS_QUERIES = [
    "NIFTY 50 India stock market",
    "Indian stock market RBI news",
    "NSE market news India",
    "Bank Nifty market news",
    "India economy market news"
]

GLOBAL_TICKERS = {
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

# ============================================================
# 6. SESSION STATE
# ============================================================

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

if "news_cache" not in st.session_state:
    st.session_state.news_cache = pd.DataFrame()

if "last_decision" not in st.session_state:
    st.session_state.last_decision = "NO TRADE"

# ============================================================
# 7. SAFE DATA CONVERSION
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        return result

    except Exception:
        return default


def safe_percent(current, previous):
    current = safe_float(current)
    previous = safe_float(previous)

    if np.isnan(current) or np.isnan(previous):
        return np.nan

    if previous == 0:
        return np.nan

    return ((current - previous) / previous) * 100


def format_number(value, decimals=2):
    value = safe_float(value)

    if np.isnan(value):
        return "Unavailable"

    return f"{value:,.{decimals}f}"


# ============================================================
# 8. DOWNLOAD MARKET DATA
# ============================================================

@st.cache_data(ttl=60)
def download_nifty_data(interval="5m", period="60d"):
    try:
        data = yf.download(
            NIFTY_SYMBOL,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [
            str(column).lower().replace(" ", "_")
            for column in data.columns
        ]

        required = ["open", "high", "low", "close"]

        for column in required:
            if column not in data.columns:
                return pd.DataFrame()

        if "volume" not in data.columns:
            data["volume"] = 0

        data = data[required + ["volume"]].copy()

        for column in required + ["volume"]:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        data = data.dropna(subset=required)

        return data

    except Exception:
        return pd.DataFrame()


# ============================================================
# 9. TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calculate_indicators(data):
    if data is None or data.empty:
        return pd.DataFrame()

    d = data.copy()

    d["ema9"] = d["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["ema21"] = d["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    d["ema50"] = d["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    d["ema200"] = d["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    d["rsi"] = calculate_rsi(d["close"])

    ema12 = d["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = d["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    d["macd"] = ema12 - ema26

    d["macd_signal"] = d["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = (
        d["macd"] - d["macd_signal"]
    )

    previous_close = d["close"].shift(1)

    tr1 = d["high"] - d["low"]

    tr2 = (d["high"] - previous_close).abs()

    tr3 = (d["low"] - previous_close).abs()

    d["true_range"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    d["atr"] = d["true_range"].rolling(
        14
    ).mean()

    d["return_1"] = d["close"].pct_change()

    d["return_3"] = d["close"].pct_change(3)

    d["return_5"] = d["close"].pct_change(5)

    d["volatility"] = d["return_1"].rolling(
        20
    ).std()

    d["candle_body"] = (
        d["close"] - d["open"]
    )

    d["candle_range"] = (
        d["high"] - d["low"]
    )

    d["body_ratio"] = (
        d["candle_body"] /
        d["candle_range"].replace(0, np.nan)
    )

    # This is a rolling proxy, not exchange VWAP.
    volume_sum = d["volume"].rolling(20).sum()

    typical_price = (
        d["high"] + d["low"] + d["close"]
    ) / 3

    d["vwap_proxy"] = (
        (typical_price * d["volume"]).rolling(20).sum()
        / volume_sum.replace(0, np.nan)
    )

    return d


# ============================================================
# 10. TECHNICAL MARKET DIRECTION
# ============================================================

def technical_analysis(data):
    if data is None or len(data) < 5:
        return {
            "direction": "UNAVAILABLE",
            "score": 0,
            "reasons": ["Insufficient market data"]
        }

    last = data.iloc[-1]

    score = 0
    reasons = []

    if last["ema9"] > last["ema21"]:
        score += 1
        reasons.append("EMA 9 above EMA 21")

    else:
        score -= 1
        reasons.append("EMA 9 below EMA 21")

    if last["ema21"] > last["ema50"]:
        score += 1
        reasons.append("EMA 21 above EMA 50")

    else:
        score -= 1
        reasons.append("EMA 21 below EMA 50")

    if last["close"] > last["ema200"]:
        score += 1
        reasons.append("Price above EMA 200")

    else:
        score -= 1
        reasons.append("Price below EMA 200")

    if last["rsi"] > 55:
        score += 1
        reasons.append("RSI bullish")

    elif last["rsi"] < 45:
        score -= 1
        reasons.append("RSI bearish")

    else:
        reasons.append("RSI neutral")

    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD bullish")

    else:
        score -= 1
        reasons.append("MACD bearish")

    if score >= 3:
        direction = "BULLISH"

    elif score <= -3:
        direction = "BEARISH"

    else:
        direction = "SIDEWAYS"

    return {
        "direction": direction,
        "score": score,
        "reasons": reasons
    }


# ============================================================
# 11. MACHINE LEARNING FEATURES
# ============================================================

ML_FEATURES = [
    "ema9",
    "ema21",
    "ema50",
    "ema200",
    "rsi",
    "macd",
    "macd_signal",
    "atr",
    "return_1",
    "return_3",
    "return_5",
    "volatility",
    "body_ratio"
]


def prepare_ml_dataset(data, horizon=3):
    if data is None or data.empty:
        return pd.DataFrame()

    d = data.copy()

    d["future_return"] = (
        d["close"].shift(-horizon) / d["close"] - 1
    )

    # ATR threshold for three classes.
    threshold = (
        0.25 * d["atr"] / d["close"]
    )

    valid = d["future_return"].notna()

    d = d.loc[valid].copy()

    threshold = threshold.loc[valid]

    d["target"] = np.select(
        [
            d["future_return"] > threshold,
            d["future_return"] < -threshold
        ],
        [
            1,
            -1
        ],
        default=0
    )

    d = d.replace(
        [np.inf, -np.inf],
        np.nan
    )

    d = d.dropna(
        subset=ML_FEATURES + ["target"]
    )

    return d


# ============================================================
# 12. MACHINE LEARNING MODEL
# ============================================================

def train_ml_model(data, horizon=3):
    result = {
        "available": False,
        "direction": "UNAVAILABLE",
        "up": 0.0,
        "down": 0.0,
        "sideways": 0.0,
        "accuracy": None,
        "samples": 0
    }

    dataset = prepare_ml_dataset(
        data,
        horizon=horizon
    )

    if len(dataset) < 150:
        return result

    split = int(len(dataset) * 0.80)

    # Purge the training boundary to reduce
    # overlap from future-label horizon.
    train_end = split - horizon

    if train_end < 100 or split >= len(dataset):
        return result

    train = dataset.iloc[:train_end]
    test = dataset.iloc[split:]

    if train["target"].nunique() < 2:
        return result

    X_train = train[ML_FEATURES]
    y_train = train["target"]

    X_test = test[ML_FEATURES]
    y_test = test["target"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    try:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        latest = dataset.iloc[[-1]][ML_FEATURES]

        probabilities = model.predict_proba(latest)[0]

        classes = model.named_steps[
            "classifier"
        ].classes_

        probability_map = {
            int(label): float(probability)
            for label, probability in zip(
                classes,
                probabilities
            )
        }

        up = probability_map.get(1, 0.0)
        down = probability_map.get(-1, 0.0)
        sideways = probability_map.get(0, 0.0)

        if up > down and up > sideways:
            direction = "UP"

        elif down > up and down > sideways:
            direction = "DOWN"

        else:
            direction = "SIDEWAYS"

        result.update({
            "available": True,
            "direction": direction,
            "up": up,
            "down": down,
            "sideways": sideways,
            "accuracy": float(accuracy),
            "samples": len(dataset)
        })

        return result

    except Exception:
        return result


# ============================================================
# 13. NEWS SENTIMENT ENGINE
# ============================================================

POSITIVE_WORDS = [
    "surge",
    "rally",
    "gain",
    "gains",
    "growth",
    "profit",
    "positive",
    "bullish",
    "upgrade",
    "record high",
    "strong",
    "recovery",
    "beat estimates",
    "inflow",
    "rate cut"
]

NEGATIVE_WORDS = [
    "crash",
    "fall",
    "falls",
    "decline",
    "loss",
    "negative",
    "bearish",
    "downgrade",
    "weak",
    "selloff",
    "sell-off",
    "inflation surge",
    "outflow",
    "recession",
    "rate hike"
]


def get_sentiment(text):
    text = str(text).lower()

    positive = sum(
        1 for word in POSITIVE_WORDS
        if word in text
    )

    negative = sum(
        1 for word in NEGATIVE_WORDS
        if word in text
    )

    if positive > negative:
        label = "POSITIVE"

    elif negative > positive:
        label = "NEGATIVE"

    else:
        label = "NEUTRAL"

    score = positive - negative

    return label, score


# ============================================================
# 14. GOOGLE NEWS RSS
# ============================================================

@st.cache_data(ttl=300)
def fetch_news():
    rows = []

    for query in NEWS_QUERIES:
        try:
            url = (
                "https://news.google.com/rss/search?q="
                + quote(query)
                + "&hl=en-IN&gl=IN&ceid=IN:en"
            )

            feed = feedparser.parse(url)

            for entry in feed.entries[:15]:
                title = entry.get("title", "")

                link = entry.get("link", "")

                published = entry.get(
                    "published",
                    "Date unavailable"
                )

                source = "Google News RSS"

                if "source" in entry:
                    source = entry.source.get(
                        "title",
                        source
                    )

                label, score = get_sentiment(title)

                rows.append({
                    "Headline": title,
                    "Source": source,
                    "Published": published,
                    "Link": link,
                    "Sentiment": label,
                    "Score": score
                })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=["Headline"]
    )

    return df.reset_index(drop=True)


def calculate_news_summary(news):
    if news is None or news.empty:
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "score": 0
        }

    counts = news["Sentiment"].value_counts()

    positive = int(counts.get("POSITIVE", 0))
    negative = int(counts.get("NEGATIVE", 0))
    neutral = int(counts.get("NEUTRAL", 0))

    total = positive + negative + neutral

    score = (
        (positive - negative) / total
        if total > 0 else 0
    )

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "score": score
    }


# ============================================================
# 15. GLOBAL MARKET DATA ENGINE
# ============================================================

@st.cache_data(ttl=60)
def get_global_markets():
    rows = []

    for name, symbol in GLOBAL_TICKERS.items():
        try:
            data = yf.download(
                symbol,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if data is None or data.empty:
                continue

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna()

            if len(close) < 2:
                continue

            current = safe_float(close.iloc[-1])
            previous = safe_float(close.iloc[-2])

            if np.isnan(current) or np.isnan(previous):
                continue

            change = current - previous

            change_pct = safe_percent(
                current,
                previous
            )

            rows.append({
                "Market": name,
                "Symbol": symbol,
                "Price": current,
                "Change": change,
                "Change %": change_pct
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# 16. NSE SESSION
# ============================================================

def create_nse_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0.0.0 "
            "Safari/537.36"
        ),
        "Accept": (
            "application/json,text/plain,*/*"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    })

    try:
        session.get(
            "https://www.nseindia.com/",
            timeout=10
        )
    except Exception:
        pass

    return session



# ============================================================
# 17. NSE OPTION CHAIN FETCH
# ============================================================

def fetch_option_chain():
    result = {
        "available": False,
        "status": "Not loaded",
        "data": pd.DataFrame(),
        "expiry": None
    }

    session = create_nse_session()

    try:
        expiry_url = (
            "https://www.nseindia.com/api/"
            "option-chain-contract-info?symbol=NIFTY"
        )

        expiry_response = session.get(
            expiry_url,
            timeout=15
        )

        expiry_response.raise_for_status()

        expiry_json = expiry_response.json()

        expiry_dates = expiry_json.get(
            "expiryDates",
            []
        )

        if not expiry_dates:
            return {
                **result,
                "status": "Expiry dates unavailable"
            }

        expiry = expiry_dates[0]

        option_url = (
            "https://www.nseindia.com/api/option-chain-v3"
        )

        response = session.get(
            option_url,
            params={
                "type": "Indices",
                "symbol": "NIFTY",
                "expiry": expiry
            },
            timeout=15
        )

        response.raise_for_status()

        payload = response.json()

        records = payload.get(
            "records",
            {}
        ).get("data", [])

        if not records:
            return {
                **result,
                "status": "Option chain records unavailable"
            }

        rows = []

        for item in records:
            strike = item.get("strikePrice")

            ce = item.get("CE", {})
            pe = item.get("PE", {})

            rows.append({
                "Strike": strike,
                "CE OI": ce.get("openInterest"),
                "CE Change OI": ce.get("changeinOpenInterest"),
                "CE LTP": ce.get("lastPrice"),
                "CE Bid": ce.get("bidprice"),
                "CE Ask": ce.get("askPrice"),
                "PE OI": pe.get("openInterest"),
                "PE Change OI": pe.get("changeinOpenInterest"),
                "PE LTP": pe.get("lastPrice"),
                "PE Bid": pe.get("bidprice"),
                "PE Ask": pe.get("askPrice")
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return {
                **result,
                "status": "Empty option chain"
            }

        for column in df.columns:
            if column != "Strike":
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        return {
            "available": True,
            "status": "NSE response parsed",
            "data": df,
            "expiry": expiry
        }

    except Exception as error:
        return {
            **result,
            "status": (
                "NSE unavailable: "
                + str(error)[:180]
            )
        }


# ============================================================
# 18. OPTION CHAIN ANALYSIS
# ============================================================

def analyse_option_chain(option_data, spot):

    result = {
        "available": False,
        "pcr": None,
        "support": None,
        "resistance": None
    }

    if option_data is None or option_data.empty:
        return result

    df = option_data.copy()

    required = [
        "Strike",
        "CE OI",
        "PE OI"
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return result

    ce_oi = df["CE OI"].fillna(0).sum()
    pe_oi = df["PE OI"].fillna(0).sum()

    pcr = (
        pe_oi / ce_oi
        if ce_oi > 0 else None
    )

    if df["PE OI"].fillna(0).sum() > 0:
        support_row = df.loc[
            df["PE OI"].fillna(0).idxmax()
        ]
        support = support_row["Strike"]
    else:
        support = None

    if df["CE OI"].fillna(0).sum() > 0:
        resistance_row = df.loc[
            df["CE OI"].fillna(0).idxmax()
        ]
        resistance = resistance_row["Strike"]
    else:
        resistance = None

    result.update({
        "available": True,
        "pcr": pcr,
        "support": support,
        "resistance": resistance
    })

    return result


# ============================================================
# 19. FII / DII DATA
# ============================================================

@st.cache_data(ttl=900)
def fetch_fii_dii():

    try:
        session = create_nse_session()

        url = (
            "https://www.nseindia.com/api/fiidiiTradeReact"
        )

        response = session.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        payload = response.json()

        if isinstance(payload, list):
            df = pd.DataFrame(payload)

        elif isinstance(payload, dict):

            if isinstance(payload.get("data"), list):
                df = pd.DataFrame(payload["data"])

            else:
                df = pd.DataFrame([payload])

        else:
            return (
                pd.DataFrame(),
                "Unexpected FII/DII response format"
            )

        if df.empty:
            return (
                pd.DataFrame(),
                "No FII/DII rows returned"
            )

        return (
            df,
            "NSE response received; verify report date and fields"
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            "FII/DII unavailable: " + str(error)[:180]
        )


# ============================================================
# 20. OPTION CONTRACT SELECTION
# ============================================================

def select_option_contract(
    option_data,
    spot,
    direction,
    strike_mode="ATM"
):

    if option_data is None or option_data.empty:
        return None

    if direction not in ["UP", "DOWN"]:
        return None

    df = option_data.copy()

    df = df.dropna(
        subset=["Strike"]
    ).reset_index(drop=True)

    if df.empty:
        return None

    atm_position = int(
        (df["Strike"] - spot).abs().argmin()
    )

    if strike_mode == "ATM":
        offset = 0

    elif strike_mode == "ITM":
        offset = 1

    else:
        offset = -1

    if direction == "UP":
        side = "CE"
        selected_position = atm_position - offset

    else:
        side = "PE"
        selected_position = atm_position + offset

    selected_position = max(
        0,
        min(selected_position, len(df) - 1)
    )

    contract = df.iloc[selected_position]

    bid = safe_float(
        contract.get(f"{side} Bid")
    )

    ask = safe_float(
        contract.get(f"{side} Ask")
    )

    ltp = safe_float(
        contract.get(f"{side} LTP")
    )

    if np.isnan(ask) or ask <= 0:
        return None

    if np.isnan(bid) or bid <= 0:
        return None

    if ask < bid:
        return None

    return {
        "side": side,
        "strike": safe_float(contract["Strike"]),
        "bid": bid,
        "ask": ask,
        "ltp": ltp,
        "spread": ask - bid
    }


# ============================================================
# END OF PART 1
# ============================================================


# ============================================================
# NIFTY AI ULTIMATE DASHBOARD
# PART 2 OF 2
# Trade engine + option premium plan + complete dashboard
# ============================================================


# ============================================================
# 21. LOAD ALL MARKET DATA
# ============================================================

st.divider()
st.header("📊 MARKET DATA ENGINE")

with st.spinner("Downloading NIFTY market data..."):

    raw5 = download_nifty_data(
        interval="5m",
        period="60d"
    )

    raw15 = download_nifty_data(
        interval="15m",
        period="60d"
    )

    df5 = calculate_indicators(raw5)
    df15 = calculate_indicators(raw15)


if df5.empty or df15.empty:

    st.error(
        "NIFTY market data unavailable. "
        "Please refresh after some time."
    )

    st.stop()


last5 = df5.iloc[-1]
last15 = df15.iloc[-1]

spot = safe_float(last5["close"])

if np.isnan(spot):
    st.error("NIFTY spot price unavailable.")
    st.stop()


# ============================================================
# 22. MARKET PRICE HEADER
# ============================================================

previous_close = safe_float(
    df5["close"].iloc[-2]
)

price_change = spot - previous_close

price_change_percent = safe_percent(
    spot,
    previous_close
)

p1, p2, p3, p4 = st.columns(4)

p1.metric(
    "NIFTY Spot",
    format_number(spot),
    format_number(price_change)
)

p2.metric(
    "Change %",
    format_number(price_change_percent) + "%"
)

p3.metric(
    "5m RSI",
    format_number(last5["rsi"])
)

p4.metric(
    "5m ATR",
    format_number(last5["atr"])
)

st.caption(
    "Market data is sourced through Yahoo Finance and may be delayed. "
    "Verify live prices with your broker before trading."
)


# ============================================================
# 23. TECHNICAL ANALYSIS
# ============================================================

st.divider()
st.header("📈 TECHNICAL ANALYSIS")

tech5 = technical_analysis(df5)
tech15 = technical_analysis(df15)

t1, t2 = st.columns(2)

with t1:

    st.subheader("5 Minute Analysis")

    st.metric(
        "5m Direction",
        tech5["direction"],
        tech5["score"]
    )

    for reason in tech5["reasons"]:
        st.write("•", reason)


with t2:

    st.subheader("15 Minute Analysis")

    st.metric(
        "15m Direction",
        tech15["direction"],
        tech15["score"]
    )

    for reason in tech15["reasons"]:
        st.write("•", reason)


# ============================================================
# 24. MACHINE LEARNING PREDICTION
# ============================================================

st.divider()
st.header("🤖 AI / MACHINE LEARNING PREDICTION")

with st.spinner("Training experimental ML models..."):

    ml5 = train_ml_model(
        df5,
        horizon=3
    )

    ml15 = train_ml_model(
        df15,
        horizon=3
    )


m1, m2 = st.columns(2)

with m1:

    st.subheader("5 Minute ML")

    if ml5["available"]:

        st.metric(
            "Prediction",
            ml5["direction"]
        )

        st.write(
            f"UP: {ml5['up']:.1%}"
        )

        st.write(
            f"DOWN: {ml5['down']:.1%}"
        )

        st.write(
            f"SIDEWAYS: {ml5['sideways']:.1%}"
        )

        st.metric(
            "Holdout Accuracy",
            f"{ml5['accuracy']:.1%}"
        )

        st.caption(
            f"Training dataset samples: {ml5['samples']}"
        )

    else:

        st.warning(
            "5-minute ML prediction unavailable."
        )


with m2:

    st.subheader("15 Minute ML")

    if ml15["available"]:

        st.metric(
            "Prediction",
            ml15["direction"]
        )

        st.write(
            f"UP: {ml15['up']:.1%}"
        )

        st.write(
            f"DOWN: {ml15['down']:.1%}"
        )

        st.write(
            f"SIDEWAYS: {ml15['sideways']:.1%}"
        )

        st.metric(
            "Holdout Accuracy",
            f"{ml15['accuracy']:.1%}"
        )

        st.caption(
            f"Training dataset samples: {ml15['samples']}"
        )

    else:

        st.warning(
            "15-minute ML prediction unavailable."
        )


st.caption(
    "ML probabilities are model classifications, not guaranteed "
    "profit probabilities. Holdout accuracy is not options win rate."
)


# ============================================================
# 25. NEWS INTELLIGENCE
# ============================================================

st.divider()
st.header("📰 NEWS INTELLIGENCE")

with st.spinner("Fetching public market news..."):

    news = fetch_news()

news_summary = calculate_news_summary(news)

n1, n2, n3, n4 = st.columns(4)

n1.metric(
    "Positive News",
    news_summary["positive"]
)

n2.metric(
    "Negative News",
    news_summary["negative"]
)

n3.metric(
    "Neutral News",
    news_summary["neutral"]
)

n4.metric(
    "News Sentiment",
    f"{news_summary['score']:.2f}"
)


if news.empty:

    st.warning("News feed unavailable.")

else:

    st.dataframe(
        news[
            [
                "Headline",
                "Source",
                "Published",
                "Sentiment"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("News Sources")

    for _, row in news.head(15).iterrows():

        with st.container(border=True):

            st.markdown(
                f"**{row['Headline']}**"
            )

            st.caption(
                f"{row['Source']} | {row['Published']} "
                f"| {row['Sentiment']}"
            )

            if row["Link"]:
                st.markdown(
                    f"[Read original news]({row['Link']})"
                )


st.caption(
    "News sentiment is based on simple keyword matching. "
    "It is not a verified financial-news interpretation system."
)


# ============================================================
# 26. OPTION CHAIN DATA
# ============================================================

st.divider()
st.header("🔗 NIFTY OPTION CHAIN")

with st.spinner("Connecting to NSE option chain..."):

    option_result = fetch_option_chain()

option_data = option_result["data"]

option_status = option_result["status"]

option_analysis = analyse_option_chain(
    option_data,
    spot
)

st.write(
    "**Option Chain Status:**",
    option_status
)

if option_result["available"]:

    st.success(
        f"Expiry returned by NSE: {option_result['expiry']}"
    )

    o1, o2, o3 = st.columns(3)

    pcr = option_analysis["pcr"]

    o1.metric(
        "Put Call Ratio",
        format_number(pcr)
    )

    o2.metric(
        "Highest Put OI Strike",
        format_number(option_analysis["support"])
    )

    o3.metric(
        "Highest Call OI Strike",
        format_number(option_analysis["resistance"])
    )

    st.dataframe(
        option_data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.error(
        "Option chain unavailable. Premium-based trade plan "
        "will not be generated."
    )

st.markdown(
    "[Open NSE Official Option Chain](https://www.nseindia.com/option-chain)"
)


# ============================================================
# 27. FII / DII DATA
# ============================================================

st.divider()
st.header("🏦 FII / DII ACTIVITY")

with st.spinner("Fetching NSE FII/DII report..."):

    fii_dii, fii_status = fetch_fii_dii()

st.write(
    "**FII/DII Status:**",
    fii_status
)

if not fii_dii.empty:

    st.dataframe(
        fii_dii,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "FII/DII data unavailable. No estimated figures are shown."
    )

st.markdown(
    "[NSE Official FII/DII Reports](https://www.nseindia.com/reports/fii-dii)"
)

st.caption(
    "FII/DII reports may be end-of-day or delayed data. "
    "They do not reveal private institutional positions."
)


# ============================================================
# 28. GLOBAL MARKET SNAPSHOT
# ============================================================

st.divider()
st.header("🌍 GLOBAL MARKET SNAPSHOT")

with st.spinner("Loading global market data..."):

    global_df = get_global_markets()

if not global_df.empty:

    st.dataframe(
        global_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    try:

        chart_df = global_df.dropna(
            subset=["Change %"]
        )

        if not chart_df.empty:

            fig_global = px.bar(
                chart_df,
                x="Market",
                y="Change %",
                title="Global Market Daily Change (%)"
            )

            fig_global.update_layout(
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                fig_global,
                use_container_width=True
            )

    except Exception:

        st.info(
            "Global comparison chart unavailable."
        )

else:

    st.warning(
        "Global market data unavailable."
    )

st.caption(
    "Nasdaq Futures is not GIFT Nifty. "
    "Global quotes may be delayed."
)


# ============================================================
# 29. AI TRADE DECISION ENGINE
# ============================================================

st.divider()
st.header("🧠 AI TRADE DECISION ENGINE")


def calculate_trade_decision(
    tech5,
    tech15,
    ml5,
    ml15,
    news_summary,
    option_analysis
):

    bullish_score = 0
    bearish_score = 0

    reasons = []
    warnings_list = []

    # 5-minute technical direction

    if tech5["direction"] == "BULLISH":

        bullish_score += 2
        reasons.append(
            "5-minute technical trend is bullish."
        )

    elif tech5["direction"] == "BEARISH":

        bearish_score += 2
        reasons.append(
            "5-minute technical trend is bearish."
        )

    else:

        warnings_list.append(
            "5-minute technical trend is not clear."
        )


    # 15-minute technical direction

    if tech15["direction"] == "BULLISH":

        bullish_score += 2
        reasons.append(
            "15-minute technical trend is bullish."
        )

    elif tech15["direction"] == "BEARISH":

        bearish_score += 2
        reasons.append(
            "15-minute technical trend is bearish."
        )

    else:

        warnings_list.append(
            "15-minute technical trend is not clear."
        )


    # 5-minute ML

    if ml5["available"]:

        if (
            ml5["direction"] == "UP"
            and ml5["up"] >= 0.50
        ):

            bullish_score += 1
            reasons.append(
                "5-minute ML model favours UP."
            )

        elif (
            ml5["direction"] == "DOWN"
            and ml5["down"] >= 0.50
        ):

            bearish_score += 1
            reasons.append(
                "5-minute ML model favours DOWN."
            )

        else:

            warnings_list.append(
                "5-minute ML prediction is not decisive."
            )

    else:

        warnings_list.append(
            "5-minute ML model unavailable."
        )


    # 15-minute ML

    if ml15["available"]:

        if (
            ml15["direction"] == "UP"
            and ml15["up"] >= 0.50
        ):

            bullish_score += 1
            reasons.append(
                "15-minute ML model favours UP."
            )

        elif (
            ml15["direction"] == "DOWN"
            and ml15["down"] >= 0.50
        ):

            bearish_score += 1
            reasons.append(
                "15-minute ML model favours DOWN."
            )

        else:

            warnings_list.append(
                "15-minute ML prediction is not decisive."
            )

    else:

        warnings_list.append(
            "15-minute ML model unavailable."
        )


    # News sentiment

    if news_summary["score"] >= 0.25:

        bullish_score += 1

        reasons.append(
            "Public news headlines show positive sentiment."
        )

    elif news_summary["score"] <= -0.25:

        bearish_score += 1

        reasons.append(
            "Public news headlines show negative sentiment."
        )

    else:

        warnings_list.append(
            "News sentiment is neutral or mixed."
        )


    # Option chain PCR

    if option_analysis["available"]:

        pcr = option_analysis["pcr"]

        if pcr is not None:

            if pcr >= 1.20:

                bullish_score += 1

                reasons.append(
                    "Option-chain PCR is above 1.20."
                )

            elif pcr <= 0.80:

                bearish_score += 1

                reasons.append(
                    "Option-chain PCR is below 0.80."
                )

            else:

                warnings_list.append(
                    "Option-chain PCR is in a neutral range."
                )

    else:

        warnings_list.append(
            "Option-chain data is unavailable."
        )


    # Final decision

    if (
        bullish_score >= 5
        and bullish_score - bearish_score >= 3
        and tech5["direction"] == "BULLISH"
        and tech15["direction"] == "BULLISH"
        and ml5["available"]
        and ml15["available"]
        and ml5["direction"] == "UP"
        and ml15["direction"] == "UP"
    ):

        decision = "BUY CE"

    elif (
        bearish_score >= 5
        and bearish_score - bullish_score >= 3
        and tech5["direction"] == "BEARISH"
        and tech15["direction"] == "BEARISH"
        and ml5["available"]
        and ml15["available"]
        and ml5["direction"] == "DOWN"
        and ml15["direction"] == "DOWN"
    ):

        decision = "BUY PE"

    else:

        decision = "NO TRADE"

    return {
        "decision": decision,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "reasons": reasons,
        "warnings": warnings_list
    }


decision_result = calculate_trade_decision(
    tech5,
    tech15,
    ml5,
    ml15,
    news_summary,
    option_analysis
)

decision = decision_result["decision"]

st.session_state.last_decision = decision


# ============================================================
# 30. DISPLAY TRADE DECISION
# ============================================================

st.subheader("Final AI Decision")

d1, d2, d3 = st.columns(3)

d1.metric(
    "Final Decision",
    decision
)

d2.metric(
    "Bullish Score",
    decision_result["bullish_score"]
)

d3.metric(
    "Bearish Score",
    decision_result["bearish_score"]
)


if decision == "BUY CE":

    st.success(
        "AI CONDITIONS FAVOUR A CE SETUP. "
        "Verify the live contract and risk before trading."
    )

elif decision == "BUY PE":

    st.error(
        "AI CONDITIONS FAVOUR A PE SETUP. "
        "Verify the live contract and risk before trading."
    )

else:

    st.warning(
        "NO TRADE — Required confirmations are not aligned."
    )


st.subheader("Reasons Behind Decision")

for reason in decision_result["reasons"]:

    st.write("✅", reason)


st.subheader("Reasons for Caution")

for warning in decision_result["warnings"]:

    st.write("⚠️", warning)


# ============================================================
# 31. OPTION PREMIUM ENTRY / SL / TARGET ENGINE
# ============================================================

def calculate_option_trade_plan(
    contract,
    capital,
    risk_percent,
    lot_size,
    stop_loss_percent
):

    if contract is None:
        return None

    entry = safe_float(contract["ask"])

    bid = safe_float(contract["bid"])

    if (
        np.isnan(entry)
        or np.isnan(bid)
        or entry <= 0
        or bid <= 0
        or bid > entry
    ):
        return None

    stop_distance = (
        entry * stop_loss_percent / 100
    )

    stop_loss = entry - stop_distance

    if stop_loss <= 0:
        return None

    risk_per_unit = entry - stop_loss

    risk_budget = (
        capital * risk_percent / 100
    )

    risk_per_lot = (
        risk_per_unit * lot_size
    )

    if risk_per_lot <= 0:
        return None

    lots_by_risk = math.floor(
        risk_budget / risk_per_lot
    )

    lots_by_capital = math.floor(
        capital / (entry * lot_size)
    )

    lots = min(
        lots_by_risk,
        lots_by_capital
    )

    if lots < 1:
        quantity = 0

    else:
        quantity = lots * lot_size

    target1 = entry + risk_per_unit * 1
    target2 = entry + risk_per_unit * 2
    target3 = entry + risk_per_unit * 3

    return {
        "side": contract["side"],
        "strike": contract["strike"],
        "entry": entry,
        "bid": bid,
        "spread": contract["spread"],
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "target3": target3,
        "risk_budget": risk_budget,
        "risk_per_unit": risk_per_unit,
        "risk_per_lot": risk_per_lot,
        "lots": lots,
        "quantity": quantity,
        "capital_required": (
            entry * quantity
        ),
        "expiry": option_result.get("expiry")
    }


# ============================================================
# 32. GENERATE OPTION TRADE PLAN
# ============================================================

st.divider()
st.header("🎯 CE / PE OPTION PREMIUM TRADE PLAN")

trade_plan = None
selected_contract = None

if decision in ["BUY CE", "BUY PE"]:

    direction = (
        "UP"
        if decision == "BUY CE"
        else "DOWN"
    )

    if option_result["available"]:

        selected_contract = select_option_contract(
            option_data=option_data,
            spot=spot,
            direction=direction,
            strike_mode=strike_mode
        )

        if selected_contract is not None:

            trade_plan = calculate_option_trade_plan(
                contract=selected_contract,
                capital=capital,
                risk_percent=risk_percent,
                lot_size=lot_size,
                stop_loss_percent=stop_loss_percent
            )

        else:

            st.warning(
                "No valid live bid/ask quote for the selected contract."
            )

    else:

        st.warning(
            "Option chain is unavailable. No premium plan generated."
        )

else:

    st.info(
        "AI decision is NO TRADE. No CE/PE entry is recommended."
    )



# ============================================================
# 33. DISPLAY OPTION PREMIUM PLAN
# ============================================================

if trade_plan is not None:

    st.success(
        "Option premium plan calculated from returned bid/ask data. "
        "Verify quote freshness with your broker."
    )

    st.subheader(
        f"{trade_plan['side']} — {trade_plan['strike']}"
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Entry Premium (Ask)",
        f"₹{trade_plan['entry']:.2f}"
    )

    a2.metric(
        "Bid Premium",
        f"₹{trade_plan['bid']:.2f}"
    )

    a3.metric(
        "Bid-Ask Spread",
        f"₹{trade_plan['spread']:.2f}"
    )

    st.subheader("Entry / Stop Loss / Targets")

    plan_table = pd.DataFrame({
        "Level": [
            "Entry",
            "Stop Loss",
            "Target 1",
            "Target 2",
            "Target 3"
        ],
        "Option Premium": [
            trade_plan["entry"],
            trade_plan["stop_loss"],
            trade_plan["target1"],
            trade_plan["target2"],
            trade_plan["target3"]
        ]
    })

    st.dataframe(
        plan_table.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Position Sizing")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Risk Budget",
        f"₹{trade_plan['risk_budget']:.2f}"
    )

    q2.metric(
        "Lots",
        trade_plan["lots"]
    )

    q3.metric(
        "Quantity",
        trade_plan["quantity"]
    )

    q4.metric(
        "Capital Required",
        f"₹{trade_plan['capital_required']:.2f}"
    )

    if trade_plan["quantity"] == 0:

        st.warning(
            "Capital/risk budget is insufficient for even one lot. "
            "Do not increase risk just to force a trade."
        )

    st.caption(
        "Targets are mathematical risk-reward levels, not predicted "
        "market prices. Brokerage, taxes, slippage and execution delays "
        "are not included. Stop loss is not guaranteed."
    )

else:

    st.warning(
        "No valid premium-based trade plan is available. "
        "Do not use estimated or manually invented option prices."
    )


# ============================================================
# 34. RISK REWARD VISUALIZATION
# ============================================================

if trade_plan is not None:

    st.subheader("Risk / Reward Levels")

    levels = pd.DataFrame({
        "Level": [
            "Stop Loss",
            "Entry",
            "Target 1",
            "Target 2",
            "Target 3"
        ],
        "Premium": [
            trade_plan["stop_loss"],
            trade_plan["entry"],
            trade_plan["target1"],
            trade_plan["target2"],
            trade_plan["target3"]
        ]
    })

    fig_levels = px.bar(
        levels,
        x="Level",
        y="Premium",
        title="Option Premium Risk / Reward Levels"
    )

    fig_levels.update_layout(
        template="plotly_dark",
        height=350
    )

    st.plotly_chart(
        fig_levels,
        use_container_width=True
    )


# ============================================================
# 35. NIFTY CANDLESTICK CHART
# ============================================================

st.divider()
st.header("📉 NIFTY 5-Minute Candlestick Chart")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df5.index,
        open=df5["open"],
        high=df5["high"],
        low=df5["low"],
        close=df5["close"],
        name="NIFTY"
    )
)

fig.add_trace(
    go.Scatter(
        x=df5.index,
        y=df5["ema9"],
        name="EMA 9",
        line=dict(width=1)
    )
)

fig.add_trace(
    go.Scatter(
        x=df5.index,
        y=df5["ema21"],
        name="EMA 21",
        line=dict(width=1)
    )
)

fig.add_trace(
    go.Scatter(
        x=df5.index,
        y=df5["ema50"],
        name="EMA 50",
        line=dict(width=1)
    )
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False,
    yaxis_title="NIFTY Price",
    xaxis_title="Time"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 36. 15-MINUTE CHART
# ============================================================

st.header("📊 NIFTY 15-Minute Chart")

fig15 = go.Figure()

fig15.add_trace(
    go.Candlestick(
        x=df15.index,
        open=df15["open"],
        high=df15["high"],
        low=df15["low"],
        close=df15["close"],
        name="NIFTY 15m"
    )
)

fig15.add_trace(
    go.Scatter(
        x=df15.index,
        y=df15["ema21"],
        name="EMA 21"
    )
)

fig15.add_trace(
    go.Scatter(
        x=df15.index,
        y=df15["ema50"],
        name="EMA 50"
    )
)

fig15.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig15,
    use_container_width=True
)


# ============================================================
# 37. TECHNICAL INDICATOR TABLE
# ============================================================

st.divider()
st.header("📋 TECHNICAL INDICATOR TABLE")

indicator_table = pd.DataFrame({
    "Indicator": [
        "EMA 9",
        "EMA 21",
        "EMA 50",
        "EMA 200",
        "RSI",
        "MACD",
        "MACD Signal",
        "MACD Histogram",
        "ATR",
        "VWAP Proxy",
        "Volatility"
    ],
    "5 Minute": [
        last5["ema9"],
        last5["ema21"],
        last5["ema50"],
        last5["ema200"],
        last5["rsi"],
        last5["macd"],
        last5["macd_signal"],
        last5["macd_hist"],
        last5["atr"],
        last5["vwap_proxy"],
        last5["volatility"]
    ],
    "15 Minute": [
        last15["ema9"],
        last15["ema21"],
        last15["ema50"],
        last15["ema200"],
        last15["rsi"],
        last15["macd"],
        last15["macd_signal"],
        last15["macd_hist"],
        last15["atr"],
        last15["vwap_proxy"],
        last15["volatility"]
    ]
})

st.dataframe(
    indicator_table.round(3),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 38. OFFICIAL MARKET SOURCES
# ============================================================

st.divider()
st.header("🌐 OFFICIAL MARKET SOURCES")

market_sources = [
    (
        "NSE India",
        "https://www.nseindia.com/"
    ),
    (
        "NSE Option Chain",
        "https://www.nseindia.com/option-chain"
    ),
    (
        "NSE FII/DII",
        "https://www.nseindia.com/reports/fii-dii"
    ),
    (
        "SEBI",
        "https://www.sebi.gov.in/"
    ),
    (
        "RBI",
        "https://www.rbi.org.in/"
    ),
    (
        "Yahoo Finance",
        "https://finance.yahoo.com/"
    ),
    (
        "Google News",
        "https://news.google.com/"
    )
]

for source_name, source_url in market_sources:

    st.markdown(
        f"- [{source_name}]({source_url})"
    )


# ============================================================
# 39. SYSTEM STATUS
# ============================================================

st.divider()
st.header("🖥️ SYSTEM STATUS")

status_data = pd.DataFrame({
    "Component": [
        "NIFTY 5m Data",
        "NIFTY 15m Data",
        "5m ML Model",
        "15m ML Model",
        "News Feed",
        "Option Chain",
        "FII/DII",
        "Global Market"
    ],
    "Status": [
        "Available" if not df5.empty else "Unavailable",
        "Available" if not df15.empty else "Unavailable",
        "Available" if ml5["available"] else "Unavailable",
        "Available" if ml15["available"] else "Unavailable",
        "Available" if not news.empty else "Unavailable",
        "Available" if option_result["available"] else "Unavailable",
        "Available" if not fii_dii.empty else "Unavailable",
        "Available" if not global_df.empty else "Unavailable"
    ]
})

st.dataframe(
    status_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 40. DISCLAIMER AND FINAL FOOTER
# ============================================================

st.divider()

st.warning(
    """
    IMPORTANT RISK DISCLOSURE

    1. This application is for education and research only.
    2. It does not provide guaranteed trading signals.
    3. Options trading can result in complete premium loss.
    4. Market data may be delayed or unavailable.
    5. ML predictions are experimental.
    6. News sentiment is keyword-based.
    7. Option chain values must be verified with a broker.
    8. Stop loss execution is not guaranteed.
    9. Brokerage, taxes and slippage are not included.
    10. Paper trade and backtest before risking real money.
    """
)

st.caption(
    "NIFTY AI ULTIMATE TERMINAL | "
    "Educational Dashboard | "
    + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)

st.caption(
    "Built with Streamlit, Python, Yahoo Finance, "
    "NSE public endpoints and Scikit-learn."
)

# ============================================================
# END OF PART 2
# ============================================================
