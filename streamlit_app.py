
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from datetime import datetime, timezone
from urllib.parse import quote
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NIFTY AI Intelligence Terminal",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 — AI Intelligence & Options Terminal")
st.caption(
    "Live public news + market data + FII/DII + option chain "
    "+ technical analysis + experimental ML"
)

# =========================================================
# CONSTANTS
# =========================================================
NSE_HOME = "https://www.nseindia.com/"
NSE_OPTION_URL = (
    "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
)
NSE_FII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

NEWS_QUERIES = [
    "NIFTY 50 Indian stock market",
    "India RBI interest rate inflation",
    "India FII DII stock market",
    "India stock market crude oil rupee",
    "NIFTY options market derivatives",
    "India economy government policy",
    "US Federal Reserve global markets"
]

# =========================================================
# HTTP SESSION
# =========================================================
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/125.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": NSE_HOME
    })
    return session


# =========================================================
# NEWS INTELLIGENCE
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_news():
    rows = []

    for query in NEWS_QUERIES:
        url = (
            "https://news.google.com/rss/search?q="
            + quote(query + " when:2d")
            + "&hl=en-IN&gl=IN&ceid=IN:en"
        )

        try:
            feed = feedparser.parse(url)

            for item in feed.entries[:12]:
                published = item.get("published", "")
                title = item.get("title", "")
                link = item.get("link", "")

                if not title or not link:
                    continue

                rows.append({
                    "Headline": title,
                    "Published": published,
                    "Source": item.get("source", {}).get(
                        "title", "Google News"
                    ),
                    "Link": link
                })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=["Headline", "Published", "Source", "Link"]
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Headline"])
    return df.head(60)


POSITIVE_WORDS = [
    "surge", "rally", "gain", "growth", "profit", "strong",
    "bullish", "upgrade", "buy", "inflow", "record high",
    "rate cut", "cuts rates", "easing", "positive",
    "beats estimates", "approval", "recovery", "deal",
    "stimulus", "support", "expansion"
]

NEGATIVE_WORDS = [
    "crash", "fall", "falls", "decline", "loss", "weak",
    "bearish", "downgrade", "sell", "outflow", "war",
    "attack", "crisis", "inflation rises", "rate hike",
    "hikes rates", "default", "recession", "fraud",
    "sanctions", "uncertainty", "misses estimates",
    "selloff", "sell-off", "volatility spike"
]


def score_news(text):
    text = str(text).lower()

    positive = sum(word in text for word in POSITIVE_WORDS)
    negative = sum(word in text for word in NEGATIVE_WORDS)

    if positive > negative:
        return "Positive", positive - negative

    if negative > positive:
        return "Negative", negative - positive

    return "Neutral", 0


def news_summary(news):
    if news.empty:
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "score": 0
        }

    sentiments = news["Headline"].apply(score_news)

    positive = sum(x[0] == "Positive" for x in sentiments)
    negative = sum(x[0] == "Negative" for x in sentiments)
    neutral = sum(x[0] == "Neutral" for x in sentiments)

    total = max(len(news), 1)
    score = (positive - negative) / total

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "score": score
    }


# =========================================================
# MARKET DATA
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_market_data(interval):
    period = "60d"

    data = yf.download(
        "^NSEI",
        period=period,
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
    data = data.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })

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

    # Exclude the latest possibly incomplete candle
    if len(data) > 2:
        data = data.iloc[:-1]

    return data


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()

    if len(df) < 30:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    typical = (high + low + close) / 3
    volume_sum = df["volume"].rolling(20).sum()

    df["vwap_proxy"] = (
        (typical * df["volume"]).rolling(20).sum()
        / volume_sum.replace(0, np.nan)
    )

    df["return1"] = close.pct_change()
    df["return3"] = close.pct_change(3)
    df["return5"] = close.pct_change(5)
    df["volatility"] = df["return1"].rolling(20).std()

    df["candle_body"] = (
        (df["close"] - df["open"]) / close
    )
    df["candle_range"] = (
        (df["high"] - df["low"]) / close
    )

    return df.replace([np.inf, -np.inf], np.nan)


# =========================================================
# MACHINE LEARNING
# =========================================================
FEATURES = [
    "ema9", "ema21", "ema50", "ema200",
    "rsi", "macd", "macd_signal", "atr",
    "return1", "return3", "return5",
    "volatility", "candle_body", "candle_range"
]


def train_predict(df, horizon=3):
    result = {
        "status": "Unavailable",
        "direction": "NO TRADE",
        "up": 0,
        "down": 0,
        "sideways": 0,
        "accuracy": None,
        "reasons": []
    }

    if df.empty or len(df) < 300:
        result["reasons"].append(
            "ML training ke liye sufficient historical candles nahi hain."
        )
        return result

    data = df.copy()
    future_return = (
        data["close"].shift(-horizon) / data["close"] - 1
    )

    # ATR-based neutral zone; avoids tiny movements being treated as trends
    threshold = (
        data["atr"] / data["close"] * 0.25
    ).clip(lower=0.0003)

    data["target"] = np.where(
        future_return > threshold, 1,
        np.where(future_return < -threshold, -1, 0)
    )

    data = data.dropna(subset=FEATURES + ["target"])

    if len(data) < 250:
        result["reasons"].append("Training dataset bahut chhota hai.")
        return result

    X = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = data["target"].astype(int)

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 250 or y.nunique() < 2:
        result["reasons"].append(
            "Training data mein required target classes nahi mile."
        )
        return result

    split = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2:
        result["reasons"].append(
            "Training period mein enough direction classes nahi hain."
        )
        return result

    model = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=8,
            min_samples_split=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        predictions = model.predict(X_test)
        result["accuracy"] = accuracy_score(
            y_test, predictions
        )

    latest = X.iloc[[-1]]
    probabilities = model.predict_proba(latest)[0]

    classes = model.named_steps["rf"].classes_
    prob_map = dict(zip(classes, probabilities))

    result["up"] = float(prob_map.get(1, 0))
    result["down"] = float(prob_map.get(-1, 0))
    result["sideways"] = float(prob_map.get(0, 0))
    result["status"] = "Available"

    highest = max(
        result["up"], result["down"], result["sideways"]
    )

    if highest < 0.50:
        result["direction"] = "NO TRADE"
    elif highest == result["up"]:
        result["direction"] = "UP"
    elif highest == result["down"]:
        result["direction"] = "DOWN"
    else:
        result["direction"] = "SIDEWAYS"

    return result


# =========================================================
# NSE OPTION CHAIN
# =========================================================
@st.cache_data(ttl=90, show_spinner=False)
def get_option_chain():
    try:
        session = get_session()
        session.get(NSE_HOME, timeout=10)

        response = session.get(
            NSE_OPTION_URL,
            timeout=15
        )
        response.raise_for_status()

        payload = response.json()
        records = payload.get("records", {})
        rows = records.get("data", [])

        output = []

        for row in rows:
            strike = row.get("strikePrice")

            ce = row.get("CE", {})
            pe = row.get("PE", {})

            output.append({
                "Strike": strike,
                "CE LTP": ce.get("lastPrice"),
                "CE OI": ce.get("openInterest"),
                "CE Change OI": ce.get("changeinOpenInterest"),
                "CE Volume": ce.get("totalTradedVolume"),
                "CE IV": ce.get("impliedVolatility"),
                "PE LTP": pe.get("lastPrice"),
                "PE OI": pe.get("openInterest"),
                "PE Change OI": pe.get("changeinOpenInterest"),
                "PE Volume": pe.get("totalTradedVolume"),
                "PE IV": pe.get("impliedVolatility")
            })

        df = pd.DataFrame(output)

        if df.empty:
            return df, "NSE returned no option data"

        expiry = records.get("expiryDates", [])
        expiry_text = expiry[0] if expiry else "Not available"

        return df, "Expiry: " + str(expiry_text)

    except Exception as e:
        return pd.DataFrame(), str(e)


def option_chain_analysis(options, spot):
    if options.empty:
        return {
            "available": False,
            "pcr": None,
            "support": None,
            "resistance": None,
            "reason": "Option chain unavailable."
        }

    df = options.copy()

    for col in [
        "CE OI", "PE OI", "CE Change OI",
        "PE Change OI", "Strike"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ce_oi = df["CE OI"].fillna(0).sum()
    pe_oi = df["PE OI"].fillna(0).sum()

    pcr = pe_oi / ce_oi if ce_oi > 0 else None

    resistance_row = df.loc[df["CE OI"].idxmax()]
    support_row = df.loc[df["PE OI"].idxmax()]

    support = support_row["Strike"]
    resistance = resistance_row["Strike"]

    if pcr is None:
        reason = "PCR calculate nahi ho saka."
    elif pcr > 1.2:
        reason = "PCR high: put OI comparatively zyada hai."
    elif pcr < 0.8:
        reason = "PCR low: call OI comparatively zyada hai."
    else:
        reason = "PCR neutral zone mein hai."

    return {
        "available": True,
        "pcr": pcr,
        "support": support,
        "resistance": resistance,
        "reason": reason
    }


# =========================================================
# FII / DII DATA
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_fii_dii():
    try:
        session = get_session()
        session.get(NSE_HOME, timeout=10)

        response = session.get(
            NSE_FII_URL,
            timeout=15
        )
        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            rows = data.get("data", data.get("rows", []))
        else:
            rows = data

        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(), "No published data returned"

        return pd.DataFrame(rows), "NSE response received"

    except Exception as e:
        return pd.DataFrame(), str(e)


# =========================================================
# DECISION ENGINE
# =========================================================
def technical_direction(row):
    score = 0
    reasons = []

    if row["ema9"] > row["ema21"]:
        score += 1
        reasons.append("EMA 9 > EMA 21: short-term bullish trend.")
    else:
        score -= 1
        reasons.append("EMA 9 < EMA 21: short-term bearish trend.")

    if row["close"] > row["ema50"]:
        score += 1
        reasons.append("Price EMA 50 ke upar hai.")
    else:
        score -= 1
        reasons.append("Price EMA 50 ke neeche hai.")

    if row["macd"] > row["macd_signal"]:
        score += 1
        reasons.append("MACD bullish crossover position mein hai.")
    else:
        score -= 1
        reasons.append("MACD bearish position mein hai.")

    if row["rsi"] > 55:
        score += 1
        reasons.append("RSI above 55: bullish momentum.")
    elif row["rsi"] < 45:
        score -= 1
        reasons.append("RSI below 45: bearish momentum.")
    else:
        reasons.append("RSI neutral zone mein hai.")

    return score, reasons


def make_decision(
    score5, score15, news_score, ml5, ml15,
    option_data
):
    reasons = []
    score = 0

    score += score5 * 1.0
    score += score15 * 1.5

    if news_score > 0.10:
        score += 1
        reasons.append("News sentiment positive hai.")
    elif news_score < -0.10:
        score -= 1
        reasons.append("News sentiment negative hai.")
    else:
        reasons.append("News sentiment neutral/mixed hai.")

    if ml5["direction"] == "UP":
        score += 1
        reasons.append("5m ML model UP direction dikha raha hai.")
    elif ml5["direction"] == "DOWN":
        score -= 1
        reasons.append("5m ML model DOWN direction dikha raha hai.")

    if ml15["direction"] == "UP":
        score += 1
        reasons.append("15m ML model UP direction dikha raha hai.")
    elif ml15["direction"] == "DOWN":
        score -= 1
        reasons.append("15m ML model DOWN direction dikha raha hai.")

    if option_data["available"] and option_data["pcr"] is not None:
        if option_data["pcr"] > 1.2:
            score += 1
            reasons.append("Option-chain PCR bullish side par hai.")
        elif option_data["pcr"] < 0.8:
            score -= 1
            reasons.append("Option-chain PCR bearish side par hai.")

    # Require multi-source confirmation
    if (
        score >= 3
        and score5 > 0
        and score15 > 0
        and ml5["direction"] == "UP"
        and ml15["direction"] == "UP"
    ):
        decision = "CE SETUP"
    elif (
        score <= -3
        and score5 < 0
        and score15 < 0
        and ml5["direction"] == "DOWN"
        and ml15["direction"] == "DOWN"
    ):
        decision = "PE SETUP"
    else:
        decision = "NO TRADE"

    if decision == "NO TRADE":
        reasons.append(
            "Multiple independent confirmations nahi mile. "
            "Capital protect karne ke liye wait."
        )

    return decision, score, reasons


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ AI Settings")

horizon = st.sidebar.selectbox(
    "Prediction horizon (candles)",
    [2, 3, 4, 5],
    index=1
)

risk_percent = st.sidebar.slider(
    "Illustrative risk % per trade",
    0.25, 2.0, 0.5, 0.25
)

st.sidebar.caption(
    "Risk % sirf planning input hai; quantity calculation ke liye "
    "actual option premium, lot size aur broker charges zaroori hain."
)

if st.sidebar.button("🔄 Refresh all data"):
    st.cache_data.clear()
    st.rerun()


# =========================================================
# LOAD ALL DATA
# =========================================================
with st.spinner("Market, news aur AI data load ho raha hai..."):
    news = get_news()

    raw5 = get_market_data("5m")
    raw15 = get_market_data("15m")

    df5 = add_indicators(raw5)
    df15 = add_indicators(raw15)

    options, option_status = get_option_chain()
    fii_dii, fii_status = get_fii_dii()


if df5.empty or df15.empty:
    st.error(
        "Market data available nahi hai. Yahoo Finance data "
        "temporarily unavailable ho sakta hai. Kuch der baad refresh karein."
    )
    st.stop()


# =========================================================
# CALCULATE
# =========================================================
last5 = df5.iloc[-1]
last15 = df15.iloc[-1]

spot = float(last5["close"])

ml5 = train_predict(df5, horizon)
ml15 = train_predict(df15, horizon)

score5, reasons5 = technical_direction(last5)
score15, reasons15 = technical_direction(last15)

news_stats = news_summary(news)

option_analysis = option_chain_analysis(options, spot)

decision, decision_score, decision_reasons = make_decision(
    score5, score15, news_stats["score"],
    ml5, ml15, option_analysis
)


# =========================================================
# TOP MARKET SUMMARY
# ========================
# =========================================================
# FINAL TRADE DECISION
# =========================================================
st.divider()
st.subheader("2. AI Final Trade Decision")

if decision == "CE SETUP":
    st.success("🟢 CE SETUP — bullish confirmations")
elif decision == "PE SETUP":
    st.error("🔴 PE SETUP — bearish confirmations")
else:
    st.warning("🟡 NO TRADE — confirmation incomplete")

st.write("**Decision score:**", decision_score)

for reason in decision_reasons:
    st.write("•", reason)

st.info(
    "CE/PE SETUP guaranteed buy signal nahi hai. "
    "Entry se pehle option premium, spread, expiry, IV aur liquidity verify karein."
)

# =========================================================
# WHY THIS TRADE?
# =========================================================
st.divider()
st.subheader("3. Trade Reasoning — Why / Why Not")

left, right = st.columns(2)

with left:
    st.markdown("### 5-minute technical reasons")
    for reason in reasons5:
        st.write("•", reason)

    st.markdown("### 5-minute ML")
    st.write("Direction:", ml5["direction"])
    st.write("UP:", f"{ml5['up']:.1%}")
    st.write("DOWN:", f"{ml5['down']:.1%}")
    st.write("SIDEWAYS:", f"{ml5['sideways']:.1%}")

with right:
    st.markdown("### 15-minute technical reasons")
    for reason in reasons15:
        st.write("•", reason)

    st.markdown("### 15-minute ML")
    st.write("Direction:", ml15["direction"])
    st.write("UP:", f"{ml15['up']:.1%}")
    st.write("DOWN:", f"{ml15['down']:.1%}")
    st.write("SIDEWAYS:", f"{ml15['sideways']:.1%}")

# =========================================================
# ML MODEL PERFORMANCE
# =========================================================
st.divider()
st.subheader("4. Machine Learning Evaluation")

m1, m2 = st.columns(2)

with m1:
    if ml5["accuracy"] is not None:
        st.metric(
            "5m chronological holdout accuracy",
            f"{ml5['accuracy']:.1%}"
        )
    else:
        st.write("5m accuracy unavailable")

with m2:
    if ml15["accuracy"] is not None:
        st.metric(
            "15m chronological holdout accuracy",
            f"{ml15['accuracy']:.1%}"
        )
    else:
        st.write("15m accuracy unavailable")

st.caption(
    "Ye classification accuracy hai, trading win rate nahi. "
    "Model probabilities calibrated profit probabilities nahi hain. "
    "Transaction costs aur slippage wala walk-forward backtest "
    "abhi included nahi hai."
)

# =========================================================
# NEWS INTELLIGENCE
# =========================================================
st.divider()
st.subheader("5. News Intelligence & Market Sentiment")

n1, n2, n3, n4 = st.columns(4)

n1.metric("Positive headlines", news_stats["positive"])
n2.metric("Negative headlines", news_stats["negative"])
n3.metric("Neutral headlines", news_stats["neutral"])
n4.metric("Sentiment score", f"{news_stats['score']:.2f}")

if news.empty:
    st.warning("News feed unavailable. News sentiment ko neutral assume na karein.")
else:
    display_news = news.copy()
    display_news["Sentiment"] = display_news["Headline"].apply(
        lambda x: score_news(x)[0]
    )

    for _, row in display_news.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Headline']}**")
            st.caption(
                f"{row['Source']} | {row['Published']} | "
                f"Sentiment: {row['Sentiment']}"
            )
            st.markdown(f"[Open original news]({row['Link']})")

st.caption(
    "Headline sentiment basic keyword analysis hai. "
    "Headline ka sentiment actual market impact ki guarantee nahi hai."
)

# =========================================================
# OPTION CHAIN
# =========================================================
st.divider()
st.subheader("6. NIFTY Option Chain — CE / PE")

st.write("**Option-chain status:**", option_status)

if option_analysis["available"]:
    o1, o2, o3 = st.columns(3)

    o1.metric(
        "PCR",
        f"{option_analysis['pcr']:.2f}"
        if option_analysis["pcr"] is not None else "N/A"
    )
    o2.metric(
        "Highest Put OI strike",
        str(option_analysis["support"])
    )
    o3.metric(
        "Highest Call OI strike",
        str(option_analysis["resistance"])
    )

    st.write("**Interpretation:**", option_analysis["reason"])

    st.dataframe(
        options.sort_values("Strike"),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "NSE option chain fetch nahi ho payi. "
        "Is condition mein option data ko live maan kar trade nahi lena."
    )

st.markdown(
    "[Open official NSE option chain](https://www.nseindia.com/option-chain)"
)

# =========================================================
# FII / DII
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
    st.warning(
        "FII/DII data automatically retrieve nahi ho paya. "
        "Published exchange data generally end-of-day hota hai."
    )

st.markdown(
    "[NSE official FII/DII reports](https://www.nseindia.com/reports/fii-dii)"
)

# =========================================================
# TECHNICAL CHART
# =========================================================
st.divider()
st.subheader("8. NIFTY Technical Chart")

chart = go.Figure()

chart.add_trace(go.Candlestick(
    x=df5.index,
    open=df5["open"],
    high=df5["high"],
    low=df5["low"],
    close=df5["close"],
    name="NIFTY 5m"
))

chart.add_trace(go.Scatter(
    x=df5.index,
    y=df5["ema9"],
    name="EMA 9",
    line=dict(width=1)
))

chart.add_trace(go.Scatter(
    x=df5.index,
    y=df5["ema21"],
    name="EMA 21",
    line=dict(width=1)
))

chart.add_trace(go.Scatter(
    x=df5.index,
    y=df5["ema50"],
    name="EMA 50",
    line=dict(width=1)
))

chart.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    yaxis_title="NIFTY Index",
    xaxis_title="Time"
)

st.plotly_chart(chart, use_container_width=True)

# =========================================================
# INDICATOR DASHBOARD
# =========================================================
st.subheader("9. Indicator Dashboard")

indicator_table = pd.DataFrame({
    "Indicator": [
        "EMA 9", "EMA 21", "EMA 50", "EMA 200",
        "RSI", "MACD", "MACD Signal", "ATR",
        "VWAP proxy"
    ],
    "5 Minute": [
        last5["ema9"], last5["ema21"],
        last5["ema50"], last5["ema200"],
        last5["rsi"], last5["macd"],
        last5["macd_signal"], last5["atr"],
        last5["vwap_proxy"]
    ],
    "15 Minute": [
        last15["ema9"], last15["ema21"],
        last15["ema50"], last15["ema200"],
        last15["rsi"], last15["macd"],
        last15["macd_signal"], last15["atr"],
        last15["vwap_proxy"]
    ]
})

st.dataframe(
    indicator_table.round(2),
    use_container_width=True,
    hide_index=True
)

# =========================================================
# DATA SOURCES
# =========================================================
st.divider()
st.subheader("10. Data Sources & Verification")

sources = [
    ("NSE India", "https://www.nseindia.com/"),
    ("NSE Option Chain", "https://www.nseindia.com/option-chain"),
    ("NSE FII/DII Reports", "https://www.nseindia.com/reports/fii-dii"),
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
    "loss ho sakta hai. Real capital se pehle paper trading karein."
)

st.caption(
    "NIFTY AI Intelligence Terminal | Last refresh: "
    + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)
