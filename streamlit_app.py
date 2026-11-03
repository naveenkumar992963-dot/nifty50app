import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# NIFTY 50 PRO MAX - FINAL DASHBOARD
# ==========================================

st.set_page_config(
    page_title="NIFTY 50 PRO MAX",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 NIFTY 50 PRO MAX")
st.caption("Advanced Multi-Timeframe Trading Analysis")

# ==========================================
# SIDEBAR SETTINGS
# ==========================================

st.sidebar.title("⚙️ Trading Settings")

sl_multiplier = st.sidebar.number_input(
    "Stop Loss ATR Multiplier",
    min_value=0.5,
    max_value=3.0,
    value=1.2,
    step=0.1
)

signal_threshold = st.sidebar.slider(
    "Minimum Signal Agreement",
    min_value=50,
    max_value=90,
    value=65,
    step=5
)

st.sidebar.info(
    "Market: NIFTY 50 Index\n\n"
    "Timeframes: 5m, 15m, 1H"
)

# ==========================================
# DATA FETCH
# ==========================================

@st.cache_data(ttl=60)
def get_data(interval, period):

    df = yf.download(
        "^NSEI",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    # Handle Yahoo Finance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):

        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)

            if "Close" in values:
                df.columns = values
                break

    required = ["Open", "High", "Low", "Close"]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.dropna(subset=required).copy()

    return df


# ==========================================
# INDICATOR CALCULATIONS
# ==========================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].fillna(0).astype(float)

    # EMA
    df["EMA9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200, adjust=False
    ).mean()

    # RSI 14
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )

    df["RSI"] = rsi

    # MACD
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # ATR 14
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    # SUPERTREND 10, 3
    factor = 3.0

    midpoint = (high + low) / 2

    basic_upper = midpoint + factor * df["ATR"]
    basic_lower = midpoint - factor * df["ATR"]

    upper = np.full(len(df), np.nan)
    lower = np.full(len(df), np.nan)
    direction = np.ones(len(df), dtype=int)

    close_values = close.to_numpy()
    upper_basic = basic_upper.to_numpy()
    lower_basic = basic_lower.to_numpy()
    atr_values = df["ATR"].to_numpy()

    for i in range(len(df)):

        if not np.isfinite(atr_values[i]):
            continue

        if i == 0 or not np.isfinite(upper[i-1]):

            upper[i] = upper_basic[i]
            lower[i] = lower_basic[i]
            direction[i] = 1

            continue

        upper[i] = (
            upper_basic[i]
            if (
                upper_basic[i] < upper[i-1]
                or close_values[i-1] > upper[i-1]
            )
            else upper[i-1]
        )

        lower[i] = (
            lower_basic[i]
            if (
                lower_basic[i] > lower[i-1]
                or close_values[i-1] < lower[i-1]
            )
            else lower[i-1]
        )

        if direction[i-1] == 1:

            direction[i] = (
                -1 if close_values[i] < lower[i] else 1
            )

        else:

            direction[i] = (
                1 if close_values[i] > upper[i] else -1
            )

    df["ST_DIRECTION"] = direction

    df["SUPERTREND"] = np.where(
        direction == 1, lower, upper
    )

    # SESSION VWAP
    typical_price = (high + low + close) / 3

    if volume.sum() > 0:

        sessions = pd.Series(
            df.index.date,
            index=df.index
        )

        pv = typical_price * volume

        cumulative_pv = pv.groupby(sessions).cumsum()

        cumulative_volume = (
            volume.groupby(sessions).cumsum()
        )

        df["VWAP"] = (
            cumulative_pv /
            cumulative_volume.replace(0, np.nan)
        )

    else:

        df["VWAP"] = np.nan

    # VOLUME AVERAGE
    df["VOL_AVG"] = volume.rolling(20).mean()

    return df


# ==========================================
# LOAD MARKET DATA
# ==========================================

if st.button("🔄 REFRESH MARKET DATA"):

    st.cache_data.clear()
    st.rerun()

try:

    with st.spinner("Loading NIFTY 50 market data..."):

        raw5 = get_data("5m", "5d")
        raw15 = get_data("15m", "1mo")
        raw60 = get_data("1h", "3mo")

        if any(
            df.empty
            for df in [raw5, raw15, raw60]
        ):

            st.error(
                "Market data unavailable. Please refresh later."
            )

            st.stop()

        df5 = calculate_indicators(raw5)
        df15 = calculate_indicators(raw15)
        df60 = calculate_indicators(raw60)

    # Latest available candles
    c5 = df5.iloc[-1]
    c15 = df15.iloc[-1]
    c60 = df60.iloc[-1]

    price = float(c5["Close"])
    atr = float(c5["ATR"])
    rsi = float(c5["RSI"])

    if not np.isfinite(atr) or atr <= 0:

        st.error("ATR data is insufficient.")
        st.stop()

    if not np.isfinite(rsi):

        st.error("RSI data is insufficient.")
        st.stop()

    # ======================================
    # SIGNAL ENGINE
    # ======================================

    buy_score = 0
    sell_score = 0
    max_score = 0

    details = []

    def add_vote(name, bullish, bearish, weight=1):

        global_dummy = None

        nonlocal_values = {
            "bullish": bullish,
            "bearish": bearish
        }

        global buy_score, sell_score, max_score

        max_score += weight

        if bullish:

            buy_score += weight
            result = "BULLISH"

        elif bearish:

            sell_score += weight
            result = "BEARISH"

        else:

            result = "NEUTRAL"

        details.append({
            "Indicator": name,
            "Signal": result,
            "Weight": weight
        })

    # 5 MINUTE EMA TREND

    ema_bull = (
        c5["EMA9"] > c5["EMA21"]
        and c5["EMA21"] > c5["EMA50"]
        and c5["Close"] > c5["EMA200"]
    )

    ema_bear = (
        c5["EMA9"] < c5["EMA21"]
        and c5["EMA21"] < c5["EMA50"]
        and c5["Close"] < c5["EMA200"]
    )

    add_vote(
        "5m EMA 9/21/50/200",
        ema_bull,
        ema_bear,
        2
    )

    # RSI

    add_vote(
        "RSI 14",
        52 <= rsi < 70,
        30 < rsi <= 48,
        1
    )

    # MACD

    add_vote(
        "MACD",
        c5["MACD"] > c5["MACD_SIGNAL"],
        c5["MACD"] < c5["MACD_SIGNAL"],
        1
    )

    # SUPERTREND

    add_vote(
        "Supertrend",
        c5["ST_DIRECTION"] == 1,
        c5["ST_DIRECTION"] == -1,
        1
    )

    # VWAP

    if pd.notna(c5["VWAP"]):

        add_vote(
            "VWAP",
            price > c5["VWAP"],
            price < c5["VWAP"],
            1
        )

    # VOLUME

    if (
        c5["Volume"] > 0
        and pd.notna(c5["VOL_AVG"])
        and c5["VOL_AVG"] > 0
    ):

        volume_bull = (
            c5["Volume"] > c5["VOL_AVG"]
            and c5["Close"] > c5["Open"]
        )

        volume_bear = (
            c5["Volume"] > c5["VOL_AVG"]
            and c5["Close"] < c5["Open"]
        )

        add_vote(
            "Volume Confirmation",
            volume_bull,
            volume_bear,
            1
        )

    # 15 MINUTE EMA

    add_vote(
        "15m EMA Trend",
        (
            c15["EMA9"] > c15["EMA21"]
            and c15["EMA21"] > c15["EMA50"]
        ),
        (
            c15["EMA9"] < c15["EMA21"]
            and c15["EMA21"] < c15["EMA50"]
        ),
        2
    )

    # 15 MINUTE SUPERTREND

    add_vote(
        "15m Supertrend",
        c15["ST_DIRECTION"] == 1,
        c15["ST_DIRECTION"] == -1,
        1
    )

    # 1 HOUR EMA

    add_vote(
        "1H EMA Trend",
        (
            c60["EMA9"] > c60["EMA21"]
            and c60["EMA21"] > c60["EMA50"]
        ),
        (
            c60["EMA9"] < c60["EMA21"]
            and c60["EMA21"] < c60["EMA50"]
        ),
        2
    )

    # 1 HOUR SUPERTREND

    add_vote(
        "1H Supertrend",
        c60["ST_DIRECTION"] == 1,
        c60["ST_DIRECTION"] == -1,
        1
    )

    # ======================================
    # FINAL SIGNAL
    # ======================================

    buy_ratio = buy_score / max_score
    sell_ratio = sell_score / max_score

    required_score = signal_threshold / 100

    required_gap = 0.20

    if (
        buy_ratio >= required_score
        and buy_ratio - sell_ratio >= required_gap
    ):

        signal = "STRONG BUY"

    elif (
        sell_ratio >= required_score
        and sell_ratio - buy_ratio >= required_gap
    ):

        signal = "STRONG SELL"

    else:

        signal = "NO TRADE"

    agreement = max(buy_ratio, sell_ratio) * 100

    # ======================================
    # MAIN DASHBOARD
    # ======================================

    st.divider()

    st.subheader("📊 LIVE MARKET OVERVIEW")

    a, b, c, d = st.columns(4)

    a.metric(
        "NIFTY 50",
        f"{price:,.2f}"
    )

    b.metric(
        "RSI 14",
        f"{rsi:.2f}"
    )

    c.metric(
        "ATR 14",
        f"{atr:.2f}"
    )

    d.metric(
        "MACD",
        f"{c5['MACD']:.2f}"
    )

    st.divider()

    st.subheader("🤖 PRO SIGNAL PREDICTION")

    if signal == "STRONG BUY":

        st.success("🟢 STRONG BUY")

    elif signal == "STRONG SELL":

        st.error("🔴 STRONG SELL")

    else:

        st.warning("🟡 NO TRADE — WAIT")

    st.metric(
        "Indicator Agreement (NOT win probability)",
        f"{agreement:.1f}%"
    )

    st.progress(
        min(1.0, agreement / 100)
    )

    x, y = st.columns(2)

    x.metric(
        "BUY SCORE",
        f"{buy_score}/{max_score}"
    )

    y.metric(
        "SELL SCORE",
        f"{sell_score}/{max_score}"
    )

    st.divider()

    # ======================================
    # INDICATOR TABLE
    # ======================================

    st.subheader("🔍 INDICATOR CONFIRMATION")

    detail_df = pd.DataFrame(details)

    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # TRADE CALCULATOR
    # ======================================

    st.divider()

    st.subheader("🎯 ENTRY / SL / TARGET CALCULATOR")

    trade_mode = st.selectbox(
        "Select Trading Mode",
        [
            "Automatic Strategy Signal",
            "Manual BUY",
            "Manual SELL"
        ]
    )

    if trade_mode == "Automatic Strategy Signal":

        trade_side = signal

    elif trade_mode == "Manual BUY":

        trade_side = "BUY"

    else:

        trade_side = "SELL"

    entry = st.number_input(
        "Entry Price",
        min_value=0.05,
        value=round(price, 2),
        step=0.05
    )

    atr_multiplier = sl_multiplier

    if trade_side in ["STRONG BUY", "BUY"]:

        risk = atr * atr_multiplier

        stop_loss = entry - risk

        target1 = entry + risk
        target2 = entry + (risk * 2)
        target3 = entry + (risk * 3)

    elif trade_side in ["STRONG SELL", "SELL"]:

        risk = atr * atr_multiplier

        stop_loss = entry + risk

        target1 = entry - risk
        target2 = entry - (risk * 2)
        target3 = entry - (risk * 3)

    else:

        risk = None
        stop_loss = None
        target1 = None
        target2 = None
        target3 = None

    if risk is not None:

        st.write(
            f"Selected Trade: **{trade_side}**"
        )

        p1, p2 = st.columns(2)

        p1.metric(
            "ENTRY PRICE",
            f"{entry:,.2f}"
        )

        p2.metric(
            "STOP LOSS",
            f"{stop_loss:,.2f}"
        )

        t1, t2, t3 = st.columns(3)

        t1.metric(
            "TARGET 1 (1R)",
            f"{target1:,.2f}"
        )

        t2.metric(
            "TARGET 2 (2R)",
            f"{target2:,.2f}"
        )

        t3.metric(
            "TARGET 3 (3R)",
            f"{target3:,.2f}"
        )

        st.info(
            f"Risk Distance: {risk:.2f} NIFTY points"
        )

    else:

        st.warning(
            "No valid automatic trade signal. "
            "Wait for market confirmation or select manual trade."
        )

    # ======================================
    # CHART
    # ======================================

    st.divider()

    st.subheader("📈 NIFTY 50 PRICE CHART")

    chart = df5[
        ["Close", "EMA9", "EMA21", "EMA50"]
    ].tail(100)

    st.line_chart(chart)

    # ======================================
    # TIMEFRAME SUMMARY
    # ======================================

    st.divider()

    st.subheader("⏱️ MULTI-TIMEFRAME ANALYSIS")

    timeframe_data = pd.DataFrame([
        {
            "Timeframe": "5 Minutes",
            "EMA Trend": (
                "Bullish" if c5["EMA9"] > c5["EMA21"]
                else "Bearish"
            ),
            "Supertrend": (
                "Bullish" if c5["ST_DIRECTION"] == 1
                else "Bearish"
            )
        },
        {
            "Timeframe": "15 Minutes",
            "EMA Trend": (
                "Bullish" if c15["EMA9"] > c15["EMA21"]
                else "Bearish"
            ),
            "Supertrend": (
                "Bullish" if c15["ST_DIRECTION"] == 1
                else "Bearish"
            )
        },
        {
            "Timeframe": "1 Hour",
            "EMA Trend": (
                "Bullish" if c60["EMA9"] > c60["EMA21"]
                else "Bearish"
            ),
            "Supertrend": (
                "Bullish" if c60["ST_DIRECTION"] == 1
                else "Bearish"
            )
        }
    ])

    st.dataframe(
        timeframe_data,
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # RECENT DATA
    # ======================================

    with st.expander("📋 Recent Market Candles"):

        st.dataframe(
            df5[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "EMA9",
                    "EMA21",
                    "RSI",
                    "MACD",
                    "ATR"
                ]
            ].tail(30),
            use_container_width=True
        )

    # ======================================
    # DISCLAIMER
    # ======================================

    st.caption(
        "Market data source: Yahoo Finance (^NSEI). "
        "Intraday data can be delayed and the latest candle may be incomplete."
    )

    st.warning(
        "IMPORTANT: This is an experimental indicator-based analysis tool. "
        "Agreement score is not a probability of winning. "
        "No strategy guarantees accurate predictions or profits. "
        "Backtest and paper trade before using real money. "
        "NIFTY index points are not option premiums."
    )

except Exception as e:

    st.error(
        f"Application Error: {str(e)}"
    )

    st.info(
        "Check internet connection, data availability "
        "and installed package versions. Then refresh the app."
    )
