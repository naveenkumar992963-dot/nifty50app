import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from math import ceil

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="NIFTY 50 PRO",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 PRO PREDICTION")
st.caption("Multi-Timeframe | EMA | RSI | MACD | VWAP | Supertrend")

# ---------------- DATA FETCH ----------------

@st.cache_data(ttl=60)
def get_data(interval, period):
    df = yf.download(
        "^NSEI",
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0

    return df.dropna(subset=needed).copy()


# ---------------- INDICATORS ----------------

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # SUPERTREND
    period = 10
    factor = 3.0

    basic_upper = (
        (high + low) / 2
    ) + factor * df["ATR"]

    basic_lower = (
        (high + low) / 2
    ) - factor * df["ATR"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    st_direction = pd.Series(
        1, index=df.index, dtype=int
    )

    for i in range(1, len(df)):

        if (
            basic_upper.iloc[i] < final_upper.iloc[i-1]
            or close.iloc[i-1] > final_upper.iloc[i-1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i-1]
            or close.iloc[i-1] < final_lower.iloc[i-1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        if close.iloc[i] > final_upper.iloc[i-1]:
            st_direction.iloc[i] = 1

        elif close.iloc[i] < final_lower.iloc[i-1]:
            st_direction.iloc[i] = -1

        else:
            st_direction.iloc[i] = st_direction.iloc[i-1]

    df["ST_DIRECTION"] = st_direction

    # VWAP - session based
    volume = df["Volume"].fillna(0)

    typical_price = (high + low + close) / 3

    if volume.sum() > 0:

        session = pd.Series(
            df.index.date, index=df.index
        )

        pv = typical_price * volume

        df["VWAP"] = (
            pv.groupby(session).cumsum()
            / volume.groupby(session).cumsum().replace(0, np.nan)
        )

    else:
        df["VWAP"] = np.nan

    # Volume confirmation
    df["VOL_AVG"] = volume.rolling(20).mean()

    return df


# ---------------- MARKET DATA ----------------

try:

    df5 = calculate_indicators(
        get_data("5m", "5d")
    )

    df15 = calculate_indicators(
        get_data("15m", "5d")
    )

    df60 = calculate_indicators(
        get_data("60m", "1mo")
    )

    if any(x.empty for x in [df5, df15, df60]):
        st.error("Market data unavailable. Please refresh.")
        st.stop()

    current = df5.iloc[-1]
    candle15 = df15.iloc[-1]
    candle60 = df60.iloc[-1]

    price = float(current["Close"])
    atr = float(current["ATR"])
    rsi = float(current["RSI"])

    if not np.isfinite(atr) or atr <= 0:
        st.error("ATR data not available.")
        st.stop()

    # ---------------- TREND SCORE ----------------

    buy_score = 0
    sell_score = 0

    details = []

    # EMA 9 / 21 / 200
    if (
        current["EMA9"] > current["EMA21"]
        and current["EMA21"] > current["EMA200"]
    ):
        buy_score += 2
        details.append(("EMA Trend", "Bullish"))

    elif (
        current["EMA9"] < current["EMA21"]
        and current["EMA21"] < current["EMA200"]
    ):
        sell_score += 2
        details.append(("EMA Trend", "Bearish"))

    else:
        details.append(("EMA Trend", "Mixed"))

    # RSI
    if 52 <= rsi < 70:
        buy_score += 1
        details.append(("RSI", "Bullish"))

    elif 30 < rsi <= 48:
        sell_score += 1
        details.append(("RSI", "Bearish"))

    else:
        details.append(("RSI", "Neutral"))

    # MACD
    if current["MACD"] > current["MACD_SIGNAL"]:
        buy_score += 1
        details.append(("MACD", "Bullish"))

    elif current["MACD"] < current["MACD_SIGNAL"]:
        sell_score += 1
        details.append(("MACD", "Bearish"))

    # Supertrend
    if current["ST_DIRECTION"] == 1:
        buy_score += 1
        details.append(("Supertrend", "Bullish"))

    else:
        sell_score += 1
        details.append(("Supertrend", "Bearish"))

    # VWAP
    vwap = current["VWAP"]

    if pd.notna(vwap):
        if price > vwap:
            buy_score += 1
            details.append(("VWAP", "Bullish"))
        elif price < vwap:
            sell_score += 1
            details.append(("VWAP", "Bearish"))

    # Volume confirmation
    vol = current["Volume"]
    vol_avg = current["VOL_AVG"]

    if (
        pd.notna(vol_avg)
        and vol_avg > 0
        and vol > vol_avg
    ):
        if current["Close"] > current["Open"]:
            buy_score += 1
            details.append(("Volume", "Bullish"))
        elif current["Close"] < current["Open"]:
            sell_score += 1
            details.append(("Volume", "Bearish"))

    # 15m confirmation
    if candle15["EMA9"] > candle15["EMA21"]:
        buy_score += 1
        details.append(("15m Trend", "Bullish"))
    else:
        sell_score += 1
        details.append(("15m Trend", "Bearish"))

    # 1h confirmation
    if candle60["EMA9"] > candle60["EMA21"]:
        buy_score += 1
        details.append(("1H Trend", "Bullish"))
    else:
        sell_score += 1
        details.append(("1H Trend", "Bearish"))

    # Available maximum score
    max_score = 2 + 1 + 1 + 1 + 1 + 1

    if pd.notna(vwap):
        max_score += 1

    if pd.notna(vol_avg) and vol_avg > 0:
        max_score += 1

    # Prediction threshold
    threshold = ceil(max_score * 0.70)

    if (
        buy_score >= threshold
        and buy_score - sell_score >= 2
    ):
        signal = "STRONG BUY"

    elif (
        sell_score >= threshold
        and sell_score - buy_score >= 2
    ):
        signal = "STRONG SELL"

    else:
        signal = "NO TRADE"

    confidence = (
        max(buy_score, sell_score) / max_score
    ) * 100

    # ---------------- DASHBOARD ----------------

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("NIFTY 50 PRICE", f"{price:,.2f}")
    c2.metric("RSI 14", f"{rsi:.2f}")
    c3.metric("ATR 14", f"{atr:.2f}")

    st.divider()

    st.subheader("🤖 STRATEGY PREDICTION")

    if signal == "STRONG BUY":
        st.success("🟢 STRONG BUY")

    elif signal == "STRONG SELL":
        st.error("🔴 STRONG SELL")

    else:
        st.warning("🟡 NO TRADE - WAIT")

    st.metric(
        "Indicator Agreement Score (not probability)",
        f"{confidence:.1f}%"
    )

    st.write(
        f"BUY Score: {buy_score}/{max_score}"
    )

    st.write(
        f"SELL Score: {sell_score}/{max_score}"
    )

    st.progress(
        min(1.0, confidence / 100)
    )

    st.divider()

    st.subheader("📊 Indicator Confirmation")

    st.dataframe(
        pd.DataFrame(
            details,
            columns=["Indicator", "Direction"]
        ),
        use_container_width=True,
        hide_index=True
    )

    # ---------------- ENTRY / SL / TARGETS ----------------

    st.divider()

    st.subheader("🎯 TRADE LEVEL CALCULATOR")

    trade_mode = st.radio(
        "Trade Side",
        ["Use Strategy Signal", "Manual BUY", "Manual SELL"],
        horizontal=True
    )

    if trade_mode == "Use Strategy Signal":

        trade_side = signal

    elif trade_mode == "Manual BUY":

        trade_side = "BUY"

    else:

        trade_side = "SELL"

    if trade_side in ["STRONG BUY", "BUY"]:

        entry = price
        sl = entry - (atr * 1.2)

        risk = entry - sl

        target1 = entry + risk
        target2 = entry + (risk * 2)
        target3 = entry + (risk * 3)

    elif trade_side in ["STRONG SELL", "SELL"]:

        entry = price
        sl = entry + (atr * 1.2)

        risk = sl - entry

        target1 = entry - risk
        target2 = entry - (risk * 2)
        target3 = entry - (risk * 3)

    else:

        entry = sl = target1 = target2 = target3 = None

    if entry is not None:

        a, b = st.columns(2)

        a.metric("ENTRY", f"{entry:,.2f}")
        b.metric("STOP LOSS", f"{sl:,.2f}")

        x, y, z = st.columns(3)

        x.metric("TARGET 1 (1R)", f"{target1:,.2f}")
        y.metric("TARGET 2 (2R)", f"{target2:,.2f}")
        z.metric("TARGET 3 (3R)", f"{target3:,.2f}")

        st.info(
            "Levels are ATR-based estimates, not guaranteed execution prices."
        )

    else:

        st.info("No trade setup. Wait for stronger confirmation.")

    # ---------------- CHART ----------------

    st.divider()

    st.subheader("📈 NIFTY 50 PRICE CHART")

    st.line_chart(
        df5[["Close", "EMA9", "EMA21"]].tail(100)
    )

    with st.expander("Recent Market Data"):
        st.dataframe(
            df5[
                ["Open", "High", "Low", "Close",
                 "RSI", "MACD", "ATR"]
            ].tail(20),
            use_container_width=True
        )

    st.caption(
        "Data source: Yahoo Finance. Intraday data may be delayed. "
        "The latest candle may still be forming."
    )

except Exception as e:

    st.error(f"Application Error: {e}")

    st.info(
        "Refresh the app, check internet connection "
        "and verify that all required libraries are installed."
    )

st.warning(
    "Risk Warning: This is an experimental indicator-based tool, "
    "not a guaranteed prediction system. A score is not a probability "
    "of profit. Backtest and paper trade before using real money."
)
