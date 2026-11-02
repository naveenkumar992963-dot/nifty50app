
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import quote

st.set_page_config(page_title="NIFTY AI Signal Desk", page_icon="📈", layout="wide")

BASE = "https://api.upstox.com"
NIFTY_KEY = "NSE_INDEX|Nifty 50"

def api_get(path, token, params=None):
    headers = {
        "Accept": "application/json",
        "Api-Version": "2.0",
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(
        BASE + path,
        headers=headers,
        params=params,
        timeout=20
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") == "error":
        raise RuntimeError(str(result))
    return result.get("data", {})

def candles_to_df(raw):
    rows = raw.get("candles", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "volume", "oi"]
    )
    df["time"] = pd.to_datetime(df["time"])

    for col in ["open", "high", "low", "close", "volume", "oi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("time").dropna(subset=["close"]).reset_index(drop=True)

def get_intraday(token, key, minutes):
    path = (
        f"/v3/historical-candle/intraday/"
        f"{quote(key, safe='')}/minutes/{minutes}"
    )
    return candles_to_df(api_get(path, token))

def get_historical(token, key, minutes, days=45):
    end = datetime.now().date()
    start = end - timedelta(days=days)

    path = (
        f"/v3/historical-candle/{quote(key, safe='')}/minutes/"
        f"{minutes}/{end.isoformat()}/{start.isoformat()}"
    )
    return candles_to_df(api_get(path, token))

def indicators(df):
    d = df.copy()

    d["EMA9"] = d.close.ewm(span=9, adjust=False).mean()
    d["EMA21"] = d.close.ewm(span=21, adjust=False).mean()
    d["EMA50"] = d.close.ewm(span=50, adjust=False).mean()

    delta = d.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        d.high - d.low,
        (d.high - d.close.shift()).abs(),
        (d.low - d.close.shift()).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    return d

def get_signal(df):
    if len(df) < 55:
        return "NO TRADE", 0, None

    d = indicators(df)
    x = d.iloc[-1]
    previous = d.iloc[-2]

    score = 0

    score += 1 if x.close > x.EMA21 else -1
    score += 1 if x.EMA9 > x.EMA21 else -1
    score += 1 if x.EMA21 > x.EMA50 else -1
    score += 1 if x.RSI > 52 else (-1 if x.RSI < 48 else 0)
    score += 1 if x.close > previous.high else (
        -1 if x.close < previous.low else 0
    )

    if score >= 3 and x.RSI < 72:
        return "BUY CE", score, d

    if score <= -3 and x.RSI > 28:
        return "BUY PE", score, d

    return "NO TRADE", score, d

def make_option_table(chain):
    rows = []

    for item in chain:
        strike = item.get("strike_price")

        for side, field in [
            ("CE", "call_options"),
            ("PE", "put_options")
        ]:
            option = item.get(field) or {}
            market = option.get("market_data") or {}
            greeks = option.get("option_greeks") or {}

            if market:
                rows.append({
                    "Strike": strike,
                    "Side": side,
                    "LTP": market.get("ltp"),
                    "Bid": market.get("bid_price"),
                    "Ask": market.get("ask_price"),
                    "Volume": market.get("volume"),
                    "OI": market.get("oi"),
                    "IV": greeks.get("iv"),
                    "Delta": greeks.get("delta"),
                    "Theta": greeks.get("theta"),
                    "Instrument": option.get("instrument_key")
                })

    return pd.DataFrame(rows)

def select_contract(options, side, spot):
    data = options[options.Side == side].copy()
    data = data.dropna(subset=["LTP", "Strike"])

    if data.empty:
        return None

    data["Distance"] = abs(data.Strike - spot)

    data["Spread"] = (
        (data.Ask - data.Bid) / data.LTP * 100
    ).replace([np.inf, -np.inf], np.nan)

    data["Volume"] = pd.to_numeric(
        data.Volume, errors="coerce"
    ).fillna(0)

    data["OI"] = pd.to_numeric(
        data.OI, errors="coerce"
    ).fillna(0)

    data["Liquidity"] = (
        np.log1p(data.Volume) + np.log1p(data.OI)
    )

    data["Score"] = (
        data.Liquidity
        - data.Distance / max(spot * 0.001, 1)
        - data.Spread.fillna(10) * 0.15
    )

    return data.sort_values(
        "Score", ascending=False
    ).iloc[0]

st.title("📈 NIFTY AI Trading Signal Platform")
st.caption("Upstox Live Option Chain | CE / PE | Risk Management")

with st.sidebar:
    st.header("Settings")

    token = st.text_input(
        "Upstox Access Token",
        type="password"
    )

    timeframe = st.selectbox(
        "Timeframe",
        [1, 3, 5, 15, 30],
        index=2
    )

    expiry = st.date_input(
        "Option Expiry Date",
        value=datetime.now().date() + timedelta(days=7)
    )

    capital = st.number_input(
        "Trading Capital ₹",
        min_value=1000,
        value=25000,
        step=1000
    )

    risk = st.slider(
        "Risk per trade %",
        0.25, 2.0, 0.5, 0.25
    )

    refresh = st.button("Refresh Live Data")

if not token:
    st.info("Live analysis ke liye Upstox Access Token enter karein.")

    st.markdown("""
    ### Upstox API kaise milegi?

    1. https://account.upstox.com/developer/apps kholo.
    2. Login karo.
    3. Create App par click karo.
    4. API application setup karo.
    5. Access Token generate karo.
    6. Token ko sidebar mein paste karo.

    API Secret ya Password kisi ko mat dena.
    """)

    st.stop()

try:
    with st.spinner("Upstox se live data aa raha hai..."):

        candles = get_intraday(
            token,
            NIFTY_KEY,
            timeframe
        )

        if len(candles) < 55:
            historical = get_historical(
                token,
                NIFTY_KEY,
                timeframe
            )

            candles = pd.concat(
                [historical, candles]
            ).drop_duplicates(
                "time"
            ).sort_values(
                "time"
            ).reset_index(drop=True)

        chain_data = api_get(
            "/v2/option/chain",
            token,
            {
                "instrument_key": NIFTY_KEY,
                "expiry_date": expiry.isoformat()
            }
        )

        chain = (
            chain_data
            if isinstance(chain_data, list)
            else chain_data.get("data", [])
        )

        options = make_option_table(chain)

except Exception as error:
    st.error("Upstox API connection error")
    st.code(str(error))
    st.stop()

if candles.empty or options.empty:
    st.error("Live candles ya option chain available nahi hai.")
    st.stop()

spot_values = [
    item.get("underlying_spot_price")
    for item in chain
    if item.get("underlying_spot_price")
]

spot = (
    float(spot_values[0])
    if spot_values
    else float(candles.close.iloc[-1])
)

signal, score, chart_data = get_signal(candles)

latest = chart_data.iloc[-1]

ce_oi = pd.to_numeric(
    options.loc[options.Side == "CE", "OI"],
    errors="coerce"
).sum()

pe_oi = pd.to_numeric(
    options.loc[options.Side == "PE", "OI"],
    errors="coerce"
).sum()

pcr = pe_oi / ce_oi if ce_oi else np.nan

bullish_oi = pcr >= 1.0 if np.isfinite(pcr) else False
bearish_oi = pcr <= 0.8 if np.isfinite(pcr) else False

if signal == "BUY CE" and not bullish_oi:
    signal = "NO TRADE"

if signal == "BUY PE" and not bearish_oi:
    signal = "NO TRADE"

st.subheader("Market Overview")

a, b, c, d = st.columns(4)

a.metric("NIFTY Spot", f"₹{spot:,.2f}")
b.metric("Trend Score", f"{score:+d}/5")
c.metric("RSI", f"{latest.RSI:.2f}")
d.metric(
    "PCR",
    f"{pcr:.2f}" if np.isfinite(pcr) else "N/A"
)

st.markdown("---")
st.subheader("AI Trading Decision")

st.markdown(f"# {signal}")

if signal == "BUY CE":
    st.success("Bullish setup detected. CE candidate.")

elif signal == "BUY PE":
    st.error("Bearish setup detected. PE candidate.")

else:
    st.warning("No qualified setup. Wait for confirmation.")

side = (
    "CE" if signal == "BUY CE"
    else "PE" if signal == "BUY PE"
    else None
)

if side:
    contract = select_contract(
        options,
        side,
        spot
    )

    if contract is not None:
        premium = float(contract.LTP)

        sl = premium * 0.80
        target1 = premium * 1.25
        target2 = premium * 1.50

        risk_amount = capital * risk / 100
        quantity = int(
            risk_amount / max(premium - sl, 0.01)
        )

        st.subheader("Selected Option")

        x1, x2, x3, x4 = st.columns(4)

        x1.metric(
            "Strike",
            f"{int(contract.Strike)} {side}"
        )

        x2.metric(
            "Premium",
            f"₹{premium:.2f}"
        )

        x3.metric(
            "Indicative SL",
            f"₹{sl:.2f}"
        )

        x4.metric(
            "Targets",
            f"₹{target1:.2f} / ₹{target2:.2f}"
        )

        st.write(
            f"Risk Budget: ₹{risk_amount:.2f}"
        )

        st.write(
            f"Theoretical quantity: {quantity} units. "
            "Actual lot-size rounding and margin must be checked."
        )

        if pd.notna(contract.Bid) and pd.notna(contract.Ask):
            spread = (
                (contract.Ask - contract.Bid)
                / premium * 100
            )

            if spread > 3:
                st.warning(
                    "Bid-Ask spread high hai. Contract avoid karne par vichar karein."
                )

st.subheader("Option Chain")

options["Distance"] = abs(
    options.Strike - spot
)

st.dataframe(
    options.sort_values("Distance").head(30),
    use_container_width=True,
    hide_index=True
)

st.subheader("Technical Chart")

st.line_chart(
    chart_data.set_index("time")[
        ["close", "EMA9", "EMA21", "EMA50"]
    ].tail(150)
)

st.subheader("Risk Warning")

st.warning("""
Ye decision-support system hai, guaranteed-profit AI nahi.

- Signal rule-based hai, trained ML model nahi.
- SL aur targets indicative percentage levels hain.
- Historical option premium backtesting abhi zaroori hai.
- Brokerage, slippage, IV aur theta ko dhyan mein rakhein.
- Pehle paper trading karein.
- App automatic order execute nahi karta.
- Access Token ko GitHub par kabhi upload na karein.
""")
