
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.upstox_client import UpstoxClient, UpstoxError
from src.strategy import build_signal, choose_option
from src.backtest import run_backtest

st.set_page_config(
    page_title="NIFTY 50 AI Trading Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NIFTY 50 AI Trading Intelligence")
st.caption("Upstox Live Data | Options Buying | CE / PE Signals")

with st.sidebar:
    st.header("Connection Settings")

    token = st.text_input(
        "Upstox Access Token",
        type="password"
    )

    timeframe = st.selectbox(
        "Timeframe",
        [1, 3, 5, 15],
        index=2
    )

    expiry = st.date_input(
        "Option Expiry",
        value=date.today() + timedelta(days=7)
    )

    capital = st.number_input(
        "Trading Capital ₹",
        min_value=1000,
        value=25000,
        step=1000
    )

    risk = st.slider(
        "Risk per Trade %",
        0.25,
        2.0,
        0.5,
        0.25
    )

if not token:
    st.info("Upstox Access Token sidebar mein enter karein.")

    st.markdown("""
    ### Upstox API Setup

    1. https://account.upstox.com/developer/apps kholo.
    2. Login karo.
    3. Developer App create karo.
    4. Official process se Access Token generate karo.
    5. Token sidebar mein paste karo.

    Apna API Secret ya Access Token kisi ke saath share mat karna.
    """)
    st.stop()

client = UpstoxClient(token)

try:
    with st.spinner("Live NIFTY data load ho raha hai..."):

        candles = client.intraday_candles(
            "NSE_INDEX|Nifty 50",
            timeframe
        )

        if len(candles) < 60:
            historical = client.historical_candles(
                "NSE_INDEX|Nifty 50",
                timeframe,
                90
            )

            if not historical.empty:
                candles = pd.concat(
                    [historical, candles]
                ).drop_duplicates(
                    "timestamp"
                ).sort_values(
                    "timestamp"
                ).reset_index(drop=True)

        chain = client.option_chain(
            "NSE_INDEX|Nifty 50",
            expiry.isoformat()
        )

except UpstoxError as error:
    st.error("Upstox API Error")
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error("Data loading error")
    st.code(str(error))
    st.stop()

if candles.empty:
    st.error("NIFTY candle data available nahi hai.")
    st.stop()

if not chain:
    st.error("Option Chain data available nahi hai.")
    st.stop()

spot_prices = [
    item.get("underlying_spot_price")
    for item in chain
    if item.get("underlying_spot_price") is not None
]

spot = (
    float(spot_prices[0])
    if spot_prices
    else float(candles.iloc[-1]["close"])
)

result = build_signal(candles, chain)

signal = result["signal"]

option = None

if signal in ["BUY CE", "BUY PE"]:
    option = choose_option(
        chain,
        signal,
        spot
    )

st.divider()

st.subheader("MARKET OVERVIEW")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "NIFTY 50",
    f"₹{spot:,.2f}"
)

c2.metric(
    "Market Regime",
    result["regime"]
)

c3.metric(
    "AI Setup Score",
    f'{result["score"]}/100'
)

c4.metric(
    "RSI",
    str(result["indicators"].get("RSI"))
)

st.divider()

st.subheader("FINAL TRADING SIGNAL")

if signal == "BUY CE":
    st.success("🟢 BUY CE")

elif signal == "BUY PE":
    st.error("🔴 BUY PE")

else:
    st.warning("🟡 NO TRADE")

st.write(result["summary"])

st.subheader("AI Confirmation")

for reason in result["reasons"]:
    if reason["passed"]:
        st.write("✅", reason["text"])
    else:
        st.write("▫️", reason["text"])

st.divider()

st.subheader("TECHNICAL INDICATORS")

st.json(result["indicators"])

if option:

    premium = float(option["ltp"])

    stop_loss = premium * 0.80
    target1 = premium * 1.30
    target2 = premium * 1.60

    risk_amount = capital * risk / 100

    quantity = int(
        risk_amount / max(premium - stop_loss, 0.01)
    )

    st.subheader("SELECTED OPTION")

    a, b, c, d = st.columns(4)

    a.metric(
        "Strike",
        f'{option["strike"]} {option["side"]}'
    )

    b.metric(
        "Premium",
        f"₹{premium:.2f}"
    )

    c.metric(
        "Indicative SL",
        f"₹{stop_loss:.2f}"
    )

    d.metric(
        "Target 1",
        f"₹{target1:.2f}"
    )

    st.metric(
        "Target 2",
        f"₹{target2:.2f}"
    )

    st.write(
        f"Risk Budget: ₹{risk_amount:.2f}"
    )

    st.write(
        f"Indicative Quantity: {quantity} units"
    )

    st.write(
        "Lot size aur actual margin Upstox mein verify karein."
    )

    if option.get("spread_pct") is not None:
        if option["spread_pct"] > 2.5:
            st.warning(
                "Bid-Ask spread high hai. Trade avoid karne par vichar karein."
            )

else:
    st.info(
        "Abhi qualified trade nahi hai. NO TRADE ka matlab wait karna hai."
    )

st.divider()

st.subheader("LIVE OPTION CHAIN")

option_table = pd.DataFrame(
    result["option_rows"]
)

if not option_table.empty:

    option_table["distance"] = abs(
        option_table["strike"] - spot
    )

    st.dataframe(
        option_table.sort_values(
            "distance"
        ).head(30),
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.subheader("NIFTY PRICE CHART")

chart = result["chart"]

st.line_chart(
    chart.set_index("timestamp")[
        ["close", "ema9", "ema21", "ema50"]
    ].tail(250)
)

st.divider()

st.subheader("BACKTEST DIAGNOSTICS")

backtest = run_backtest(candles)

if backtest:

    x, y, z = st.columns(3)

    x.metric(
        "Directional Hit Rate",
        f'{backtest["hit_rate"]:.1f}%'
    )

    y.metric(
        "Tested Signals",
        backtest["trades"]
    )

    z.metric(
        "Max Drawdown",
        f'{backtest["max_drawdown_points"]:.2f}'
    )

else:
    st.info(
        "Backtest ke liye aur historical candles chahiye."
    )

st.warning("""
IMPORTANT RISK NOTICE:

- Ye rule-based decision-support engine hai, trained ML model nahi.
- Setup score profit ki probability nahi hai.
- Option SL aur targets indicative estimates hain.
- Historical options premium backtesting abhi zaroori hai.
- Brokerage, slippage, IV aur Theta ka impact hota hai.
- Pehle paper trading karein.
- App automatic orders execute nahi karta.
""")

st.caption(
    "Data Source: Upstox API | NIFTY 50 Only"
)
