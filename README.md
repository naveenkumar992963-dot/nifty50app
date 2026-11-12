NIFTY 50 PRO MAX — Five Strategy Detailed Build
Included five strategies
Trend Following — EMA 9/21/50 + Supertrend across 3m/5m/15m.
VWAP + Volume — price acceptance above/below VWAP with relative volume.
Mean Reversion — Bollinger deviation + RSI; contextual vote only, not allowed to override hard higher-timeframe trend filters.
Donchian Breakout — breakout of prior 20 completed 5m candles.
Options OI + PCR — OI positioning context with more conservative PCR thresholds.
Also retained from the original uploaded app:
Upstox V3 REST candles/LTP and WebSocket streams.
1m/3m/5m/15m technical indicators, chart analysis, price action and ML research.
Option chain, Greeks, OI, PCR and max pain.
CE/PE contract ranking and manual execution only.
Important logic improvements
1m analysis now uses completed candles to reduce intrabar repainting.
15m is mandatory; no silent 5m fallback.
CE/PE decision requires five-strategy directional support plus price action and 5m/15m trend alignment.
PE threshold was made symmetric with CE and the excessive bearish evidence threshold was reduced from 10/24 to 8/24, while keeping trend and price-action gates.
Added detailed diagnostics explaining why a signal was blocked.
Option candidate selection applies a configurable bid/ask spread cap when valid quotes are available and prefers Ask as indicative buy entry.
Added configurable capital/risk planning; quantity estimate is pre-lot-rounding and excludes charges/slippage.
Run
Python 3.10+
pip install -r requirements.txt
Create .streamlit/secrets.toml using .streamlit/secrets.toml.example and your own Upstox token.
streamlit run app.py
Critical limitations
Python syntax validation is not live Upstox validation. This app has NOT been tested with the user's live account, entitlements, WebSocket subscription, option payloads or actual market.
AI confidence and strategy CALL/PUT setup scores are heuristic scores, NOT calibrated win probabilities.
The 5-strategy ensemble is a research heuristic; OI/PCR alone cannot predict direction.
Premium SL/targets are illustrative and are not dynamically calibrated to option Greeks or volatility.
Quantity estimate excludes current exchange lot-size rounding, brokerage, STT, GST, exchange charges, slippage, gaps and partial fills. Actual loss may exceed planned risk.
No auto order placement. Manual execution only.
No strategy guarantees correct signals or profit. Backtest out-of-sample and paper trade before risking money.
Option premium improvements (this build)
Candidate ranking uses Delta band, bid/ask spread, volume, OI and ATM distance.
Configurable minimum volume/OI, maximum spread, Delta range and premium stop percentage.
Buy entry uses Ask when a valid Ask exists; LTP fallback is explicitly warned as non-executable.
WebSocket option ticks older than 10 seconds are not allowed to overwrite the REST chain snapshot.
Option chain table shows CE/PE Bid, Ask and spread.
Premium SL/targets are percentage-based illustrative planning levels. No true premium ATR candle history is yet stored, so they are NOT described as volatility-adaptive.
