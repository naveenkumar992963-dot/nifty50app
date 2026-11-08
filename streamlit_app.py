# NIFTY 50 Live Options Scalping Signal Dashboard — Upstox WebSocket + Chart Analysis
# Research/education only. No broker order-placement endpoints are used.
import streamlit as st
import threading, json
import upstox_client
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="NIFTY 50 AI Desk", page_icon="📊", layout="wide")

@st.cache_resource
def get_upstox_stream_state(access_token, instrument_key):
    """One process-wide Upstox V3 WebSocket stream; callback stores the latest NIFTY tick."""
    state = {"ltp": None, "cp": None, "ltt": None, "connected": False,
             "error": None, "last_message": None}
    lock = threading.Lock()
    try:
        cfg = upstox_client.Configuration()
        cfg.access_token = access_token
        api_client = upstox_client.ApiClient(cfg)
        streamer = upstox_client.MarketDataStreamerV3(api_client, [instrument_key], "ltpc")

        def on_open():
            with lock:
                state["connected"] = True
                state["error"] = None

        def on_message(message):
            try:
                obj = message
                if isinstance(obj, (bytes, bytearray)):
                    obj = json.loads(bytes(obj).decode("utf-8"))
                elif isinstance(obj, str):
                    obj = json.loads(obj)
                feeds = obj.get("feeds", {}) if isinstance(obj, dict) else {}
                feed = feeds.get(instrument_key, {})
                ltpc = feed.get("ltpc") or feed.get("ff", {}).get("indexFF", {}).get("ltpc") or {}
                if ltpc.get("ltp") is not None:
                    with lock:
                        state["ltp"] = float(ltpc["ltp"])
                        state["cp"] = float(ltpc["cp"]) if ltpc.get("cp") is not None else state["cp"]
                        state["ltt"] = ltpc.get("ltt")
                        state["last_message"] = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
            except Exception as exc:
                with lock:
                    state["error"] = f"Tick decode: {exc}"

        def on_error(error):
            with lock:
                state["error"] = str(error)
                state["connected"] = False

        def on_close(*args):
            with lock:
                state["connected"] = False

        streamer.on("open", on_open)
        streamer.on("message", on_message)
        streamer.on("error", on_error)
        streamer.on("close", on_close)
        try:
            streamer.auto_reconnect(True, 2, 10)
        except Exception:
            pass
        state["streamer"] = streamer

        def run_stream():
            try:
                streamer.connect()
            except Exception as exc:
                with lock:
                    state["error"] = str(exc)
                    state["connected"] = False

        threading.Thread(target=run_stream, daemon=True, name="upstox-market-feed").start()
    except Exception as exc:
        state["error"] = str(exc)
    state["_lock"] = lock
    return state


@st.fragment(run_every="1s")
def live_dashboard():
    IST = ZoneInfo("Asia/Kolkata")
    BASE = "https://api.upstox.com"
    INDEX_DEFAULT = "NSE_INDEX|Nifty 50"

    st.title("🏛️ NIFTY 50 Live Options Scalping Desk")
    st.caption("Upstox tick-by-tick NIFTY • 1m/3m/5m chart analysis • CE/PE scalping signals • Signal-only")

    # -------------------- SESSION STATE --------------------
    defaults = {"last_signal_key": None, "signal_log": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # -------------------- SETTINGS --------------------
    with st.sidebar:
        st.header("Configuration")
        expiry_choice = st.selectbox("Option expiry", ["current_week", "next_week", "current_month", "next_month"])
        strike_range = st.slider("Option-chain strikes around ATM", 5, 20, 10)
        option_pick_mode = st.selectbox("Option contract selection", ["Best Auto Select", "Manual Strike"])
        preferred_delta = st.slider("Preferred option |Delta|", 0.30, 0.70, 0.50, 0.05)
        max_spread_pct = st.slider("Maximum bid-ask spread (%)", 1.0, 15.0, 5.0, 0.5)
        ml_threshold = st.slider("Minimum ML probability", 0.50, 0.80, 0.58, 0.01)
        signal_mode = st.selectbox("Signal display", ["Both: Moderate + Strong", "Strong only"], index=0)
        bt_cost_pct = st.number_input("Estimated round-trip cost (%) for index study", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        if st.button("Refresh market data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("No place-order API or broker order code exists in this app.")

    # -------------------- API HELPERS --------------------
    def get_secret(name, default=""):
        try:
            return str(st.secrets.get(name, default)).strip()
        except Exception:
            return default

    TOKEN = get_secret("UPSTOX_ACCESS_TOKEN")
    INDEX_KEY = get_secret("UPSTOX_INSTRUMENT_KEY", INDEX_DEFAULT)

    def headers():
        if not TOKEN:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN missing. Streamlit Cloud → Manage app → Settings → Secrets में token जोड़ें.")
        return {"Accept": "application/json", "Authorization": f"Bearer {TOKEN}"}

    def api_get(url, params=None, timeout=20):
        r = requests.get(url, headers=headers(), params=params, timeout=timeout)
        if r.status_code == 401:
            raise RuntimeError("Upstox token invalid/expired. नया Analytics Token बनाकर Secrets update करें.")
        if r.status_code == 403:
            raise RuntimeError("Upstox ने इस API के लिए permission नहीं दी. App/API entitlement जाँचें.")
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") == "error":
            raise RuntimeError(str(payload.get("errors", payload)))
        return payload.get("data", {})

    def candles_frame(candles):
        if not candles:
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        df = pd.DataFrame(candles)
        df = df.iloc[:, :6]
        df.columns = ["Timestamp","Open","High","Low","Close","Volume"]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")
        df = df.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Open","High","Low","Close"])

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_history(token, key, from_date, to_date):
        enc = quote(key, safe="")
        url = f"{BASE}/v3/historical-candle/{enc}/minutes/1/{to_date}/{from_date}"
        r = requests.get(url, headers={"Accept":"application/json","Authorization":f"Bearer {token}"}, timeout=25)
        if r.status_code == 401: raise RuntimeError("Upstox token invalid/expired.")
        r.raise_for_status()
        return candles_frame(r.json().get("data",{}).get("candles",[]))

    @st.cache_data(ttl=10, show_spinner=False)
    def fetch_intraday(token, key, cache_bucket):
        enc = quote(key, safe="")
        url = f"{BASE}/v3/historical-candle/intraday/{enc}/minutes/1"
        r = requests.get(url, headers={"Accept":"application/json","Authorization":f"Bearer {token}"}, timeout=20)
        if r.status_code == 401: raise RuntimeError("Upstox token invalid/expired.")
        r.raise_for_status()
        return candles_frame(r.json().get("data",{}).get("candles",[]))

    @st.cache_data(ttl=8, show_spinner=False)
    def fetch_ltp(token, key, bucket):
        enc = quote(key, safe="")
        url = f"{BASE}/v3/market-quote/ltp?instrument_key={enc}"
        r = requests.get(url, headers={"Accept":"application/json","Authorization":f"Bearer {token}"}, timeout=15)
        if r.status_code == 401: raise RuntimeError("Upstox token invalid/expired.")
        r.raise_for_status()
        data = r.json().get("data",{})
        q = data.get(key.replace("|",":"))
        if not q and len(data) == 1: q = next(iter(data.values()))
        if not q or q.get("last_price") is None:
            raise RuntimeError("NIFTY LTP response में last_price नहीं मिला.")
        return {"ltp":float(q["last_price"]), "cp":q.get("cp"), "volume":q.get("volume"), "ltq":q.get("ltq")}

    @st.cache_data(ttl=25, show_spinner=False)
    def fetch_option_chain(token, key, expiry, bucket):
        url = f"{BASE}/v2/option/chain"
        r = requests.get(url, headers={"Accept":"application/json","Authorization":f"Bearer {token}"},
                         params={"instrument_key":key,"expiry_date":expiry}, timeout=20)
        if r.status_code == 401: raise RuntimeError("Option chain: token invalid/expired.")
        r.raise_for_status()
        return r.json().get("data", [])

    # -------------------- INDICATORS --------------------
    def resample_ohlcv(raw, minutes):
        # Normalize provider payload columns before aggregation; volume can arrive as object/string.
        raw = raw.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        if "Volume" not in raw.columns:
            raw["Volume"] = 0.0
        raw["Volume"] = raw["Volume"].fillna(0.0).astype("float64")
        out = raw.resample(f"{minutes}min", origin="start_day", offset="15min").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(subset=["Open","High","Low","Close"])
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        if len(out) and out.index[-1] + pd.Timedelta(minutes=minutes) > now:
            out = out.iloc[:-1]  # completed candles only
        return out

    def indicators(data):
        d = data.copy()
        # Ensure OHLCV are numeric before rolling, VWAP and cumulative calculations.
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        if "Volume" not in d.columns:
            d["Volume"] = 0.0
        d["Volume"] = d["Volume"].fillna(0.0).astype("float64")
        d = d.dropna(subset=["Open", "High", "Low", "Close"])
        c,h,l = d.Close,d.High,d.Low
        d["EMA9"] = c.ewm(span=9,adjust=False).mean()
        d["EMA21"] = c.ewm(span=21,adjust=False).mean()
        d["EMA50"] = c.ewm(span=50,adjust=False).mean()
        delta = c.diff()
        ag = delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
        al = (-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        d["RSI"] = 100 - 100/(1+ag/al.replace(0,np.nan))
        d["RSI"] = d["RSI"].fillna(50)
        e12,e26 = c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean()
        d["MACD"] = e12-e26
        d["MACD_SIGNAL"] = d.MACD.ewm(span=9,adjust=False).mean()
        d["MACD_HIST"] = d.MACD-d.MACD_SIGNAL
        pc = c.shift(1)
        tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
        d["ATR"] = tr.ewm(alpha=1/14,adjust=False).mean()
        # ADX / directional movement
        up, down = h.diff(), -l.diff()
        plus_dm = pd.Series(np.where((up>down)&(up>0),up,0.0),index=d.index)
        minus_dm = pd.Series(np.where((down>up)&(down>0),down,0.0),index=d.index)
        atr_safe = d.ATR.replace(0,np.nan)
        pdi = 100*plus_dm.ewm(alpha=1/14,adjust=False).mean()/atr_safe
        mdi = 100*minus_dm.ewm(alpha=1/14,adjust=False).mean()/atr_safe
        dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
        d["ADX"] = dx.ewm(alpha=1/14,adjust=False).mean().fillna(0)
        # Supertrend
        mid=(h+l)/2; bu=mid+3*d.ATR; bl=mid-3*d.ATR
        fu,fl=bu.copy(),bl.copy()
        for i in range(1,len(d)):
            fu.iloc[i] = bu.iloc[i] if (bu.iloc[i]<fu.iloc[i-1] or c.iloc[i-1]>fu.iloc[i-1]) else fu.iloc[i-1]
            fl.iloc[i] = bl.iloc[i] if (bl.iloc[i]>fl.iloc[i-1] or c.iloc[i-1]<fl.iloc[i-1]) else fl.iloc[i-1]
        direction=np.ones(len(d)); stline=np.zeros(len(d))
        for i in range(len(d)):
            if i==0: stline[i]=fl.iloc[i]
            elif direction[i-1]>0:
                direction[i],stline[i]=( (-1,fu.iloc[i]) if c.iloc[i]<fl.iloc[i] else (1,fl.iloc[i]) )
            else:
                direction[i],stline[i]=( (1,fl.iloc[i]) if c.iloc[i]>fu.iloc[i] else (-1,fu.iloc[i]) )
        d["ST_DIR"]=direction; d["Supertrend"]=stline
        # Session VWAP (only meaningful if provider supplies volume)
        typical=((h.astype("float64")+l.astype("float64")+c.astype("float64"))/3.0).astype("float64")
        volume=d["Volume"].astype("float64").fillna(0.0)
        day_key=pd.Series(d.index.date,index=d.index)
        pv=(typical*volume).groupby(day_key).cumsum()
        vv=volume.groupby(day_key).cumsum()
        d["VWAP"]=pv/vv.replace(0,np.nan)
        d["VOL_AVG"]=d.Volume.rolling(20,min_periods=20).mean()
        d["RET1"]=c.pct_change()
        d["RET3"]=c.pct_change(3)
        d["RET6"]=c.pct_change(6)
        d["EMA_SPREAD"]=(d.EMA9-d.EMA21)/c
        d["EMA_TREND"]=(d.EMA21-d.EMA50)/c
        d["ATR_PCT"]=d.ATR/c
        d["VWAP_DIST"]=(c-d.VWAP)/c
        d["VOL_RATIO"]=d.Volume/d.VOL_AVG.replace(0,np.nan)
        return d.replace([np.inf,-np.inf],np.nan)

    FEATURES=["RSI","MACD_HIST","ADX","RET1","RET3","RET6","EMA_SPREAD","EMA_TREND","ATR_PCT","VWAP_DIST","VOL_RATIO"]

    def tf_summary(d):
        if len(d)<55: return None
        x=d.iloc[-1]
        # Separate directional bias from a fully aligned trend. This catches early
        # bearish pressure before EMA9/21/50 have all crossed into perfect order.
        checks = [
            1 if x.Close > x.EMA21 else -1,
            1 if x.EMA9 > x.EMA21 else -1,
            1 if x.EMA21 > x.EMA50 else -1,
            1 if pd.notna(x.VWAP) and x.Close > x.VWAP else -1 if pd.notna(x.VWAP) else 0,
            1 if x.MACD > x.MACD_SIGNAL else -1,
            1 if x.RSI >= 52 else -1 if x.RSI <= 48 else 0,
            1 if x.ST_DIR > 0 else -1,
            1 if x.Close > x.Open else -1 if x.Close < x.Open else 0
        ]
        bias_score = float(sum(checks) / max(1, sum(v != 0 for v in checks)))
        bull=bool(bias_score >= 0.25 and x.Close > x.EMA21)
        bear=bool(bias_score <= -0.25 and x.Close < x.EMA21)
        trend_bull=bool(x.EMA9>x.EMA21>x.EMA50 and x.Close>x.EMA21 and x.ST_DIR>0)
        trend_bear=bool(x.EMA9<x.EMA21<x.EMA50 and x.Close<x.EMA21 and x.ST_DIR<0)
        return {"close":float(x.Close),"rsi":float(x.RSI),"atr":float(x.ATR),"adx":float(x.ADX),
                "ema9":float(x.EMA9),"ema21":float(x.EMA21),"ema50":float(x.EMA50),
                "macd_bull":bool(x.MACD>x.MACD_SIGNAL),"macd_bear":bool(x.MACD<x.MACD_SIGNAL),
                "bull":bull,"bear":bear,"trend_bull":trend_bull,"trend_bear":trend_bear,
                "bias_score":bias_score,
                "bias":"BULLISH" if bias_score >= 0.25 else "BEARISH" if bias_score <= -0.25 else "NEUTRAL",
                "vwap":float(x.VWAP) if pd.notna(x.VWAP) else np.nan,
                "vol_ratio":float(x.VOL_RATIO) if pd.notna(x.VOL_RATIO) else np.nan,"time":str(d.index[-1])}
        # -------------------- ML RESEARCH --------------------
    def ml_dataset(d):
        x=d[FEATURES].replace([np.inf,-np.inf],np.nan).copy()
        # Predict 3 completed 5-minute bars ahead; ignore tiny moves inside a volatility-scaled dead zone.
        future=d.Close.shift(-3)
        threshold=(d.ATR*0.12).clip(lower=0.5)
        move=future-d.Close
        y=pd.Series(np.where(move>threshold,1,np.where(move < -threshold,0,np.nan)),index=d.index)
        z=x.join(y.rename("target")).dropna()
        return z[FEATURES],z["target"].astype(int)

    def train_ml(d):
        X,y=ml_dataset(d)
        if len(X)<180 or y.nunique()<2:
            return None, {"status":"Insufficient labelled data","samples":len(X)}
        split=max(int(len(X)*0.80),100)
        if len(X)-split<25: split=len(X)-25
        # Purge the final 3 training rows: their forward labels overlap the test period.
        train_end = max(0, split - 3)
        Xtr,Xte,ytr,yte=X.iloc[:train_end],X.iloc[split:],y.iloc[:train_end],y.iloc[split:]
        if len(Xtr) < 100 or ytr.nunique()<2 or yte.nunique()<2:
            return None, {"status":"Not enough class diversity in time split","samples":len(X)}
        model=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.06,max_leaf_nodes=15,
                                             l2_regularization=1.0,random_state=42)
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        metrics={"status":"Temporal holdout evaluated","samples":len(X),"train":len(Xtr),"test":len(Xte),
                 "accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),
                 "recall":recall_score(yte,pred,zero_division=0)}
        # Refit on all labelled rows for current estimate after recording holdout metrics.
        final=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.06,max_leaf_nodes=15,
                                             l2_regularization=1.0,random_state=42)
        final.fit(X,y)
        last=d[FEATURES].replace([np.inf,-np.inf],np.nan).dropna().tail(1)
        if last.empty: return None,metrics
        p=float(final.predict_proba(last[FEATURES])[0][1])
        metrics["p_up"]=p
        return final,metrics

    # -------------------- SIGNAL ENGINE --------------------
    def decide_signal(s3,s5,s15,ml,chain_stats,stale):
        reasons=[]
        if stale: return "NO TRADE",0.0,["Market feed stale: new signal blocked."]
        bulls=sum([s3["bull"],s5["bull"],s15["bull"]])
        bears=sum([s3["bear"],s5["bear"],s15["bear"]])
        if bulls==3: direction="BUY"
        elif bears==3: direction="SELL"
        else: return "NO TRADE",0.0,[f"Trend alignment incomplete (bull {bulls}/3, bear {bears}/3)."]
        rsi_bull=sum(x["rsi"]>50 for x in [s3,s5,s15])
        rsi_bear=sum(x["rsi"]<50 for x in [s3,s5,s15])
        macd_bull=sum(x["macd_bull"] for x in [s3,s5,s15])
        macd_bear=sum(x["macd_bear"] for x in [s3,s5,s15])
        if direction=="BUY" and (rsi_bull<2 or macd_bull<2):
            return "NO TRADE",0.0,["RSI/MACD confirmation insufficient."]
        if direction=="SELL" and (rsi_bear<2 or macd_bear<2):
            return "NO TRADE",0.0,["RSI/MACD confirmation insufficient."]
        if s3["adx"]<15:
            return "NO TRADE",0.0,["ADX below 15: weak trend / possible range."]
        if direction=="BUY" and s3["rsi"]>75:
            return "NO TRADE",0.0,["3m RSI overheated."]
        if direction=="SELL" and s3["rsi"]<25:
            return "NO TRADE",0.0,["3m RSI oversold; avoid chasing."]
        if ml and "p_up" in ml:
            p=ml["p_up"]
            if direction=="BUY" and p<0.5: return "NO TRADE",p,["ML direction conflicts with bullish trend."]
            if direction=="SELL" and p>0.5: return "NO TRADE",1-p,["ML direction conflicts with bearish trend."]
            prob=p if direction=="BUY" else 1-p
            if prob<0.53: return "NO TRADE",prob,["ML directional probability too weak."]
        else:
            prob=0.0
            reasons.append("ML estimate unavailable; rule-based signal only.")
        if chain_stats:
            if direction=="BUY" and chain_stats["pcr"]<0.65:
                return "NO TRADE",prob,["Option-chain PCR indicates weak put-side support."]
            if direction=="SELL" and chain_stats["pcr"]>1.6:
                return "NO TRADE",prob,["Option-chain PCR is elevated; bearish setup needs caution."]
        reasons.extend([f"3m/5m/15m {direction.lower()} trend aligned.","RSI, MACD and ADX filters passed."])
        return direction,prob,reasons

    # -------------------- OPTION CHAIN NORMALIZER --------------------
    def normalize_chain(rows):
        out=[]
        for row in rows or []:
            try:
                c=row.get("call_options") or {}; p=row.get("put_options") or {}
                cm=c.get("market_data") or {}; pm=p.get("market_data") or {}
                cg=c.get("option_greeks") or {}; pg=p.get("option_greeks") or {}
                out.append({
                    "Expiry":row.get("expiry"),"Strike":float(row.get("strike_price")),
                    "Spot":float(row.get("underlying_spot_price") or np.nan),
                    "CE LTP":float(cm.get("ltp") or 0),"CE OI":float(cm.get("oi") or 0),
                    "CE Prev OI":float(cm.get("prev_oi") or 0),"CE Volume":float(cm.get("volume") or 0),
                    "CE Bid":float(cm.get("bid_price") or 0),"CE Ask":float(cm.get("ask_price") or 0),
                    "CE IV":float(cg.get("iv") or 0),"CE Delta":float(cg.get("delta") or 0),
                    "CE Gamma":float(cg.get("gamma") or 0),"CE Theta":float(cg.get("theta") or 0),
                    "CE Vega":float(cg.get("vega") or 0),
                    "CE Key":c.get("instrument_key"),
                    "PE LTP":float(pm.get("ltp") or 0),"PE OI":float(pm.get("oi") or 0),
                    "PE Prev OI":float(pm.get("prev_oi") or 0),"PE Volume":float(pm.get("volume") or 0),
                    "PE Bid":float(pm.get("bid_price") or 0),"PE Ask":float(pm.get("ask_price") or 0),
                    "PE IV":float(pg.get("iv") or 0),"PE Delta":float(pg.get("delta") or 0),
                    "PE Gamma":float(pg.get("gamma") or 0),"PE Theta":float(pg.get("theta") or 0),
                    "PE Vega":float(pg.get("vega") or 0),
                    "PE Key":p.get("instrument_key")
                })
            except (TypeError,ValueError):
                continue
        return pd.DataFrame(out).sort_values("Strike") if out else pd.DataFrame()

    def option_stats(chain):
        if chain.empty: return None
        ce_oi=chain["CE OI"].sum(); pe_oi=chain["PE OI"].sum()
        pcr=pe_oi/ce_oi if ce_oi else np.nan
        max_ce=chain.loc[chain["CE OI"].idxmax(),"Strike"] if ce_oi else np.nan
        max_pe=chain.loc[chain["PE OI"].idxmax(),"Strike"] if pe_oi else np.nan
        return {"pcr":float(pcr),"ce_oi":ce_oi,"pe_oi":pe_oi,"max_ce":max_ce,"max_pe":max_pe}

    def max_pain(chain):
        if chain.empty: return None
        strikes=chain.Strike.to_numpy()
        ceoi=chain["CE OI"].to_numpy(); peoi=chain["PE OI"].to_numpy()
        payouts=[]
        for settle in strikes:
            call_loss=np.maximum(settle-strikes,0)*ceoi
            put_loss=np.maximum(strikes-settle,0)*peoi
            payouts.append(call_loss.sum()+put_loss.sum())
        return float(strikes[int(np.argmin(payouts))])

    # -------------------- FIVE-STRATEGY ENSEMBLE --------------------
    def clamp(v, lo=-1.0, hi=1.0):
        return float(max(lo, min(hi, v)))

    def combined_strategy_scores(d1, d3, d5, chain_stats, ml_metrics):
        """Five strategy votes with regime-aware filters; scores are heuristic, not win probabilities."""
        if min(len(d1), len(d3), len(d5)) < 25:
            names = ["Trend Following", "VWAP + Volume", "Mean Reversion", "Donchian Breakout", "Options OI + PCR"]
            weights = dict(zip(names, [0.30, 0.20, 0.10, 0.20, 0.20]))
            scores = {k: 0.0 for k in names}
            return scores, weights, 0.0, 0.0

        last5 = d5.iloc[-1]
        # 1) Trend following: require directional EMA structure and Supertrend on each TF.
        trend_votes = []
        for frame in (d1, d3, d5):
            x = frame.iloc[-1]
            if x.EMA9 > x.EMA21 > x.EMA50 and x.ST_DIR > 0:
                trend_votes.append(1.0)
            elif x.EMA9 < x.EMA21 < x.EMA50 and x.ST_DIR < 0:
                trend_votes.append(-1.0)
            else:
                trend_votes.append(0.0)
        trend = float(np.mean(trend_votes))

        # 2) VWAP + volume: use completed-bar volume and candle direction; no volume = neutral.
        vwap_score = 0.0
        vr = last5.get("VOL_RATIO", np.nan)
        if pd.notna(last5.get("VWAP")) and np.isfinite(vr):
            side = 1 if last5.Close > last5.VWAP else -1 if last5.Close < last5.VWAP else 0
            candle_side = 1 if last5.Close > last5.Open else -1 if last5.Close < last5.Open else 0
            if side and vr >= 1.35 and candle_side == side:
                vwap_score = float(side * min(1.0, 0.65 + (vr - 1.35) * 0.25))
            elif side and vr >= 0.9:
                vwap_score = 0.25 * side

        # 3) Mean reversion: only active in a weak/ranging regime and with reversal evidence.
        mean_score = 0.0
        mid = d5.Close.rolling(20, min_periods=20).mean().iloc[-1]
        sd = d5.Close.rolling(20, min_periods=20).std().iloc[-1]
        adx = float(last5.ADX) if pd.notna(last5.ADX) else 0.0
        if pd.notna(mid) and pd.notna(sd) and sd > 0 and adx < 22:
            z = (float(last5.Close) - float(mid)) / float(sd)
            prev5 = d5.iloc[-2]
            bullish_reversal = last5.Close > last5.Open and last5.Close > prev5.Close
            bearish_reversal = last5.Close < last5.Open and last5.Close < prev5.Close
            if z <= -1.6 and last5.RSI < 38 and bullish_reversal:
                mean_score = 0.75
            elif z >= 1.6 and last5.RSI > 62 and bearish_reversal:
                mean_score = -0.75
            elif z <= -1.25 and last5.RSI < 32:
                mean_score = 0.35
            elif z >= 1.25 and last5.RSI > 68:
                mean_score = -0.35

        # 4) Donchian: breakout must close beyond prior range with ATR buffer and volume.
        prior = d5.iloc[:-1].tail(20)
        donchian = 0.0
        if len(prior) >= 15:
            rh, rl = float(prior.High.max()), float(prior.Low.min())
            atr = float(last5.ATR) if pd.notna(last5.ATR) else 0.0
            buffer = max(atr * 0.05, 0.5)
            vol_ok = np.isfinite(vr) and vr >= 1.15
            if last5.Close > rh + buffer and vol_ok and last5.Close > last5.Open:
                donchian = 1.0
            elif last5.Close < rl - buffer and vol_ok and last5.Close < last5.Open:
                donchian = -1.0
            elif last5.High > rh and last5.Close <= rh:
                donchian = -0.20  # failed upside break
            elif last5.Low < rl and last5.Close >= rl:
                donchian = 0.20   # failed downside break

        # 5) Options positioning: PCR is contextual, not a standalone direction signal.
        oi_score = 0.0
        if chain_stats and np.isfinite(chain_stats.get("pcr", np.nan)):
            pcr = float(chain_stats["pcr"])
            if 0.0 < pcr < 3.0:
                oi_score = float(np.clip((pcr - 1.0) / 0.65, -0.65, 0.65))

        weights = {"Trend Following": 0.30, "VWAP + Volume": 0.20, "Mean Reversion": 0.10,
                   "Donchian Breakout": 0.20, "Options OI + PCR": 0.20}
        scores = {"Trend Following": trend, "VWAP + Volume": vwap_score,
                  "Mean Reversion": mean_score, "Donchian Breakout": donchian,
                  "Options OI + PCR": oi_score}
        weighted = sum(scores[k] * weights[k] for k in scores)
        ml_adj = 0.0
        if ml_metrics and "p_up" in ml_metrics and np.isfinite(ml_metrics["p_up"]):
            ml_adj = float(np.clip((float(ml_metrics["p_up"]) - 0.5) * 2, -1, 1)) * 0.08
        final_score = clamp(weighted + ml_adj)
        agreement = sum(weights[k] for k, v in scores.items() if (v > 0.15 and final_score > 0) or (v < -0.15 and final_score < 0))
        confidence = clamp(abs(final_score) * 0.75 + agreement * 0.25, 0, 1) * 100
        return scores, weights, final_score, confidence

    # -------------------- SCALPING PRICE-ACTION SCORE --------------------
    def support_resistance(frame):
        """Confirmed swing-pivot zones; right-side confirmation avoids using unconfirmed pivots."""
        empty = {"support": np.nan, "resistance": np.nan, "score": 0.0, "state": "Levels unavailable",
                 "support_touches": 0, "resistance_touches": 0}
        if frame is None or len(frame) < 35:
            return empty
        data = frame.iloc[:-1].tail(180).copy()  # use completed candles only
        if len(data) < 25:
            return empty
        atrv = data.ATR.iloc[-1] if "ATR" in data else np.nan
        if not np.isfinite(atrv) or atrv <= 0:
            atrv = max(float(data.Close.iloc[-1]) * 0.0005, 1.0)
        tol = max(float(atrv) * 0.22, 2.5)
        highs, lows = data.High.to_numpy(float), data.Low.to_numpy(float)
        pivots = []
        left = right = 3
        for i in range(left, len(data) - right):
            if highs[i] >= np.max(highs[i-left:i+right+1]):
                pivots.append((float(highs[i]), "R"))
            if lows[i] <= np.min(lows[i-left:i+right+1]):
                pivots.append((float(lows[i]), "S"))
        close = float(data.Close.iloc[-1])
        def cluster(kind, relation):
            vals = sorted([v for v, k in pivots if k == kind and relation(v)])
            groups = []
            for value in vals:
                if not groups or abs(value - np.mean(groups[-1])) > tol:
                    groups.append([value])
                else:
                    groups[-1].append(value)
            ranked = []
            for group in groups:
                level = float(np.mean(group))
                touches = sum(abs(float(v) - level) <= tol for v, k in pivots if k == kind)
                # Favor repeated, recent reactions; require at least one confirmed pivot.
                ranked.append((level, touches))
            return ranked
        supports = cluster("S", lambda v: v < close)
        resistances = cluster("R", lambda v: v > close)
        support, st = max(supports, key=lambda z: z[0]) if supports else (np.nan, 0)
        resistance, rt = min(resistances, key=lambda z: z[0]) if resistances else (np.nan, 0)
        x = data.iloc[-1]
        buffer = max(float(atrv) * 0.08, 1.0)
        vr = x.get("VOL_RATIO", np.nan)
        vol_ok = pd.notna(vr) and np.isfinite(vr) and float(vr) >= 1.15
        score, state = 0.0, "Between zones"
        if np.isfinite(resistance) and close > resistance + buffer and vol_ok and x.Close > x.Open:
            score, state = 0.30, "Confirmed resistance breakout"
        elif np.isfinite(support) and close < support - buffer and vol_ok and x.Close < x.Open:
            score, state = -0.30, "Confirmed support breakdown"
        elif np.isfinite(resistance) and abs(close - resistance) <= tol and x.Close < x.Open:
            score, state = -0.18, "Resistance rejection / caution"
        elif np.isfinite(support) and abs(close - support) <= tol and x.Close > x.Open:
            score, state = 0.18, "Support rejection / bounce"
        return {"support": support, "resistance": resistance, "score": score, "state": state,
                "support_touches": int(st), "resistance_touches": int(rt)}

    def price_action_score(frame):
        """Directional chart score from structure, candle, VWAP, Supertrend and volume."""
        if frame is None or len(frame) < 25:
            return 0.0, ["Not enough candles"], 0
        x, prev = frame.iloc[-1], frame.iloc[-2]
        score, notes, confirmations = 0.0, [], 0
        # Trend / market structure proxy
        if x.EMA9 > x.EMA21 and x.Close > x.EMA21:
            score += 0.18; confirmations += 1; notes.append("Short-term bullish structure")
        elif x.EMA9 < x.EMA21 and x.Close < x.EMA21:
            score -= 0.18; confirmations += 1; notes.append("Short-term bearish structure")
        # Supertrend
        if x.ST_DIR > 0:
            score += 0.14; confirmations += 1; notes.append("Supertrend bullish")
        elif x.ST_DIR < 0:
            score -= 0.14; confirmations += 1; notes.append("Supertrend bearish")
        # VWAP
        if pd.notna(x.VWAP):
            if x.Close > x.VWAP:
                score += 0.12; confirmations += 1; notes.append("Price above VWAP")
            elif x.Close < x.VWAP:
                score -= 0.12; confirmations += 1; notes.append("Price below VWAP")
        # Volume spike
        vr = x.VOL_RATIO if pd.notna(x.VOL_RATIO) else np.nan
        if np.isfinite(vr) and vr >= 1.5:
            if x.Close > x.Open:
                score += 0.14; confirmations += 1; notes.append(f"Bullish volume spike ({vr:.1f}x)")
            elif x.Close < x.Open:
                score -= 0.14; confirmations += 1; notes.append(f"Bearish volume spike ({vr:.1f}x)")
        # Engulfing / rejection candles
        body = abs(x.Close-x.Open)
        rng = max(x.High-x.Low, 1e-9)
        lower = min(x.Open,x.Close)-x.Low
        upper = x.High-max(x.Open,x.Close)
        if x.Close > x.Open and prev.Close < prev.Open and x.Close >= prev.Open and x.Open <= prev.Close:
            score += 0.12; confirmations += 1; notes.append("Bullish engulfing")
        elif x.Close < x.Open and prev.Close > prev.Open and x.Open >= prev.Close and x.Close <= prev.Open:
            score -= 0.12; confirmations += 1; notes.append("Bearish engulfing")
        elif lower > max(body*1.8, rng*0.45) and upper < rng*0.2:
            score += 0.08; confirmations += 1; notes.append("Bullish rejection wick")
        elif upper > max(body*1.8, rng*0.45) and lower < rng*0.2:
            score -= 0.08; confirmations += 1; notes.append("Bearish rejection wick")
        # Breakout / fake-breakout check against prior 20 bars (current excluded)
        prior = frame.iloc[-21:-1]
        if len(prior) >= 15:
            rh, rl = prior.High.max(), prior.Low.min()
            if x.Close > rh:
                score += 0.20; confirmations += 1; notes.append("Donchian resistance breakout")
            elif x.High > rh and x.Close <= rh:
                score -= 0.04; notes.append("Possible failed upside breakout")
            elif x.Close < rl:
                score -= 0.20; confirmations += 1; notes.append("Donchian support breakdown")
            elif x.Low < rl and x.Close >= rl:
                score += 0.04; notes.append("Possible failed downside breakdown")
        # Momentum filter: avoid chasing extremes, but don't demand perfect alignment
        if x.RSI >= 52 and x.MACD > x.MACD_SIGNAL:
            score += 0.08; confirmations += 1; notes.append("Bullish RSI/MACD momentum")
        elif x.RSI <= 48 and x.MACD < x.MACD_SIGNAL:
            score -= 0.08; confirmations += 1; notes.append("Bearish RSI/MACD momentum")
        sr = support_resistance(frame)
        if sr["score"]:
            score += sr["score"]
            confirmations += 1
            notes.append(f"S/R: {sr['state']}")
        return clamp(score), notes, confirmations

    def bearish_evidence(frame):
        """Count independent bearish conditions without requiring every timeframe to be perfectly aligned."""
        if frame is None or len(frame) < 25:
            return 0, []
        x = frame.iloc[-1]
        prior = frame.iloc[:-1].tail(20)
        checks = []
        checks.append(("Close below EMA21", x.Close < x.EMA21))
        checks.append(("EMA9 below EMA21", x.EMA9 < x.EMA21))
        checks.append(("EMA21 below EMA50", x.EMA21 < x.EMA50))
        checks.append(("Price below VWAP", pd.notna(x.VWAP) and x.Close < x.VWAP))
        checks.append(("Bearish MACD", x.MACD < x.MACD_SIGNAL))
        checks.append(("RSI below 48", x.RSI < 48))
        checks.append(("Support breakdown", len(prior) >= 15 and x.Close < prior.Low.min()))
        checks.append(("Bearish volume expansion", pd.notna(x.VOL_RATIO) and x.VOL_RATIO >= 1.2 and x.Close < x.Open))
        notes = [name for name, passed in checks if passed]
        return len(notes), notes


    # -------------------- FETCH & PROCESS --------------------
    try:
        now=datetime.now(IST)
        from_date=(now.date()-timedelta(days=25)).isoformat()
        to_date=(now.date()-timedelta(days=1)).isoformat()
        hist=fetch_history(TOKEN,INDEX_KEY,from_date,to_date)
        today=fetch_intraday(TOKEN,INDEX_KEY,int(now.timestamp()//10))
        raw=pd.concat([hist,today])
        if raw.empty: st.error("Upstox candle data empty."); st.stop()
        raw=raw[~raw.index.duplicated(keep="last")].sort_index()
        quote_data=fetch_ltp(TOKEN,INDEX_KEY,int(now.timestamp()//8))
        stream_state=get_upstox_stream_state(TOKEN,INDEX_KEY)
        with stream_state["_lock"]:
            stream_ltp=stream_state.get("ltp")
            stream_cp=stream_state.get("cp")
            stream_ltt=stream_state.get("ltt")
            stream_connected=stream_state.get("connected")
            stream_error=stream_state.get("error")
        # Prefer tick-by-tick WebSocket LTP; REST quote is fallback only.
        if stream_ltp is not None:
            quote_data["ltp"]=stream_ltp
            if stream_cp is not None:
                quote_data["cp"]=stream_cp
        d1=indicators(raw.copy())
        d3=indicators(resample_ohlcv(raw,3))
        d5=indicators(resample_ohlcv(raw,5))
        d15=indicators(resample_ohlcv(raw,15))
        if min(len(d1),len(d3),len(d5))<55: st.error("Scalping indicators के लिए पर्याप्त completed 1m/3m/5m candles नहीं हैं."); st.stop()
        s1,s3,s5=map(tf_summary,[d1,d3,d5])
        s15=tf_summary(d15) if len(d15)>=55 else s5
        latest=raw.index.max()
        now_ts=pd.Timestamp.now(tz="Asia/Kolkata")
        market_open=now_ts.weekday()<5 and (now_ts.hour,now_ts.minute)>=(9,15) and (now_ts.hour,now_ts.minute)<(15,30)
        age=(now_ts-latest).total_seconds()
        stale=bool(market_open and age>180)
        # Option chain
        chain_rows=fetch_option_chain(TOKEN,INDEX_KEY,expiry_choice,int(now.timestamp()//25))
        chain_all=normalize_chain(chain_rows)
        stats=option_stats(chain_all)
        spot=float(quote_data["ltp"])
        if not chain_all.empty:
            atm=float(chain_all.iloc[(chain_all.Strike-spot).abs().argsort()[:1]]["Strike"].iloc[0])
            chain=chain_all[(chain_all.Strike>=atm-strike_range*50)&(chain_all.Strike<=atm+strike_range*50)].copy()
        else:
            chain=pd.DataFrame()
        # WebSocket is the required live signal source. REST LTP is display-only fallback.
        ws_tick_age = np.inf
        if stream_state.get("last_message"):
            try:
                last_tick_ts = pd.Timestamp(stream_state["last_message"])
                if last_tick_ts.tzinfo is None:
                    last_tick_ts = last_tick_ts.tz_localize("Asia/Kolkata")
                ws_tick_age = max(0.0, (now_ts - last_tick_ts).total_seconds())
            except Exception:
                ws_tick_age = np.inf
        ws_live_ok = bool(stream_connected and stream_ltp is not None and ws_tick_age <= 20)
        feed_block = bool(market_open and (stale or not ws_live_ok))
        market_block = not market_open

        # ML on completed 5m features.
        ml_model,ml_metrics=train_ml(d5)
        strategy_scores,strategy_weights,combined_score,ai_confidence=combined_strategy_scores(
            d1,d3,d5,stats,ml_metrics if ml_model is not None else None)

        # Price-action evidence from all requested timeframes.
        pa1, pa_notes1, pa_conf1 = price_action_score(d1)
        pa3, pa_notes3, pa_conf3 = price_action_score(d3)
        pa5, pa_notes5, pa_conf5 = price_action_score(d5)
        chart_score = 0.50*pa1 + 0.30*pa3 + 0.20*pa5
        final_direction_score = clamp(0.60*chart_score + 0.40*combined_score)
        bullish_votes = sum(v > 0.08 for v in [pa1, pa3, pa5])
        bearish_votes = sum(v < -0.08 for v in [pa1, pa3, pa5])
        total_confirmations = pa_conf1 + pa_conf3 + pa_conf5
        bear_ev1, bear_notes1 = bearish_evidence(d1)
        bear_ev3, bear_notes3 = bearish_evidence(d3)
        bear_ev5, bear_notes5 = bearish_evidence(d5)
        bearish_evidence_total = bear_ev1 + bear_ev3 + bear_ev5
        bull_ev1 = sum([
            d1.iloc[-1].Close > d1.iloc[-1].EMA21,
            d1.iloc[-1].EMA9 > d1.iloc[-1].EMA21,
            d1.iloc[-1].EMA21 > d1.iloc[-1].EMA50,
            pd.notna(d1.iloc[-1].VWAP) and d1.iloc[-1].Close > d1.iloc[-1].VWAP,
            d1.iloc[-1].MACD > d1.iloc[-1].MACD_SIGNAL,
            d1.iloc[-1].RSI > 52,
            d1.iloc[-1].ST_DIR > 0,
            pd.notna(d1.iloc[-1].VOL_RATIO) and d1.iloc[-1].VOL_RATIO >= 1.2 and d1.iloc[-1].Close > d1.iloc[-1].Open
        ])
        bull_ev3 = sum([
            d3.iloc[-1].Close > d3.iloc[-1].EMA21,
            d3.iloc[-1].EMA9 > d3.iloc[-1].EMA21,
            d3.iloc[-1].EMA21 > d3.iloc[-1].EMA50,
            pd.notna(d3.iloc[-1].VWAP) and d3.iloc[-1].Close > d3.iloc[-1].VWAP,
            d3.iloc[-1].MACD > d3.iloc[-1].MACD_SIGNAL,
            d3.iloc[-1].RSI > 52,
            d3.iloc[-1].ST_DIR > 0,
            pd.notna(d3.iloc[-1].VOL_RATIO) and d3.iloc[-1].VOL_RATIO >= 1.2 and d3.iloc[-1].Close > d3.iloc[-1].Open
        ])
        bull_ev5 = sum([
            d5.iloc[-1].Close > d5.iloc[-1].EMA21,
            d5.iloc[-1].EMA9 > d5.iloc[-1].EMA21,
            d5.iloc[-1].EMA21 > d5.iloc[-1].EMA50,
            pd.notna(d5.iloc[-1].VWAP) and d5.iloc[-1].Close > d5.iloc[-1].VWAP,
            d5.iloc[-1].MACD > d5.iloc[-1].MACD_SIGNAL,
            d5.iloc[-1].RSI > 52,
            d5.iloc[-1].ST_DIR > 0,
            pd.notna(d5.iloc[-1].VOL_RATIO) and d5.iloc[-1].VOL_RATIO >= 1.2 and d5.iloc[-1].Close > d5.iloc[-1].Open
        ])
        bullish_evidence_total = int(bull_ev1 + bull_ev3 + bull_ev5)

        # Direction lock: 5m + 3m establish direction; 1m is the entry trigger.
        # A 1m counter-move cannot reverse the higher-timeframe direction.
        # Higher-timeframe bias is based on multiple independent clues, not
        # a requirement that all three EMAs have already formed perfect alignment.
        higher_tf_bull = bool(s5["bias_score"] >= 0.25 and s3["bias_score"] >= 0.15)
        higher_tf_bear = bool(s5["bias_score"] <= -0.25 and s3["bias_score"] <= -0.15)
        if higher_tf_bear:
            market_bias = "BEARISH"
            market_bias_reason = "5m + 3m bearish pressure confirmed"
        elif higher_tf_bull:
            market_bias = "BULLISH"
            market_bias_reason = "5m + 3m bullish pressure confirmed"
        elif s5["bias_score"] <= -0.15 or s3["bias_score"] <= -0.15:
            market_bias = "BEARISH LEAN"
            market_bias_reason = "Bearish pressure present, but higher-timeframe confirmation incomplete"
        elif s5["bias_score"] >= 0.15 or s3["bias_score"] >= 0.15:
            market_bias = "BULLISH LEAN"
            market_bias_reason = "Bullish pressure present, but higher-timeframe confirmation incomplete"
        else:
            market_bias = "SIDEWAYS / MIXED"
            market_bias_reason = "No clear higher-timeframe directional edge"
        signal = "NO TRADE"
        signal_strength = "NONE"
        reasons = []
        if market_block:
            reasons = ["Market is outside regular NSE session; signals disabled."]
        elif feed_block:
            reasons = ["LIVE FEED BLOCK: WebSocket disconnected/stale or candle data stale. Automatic NO TRADE."]
        elif higher_tf_bull and pa1 >= 0.05 and pa3 > -0.05 and pa5 > -0.05 and final_direction_score >= 0.10 and s1["rsi"] < 82:
            # Moderate is allowed when 5m+3m agree and 1m gives a non-conflicting trigger.
            strong = bool(pa1 >= 0.18 and pa3 >= 0.12 and pa5 >= 0.12 and
                          final_direction_score >= 0.22 and bullish_votes == 3 and
                          bullish_evidence_total >= 15 and min(s3["adx"],s5["adx"]) >= 18)
            signal = "BUY CE"
            signal_strength = "STRONG" if strong else "MODERATE"
            reasons = ["5m + 3m bullish direction lock passed; 1m entry trigger is bullish."]
        elif higher_tf_bear and pa1 <= -0.05 and pa3 < 0.05 and pa5 < 0.05 and final_direction_score <= -0.10 and s1["rsi"] > 18:
            strong = bool(pa1 <= -0.18 and pa3 <= -0.12 and pa5 <= -0.12 and
                          final_direction_score <= -0.22 and bearish_votes == 3 and
                          bearish_evidence_total >= 15 and min(s3["adx"],s5["adx"]) >= 18)
            signal = "BUY PE"
            signal_strength = "STRONG" if strong else "MODERATE"
            reasons = ["5m + 3m bearish direction lock passed; 1m entry trigger is bearish."]
        else:
            reasons = [
                f"Direction lock not satisfied: 5m/3m bull={higher_tf_bull}, bear={higher_tf_bear}; "
                f"1m PA={pa1:+.2f}, 3m PA={pa3:+.2f}, 5m PA={pa5:+.2f}, combined={final_direction_score:+.2f}."
            ]

        # ML is a conservative veto only for a strong disagreement; it cannot create a trade.
        if ml_model is not None and signal != "NO TRADE":
            p_up = float(ml_metrics.get("p_up", 0.5))
            if signal == "BUY CE" and p_up < 0.38:
                signal = "NO TRADE"; signal_strength = "NONE"; reasons.append("ML strongly conflicts with bullish setup.")
            elif signal == "BUY PE" and p_up > 0.72:
                signal = "NO TRADE"; signal_strength = "NONE"; reasons.append("ML strongly conflicts with bearish setup.")

        if signal != "NO TRADE" and signal_mode == "Strong only" and signal_strength != "STRONG":
            signal = "NO TRADE"; signal_strength = "NONE"
            reasons.append("Moderate setup hidden because Strong only mode is selected.")

        # Confidence is a heuristic confluence score, never a calibrated win probability.
        directional_agreement = max(bullish_votes, bearish_votes) / 3
        ai_confidence = min(95.0, max(0.0, abs(final_direction_score)*65 +
                                      directional_agreement*20 + min(total_confirmations, 12)*1.0))
        strategy_scores["Chart Price Action"] = chart_score
        strategy_weights["Chart Price Action"] = 0.0
        combined_score = final_direction_score
        reasons.extend((pa_notes1 + pa_notes3 + pa_notes5)[:8])

        # Keep a compact signal audit trail, one record per latest completed 1m candle.
        if "signal_log" not in st.session_state:
            st.session_state["signal_log"] = []
        signal_key = f"{latest}|{signal}|{signal_strength}"
        if signal_key != st.session_state.get("last_signal_key"):
            st.session_state["signal_log"].append({
                "Time": str(latest), "Signal": signal, "Strength": signal_strength,
                "Score": round(float(final_direction_score), 3),
                "Feed": "LIVE" if ws_live_ok else "BLOCKED"
            })
            st.session_state["signal_log"] = st.session_state["signal_log"][-200:]
            st.session_state["last_signal_key"] = signal_key
    except Exception as e:
        st.error(f"Data / API error: {e}")
        st.info("Secrets में UPSTOX_ACCESS_TOKEN और UPSTOX_INSTRUMENT_KEY = NSE_INDEX|Nifty 50 जाँचें. Option chain API access भी आवश्यक है.")
        st.stop()
        # -------------------- OVERVIEW --------------------
    st.divider()
    st.subheader("Market Snapshot")
    c=st.columns(5)
    c[0].metric("NIFTY 50 LTP",f"{spot:,.2f}",None if quote_data.get("cp") is None else f"Prev close {float(quote_data['cp']):,.2f}")
    c[1].metric("1m RSI",f"{s1['rsi']:.1f}")
    c[2].metric("3m RSI",f"{s3['rsi']:.1f}")
    c[3].metric("5m RSI",f"{s5['rsi']:.1f}")
    c[4].metric("3m ADX",f"{s3['adx']:.1f}")
    st.caption(f"Last completed 1m candle: {latest} | Feed age: {age:.0f}s | Option expiry: {chain_all['Expiry'].iloc[0] if not chain_all.empty else expiry_choice}")
    if ws_live_ok:
        st.success(f"🟢 UPSTOX WEBSOCKET LIVE • Tick LTP: ₹{stream_ltp:,.2f} • Tick age: {ws_tick_age:.1f}s")
    else:
        st.error(f"🔴 LIVE WEBSOCKET NOT VERIFIED • Connected={stream_connected} • Tick age={ws_tick_age:.1f}s • {stream_error or 'No fresh tick'}")
    if stale:
        st.error("STALE CANDLE DATA: latest completed candle is older than 3 minutes.")
    if market_open and not ws_live_ok:
        st.error("AUTOMATIC NO TRADE: live WebSocket tick missing, disconnected or older than 20 seconds. REST quote is display-only.")
    elif not market_open:
        st.info("Market closed/outside regular NSE session: automatic NO TRADE.")
    elif not stale:
        st.success("Candle freshness check passed.")

    # -------------------- MULTI-TIMEFRAME --------------------
    st.subheader("Multi-Timeframe Regime")
    tf_rows=[]
    for name,s in [("1 min",s1),("3 min",s3),("5 min",s5)]:
        tf_rows.append({"Timeframe":name,"Directional bias":s["bias"],
                        "Full EMA trend":"Bullish" if s["trend_bull"] else "Bearish" if s["trend_bear"] else "Not aligned",
                        "Bias score":round(s["bias_score"],2),
                        "Close":round(s["close"],2),"RSI":round(s["rsi"],1),"ADX":round(s["adx"],1),
                        "MACD":"Bullish" if s["macd_bull"] else "Bearish","VWAP":round(s["vwap"],2) if np.isfinite(s["vwap"]) else None})
    st.dataframe(pd.DataFrame(tf_rows),use_container_width=True,hide_index=True)
    st.subheader("Market Direction Bias — separate from entry signal")
    mb1,mb2,mb3=st.columns(3)
    mb1.metric("Overall market bias", market_bias)
    mb2.metric("3m directional score", f"{s3['bias_score']:+.2f}")
    mb3.metric("5m directional score", f"{s5['bias_score']:+.2f}")
    if market_bias == "BEARISH":
        st.error(f"Downward bias: {market_bias_reason}. PE entry still waits for 1m trigger and feed confirmation.")
    elif market_bias == "BEARISH LEAN":
        st.warning(f"Possible downward pressure: {market_bias_reason}. No forced PE entry.")
    elif market_bias == "BULLISH":
        st.success(f"Upward bias: {market_bias_reason}. CE entry still waits for 1m trigger and feed confirmation.")
    elif market_bias == "BULLISH LEAN":
        st.info(f"Possible upward pressure: {market_bias_reason}. No forced CE entry.")
    else:
        st.info(market_bias_reason + ". Prefer NO TRADE until direction improves.")

    # -------------------- ML --------------------
    st.divider()
    st.subheader("Machine-Learning Research Model")
    if ml_model is not None:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Temporal holdout accuracy",f"{ml_metrics['accuracy']*100:.1f}%")
        m2.metric("Precision (Up class)",f"{ml_metrics['precision']*100:.1f}%")
        m3.metric("Recall (Up class)",f"{ml_metrics['recall']*100:.1f}%")
        m4.metric("Labelled samples",f"{ml_metrics['samples']:,}")
        p_up=ml_metrics["p_up"]
        st.write(f"Model direction estimate: **{p_up*100:.1f}% Up / {(1-p_up)*100:.1f}% Down**")
        st.caption("This is a model score, not a calibrated probability or guarantee. Holdout metrics are historical and can change by regime.")
    else:
        st.warning(f"ML unavailable: {ml_metrics.get('status')} (samples: {ml_metrics.get('samples',0)}). Rule-based filters remain active.")

    # -------------------- OPTION CHAIN --------------------
    st.divider()
    st.subheader("Live Option Chain & OI Dashboard")
    if chain.empty:
        st.warning("Option chain data unavailable for selected expiry.")
    else:
        pcr=stats["pcr"]
        a,b,c,d=st.columns(4)
        a.metric("PCR (OI)",f"{pcr:.2f}" if np.isfinite(pcr) else "N/A")
        b.metric("Total CE OI",f"{stats['ce_oi']:,.0f}")
        c.metric("Total PE OI",f"{stats['pe_oi']:,.0f}")
        d.metric("Max Pain",f"{max_pain(chain_all):,.0f}" if max_pain(chain_all) else "N/A")
        e,f=st.columns(2)
        e.metric("Highest CE OI strike",f"{stats['max_ce']:,.0f}")
        f.metric("Highest PE OI strike",f"{stats['max_pe']:,.0f}")
        view_cols=["Strike","CE LTP","CE OI","CE Prev OI","CE Volume","CE IV","CE Delta","CE Gamma","CE Theta","CE Vega","PE LTP","PE OI","PE Prev OI","PE Volume","PE IV","PE Delta","PE Gamma","PE Theta","PE Vega"]
        view=chain[view_cols].copy()
        view["CE OI Chg"]=view["CE OI"]-view["CE Prev OI"]
        view["PE OI Chg"]=view["PE OI"]-view["PE Prev OI"]
        st.dataframe(view[["Strike","CE LTP","CE OI","CE OI Chg","CE Volume","CE IV","CE Delta","CE Gamma","CE Theta","CE Vega",
                           "PE LTP","PE OI","PE OI Chg","PE Volume","PE IV","PE Delta","PE Gamma","PE Theta","PE Vega"]],
                     use_container_width=True,hide_index=True)
        st.caption("OI/PCR are context, not standalone buy/sell signals. API values may be delayed or stale outside market hours.")

    # -------------------- FINAL SIGNAL --------------------
    st.divider()
    st.subheader("FINAL SCALPING SIGNAL — OPTIONS ONLY")
    if signal=="BUY CE":
        st.success(f"🟢 FINAL SIGNAL: BUY CE — {signal_strength}")
    elif signal=="BUY PE":
        st.error(f"🔴 FINAL SIGNAL: BUY PE — {signal_strength}")
    else:
        st.warning("🟡 FINAL SIGNAL: NO TRADE")
    st.caption(f"Market bias: {market_bias} ({market_bias_reason}). Entry lock: 5m+3m establish bias; 1m confirms entry. WebSocket live required during market hours. Signal tier: {signal_strength}.")
    f1,f2,f3=st.columns(3)
    f1.metric("AI confidence score",f"{ai_confidence:.1f}%",help="Heuristic agreement score, not calibrated win probability.")
    f2.metric("Combined strategy score",f"{combined_score:+.3f}",help="Weighted directional score from -1 (bearish) to +1 (bullish).")
    f3.metric("Option selection","CE" if signal=="BUY CE" else "PE" if signal=="BUY PE" else "NONE")
    st.caption("One consolidated signal is shown. Five strategy votes plus chart price-action score are diagnostic components.")
    score_rows=[]
    for name,val in strategy_scores.items():
        score_rows.append({"Strategy":name,"Weight":f"{strategy_weights[name]*100:.0f}%","Directional score":round(val,2),
                           "Contribution":round(val*strategy_weights[name],3)})
    st.dataframe(pd.DataFrame(score_rows),use_container_width=True,hide_index=True)
    for r in reasons: st.write("• "+r)
    st.caption(f"Bullish checks: 1m {bull_ev1}/8 · 3m {bull_ev3}/8 · 5m {bull_ev5}/8 · Total {bullish_evidence_total}/24 | Bearish checks: 1m {bear_ev1}/8 · 3m {bear_ev3}/8 · 5m {bear_ev5}/8 · Total {bearish_evidence_total}/24")
    with st.expander("Bullish/Bearish signal diagnostics"):
        st.write("**Bullish checks:** " + f"1m {bull_ev1}/8, 3m {bull_ev3}/8, 5m {bull_ev5}/8")
        st.write("**Bearish 1m:** " + (", ".join(bear_notes1) if bear_notes1 else "No bearish checks passed"))
        st.write("**Bearish 3m:** " + (", ".join(bear_notes3) if bear_notes3 else "No bearish checks passed"))
        st.write("**Bearish 5m:** " + (", ".join(bear_notes5) if bear_notes5 else "No bearish checks passed"))
    with st.expander("Recent signal history"):
        if st.session_state.get("signal_log"):
            st.dataframe(pd.DataFrame(st.session_state["signal_log"]).tail(50), use_container_width=True, hide_index=True)
        else:
            st.info("No signal records yet.")
    with st.expander("Legacy bearish diagnostics"):
        st.write("**1m:** " + (", ".join(bear_notes1) if bear_notes1 else "No bearish checks passed"))
        st.write("**3m:** " + (", ".join(bear_notes3) if bear_notes3 else "No bearish checks passed"))
        st.write("**5m:** " + (", ".join(bear_notes5) if bear_notes5 else "No bearish checks passed"))
    if ml_model is not None:
        st.caption(f"ML model estimate: {ml_metrics['p_up']*100:.1f}% Up / {(1-ml_metrics['p_up'])*100:.1f}% Down. Holdout accuracy: {ml_metrics['accuracy']*100:.1f}%.")

    # -------------------- BEST OPTION CONTRACT SELECTOR --------------------
    st.subheader("Best Option Auto Selector — Greeks + Liquidity")
    candidate = None
    side_name = "CE" if signal == "BUY CE" else "PE" if signal == "BUY PE" else None

    if side_name and not chain.empty:
        key_col = f"{side_name} Key"
        ltp_col = f"{side_name} LTP"
        vol_col = f"{side_name} Volume"
        oi_col = f"{side_name} OI"
        delta_col = f"{side_name} Delta"
        gamma_col = f"{side_name} Gamma"
        theta_col = f"{side_name} Theta"
        vega_col = f"{side_name} Vega"
        iv_col = f"{side_name} IV"
        bid_col = f"{side_name} Bid"
        ask_col = f"{side_name} Ask"

        candidates = chain[
            (pd.to_numeric(chain[ltp_col], errors="coerce") > 0) &
            chain[key_col].notna()
        ].copy()

        if not candidates.empty:
            candidates["atm_dist"] = (pd.to_numeric(candidates["Strike"], errors="coerce") - float(spot)).abs()
            for col in [delta_col, gamma_col, theta_col, vega_col, iv_col, vol_col, oi_col, bid_col, ask_col, ltp_col]:
                candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

            candidates["delta_abs"] = candidates[delta_col].abs().fillna(0)
            candidates["gamma_val"] = candidates[gamma_col].clip(lower=0).fillna(0)
            candidates["theta_drag"] = candidates[theta_col].abs()
            candidates["iv_val"] = candidates[iv_col].fillna(0)
            candidates["volume_val"] = candidates[vol_col].fillna(0)
            candidates["oi_val"] = candidates[oi_col].fillna(0)

            valid_quote = (
                candidates[bid_col].gt(0) &
                candidates[ask_col].ge(candidates[bid_col]) &
                candidates[ltp_col].gt(0)
            )
            candidates["spread_pct"] = np.where(
                valid_quote,
                (candidates[ask_col] - candidates[bid_col]) /
                candidates[ltp_col].replace(0, np.nan) * 100,
                np.nan
            )

            def normalized_rank(series, higher_is_better=True):
                values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
                if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
                    return pd.Series(0.5, index=series.index)
                return values.rank(pct=True, ascending=not higher_is_better).fillna(0.0)

            # Delta near the chosen target; penalize excessive theta and IV.
            candidates["delta_score"] = (
                1 - (candidates["delta_abs"] - float(preferred_delta)).abs() / 0.50
            ).clip(0, 1)
            candidates["gamma_score"] = normalized_rank(candidates["gamma_val"], True)
            candidates["theta_score"] = normalized_rank(candidates["theta_drag"], False)
            candidates["volume_score"] = normalized_rank(candidates["volume_val"], True)
            candidates["oi_score"] = normalized_rank(candidates["oi_val"], True)
            candidates["spread_score"] = normalized_rank(candidates["spread_pct"], False)
            candidates["iv_score"] = normalized_rank(candidates["iv_val"], False)
            candidates["atm_score"] = normalized_rank(candidates["atm_dist"], False)

            # Weighted relative score: directional response, liquidity, and execution quality.
            candidates["selection_score"] = (
                0.25 * candidates["delta_score"] +
                0.10 * candidates["gamma_score"] +
                0.15 * candidates["theta_score"] +
                0.15 * candidates["volume_score"] +
                0.10 * candidates["oi_score"] +
                0.15 * candidates["spread_score"] +
                0.05 * candidates["iv_score"] +
                0.05 * candidates["atm_score"]
            )

            # Avoid contracts with extreme spreads when usable quotes exist.
            quoted = candidates["spread_pct"].notna()
            if quoted.any():
                tradable = candidates[~quoted | (candidates["spread_pct"] <= float(max_spread_pct))].copy()
                if not tradable.empty:
                    candidates = tradable

            has_greeks = (
                candidates[delta_col].fillna(0).abs().max() > 0 or
                candidates[gamma_col].fillna(0).abs().max() > 0 or
                candidates[theta_col].fillna(0).abs().max() > 0
            )

            if option_pick_mode == "Manual Strike":
                strike_options = sorted(candidates["Strike"].dropna().unique().tolist())
                selected_strike = st.selectbox(
                    f"Choose {side_name} strike",
                    strike_options,
                    index=min(range(len(strike_options)), key=lambda i: abs(strike_options[i] - float(spot)))
                )
                candidate = candidates[candidates["Strike"] == selected_strike].sort_values(
                    ["selection_score", vol_col], ascending=[False, False]
                ).iloc[0].to_dict()
                selection_method = "Manual strike; ranked contract details shown below"
            else:
                candidates = candidates.sort_values(
                    ["selection_score", vol_col], ascending=[False, False]
                )
                candidate = candidates.iloc[0].to_dict()
                selection_method = "Auto-ranked by Delta, Gamma, Theta, IV, OI, volume, spread and ATM distance"

            option_entry = float(candidate[ltp_col])
            option_sl = max(0.05, option_entry * 0.80)
            option_risk = option_entry - option_sl
            option_t1 = option_entry + option_risk
            option_t2 = option_entry + 2 * option_risk
            option_rr = (option_t1 - option_entry) / option_risk if option_risk > 0 else None

            st.success(f"Best available contract: BUY {side_name} {candidate['Strike']:.0f}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Contract", f"{side_name} {candidate['Strike']:.0f}")
            c2.metric("Premium (LTP)", f"₹{option_entry:.2f}")
            c3.metric("Illustrative SL", f"₹{option_sl:.2f}")
            c4.metric("Target 1", f"₹{option_t1:.2f}")
            c5.metric("Target 2", f"₹{option_t2:.2f}")

            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Delta", f"{candidate.get(delta_col, 0):.3f}")
            g2.metric("Gamma", f"{candidate.get(gamma_col, 0):.4f}")
            g3.metric("Theta", f"{candidate.get(theta_col, 0):.3f}")
            g4.metric("IV", f"{candidate.get(iv_col, 0):.2f}")
            g5.metric("Selection score", f"{candidate.get('selection_score', 0) * 100:.1f}/100")

            l1, l2, l3 = st.columns(3)
            l1.metric("Volume", f"{candidate.get(vol_col, 0):,.0f}")
            l2.metric("Open Interest", f"{candidate.get(oi_col, 0):,.0f}")
            spread_value = candidate.get("spread_pct", np.nan)
            l3.metric("Bid-ask spread", f"{spread_value:.2f}%" if pd.notna(spread_value) else "Quote unavailable")

            st.caption(f"Selection mode: {selection_method}. Greek data present: {'Yes' if has_greeks else 'No — ranking relies more on liquidity/quotes/ATM distance'}.")
            st.caption("Score is a relative ranking, not a win probability. Greeks and option-chain values are snapshots. The displayed 20% stop and targets are illustrative only; verify live bid/ask, lot size, slippage and risk before any order.")
            if not has_greeks:
                st.warning("Upstox snapshot में usable Greeks नहीं मिले. Auto-selection जारी है, लेकिन Greek-based ranking विश्वसनीय नहीं मानी जाएगी.")
        else:
            st.warning(f"{side_name} signal मिला, लेकिन selected expiry में usable contract/instrument key नहीं मिला.")
    elif side_name:
        st.warning("Option chain unavailable: contract selection नहीं हो सकी; entry/SL/targets withheld.")
    else:
        st.info("NO TRADE: CE/PE signal नहीं है, इसलिए कोई option contract recommend नहीं किया गया.")
        # -------------------- CHART ANALYSIS --------------------
    st.divider()
    st.subheader("Scalping Chart Analysis — 1m / 3m / 5m")
    def chart_levels(frame):
        levels = support_resistance(frame)
        return levels["resistance"], levels["support"], levels
    for tf_name, frame in [("1 Minute",d1),("3 Minute",d3),("5 Minute",d5)]:
        if len(frame)<10: continue
        last=frame.iloc[-1]
        resistance,support,sr_info=chart_levels(frame)
        prev=frame.iloc[-2]
        body=abs(last.Close-last.Open); rng=max(last.High-last.Low,1e-9)
        lower_wick=min(last.Open,last.Close)-last.Low
        upper_wick=last.High-max(last.Open,last.Close)
        pattern="No clear pattern"
        if last.Close>last.Open and prev.Close<prev.Open and last.Close>=prev.Open and last.Open<=prev.Close:
            pattern="Bullish Engulfing"
        elif last.Close<last.Open and prev.Close>prev.Open and last.Open>=prev.Close and last.Close<=prev.Open:
            pattern="Bearish Engulfing"
        elif lower_wick>body*2 and upper_wick<body:
            pattern="Hammer-like candle"
        elif upper_wick>body*2 and lower_wick<body:
            pattern="Shooting-star-like candle"
        trend="Bullish" if last.EMA9>last.EMA21>last.EMA50 else "Bearish" if last.EMA9<last.EMA21<last.EMA50 else "Mixed"
        vwap_state="Above VWAP" if pd.notna(last.VWAP) and last.Close>last.VWAP else "Below VWAP" if pd.notna(last.VWAP) else "VWAP unavailable"
        vol_ratio=last.VOL_RATIO if pd.notna(last.VOL_RATIO) else np.nan
        support_txt=f"{support:.2f}" if np.isfinite(support) else "N/A"
        resistance_txt=f"{resistance:.2f}" if np.isfinite(resistance) else "N/A"
        st.write(f"**{tf_name}:** Trend **{trend}** · {vwap_state} · RSI **{last.RSI:.1f}** · ADX **{last.ADX:.1f}** · Candle **{pattern}** · Support zone **{support_txt}** · Resistance zone **{resistance_txt}** · {sr_info['state']}" + (f" · Volume ratio **{vol_ratio:.2f}x**" if np.isfinite(vol_ratio) else ""))

    # -------------------- SIGNAL-ONLY MODE --------------------
    st.info("Signal-only dashboard: Paper Trading, virtual balance, trade history and automatic broker orders are disabled.")

    # -------------------- BACKTEST (COST-ADJUSTED UNDERLYING DIRECTIONAL PROXY) --------------------
    st.divider()
    st.subheader("Historical Signal Study — 5-minute directional proxy")
    st.caption("Signals use completed 5m bars; entry is next bar open and exit is the close 3 bars later. Trades are non-overlapping. This tests NIFTY direction, NOT option-premium P&L.")
    bt = d5.copy()
    bt["next_open"] = bt.Open.shift(-1)
    bt["exit_close"] = bt.Close.shift(-3)
    bt["fwd_ret"] = bt["exit_close"] / bt["next_open"] - 1
    bt["bull_setup"] = ((bt.EMA9 > bt.EMA21) & (bt.EMA21 > bt.EMA50) &
                        (bt.Close > bt.VWAP) & (bt.RSI.between(52, 72)) &
                        (bt.ADX >= 18) & (bt.ST_DIR > 0) & (bt.MACD > bt.MACD_SIGNAL))
    bt["bear_setup"] = ((bt.EMA9 < bt.EMA21) & (bt.EMA21 < bt.EMA50) &
                        (bt.Close < bt.VWAP) & (bt.RSI.between(28, 48)) &
                        (bt.ADX >= 18) & (bt.ST_DIR < 0) & (bt.MACD < bt.MACD_SIGNAL))
    bt = bt.dropna(subset=["fwd_ret", "next_open", "exit_close"])
    cost = float(bt_cost_pct) / 100.0
    trades = []
    next_allowed = 0
    for pos, (ts, row) in enumerate(bt.iterrows()):
        if pos < next_allowed:
            continue
        side = 1 if bool(row.bull_setup) else -1 if bool(row.bear_setup) else 0
        if side == 0:
            continue
        gross = float(row.fwd_ret) * side
        net = gross - cost
        trades.append({"Entry candle": str(ts), "Side": "UP" if side > 0 else "DOWN",
                       "Gross return %": gross * 100, "Net return %": net * 100})
        next_allowed = pos + 3
    if len(trades) >= 5:
        bt_trades = pd.DataFrame(trades)
        rets = bt_trades["Net return %"] / 100.0
        equity = (1 + rets).cumprod()
        drawdown = equity / equity.cummax() - 1
        wins = int((rets > 0).sum())
        gross_profit = float(rets[rets > 0].sum())
        gross_loss = abs(float(rets[rets < 0].sum()))
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
        b1,b2,b3,b4,b5 = st.columns(5)
        b1.metric("Trades", len(bt_trades))
        b2.metric("Win rate", f"{wins/len(bt_trades)*100:.1f}%")
        b3.metric("Average net/trade", f"{rets.mean()*100:.3f}%")
        b4.metric("Profit factor", f"{pf:.2f}" if np.isfinite(pf) else "∞")
        b5.metric("Max drawdown", f"{drawdown.min()*100:.2f}%")
        st.dataframe(bt_trades.tail(30), use_container_width=True, hide_index=True)
        st.caption(f"Estimated round-trip cost assumed: {bt_cost_pct:.2f}% of underlying notional per trade. Results are historical, sample-dependent, and not an options strategy backtest.")
    else:
        st.info(f"Historical signal sample insufficient for meaningful metrics ({len(trades)} trades; need at least 5).")
    # -------------------- CHARTS --------------------
    st.divider()
    st.subheader("NIFTY Scalping Charts — 1m / 3m / 5m")
    tab1,tab3,tab5=st.tabs(["1 Minute","3 Minute","5 Minute"])
    for tab,frame in [(tab1,d1),(tab3,d3),(tab5,d5)]:
        with tab:
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=frame.index,open=frame.Open,high=frame.High,low=frame.Low,close=frame.Close,name="NIFTY"))
            for col in ["EMA9","EMA21","EMA50","Supertrend","VWAP"]:
                if col in frame and frame[col].notna().any():
                    fig.add_trace(go.Scatter(x=frame.index,y=frame[col],name=col))
            sr_plot = support_resistance(frame)
            if np.isfinite(sr_plot["support"]):
                fig.add_hline(y=sr_plot["support"], line_dash="dot", annotation_text="Support zone")
            if np.isfinite(sr_plot["resistance"]):
                fig.add_hline(y=sr_plot["resistance"], line_dash="dot", annotation_text="Resistance zone")
            fig.update_layout(height=500,template="plotly_dark",xaxis_rangeslider_visible=False)
            st.plotly_chart(fig,use_container_width=True)
            rfig=go.Figure()
            rfig.add_trace(go.Scatter(x=frame.index,y=frame.RSI,name="RSI"))
            rfig.add_hline(y=70,line_dash="dash"); rfig.add_hline(y=30,line_dash="dash"); rfig.add_hline(y=50,line_dash="dot")
            rfig.update_layout(height=220,template="plotly_dark")
            st.plotly_chart(rfig,use_container_width=True)

    # -------------------- STATUS --------------------
    st.divider()
    st.subheader("System Health & Limitations")
    h1,h2,h3=st.columns(3)
    h1.metric("1m candles",len(raw))
    h2.metric("5m bars",len(d5))
    h3.metric("Option strikes loaded",len(chain_all))
    st.write(f"- Data source: Upstox REST V3 candles + V3 LTP + V2 option chain")
    st.write(f"- Last candle timestamp: {latest}")
    st.write(f"- Data stale guard: {'TRIGGERED' if stale else 'Not triggered'}")
    st.write(f"- WebSocket live guard: {'PASS' if ws_live_ok else 'BLOCKED'} (tick age {ws_tick_age:.1f}s; max 20s)")
    st.write(f"- Automatic trade block: {'ACTIVE' if (feed_block or market_block) else 'Not active'}")
    st.write("- Five strategy ensemble uses heuristic weights; the displayed confidence is not calibrated probability. ML is a recent-data research model, not a production-grade institutional model. News, order book and exchange tick replay are not included.")
    st.write("- Option chain is a snapshot, not a tick-by-tick feed. Option-chain values are snapshots; premium estimates exclude spread impact, slippage and lot-size constraints.")
    st.error("Educational signal research only. No guaranteed prediction or accuracy. No paper trading and no broker orders are executed.")


live_dashboard()
