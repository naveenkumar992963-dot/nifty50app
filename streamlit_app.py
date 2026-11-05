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
        ml_threshold = st.slider("Minimum ML probability", 0.50, 0.80, 0.58, 0.01)
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
        out = raw.resample(f"{minutes}min", origin="start_day", offset="15min").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(subset=["Open","High","Low","Close"])
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        if len(out) and out.index[-1] + pd.Timedelta(minutes=minutes) > now:
            out = out.iloc[:-1]  # completed candles only
        return out
        
    def indicators(data):
        d = data.copy()

        # Convert all OHLCV columns to numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            d[col] = pd.to_numeric(
                d[col], errors="coerce"
            ).astype("float64")

        d = d.dropna(subset=["Open", "High", "Low", "Close"])
        d["Volume"] = d["Volume"].fillna(0.0)

        c, h, l = d.Close, d.High, d.Low

        # EMA
        d["EMA9"] = c.ewm(span=9, adjust=False).mean()
        d["EMA21"] = c.ewm(span=21, adjust=False).mean()
        d["EMA50"] = c.ewm(span=50, adjust=False).mean()

        # RSI
        delta = c.diff()
        ag = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        al = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        d["RSI"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
        d["RSI"] = d["RSI"].fillna(50)

        # MACD
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        d["MACD"] = e12 - e26
        d["MACD_SIGNAL"] = d.MACD.ewm(span=9, adjust=False).mean()
        d["MACD_HIST"] = d.MACD - d.MACD_SIGNAL

        # ATR
        pc = c.shift(1)
        tr = pd.concat(
            [h-l, (h-pc).abs(), (l-pc).abs()],
            axis=1
        ).max(axis=1)
        d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

        # ADX / Directional Movement
        up, down = h.diff(), -l.diff()
        plus_dm = pd.Series(
            np.where((up > down) & (up > 0), up, 0.0),
            index=d.index
        )
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0),
            index=d.index
        )

        atr_safe = d.ATR.replace(0, np.nan)
        pdi = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_safe
        mdi = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_safe
        dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
        d["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)

        # Supertrend
        mid = (h+l)/2
        bu = mid + 3*d.ATR
        bl = mid - 3*d.ATR
        fu, fl = bu.copy(), bl.copy()

        for i in range(1, len(d)):
            fu.iloc[i] = (
                bu.iloc[i]
                if (bu.iloc[i] < fu.iloc[i-1] or c.iloc[i-1] > fu.iloc[i-1])
                else fu.iloc[i-1]
            )
            fl.iloc[i] = (
                bl.iloc[i]
                if (bl.iloc[i] > fl.iloc[i-1] or c.iloc[i-1] < fl.iloc[i-1])
                else fl.iloc[i-1]
            )

        direction = np.ones(len(d))
        stline = np.zeros(len(d))

        for i in range(len(d)):
            if i == 0:
                stline[i] = fl.iloc[i]
            elif direction[i-1] > 0:
                direction[i], stline[i] = (
                    (-1, fu.iloc[i]) if c.iloc[i] < fl.iloc[i]
                    else (1, fl.iloc[i])
                )
            else:
                direction[i], stline[i] = (
                    (1, fl.iloc[i]) if c.iloc[i] > fu.iloc[i]
                    else (-1, fu.iloc[i])
                )

        d["ST_DIR"] = direction
        d["Supertrend"] = stline

        # Session VWAP - numeric Volume prevents cumsum object error
        typical = (h+l+c)/3
        day_key = pd.Series(d.index.date, index=d.index)

        pv = (typical * d["Volume"]).groupby(day_key).cumsum()
        vv = d["Volume"].groupby(day_key).cumsum()

        d["VWAP"] = pv / vv.replace(0, np.nan)

        # Volume average and returns
        d["VOL_AVG"] = d["Volume"].rolling(20, min_periods=20).mean()
        d["RET1"] = c.pct_change()
        d["RET3"] = c.pct_change(3)
        d["RET6"] = c.pct_change(6)

        # Model features
        d["EMA_SPREAD"] = (d.EMA9-d.EMA21)/c
        d["EMA_TREND"] = (d.EMA21-d.EMA50)/c
        d["ATR_PCT"] = d.ATR/c
        d["VWAP_DIST"] = (c-d.VWAP)/c
        d["VOL_RATIO"] = d.Volume/d.VOL_AVG.replace(0, np.nan)

        return d.replace([np.inf, -np.inf], np.nan)
        

    

d = d.dropna(subset=["Open", "High", "Low", "Close"])
d["Volume"] = d["Volume"].fillna(0.0)
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
        typical=(h+l+c)/3
        day_key=pd.Series(d.index.date,index=d.index)
        pv=(typical*d.Volume.fillna(0)).groupby(day_key).cumsum()
        vv=d.Volume.fillna(0).groupby(day_key).cumsum()
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
        bull=bool(x.EMA9>x.EMA21>x.EMA50 and x.Close>x.EMA21 and x.ST_DIR>0)
        bear=bool(x.EMA9<x.EMA21<x.EMA50 and x.Close<x.EMA21 and x.ST_DIR<0)
        return {"close":float(x.Close),"rsi":float(x.RSI),"atr":float(x.ATR),"adx":float(x.ADX),
                "ema9":float(x.EMA9),"ema21":float(x.EMA21),"ema50":float(x.EMA50),
                "macd_bull":bool(x.MACD>x.MACD_SIGNAL),"macd_bear":bool(x.MACD<x.MACD_SIGNAL),
                "bull":bull,"bear":bear,"vwap":float(x.VWAP) if pd.notna(x.VWAP) else np.nan,
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
        Xtr,Xte,ytr,yte=X.iloc[:split],X.iloc[split:],y.iloc[:split],y.iloc[split:]
        if ytr.nunique()<2 or yte.nunique()<2:
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
                    "CE Key":c.get("instrument_key"),
                    "PE LTP":float(pm.get("ltp") or 0),"PE OI":float(pm.get("oi") or 0),
                    "PE Prev OI":float(pm.get("prev_oi") or 0),"PE Volume":float(pm.get("volume") or 0),
                    "PE Bid":float(pm.get("bid_price") or 0),"PE Ask":float(pm.get("ask_price") or 0),
                    "PE IV":float(pg.get("iv") or 0),"PE Delta":float(pg.get("delta") or 0),
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

    def combined_strategy_scores(d3, d5, d15, chain_stats, ml_metrics):
        """Five independent strategy votes. Scores are heuristic, not calibrated probabilities."""
        last5=d5.iloc[-1]; last3=d3.iloc[-1]
        # 1) Trend-following: multi-timeframe EMA alignment + supertrend
        trend_votes=[]
        for d in (d3,d5,d15):
            x=d.iloc[-1]
            trend_votes.append(1.0 if x.EMA9>x.EMA21>x.EMA50 and x.ST_DIR>0 else
                               -1.0 if x.EMA9<x.EMA21<x.EMA50 and x.ST_DIR<0 else 0.0)
        trend=float(np.mean(trend_votes))
        # 2) VWAP + volume breakout/acceptance
        vwap_score=0.0
        if pd.notna(last5.get("VWAP")) and pd.notna(last5.get("VOL_RATIO")):
            if last5.Close>last5.VWAP and last5.VOL_RATIO>=1.2: vwap_score=1.0
            elif last5.Close<last5.VWAP and last5.VOL_RATIO>=1.2: vwap_score=-1.0
            elif last5.Close>last5.VWAP: vwap_score=0.35
            elif last5.Close<last5.VWAP: vwap_score=-0.35
        # 3) Mean reversion: Bollinger z-score + RSI, contrarian direction
        mean_score=0.0
        mid=d5.Close.rolling(20).mean().iloc[-1]
        sd=d5.Close.rolling(20).std().iloc[-1]
        if pd.notna(mid) and pd.notna(sd) and sd>0:
            z=(last5.Close-mid)/(2*sd)
            if z<=-1 and last5.RSI<38: mean_score=1.0
            elif z>=1 and last5.RSI>62: mean_score=-1.0
            elif z<=-0.6: mean_score=0.35
            elif z>=0.6: mean_score=-0.35
        # 4) Donchian breakout using previous 20 completed bars, current bar excluded
        prior=d5.iloc[:-1].tail(20)
        donchian=0.0
        if len(prior)>=15:
            if last5.Close>prior.High.max(): donchian=1.0
            elif last5.Close<prior.Low.min(): donchian=-1.0
        # 5) Option chain positioning: PCR plus dominant OI walls relative to spot
        oi_score=0.0
        if chain_stats and np.isfinite(chain_stats.get("pcr",np.nan)):
            pcr=chain_stats["pcr"]
            if pcr>=1.25: oi_score=0.75
            elif pcr<=0.75: oi_score=-0.75
            elif pcr>1.05: oi_score=0.35
            elif pcr<0.95: oi_score=-0.35
        weights={"Trend Following":0.25,"VWAP + Volume":0.20,"Mean Reversion":0.15,
                 "Donchian Breakout":0.15,"Options OI + PCR":0.25}
        scores={"Trend Following":trend,"VWAP + Volume":vwap_score,
                "Mean Reversion":mean_score,"Donchian Breakout":donchian,
                "Options OI + PCR":oi_score}
        weighted=sum(scores[k]*weights[k] for k in scores)
        ml_adj=0.0
        if ml_metrics and "p_up" in ml_metrics:
            ml_adj=clamp((ml_metrics["p_up"]-0.5)*2)*0.10
        final_score=clamp(weighted+ml_adj)
        # Confidence is a heuristic agreement score, explicitly not calibrated probability.
        agreement=sum(weights[k] for k,v in scores.items() if (v>0 and final_score>0) or (v<0 and final_score<0))
        confidence=clamp(abs(final_score)*0.75+agreement*0.25,0,1)*100
        return scores,weights,final_score,confidence

    # -------------------- SCALPING PRICE-ACTION SCORE --------------------
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
        return clamp(score), notes, confirmations

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
        # ML on 5m features
        ml_model,ml_metrics=train_ml(d5)
        strategy_scores,strategy_weights,combined_score,ai_confidence=combined_strategy_scores(
            d3,d5,d15,stats,ml_metrics if ml_model is not None else None)
        # Flexible multi-factor scalping decision: chart price action is a major input.
        pa1, pa_notes1, pa_conf1 = price_action_score(d1)
        pa3, pa_notes3, pa_conf3 = price_action_score(d3)
        pa5, pa_notes5, pa_conf5 = price_action_score(d5)
        chart_score = 0.50*pa1 + 0.30*pa3 + 0.20*pa5
        # Blend chart evidence with the five-strategy ensemble; this is a directional score.
        final_direction_score = clamp(0.60*chart_score + 0.40*combined_score)
        bullish_votes = sum(v > 0.08 for v in [pa1, pa3, pa5])
        bearish_votes = sum(v < -0.08 for v in [pa1, pa3, pa5])
        total_confirmations = pa_conf1 + pa_conf3 + pa_conf5
        reasons = []
        if stale:
            signal = "NO TRADE"
            reasons = ["Market feed stale: fresh scalping signals are blocked."]
        elif final_direction_score >= 0.16 and bullish_votes >= 2 and total_confirmations >= 5 and s1["rsi"] < 82:
            signal = "BUY CE"
            reasons = ["Bullish price-action confluence across 1m/3m/5m charts."]
        elif final_direction_score <= -0.16 and bearish_votes >= 2 and total_confirmations >= 5 and s1["rsi"] > 18:
            signal = "BUY PE"
            reasons = ["Bearish price-action confluence across 1m/3m/5m charts."]
        else:
            signal = "NO TRADE"
            reasons = [f"Mixed/weak setup: chart score {final_direction_score:+.2f}; bullish votes {bullish_votes}, bearish votes {bearish_votes}."]
        # ML is a soft veto only when it strongly contradicts a chart setup.
        if ml_model is not None:
            if signal == "BUY CE" and ml_metrics["p_up"] < 0.38:
                signal = "NO TRADE"; reasons.append("ML strongly conflicts with CE setup.")
            elif signal == "BUY PE" and ml_metrics["p_up"] > 0.62:
                signal = "NO TRADE"; reasons.append("ML strongly conflicts with PE setup.")
        # Update displayed confidence from confluence, not as a claimed win probability.
        directional_agreement = max(bullish_votes, bearish_votes) / 3
        ai_confidence = min(95.0, max(5.0, abs(final_direction_score)*65 + directional_agreement*25 +
                                      min(total_confirmations, 12)*1.0))
        strategy_scores["Chart Price Action"] = chart_score
        strategy_weights["Chart Price Action"] = 0.0  # separately displayed; blended into final score above
        combined_score = final_direction_score
        reasons.extend((pa_notes1 + pa_notes3 + pa_notes5)[:8])
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
    if stream_ltp is not None and stream_connected:
        st.success(f"🟢 UPSTOX WEBSOCKET LIVE • Tick LTP: ₹{stream_ltp:,.2f} • Last tick: {stream_ltt or 'received'}")
    elif stream_error:
        st.warning(f"WebSocket not connected: {stream_error}. Showing REST quote fallback; live tick feed not confirmed.")
    else:
        st.info("Connecting to Upstox WebSocket… REST quote is shown until the first live tick arrives.")
    if stale: st.error("STALE FEED: Market open but latest candle is older than 3 minutes. Signal blocked.")
    elif not market_open: st.info("Market currently closed or outside regular NSE session. Values show last available exchange data.")
    else: st.success("Market data timestamps are within freshness guard.")

    # -------------------- MULTI-TIMEFRAME --------------------
    st.subheader("Multi-Timeframe Regime")
    tf_rows=[]
    for name,s in [("1 min",s1),("3 min",s3),("5 min",s5)]:
        tf_rows.append({"Timeframe":name,"Trend":"Bullish" if s["bull"] else "Bearish" if s["bear"] else "Neutral",
                        "Close":round(s["close"],2),"RSI":round(s["rsi"],1),"ADX":round(s["adx"],1),
                        "MACD":"Bullish" if s["macd_bull"] else "Bearish","VWAP":round(s["vwap"],2) if np.isfinite(s["vwap"]) else None})
    st.dataframe(pd.DataFrame(tf_rows),use_container_width=True,hide_index=True)

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
        view_cols=["Strike","CE LTP","CE OI","CE Prev OI","CE Volume","CE IV","CE Delta","PE LTP","PE OI","PE Prev OI","PE Volume","PE IV","PE Delta"]
        view=chain[view_cols].copy()
        view["CE OI Chg"]=view["CE OI"]-view["CE Prev OI"]
        view["PE OI Chg"]=view["PE OI"]-view["PE Prev OI"]
        st.dataframe(view[["Strike","CE LTP","CE OI","CE OI Chg","CE Volume","CE IV","CE Delta",
                           "PE LTP","PE OI","PE OI Chg","PE Volume","PE IV","PE Delta"]],
                     use_container_width=True,hide_index=True)
        st.caption("OI/PCR are context, not standalone buy/sell signals. API values may be delayed or stale outside market hours.")
        # -------------------- FINAL SIGNAL --------------------
    st.divider()
    st.subheader("FINAL SCALPING SIGNAL — OPTIONS ONLY")
    if signal=="BUY CE": st.success("🟢 FINAL SIGNAL: BUY CE")
    elif signal=="BUY PE": st.error("🔴 FINAL SIGNAL: BUY PE")
    else: st.warning("🟡 FINAL SIGNAL: NO TRADE")
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
    if ml_model is not None:
        st.caption(f"ML model estimate: {ml_metrics['p_up']*100:.1f}% Up / {(1-ml_metrics['p_up'])*100:.1f}% Down. Holdout accuracy: {ml_metrics['accuracy']*100:.1f}%.")

    # -------------------- OPTION BUYING CANDIDATE --------------------
    st.subheader("Scalping Option Contract")
    candidate=None
    side_name="CE" if signal=="BUY CE" else "PE" if signal=="BUY PE" else None
    if side_name and not chain.empty:
        key_col=f"{side_name} Key"; ltp_col=f"{side_name} LTP"; vol_col=f"{side_name} Volume"
        candidates=chain[(chain[ltp_col]>0)&chain[key_col].notna()].copy()
        if not candidates.empty:
            candidates["atm_dist"]=(candidates.Strike-spot).abs()
            candidates=candidates.sort_values(["atm_dist",vol_col],ascending=[True,False])
            candidate=candidates.iloc[0].to_dict()
            option_entry=float(candidate[ltp_col])
            # Short-horizon illustrative premium risk: 20% stop, 1R and 2R targets.
            option_sl=max(0.05,option_entry*0.80)
            option_risk=option_entry-option_sl
            option_t1=option_entry+option_risk
            option_t2=option_entry+2*option_risk
            option_rr=(option_t1-option_entry)/option_risk if option_risk>0 else None
            a,b,c,d,e=st.columns(5)
            a.metric("Selection",f"BUY {side_name}")
            b.metric("Strike",f"{candidate['Strike']:.0f}")
            c.metric("Entry premium",f"₹{option_entry:.2f}")
            d.metric("Stop Loss",f"₹{option_sl:.2f}")
            e.metric("T1 / T2",f"₹{option_t1:.2f} / ₹{option_t2:.2f}")
            st.metric("Risk : Reward",f"1 : {option_rr:.2f}" if option_rr is not None else "N/A")
            st.caption("Illustrative levels only, derived from option-chain LTP snapshot; not a live executable order. Verify bid/ask spread, liquidity, lot size and slippage.")
        else:
            st.warning(f"Signal {side_name} है, लेकिन selected expiry में usable option contract नहीं मिला. Contract unavailable होने पर trade setup confirm नहीं है.")
    elif side_name:
        st.warning("Option chain unavailable: CE/PE signal को contract-confirmed नहीं माना जा सकता; entry/SL/targets withheld.")
    else:
        st.info("NO TRADE: कोई option entry, stop loss या target generate नहीं किया गया.")
        # -------------------- CHART ANALYSIS --------------------
    st.divider()
    st.subheader("Scalping Chart Analysis — 1m / 3m / 5m")
    def chart_levels(frame):
        recent=frame.iloc[:-1].tail(30) if len(frame) > 1 else frame.tail(30)
        return float(recent.High.max()), float(recent.Low.min())
    for tf_name, frame in [("1 Minute",d1),("3 Minute",d3),("5 Minute",d5)]:
        if len(frame)<10: continue
        last=frame.iloc[-1]
        resistance,support=chart_levels(frame)
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
        st.write(f"**{tf_name}:** Trend **{trend}** · {vwap_state} · RSI **{last.RSI:.1f}** · ADX **{last.ADX:.1f}** · Candle **{pattern}** · Support **{support:.2f}** · Resistance **{resistance:.2f}** · Volume ratio **{vol_ratio:.2f}x**" if np.isfinite(vol_ratio) else f"**{tf_name}:** Trend **{trend}** · {vwap_state} · RSI **{last.RSI:.1f}** · ADX **{last.ADX:.1f}** · Candle **{pattern}** · Support **{support:.2f}** · Resistance **{resistance:.2f}**")

    # -------------------- SIGNAL-ONLY MODE --------------------
    st.info("Signal-only dashboard: Paper Trading, virtual balance, trade history and automatic broker orders are disabled.")

    # -------------------- BACKTEST (SIMPLE, NON-OVERLAPPING SIGNAL STUDY) --------------------
    st.divider()
    st.subheader("Historical Signal Study (5-minute bars)")
    st.caption("A simple chronological, non-overlapping research simulation using trend filters. Excludes brokerage, slippage and taxes; not a live execution backtest.")
    bt=d5.copy()
    bt["fwd_ret"] = bt.Close.shift(-3)/bt.Close-1
    bt["bull_setup"]=(bt.EMA9>bt.EMA21)&(bt.EMA21>bt.EMA50)&(bt.RSI.between(50,72))&(bt.ADX>=15)&(bt.ST_DIR>0)
    bt["bear_setup"]=(bt.EMA9<bt.EMA21)&(bt.EMA21<bt.EMA50)&(bt.RSI.between(28,50))&(bt.ADX>=15)&(bt.ST_DIR<0)
    bt=bt.dropna(subset=["fwd_ret"])
    bt["strategy_ret"]=np.where(bt.bull_setup,bt.fwd_ret,np.where(bt.bear_setup,-bt.fwd_ret,0))
    active=bt[bt.bull_setup|bt.bear_setup]
    if len(active)>10:
        wins=int((active.strategy_ret>0).sum())
        st.write(f"Signals: **{len(active)}** | Hit ratio: **{wins/len(active)*100:.1f}%** | Mean return per signal: **{active.strategy_ret.mean()*100:.3f}%**")
        st.caption("Signals can overlap and use close-to-close returns; this is a screening study, not a fully costed portfolio backtest.")
    else: st.info("Historical signal sample अभी पर्याप्त नहीं है.")

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
    st.write("- Five strategy ensemble uses heuristic weights; the displayed confidence is not calibrated probability. ML is a recent-data research model, not a production-grade institutional model. News, order book and exchange tick replay are not included.")
    st.write("- Option chain is a snapshot, not a tick-by-tick feed. Option-chain values are snapshots; premium estimates exclude spread impact, slippage and lot-size constraints.")
    st.error("Educational signal research only. No guaranteed prediction or accuracy. No paper trading and no broker orders are executed.")


live_dashboard()
