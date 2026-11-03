# NIFTY 50 Institutional-Style AI Research & Paper Trading Dashboard
# Research/education only. No broker order-placement endpoints are used.
import streamlit as st
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
IST = ZoneInfo("Asia/Kolkata")
BASE = "https://api.upstox.com"
INDEX_DEFAULT = "NSE_INDEX|Nifty 50"

st.title("🏛️ NIFTY 50 AI Research & Paper Trading Desk")
st.caption("Upstox market data • Multi-timeframe • Option chain/OI • ML research • Walk-forward evaluation • Paper only")
st_autorefresh(interval=15000, key="institutional_refresh")

# -------------------- SESSION STATE --------------------
defaults = {
    "paper_cash": 100000.0, "paper_trades": [], "open_paper": None,
    "last_signal_key": None, "signal_log": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- SETTINGS --------------------
with st.sidebar:
    st.header("Configuration")
    risk_pct = st.slider("Virtual risk per trade (%)", 0.25, 2.0, 0.5, 0.25)
    max_daily_loss = st.number_input("Daily loss stop (₹ virtual)", 500, 20000, 3000, 250)
    expiry_choice = st.selectbox("Option expiry", ["current_week", "next_week", "current_month", "next_month"])
    strike_range = st.slider("Option-chain strikes around ATM", 5, 20, 10)
    ml_threshold = st.slider("Minimum ML probability", 0.50, 0.80, 0.58, 0.01)
    st.divider()
    st.subheader("Paper account")
    st.metric("Virtual balance", f"₹{st.session_state.paper_cash:,.2f}")
    reset_amount = st.number_input("Reset balance (₹)", min_value=10000, value=100000, step=10000)
    if st.button("Reset paper account", use_container_width=True):
        st.session_state.paper_cash = float(reset_amount)
        st.session_state.paper_trades = []
        st.session_state.open_paper = None
        st.session_state.signal_log = []
        st.rerun()
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

# -------------------- PAPER TRADING --------------------
def save_close(exit_px, reason):
    t=st.session_state.open_paper
    if not t: return
    side=1 if t["side"]=="BUY" else -1
    pts=(float(exit_px)-t["entry"])*side
    pnl=pts*t["units"]
    st.session_state.paper_cash += pnl
    st.session_state.paper_trades.insert(0,{"Instrument":t["instrument"],"Side":t["side"],
        "Entry time":t["time"],"Exit time":str(pd.Timestamp.now(tz="Asia/Kolkata")),
        "Entry":round(t["entry"],2),"Exit":round(float(exit_px),2),"Units":round(t["units"],3),
        "Points":round(pts,2),"Virtual P&L ₹":round(pnl,2),"Reason":reason})
    st.session_state.open_paper=None

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
    d3=indicators(resample_ohlcv(raw,3))
    d5=indicators(resample_ohlcv(raw,5))
    d15=indicators(resample_ohlcv(raw,15))
    if min(len(d3),len(d5),len(d15))<55: st.error("Indicators के लिए पर्याप्त completed candles नहीं हैं."); st.stop()
    s3,s5,s15=map(tf_summary,[d3,d5,d15])
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
    signal,ml_directional_prob,reasons=decide_signal(s3,s5,s15,ml_metrics if ml_model is not None else None,stats,stale)
except Exception as e:
    st.error(f"Data / API error: {e}")
    st.info("Secrets में UPSTOX_ACCESS_TOKEN और UPSTOX_INSTRUMENT_KEY = NSE_INDEX|Nifty 50 जाँचें. Option chain API access भी आवश्यक है.")
    st.stop()

# -------------------- OVERVIEW --------------------
st.divider()
st.subheader("Market Snapshot")
c=st.columns(5)
c[0].metric("NIFTY 50 LTP",f"{spot:,.2f}",None if quote_data.get("cp") is None else f"Prev close {float(quote_data['cp']):,.2f}")
c[1].metric("3m RSI",f"{s3['rsi']:.1f}")
c[2].metric("5m RSI",f"{s5['rsi']:.1f}")
c[3].metric("15m RSI",f"{s15['rsi']:.1f}")
c[4].metric("3m ADX",f"{s3['adx']:.1f}")
st.caption(f"Last completed 1m candle: {latest} | Feed age: {age:.0f}s | Option expiry: {chain_all['Expiry'].iloc[0] if not chain_all.empty else expiry_choice}")
if stale: st.error("STALE FEED: Market open but latest candle is older than 3 minutes. Signal blocked.")
elif not market_open: st.info("Market currently closed or outside regular NSE session. Values show last available exchange data.")
else: st.success("Market data timestamps are within freshness guard.")

# -------------------- MULTI-TIMEFRAME --------------------
st.subheader("Multi-Timeframe Regime")
tf_rows=[]
for name,s in [("3 min",s3),("5 min",s5),("15 min",s15)]:
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
st.subheader("AI + Rules + Options Decision")
if signal=="BUY": st.success("🟢 NIFTY BIAS: BULLISH")
elif signal=="SELL": st.error("🔴 NIFTY BIAS: BEARISH")
else: st.warning("🟡 NO TRADE")
for r in reasons: st.write("• "+r)
if ml_model is not None:
    st.write(f"ML direction score: {ml_metrics['p_up']*100:.1f}% Up, {(1-ml_metrics['p_up'])*100:.1f}% Down")

# -------------------- OPTION BUYING CANDIDATE --------------------
st.subheader("CE / PE Paper-Buy Candidate")
candidate=None
if not chain.empty and signal in ("BUY","SELL"):
    side_col="CE" if signal=="BUY" else "PE"
    key_col=f"{side_col} Key"; ltp_col=f"{side_col} LTP"; vol_col=f"{side_col} Volume"
    candidates=chain[(chain[ltp_col]>0)&chain[key_col].notna()].copy()
    if not candidates.empty:
        candidates["atm_dist"]=(candidates.Strike-spot).abs()
        candidates=candidates.sort_values(["atm_dist",vol_col],ascending=[True,False])
        candidate=candidates.iloc[0].to_dict()
        opt_ltp=float(candidate[ltp_col])
        opt_key=str(candidate[key_col])
        st.write(f"Directional setup: **BUY {side_col}** | Candidate strike: **{candidate['Strike']:.0f}** | Chain LTP: **₹{opt_ltp:.2f}**")
        st.caption(f"Instrument key: {opt_key} | IV: {candidate.get(side_col+' IV',0):.2f} | Delta: {candidate.get(side_col+' Delta',0):.3f}")
        if candidate.get(side_col+" Ask",0)>0 and candidate.get(side_col+" Bid",0)>0:
            st.caption(f"Bid/Ask: ₹{candidate[side_col+' Bid']:.2f} / ₹{candidate[side_col+' Ask']:.2f}")
    else:
        st.info("Directional signal मिला, लेकिन usable CE/PE contract data नहीं मिला.")
else:
    st.info("Option candidate तभी बनेगा जब NIFTY directional signal और usable option chain दोनों उपलब्ध हों.")
    
# -------------------- LEVELS & RISK --------------------
st.divider()
st.subheader("Risk & Index Levels (Research only)")
entry=spot
atr=max(s3["atr"],s5["atr"]*0.5)
risk_pts=max(1.5*atr,1.0)
if signal in ("BUY","SELL"):
    if signal=="BUY": sl=entry-risk_pts; t1=entry+risk_pts; t2=entry+2*risk_pts
    else: sl=entry+risk_pts; t1=entry-risk_pts; t2=entry-2*risk_pts
    x1,x2,x3,x4=st.columns(4)
    x1.metric("Reference entry",f"{entry:,.2f}")
    x2.metric("ATR stop",f"{sl:,.2f}")
    x3.metric("Target 1",f"{t1:,.2f}")
    x4.metric("Target 2",f"{t2:,.2f}")
    risk_rupees=st.session_state.paper_cash*risk_pct/100
    units=risk_rupees/risk_pts
    st.caption(f"Virtual sizing: risk budget ₹{risk_rupees:,.0f} / {risk_pts:.1f} index points = {units:.2f} illustrative units. Not actual NIFTY lot sizing.")
else:
    sl=t1=t2=None
    st.info("NO TRADE: no entry/SL/target generated.")

# -------------------- PAPER TRADING --------------------
st.divider()
st.header("🧪 Paper Trading Simulator")
st.warning("Paper-only simulation. No broker orders are placed. Index trades use point-based illustrative units; option trades use observed option LTP and virtual units. Fees, slippage and taxes are not included.")
daily_loss=sum(float(x["Virtual P&L ₹"]) for x in st.session_state.paper_trades
               if str(x["Exit time"])[:10]==str(pd.Timestamp.now(tz="Asia/Kolkata").date()))
st.metric("Today's realized virtual P&L",f"₹{daily_loss:+,.2f}")
if daily_loss <= -max_daily_loss:
    st.error("Daily virtual loss limit reached. New paper trades disabled.")
trade=st.session_state.open_paper
if trade is None:
    if signal in ("BUY","SELL") and sl is not None:
        if st.button(f"Open NIFTY {signal} paper position",use_container_width=True,disabled=(daily_loss<=-max_daily_loss)):
            st.session_state.open_paper={"instrument":"NIFTY 50 INDEX","side":signal,"entry":entry,"sl":sl,
                "target":t1,"units":(st.session_state.paper_cash*risk_pct/100)/risk_pts,
                "time":str(pd.Timestamp.now(tz="Asia/Kolkata"))}
            st.rerun()
    if candidate is not None:
        optside="BUY"
        optprice=float(candidate["CE LTP"] if signal=="BUY" else candidate["PE LTP"])
        optname=f"NIFTY {candidate['Strike']:.0f} {('CE' if signal=='BUY' else 'PE')}"
        optkey=str(candidate["CE Key"] if signal=="BUY" else candidate["PE Key"])
        opt_sl=max(0.05,optprice*0.70)
        opt_target=optprice*1.50
        if st.button(f"Open {optname} option paper position @ ₹{optprice:.2f}",use_container_width=True,
                     disabled=(daily_loss<=-max_daily_loss)):
            opt_risk=max(optprice-opt_sl,0.01)
            units=(st.session_state.paper_cash*risk_pct/100)/opt_risk
            st.session_state.open_paper={"instrument":optname,"instrument_key":optkey,"side":"BUY",
                "entry":optprice,"sl":opt_sl,"target":opt_target,"units":units,
                "time":str(pd.Timestamp.now(tz="Asia/Kolkata"))}
            st.rerun()
else:
    is_option=bool(trade.get("instrument_key"))
    current_price=None
    if is_option:
        try:
            oq=fetch_ltp(TOKEN,trade["instrument_key"],int(datetime.now(IST).timestamp()//8))
            current_price=oq["ltp"]
        except Exception as e:
            st.warning(f"Option LTP refresh failed: {e}")
    else:
        current_price=spot
    if current_price is not None:
        direction=1 if trade["side"]=="BUY" else -1
        points=(current_price-trade["entry"])*direction
        unreal=points*trade["units"]
        q1,q2,q3,q4=st.columns(4)
        q1.metric("Open position",trade["instrument"])
        q2.metric("Entry",f"₹{trade['entry']:,.2f}")
        q3.metric("Current",f"₹{current_price:,.2f}")
        q4.metric("Unrealized virtual P&L",f"₹{unreal:+,.2f}")
        st.write(f"Stop: ₹{trade['sl']:,.2f} | Target: ₹{trade['target']:,.2f} | Units: {trade['units']:.3f}")
        sl_hit=current_price<=trade["sl"] if trade["side"]=="BUY" else current_price>=trade["sl"]
        tgt_hit=current_price>=trade["target"] if trade["side"]=="BUY" else current_price<=trade["target"]
        if sl_hit:
            save_close(current_price,"Paper stop-loss condition"); st.rerun()
        elif tgt_hit:
            save_close(current_price,"Paper target condition"); st.rerun()
        elif st.button("Close paper position at current quote",use_container_width=True):
            save_close(current_price,"Manual paper exit"); st.rerun()
    else:
        st.info("Open position active; waiting for a valid quote. No simulated exit is assumed.")

st.subheader("Closed Paper Trades")
if st.session_state.paper_trades:
    hist_df=pd.DataFrame(st.session_state.paper_trades)
    st.dataframe(hist_df,use_container_width=True,hide_index=True)
    wins=sum(1 for x in st.session_state.paper_trades if x["Virtual P&L ₹"]>0)
    total=len(st.session_state.paper_trades)
    gross_profit=sum(x["Virtual P&L ₹"] for x in st.session_state.paper_trades if x["Virtual P&L ₹"]>0)
    gross_loss=abs(sum(x["Virtual P&L ₹"] for x in st.session_state.paper_trades if x["Virtual P&L ₹"]<0))
    pf=gross_profit/gross_loss if gross_loss else np.nan
    a,b,c=st.columns(3)
    a.metric("Win rate",f"{wins/total*100:.1f}%")
    b.metric("Profit factor",f"{pf:.2f}" if np.isfinite(pf) else "N/A")
    c.metric("Closed trades",str(total))
else: st.info("No closed paper trades yet.")
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
st.subheader("NIFTY 3-minute Price & Indicators")
fig=go.Figure()
fig.add_trace(go.Candlestick(x=d3.index,open=d3.Open,high=d3.High,low=d3.Low,close=d3.Close,name="NIFTY"))
for col in ["EMA9","EMA21","EMA50","Supertrend","VWAP"]:
    if col in d3 and d3[col].notna().any():
        fig.add_trace(go.Scatter(x=d3.index,y=d3[col],name=col))
fig.update_layout(height=560,template="plotly_dark",xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)
rfig=go.Figure()
rfig.add_trace(go.Scatter(x=d3.index,y=d3.RSI,name="RSI"))
rfig.add_hline(y=70,line_dash="dash"); rfig.add_hline(y=30,line_dash="dash"); rfig.add_hline(y=50,line_dash="dot")
rfig.update_layout(height=250,template="plotly_dark")
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
st.write("- Session-state paper history may reset if Streamlit session/server restarts; export or persistent storage is not configured.")
st.write("- Model is trained on recent intraday data in-session. It is not a production-grade institutional model and does not include news, order book, exchange tick replay, or a separately validated calibrated probability.")
st.write("- Option chain is a snapshot, not a tick-by-tick feed. Paper option P&L excludes costs, spread impact and lot-size constraints.")
st.error("Educational research tool only. No guaranteed prediction. Never treat a model score or paper result as a promise of live returns.")
