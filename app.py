# app.py
import warnings, datetime as dt, numpy as np, pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ------------------------- UI -------------------------
st.set_page_config(page_title="Simple Quant Trader", layout="wide")
st.title("🧠 Simple Quant Trader — demo app")

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Ticker (examples: AAPL, NVDA, ^GSPC, BTC-USD, EURUSD=X)", "AAPL")
    start = st.date_input("Start date", dt.date.today() - dt.timedelta(days=365*2))
    end = st.date_input("End date", dt.date.today())
    interval = st.selectbox("Interval", ["1d","1h","30m","15m"], index=0)
    horizon = st.number_input("Prediction horizon (bars ahead)", 1, 10, 1)
    prob_long = st.slider("Prob. threshold LONG", 0.50, 0.70, 0.55, 0.01)
    prob_short = st.slider("Prob. threshold SHORT", 0.50, 0.70, 0.55, 0.01)
    cost_bps = st.number_input("Txn cost (bps per trade)", 0, 50, 2)
    account = st.number_input("Account size (USD)", 1000, 1000000, 10000, 100)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 2.0, 1.0, 0.1)
    atr_mult = st.slider("ATR stop multiple", 0.5, 5.0, 1.5, 0.1)
    max_positions = st.number_input("Max concurrent positions", 1, 10, 3)

# ------------------------- Data -------------------------
@st.cache_data(show_spinner=False)
def load(symbol, start, end, interval):
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df = df.dropna().copy()
    df.columns = [c.capitalize() for c in df.columns]
    return df

df = load(symbol, start, end + dt.timedelta(days=1), interval)
if df.empty:
    st.error("No data loaded. Try a different ticker or interval.")
    st.stop()

# ------------------------- Features -------------------------
def add_features(df):
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["sma_10"] = SMAIndicator(out["Close"], 10).sma_indicator()
    out["sma_50"] = SMAIndicator(out["Close"], 50).sma_indicator()
    out["ema_20"] = EMAIndicator(out["Close"], 20).ema_indicator()
    rsi = RSIIndicator(out["Close"], 14)
    out["rsi_14"] = rsi.rsi()
    macd = MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_sig"] = macd.macd_signal()
    bb = BollingerBands(out["Close"], 20, 2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["atr"] = (
        (out["High"] - out["Low"])
        .combine((out["High"] - out["Close"].shift()).abs(), np.maximum)
        .combine((out["Low"] - out["Close"].shift()).abs(), np.maximum)
        .rolling(14).mean()
    )
    out["target"] = (out["Close"].shift(-horizon) > out["Close"]).astype(int)  # 1 = up
    return out.dropna()

df_feat = add_features(df)

# Train/test split via time-series folds
features = ["ret_1","ret_5","vol_10","sma_10","sma_50","ema_20","rsi_14","macd","macd_sig","bb_high","bb_low"]
X = df_feat[features].values
y = df_feat["target"].values
idx = df_feat.index

tscv = TimeSeriesSplit(n_splits=5)
pred = np.full(len(y), np.nan)
proba = np.full(len(y), np.nan)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y[train_idx]
    model = RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:,1]
    proba[test_idx] = p
    pred[test_idx] = (p >= 0.5).astype(int)

df_feat["pred_up"] = pred
df_feat["proba_up"] = proba

# ------------------------- Backtest -------------------------
def backtest(df, prob_long, prob_short, cost_bps):
    d = df.copy()
    d["signal"] = 0
    d.loc[d["proba_up"] >= prob_long, "signal"] = 1
    d.loc[d["proba_up"] <= (1 - prob_short), "signal"] = -1   # short if high prob down
    d["signal"] = d["signal"].shift().fillna(0)  # enter next bar open

    # returns at next close
    r = d["Close"].pct_change().fillna(0)
    trade_cost = (abs(d["signal"].diff().fillna(0)) * (cost_bps/10000.0))
    strat = d["signal"] * r - trade_cost
    d["equity"] = (1 + strat).cumprod()
    acc = accuracy_score(d["target"].dropna(), (d["proba_up"].dropna()>=0.5).astype(int))
    prec = precision_score(d["target"].dropna(), (d["proba_up"].dropna()>=0.5).astype(int))
    sharpe = np.sqrt(252 if interval=="1d" else 252*6.5) * strat.mean() / (strat.std() + 1e-9)
    return d, acc, prec, sharpe

bt, acc, prec, sharpe = backtest(df_feat, prob_long, prob_short, cost_bps)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (clf)", f"{acc:.2%}")
col2.metric("Precision (up)", f"{prec:.2%}")
col3.metric("Sharpe (naive)", f"{sharpe:.2f}")
col4.metric("CAGR-ish", f"{(bt['equity'].iloc[-1]-1):.2%}")

# ------------------------- Chart -------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["sma_50"], line=dict(width=1), name="SMA50", yaxis="y1"))
fig.add_trace(go.Scatter(
    x=bt.index, y=bt["equity"] / bt["equity"].iloc[0] * df["Close"].iloc[0],
    line=dict(width=2), name="Strategy (scaled)"
))
st.plotly_chart(fig, use_container_width=True)

# ------------------------- Today’s Signals & Sizing -------------------------
last = df_feat.iloc[-1]
atr = float(last["atr"])
px = float(df["Close"].iloc[-1])

# ATR stop distance (price units)
stop_dist = atr_mult * atr if atr > 0 else 0.01 * px
usd_risk = account * (risk_pct/100.0)
qty = max(int(usd_risk / stop_dist), 1)

side = "FLAT"
if last["proba_up"] >= prob_long:
    side = "LONG"
elif last["proba_up"] <= (1 - prob_short):
    side = "SHORT"

tp_pts = 3 * stop_dist   # 1:3 R:R by default
sl_lvl_long = px - stop_dist
tp_lvl_long = px + tp_pts
sl_lvl_short = px + stop_dist
tp_lvl_short = px - tp_pts

signals = []
if side == "LONG":
    signals.append(dict(symbol=symbol, side="BUY", px=round(px,4), qty=qty,
                        SL=round(sl_lvl_long,4), TP=round(tp_lvl_long,4),
                        prob_up=round(float(last["proba_up"]),4)))
elif side == "SHORT":
    signals.append(dict(symbol=symbol, side="SELL", px=round(px,4), qty=qty,
                        SL=round(sl_lvl_short,4), TP=round(tp_lvl_short,4),
                        prob_up=round(float(last["proba_up"]),4)))
else:
    signals.append(dict(symbol=symbol, side="NO TRADE", px=round(px,4), qty=0,
                        SL=None, TP=None, prob_up=round(float(last["proba_up"]),4)))

sig_df = pd.DataFrame(signals)
st.subheader("📋 Today’s signal")
st.dataframe(sig_df)

# Export CSV you can open or send to MT5 manually
csv = sig_df.to_csv(index=False).encode()
st.download_button("Download signal CSV", data=csv, file_name=f"signal_{symbol}.csv", mime="text/csv")

st.caption("Tip: change ticker and interval on the left to scan multiple markets quickly.")
