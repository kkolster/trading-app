# app.py
import warnings, datetime as dt
import numpy as np
import pandas as pd
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
st.set_page_config(page_title="Simple Quant Trader", layout="wide")
st.title("🧠 Simple Quant Trader — demo app")

# ---------- helpers ----------
def _ensure_datetime(x):
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return dt.datetime.combine(x, dt.time.min)
    return x

def _find_ohlc_cols(df: pd.DataFrame):
    """Return (open, high, low, close, volume) column names present in df.
    Works case-insensitively and with common aliases like 'Adj Close'."""
    cols_map = {str(c).strip().lower(): c for c in df.columns}

    def pick(candidates):
        for c in candidates:
            key = str(c).strip().lower()
            if key in cols_map:
                return cols_map[key]
        return None

    open_c  = pick(["open", "o"])
    high_c  = pick(["high", "h"])
    low_c   = pick(["low", "l"])
    close_c = pick(["close", "adj close", "c"])  # accept Adj Close
    vol_c   = pick(["volume", "vol"])

    # ultimate fallback for close: last numeric column
    if close_c is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        close_c = numeric[-1] if numeric else None

    return open_c, high_c, low_c, close_c, vol_c

def load(symbol, start, end, interval):
    start = _ensure_datetime(start)
    end   = _ensure_datetime(end)
    now = dt.datetime.utcnow()
    if end > now: end = now

    alias = {"SPX":"^GSPC","GSPC":"^GSPC","SP500":"^GSPC",
             "NDX":"^NDX","US100":"^NDX","NASDAQ":"^IXIC",
             "DOW":"^DJI","DJI":"^DJI"}
    sym = alias.get(symbol.strip().upper(), symbol.strip())

    df = yf.download(sym, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False, threads=True)
    if df is None or df.empty:
        raise ValueError(f"No data for '{sym}' (interval={interval}).")

    # Flatten multi-index, normalize names
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    df = df.rename(columns=lambda c: str(c).strip().title())
    df.index = pd.to_datetime(df.index)
    return df

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    symbol  = st.text_input("Ticker (AAPL, NVDA, ^GSPC, ^NDX, BTC-USD, EURUSD=X)", "AAPL")
    start   = st.date_input("Start date", dt.date.today() - dt.timedelta(days=365*2))
    end     = st.date_input("End date",   dt.date.today())
    interval = st.selectbox("Interval", ["1d","1h","15m"])
    horizon  = st.number_input("Prediction horizon (bars ahead)", 1, 10, 1)
    prob_long  = st.slider("Prob. threshold LONG", 0.0, 1.0, 0.55, 0.01)
    prob_short = st.slider("Prob. threshold SHORT",0.0, 1.0, 0.55, 0.01)

# ---------- load ----------
try:
    df = load(symbol, start, end + dt.timedelta(days=1), interval)
except Exception as e:
    st.error(f"❌ Could not load data: {e}")
    st.stop()

st.success(f"Loaded {len(df)} rows for {symbol}")
st.caption(f"Columns: {list(df.columns)}")

# ---------- resolve column names robustly ----------
OPEN, HIGH, LOW, CLOSE, VOL = _find_ohlc_cols(df)
missing = [name for name,real in
           {"Open":OPEN,"High":HIGH,"Low":LOW,"Close/Adj":CLOSE}.items() if real is None]
if missing:
    st.error(f"Could not find required columns: {', '.join(missing)}. "
             f"Available columns: {list(df.columns)}")
    st.stop()

# ---------- plot ----------
fig = go.Figure(data=[go.Candlestick(
    x=df.index, open=df[OPEN], high=df[HIGH], low=df[LOW], close=df[CLOSE], name="Price"
)])
st.plotly_chart(fig, use_container_width=True)

# (You can keep building features below, all using CLOSE/HIGH/LOW/OPEN safely)
