import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Pro Market Analyst", layout="wide", page_icon="🧠")

st.title("🧠 Pro Market Analyst & Signal Engine")
st.markdown("""
**Advanced Multi-Factor Analysis.** This engine combines **Technical Trend** (SMA), **Momentum** (RSI/MACD), and **Macro Risks** (Yield Curve/VIX) to generate a weighted investment score.
""")

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
days_lookback = st.sidebar.slider("Lookback Period (Days)", 365, 365*5, 730)
start_date = datetime.now() - timedelta(days=days_lookback)

st.sidebar.markdown("---")
st.sidebar.info("""
**Scoring Logic (0 to 100):**
* **0-40:** Bearish (Sell/Avoid)
* **40-60:** Neutral (Watch)
* **60-100:** Bullish (Accumulate)

**Macro Penalty:**
Scores are penalized if the Yield Curve is inverted or Volatility (VIX) is high.
""")

# --- Assets ---
DECISION_ASSETS = {
    '🇺🇸 Equities (S&P 500)': 'SPY',
    '🌍 Dev. Markets (EAFE)': 'EFA',
    '🏛 Bonds (20Y Treasury)': 'TLT',
    '🧈 Gold': 'GLD',
    '🏠 Real Estate': 'VNQ',
    '₿ Bitcoin': 'BTC-USD'
}

MACRO_ASSETS = {
    '10Y Yield': '^TNX',
    '5Y Yield': '^FVX',
    'Volatility (VIX)': '^VIX'
}

# --- Advanced Calculation Functions ---

@st.cache_data
def fetch_data(ticker):
    """Fetches data and calculates advanced indicators."""
    try:
        # Fetch extra buffer for calculations
        df = yf.download(ticker, start=start_date - timedelta(days=100), progress=False)
        if df.empty: return None
        
        # Ensure we are working with 1D series (Close)
        df = df['Close'].to_frame() if isinstance(df['Close'], pd.Series) else df[['Close']]
        df.columns = ['Close']
        
        # 1. SMAs
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 2. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 4. Drawdown
        rolling_max = df['Close'].cummax()
        df['Drawdown'] = (df['Close'] - rolling_max) / rolling_max
        
        return df
    except Exception as e:
        return None

def analyze_asset(asset_name, ticker, macro_penalty=0):
    """
    Generates a score (0-100) based on weighted factors.
    """
    df = fetch_data(ticker)
    if df is None: return None
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 50 # Start Neutral
    reasons = []
    
    # --- Factor 1: Trend (Weight: 40%) ---
    if current['Close'] > current['SMA_200']:
        score += 20
        reasons.append("Price > 200 SMA (Long Term Bullish)")
    else:
        score -= 20
        reasons.append("Price < 200 SMA (Long Term Bearish)")
        
    if current['Close'] > current['SMA_50']:
        score += 10
        if current['SMA_50'] > current['SMA_200']:
            reasons.append("Golden Cross Active")
    else:
        score -= 10
    
    # --- Factor 2: Momentum (RSI) (Weight: 20%) ---
    rsi = current['RSI']
    if rsi < 30:
        score += 10
        reasons.append("RSI Oversold (Value Buy?)")
    elif rsi > 70:
        score -= 10
        reasons.append("RSI Overbought (Risk)")
    else:
        reasons.append(f"RSI Neutral ({rsi:.0f})")
        
    # --- Factor 3: MACD (Weight: 20%) ---
    if current['MACD'] > current['Signal_Line']:
        score += 10
        reasons.append("MACD Bullish Crossover")
    else:
        score -= 10
    
    # --- Factor 4: Macro Penalty (Weight: 20%) ---
    # We subtract the macro penalty passed from the main loop
    if macro_penalty > 0:
        score -= macro_penalty
        reasons.append(f"⚠️ Macro Headwind (-{macro_penalty} pts)")
    
    # Cap score
    score = max(0, min(100, score))
    
    # Determine Verdict
    if score >= 75: verdict = "STRONG BUY 🚀"
    elif score >= 60: verdict = "BUY / ACCUMULATE 🟢"
    elif score >= 40: verdict = "HOLD / NEUTRAL 🟡"
    elif score >= 25: verdict = "SELL / REDUCE 🔴"
    else: verdict = "STRONG SELL 📉"

    return {
        "Asset": asset_name,
        "Price": current['Close'],
        "Score": score,
        "Verdict": verdict,
        "Drawdown": f"{current['Drawdown']*100:.2f}%",
        "RSI": f"{rsi:.1f}",
        "Details": "; ".join(reasons)
    }

def get_macro_environment():
    """Analyzes the 'Weather' of the market to set penalties."""
    try:
        # Fetch Data
        # We wrap in float() to ensure we have a simple number, not a data Series
        ten_yr_data = yf.download(MACRO_ASSETS['10Y Yield'], period="5d", progress=False)['Close']
        five_yr_data = yf.download(MACRO_ASSETS['5Y Yield'], period="5d", progress=False)['Close']
        vix_data = yf.download(MACRO_ASSETS['Volatility (VIX)'], period="5d", progress=False)['Close']
        
        # Check if data is empty to prevent crashes
        if ten_yr_data.empty or five_yr_data.empty or vix_data.empty:
            return 0, ["⚠️ insufficient macro data"]

        # Extract the very last value as a pure Python float
        # .iloc[-1] gets the last row. float() converts it to a raw number.
        ten_yr = float(ten_yr_data.iloc[-1])
        five_yr = float(five_yr_data.iloc[-1])
        vix = float(vix_data.iloc[-1])
        
        penalty = 0
        status = []
        
        # 1. Yield Curve Check
        if five_yr > ten_yr:
            penalty += 15
            status.append(f"Yield Curve Inverted (Recession Risk) [5Y: {five_yr:.2f}% > 10Y: {ten_yr:.2f}%]")
        
        # 2. Fear Check
        if vix > 30:
            penalty += 15
            status.append(f"High Volatility (VIX: {vix:.0f})")
        elif vix > 20:
            penalty += 5
            status.append(f"Elevated Volatility (VIX: {vix:.0f})")
            
        return penalty, status

    except Exception as e:
        # Fallback in case of API errors
        print(f"Macro Error: {e}")
        return 0, [f"⚠️ Macro Data Unavailable: {e}"]

# --- Main Logic ---

# 1. Get Macro Weather Report
st.subheader("🌩️ Macro Environment Check")
macro_penalty, macro_status = get_macro_environment()

col1, col2, col3 = st.columns(3)
col1.metric("Macro Penalty Score", f"-{macro_penalty}", help="Points deducted from assets due to macro risk.")
col2.write("**Risk Factors Detected:**")
if macro_status:
    for s in macro_status: col2.error(s)
else:
    col2.success("No major macro alarms detected (Curve Normal, VIX Stable).")

st.markdown("---")

# 2. Asset Analysis
st.subheader("📋 Asset Allocation Matrix")

results = []
progress = st.progress(0)

for i, (name, ticker) in enumerate(DECISION_ASSETS.items()):
    # Assets like Bonds/Gold might act as hedges, so we reduce macro penalty for them
    local_penalty = macro_penalty
    if "Bonds" in name or "Gold" in name:
        local_penalty = 0 # Safe havens don't get penalized for fear
        
    res = analyze_asset(name, ticker, local_penalty)
    if res: results.append(res)
    progress.progress((i+1)/len(DECISION_ASSETS))

progress.empty()

# 3. Display Results
if results:
    df_res = pd.DataFrame(results)
    
    # Visual Styling
    def color_verdict(val):
        color = 'white'
        if 'STRONG BUY' in val: color = '#28a745' # Dark Green
        elif 'ACCUMULATE' in val: color = '#90ee90' # Light Green
        elif 'NEUTRAL' in val: color = '#ffc107' # Yellow
        elif 'SELL' in val: color = '#dc3545' # Red
        return f'background-color: {color}; color: black; font-weight: bold'

    # Show Main Table
    st.dataframe(
        df_res[['Asset', 'Price', 'Score', 'Verdict', 'Drawdown', 'RSI']].style.applymap(color_verdict, subset=['Verdict']),
        use_container_width=True,
        height=300
    )
    
    # Show Details Expander
    with st.expander("🔎 Click for Deep Dive Analysis (Why this score?)"):
        for index, row in df_res.iterrows():
            st.markdown(f"**{row['Asset']} (Score: {row['Score']})**")
            st.caption(f"Analysis: {row['Details']}")
            st.markdown("---")

# 4. Charting
st.subheader("📉 Relative Performance Chart")
df_hist = yf.download(list(DECISION_ASSETS.values()), start=start_date, progress=False)['Close']
df_hist.columns = list(DECISION_ASSETS.keys())
df_norm = df_hist / df_hist.iloc[0] * 100
st.plotly_chart(px.line(df_norm), use_container_width=True)

st.sidebar.info("Disclaimer: This tool aggregates technical indicators (RSI, MACD, SMA) and macro data (Yields, VIX). Financial markets are probabilistic, not deterministic. Always do your own research.")
