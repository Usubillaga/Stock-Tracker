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
**The Complete Picture.** This dashboard integrates **Algorithmic Trading Signals** (Trend + Momentum + Macro) with deep **Macro-Economic Data** (Yields, Debt, Sectors).
""")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
days_lookback = st.sidebar.slider("Lookback Period (Days)", 180, 365*5, 730)
start_date = datetime.now() - timedelta(days=days_lookback)

st.sidebar.markdown("---")
st.sidebar.info("""
**Signal Scoring Logic (0-100):**
* **0-40:** Bearish (Sell)
* **40-60:** Neutral (Watch)
* **60-100:** Bullish (Accumulate)

**Macro Penalty:**
Scores are automatically reduced if the Yield Curve is inverted or VIX > 20.
""")

# --- Asset Dictionaries ---

# 1. Decision Assets (The things you want to buy/sell)
DECISION_ASSETS = {
    '🇺🇸 Equities (S&P 500)': 'SPY',
    '🌍 Dev. Markets (EAFE)': 'EFA',
    '🏛 Bonds (20Y Treasury)': 'TLT',
    '🧈 Gold': 'GLD',
    '🏠 Real Estate': 'VNQ',
    '₿ Bitcoin': 'BTC-USD'
}

# 2. Macro "Weather" Assets
MACRO_ASSETS = {
    '10Y Yield': '^TNX',
    '5Y Yield': '^FVX',
    'Volatility (VIX)': '^VIX'
}

# 3. Market Sectors
SECTORS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Healthcare': 'XLV',
    'Energy': 'XLE', 'Consumer Disc.': 'XLY', 'Consumer Staples': 'XLP',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Industrials': 'XLI',
    'Real Estate': 'XLRE', 'Comm. Services': 'XLC'
}

# 4. Leading Indicators
LEADING_INDICATORS = {
    'Copper (Economy)': 'HG=F',
    'Gold (Fear)': 'GC=F',
    'Semiconductors (Tech)': 'SMH',
    'Staples (Defensive)': 'XLP'
}

# 5. Global Indexes
GLOBAL_INDEXES = {
    'S&P 500 (USA)': '^GSPC', 'DAX (Germany)': '^GDAXI', 
    'FTSE 100 (UK)': '^FTSE', 'Nikkei 225 (Japan)': '^N225', 
    'Shanghai (China)': '000001.SS'
}

# --- Core Functions ---

@st.cache_data
def fetch_price_history(tickers, start_date):
    """Fetches simple closing price history for plotting."""
    try:
        # yfinance download
        data = yf.download(list(tickers.values()), start=start_date, progress=False)['Close']
        
        # Check if we got a single column (Series) or multiple (DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = list(tickers.values())
            
        # Rename columns to friendly names
        reverse_map = {v: k for k, v in tickers.items()}
        data.rename(columns=reverse_map, inplace=True)
        return data
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_advanced_data(ticker):
    """Fetches data and calculates indicators (RSI, MACD, Drawdown)."""
    try:
        # Buffer start date for moving averages
        buffer_date = start_date - timedelta(days=200)
        df = yf.download(ticker, start=buffer_date, progress=False)
        
        if df.empty: return None
        
        # Ensure 1D structure
        df = df['Close'].to_frame() if isinstance(df['Close'], pd.Series) else df[['Close']]
        df.columns = ['Close']
        
        # 1. SMAs
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 2. RSI
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
    """Generates a score (0-100) based on Technicals + Macro Penalty."""
    df = fetch_advanced_data(ticker)
    if df is None: return None
    
    current = df.iloc[-1]
    
    score = 50 # Start Neutral
    reasons = []
    
    # 1. Trend (40%)
    if current['Close'] > current['SMA_200']:
        score += 20
        reasons.append("Bullish Trend (>200 SMA)")
    else:
        score -= 20
        reasons.append("Bearish Trend (<200 SMA)")
        
    if current['Close'] > current['SMA_50']:
        score += 10
    else:
        score -= 10
    
    # 2. RSI (20%)
    rsi = current['RSI']
    if rsi < 30:
        score += 10
        reasons.append("Oversold (Value?)")
    elif rsi > 70:
        score -= 10
        reasons.append("Overbought (Risk)")
        
    # 3. MACD (20%)
    if current['MACD'] > current['Signal_Line']:
        score += 10
        reasons.append("MACD Bullish")
    else:
        score -= 10
        
    # 4. Macro Penalty (20%)
    if macro_penalty > 0:
        score -= macro_penalty
        reasons.append(f"Macro Headwind (-{macro_penalty})")
        
    # Cap Score
    score = max(0, min(100, score))
    
    # Verdict
    if score >= 75: verdict = "STRONG BUY 🚀"
    elif score >= 60: verdict = "BUY 🟢"
    elif score >= 40: verdict = "HOLD 🟡"
    elif score >= 25: verdict = "SELL 🔴"
    else: verdict = "STRONG SELL 📉"
    
    return {
        "Asset": asset_name,
        "Price": current['Close'],
        "Score": score,
        "Verdict": verdict,
        "Drawdown": f"{current['Drawdown']*100:.2f}%",
        "RSI": f"{rsi:.1f}",
        "Details": ", ".join(reasons)
    }

def get_macro_environment():
    """Analyzes Yield Curve and VIX to determine market penalty."""
    try:
        # Fetch Data - use .iloc[-1] and float() to ensure scalars
        ten = yf.download(MACRO_ASSETS['10Y Yield'], period="5d", progress=False)['Close']
        five = yf.download(MACRO_ASSETS['5Y Yield'], period="5d", progress=False)['Close']
        vix = yf.download(MACRO_ASSETS['Volatility (VIX)'], period="5d", progress=False)['Close']
        
        if ten.empty or five.empty or vix.empty:
            return 0, ["⚠️ Missing Macro Data"]
            
        ten_val = float(ten.iloc[-1])
        five_val = float(five.iloc[-1])
        vix_val = float(vix.iloc[-1])
        
        penalty = 0
        status = []
        
        # Yield Curve Logic
        if five_val > ten_val:
            penalty += 15
            status.append(f"Inverted Yield Curve (Recession Risk)")
        
        # VIX Logic
        if vix_val > 30:
            penalty += 15
            status.append(f"Extreme Fear (VIX: {vix_val:.0f})")
        elif vix_val > 20:
            penalty += 5
            status.append(f"Elevated Fear (VIX: {vix_val:.0f})")
            
        return penalty, status
        
    except Exception as e:
        return 0, [f"Error: {e}"]

# --- Application Layout ---

# Tab Structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚦 AI Signals (Action)", 
    "📈 Sector Rotation", 
    "💸 Yields & Macro", 
    "🔮 Leading Indicators",
    "🌍 Global View"
])

# --- TAB 1: AI SIGNALS ---
with tab1:
    st.header("🤖 Multi-Factor Investment Signals")
    
    # Macro Check
    macro_penalty, macro_status = get_macro_environment()
    
    # Display Macro Header
    col_m1, col_m2 = st.columns([1, 3])
    col_m1.metric("Market Penalty", f"-{macro_penalty}", help="Points deducted from risk assets due to macro conditions.")
    if macro_status:
        for s in macro_status: col_m2.error(s)
    else:
        col_m2.success("Macro Conditions Stable (Normal Curve, Low Volatility)")
    
    st.divider()
    
    # Run Analysis
    results = []
    progress = st.progress(0)
    for i, (name, ticker) in enumerate(DECISION_ASSETS.items()):
        # Bonds and Gold are hedges, so we ignore macro penalty for them
        local_penalty = macro_penalty if "Bonds" not in name and "Gold" not in name else 0
        
        res = analyze_asset(name, ticker, local_penalty)
        if res: results.append(res)
        progress.progress((i+1)/len(DECISION_ASSETS))
    progress.empty()
    
    # Display DataFrame
    if results:
        df_res = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        
        def color_verdict(val):
            color = 'white'
            if 'STRONG BUY' in val: color = '#28a745'
            elif 'BUY' in val: color = '#90ee90'
            elif 'HOLD' in val: color = '#ffc107'
            elif 'SELL' in val: color = '#dc3545'
            return f'background-color: {color}; color: black; font-weight: bold'

        st.dataframe(
            df_res[['Asset', 'Price', 'Score', 'Verdict', 'Drawdown', 'RSI', 'Details']].style.applymap(color_verdict, subset=['Verdict']),
            use_container_width=True,
            height=300
        )
        
    # Relative Performance Chart for these assets
    st.subheader("Asset Performance Comparison")
    hist_data = fetch_price_history(DECISION_ASSETS, start_date)
    if not hist_data.empty:
        norm_data = hist_data / hist_data.iloc[0] * 100
        st.plotly_chart(px.line(norm_data, title="Growth of $100 (Rebased)"), use_container_width=True)

# --- TAB 2: SECTORS ---
with tab2:
    st.subheader("🇺🇸 Sector Rotation Analysis")
    st.caption("Identify if money is flowing into 'Offensive' (Tech) or 'Defensive' (Utilities) sectors.")
    
    sec_data = fetch_price_history(SECTORS, start_date)
    if not sec_data.empty:
        norm_sec = sec_data / sec_data.iloc[0] * 100
        st.plotly_chart(px.line(norm_sec, title="Sector Performance (Normalized)"), use_container_width=True)

# --- TAB 3: YIELDS & DEBT ---
with tab3:
    st.subheader("💸 The Bond Market (The Truth Teller)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Yield Curve**")
        # Fetch current yields again for plotting
        tickers = list(MACRO_ASSETS.values())
        try:
            yd = yf.download(['^IRX', '^FVX', '^TNX', '^TYX'], period="5d", progress=False)['Close'].iloc[-1]
            # Manual mapping because download returns tickers as columns
            yc_data = pd.Series({
                '13 Week': float(yd['^IRX']),
                '5 Year': float(yd['^FVX']),
                '10 Year': float(yd['^TNX']),
                '30 Year': float(yd['^TYX'])
            })
            st.plotly_chart(px.line(x=yc_data.index, y=yc_data.values, markers=True, title="US Treasury Yields"), use_container_width=True)
        except:
            st.warning("Yield curve data temporarily unavailable.")

    with col2:
        st.write("**10-Year Yield Trend**")
        ten_hist = fetch_price_history({'10 Year': '^TNX'}, start_date)
        if not ten_hist.empty:
            st.plotly_chart(px.area(ten_hist, title="10-Year Yield History"), use_container_width=True)

    st.divider()
    st.subheader("🏛 Sovereign Debt Proxies")
    st.caption("Falling lines = Rising Yields (Bad for debt holders).")
    debt_proxies = {'US Gov Bonds': 'GOVT', 'Intl Bonds': 'BWX', 'Emerging Mkts': 'EMB'}
    debt_hist = fetch_price_history(debt_proxies, start_date)
    if not debt_hist.empty:
        st.plotly_chart(px.line(debt_hist / debt_hist.iloc[0] * 100), use_container_width=True)

# --- TAB 4: LEADING INDICATORS ---
with tab4:
    st.subheader("🔮 Economic Crystal Ball")
    
    leads = fetch_price_history(LEADING_INDICATORS, start_date)
    if not leads.empty:
        col_l1, col_l2 = st.columns(2)
        
        with col_l1:
            st.markdown("### Dr. Copper vs Gold")
            st.caption("Rising = Economic Expansion. Falling = Fear/Recession.")
            if 'Copper (Economy)' in leads and 'Gold (Fear)' in leads:
                leads['Ratio'] = leads['Copper (Economy)'] / leads['Gold (Fear)']
                st.plotly_chart(px.line(leads['Ratio'], title="Copper / Gold Ratio"), use_container_width=True)
                
        with col_l2:
            st.markdown("### Semi vs Staples (Risk Appetite)")
            st.caption("Rising = Investors love risk. Falling = Investors want safety.")
            if 'Semiconductors (Tech)' in leads and 'Staples (Defensive)' in leads:
                leads['Risk'] = leads['Semiconductors (Tech)'] / leads['Staples (Defensive)']
                st.plotly_chart(px.line(leads['Risk'], title="Tech / Staples Ratio"), use_container_width=True)

# --- TAB 5: GLOBAL VIEW ---
with tab5:
    st.subheader("🌍 Global Equity Markets")
    global_data = fetch_price_history(GLOBAL_INDEXES, start_date)
    if not global_data.empty:
        norm_global = global_data / global_data.iloc[0] * 100
        st.plotly_chart(px.line(norm_global, title="Global Indexes (Rebased to 100)"), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance. Analysis is algorithmic only.")
