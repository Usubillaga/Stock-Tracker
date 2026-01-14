import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. Page Configuration ---
st.set_page_config(page_title="Global Macro Hedge Dashboard", layout="wide", page_icon="🏦")

st.title("🏦 Global Macro Hedge Dashboard")
st.markdown("""
**Institutional Grade Analysis.** Combines **Economic Regime Detection** (Stagflation/Reflation), **Market Breadth**, and **Multi-Factor Asset Scoring** into one unified view.
""")

# --- 2. Sidebar Settings ---
st.sidebar.header("⚙️ System Settings")
lookback_years = st.sidebar.slider("Analysis Lookback (Years)", 1, 5, 2)
start_date = datetime.now() - timedelta(days=lookback_years*365)

st.sidebar.markdown("---")
st.sidebar.info("""
**Regime Logic:**
* **Reflation:** Growth ↑ + Inflation ↑
* **Stagflation:** Growth ↓ + Inflation ↑
* **Deflation:** Growth ↓ + Inflation ↓
* **Goldilocks:** Growth ↑ + Inflation ↓
""")

# --- 3. Asset Universes ---

# Core Portfolio for Scoring
DECISION_ASSETS = {
    '🇺🇸 S&P 500 (Equities)': 'SPY',
    '🌍 EAFE (Dev Markets)': 'EFA',
    '🏛 20Y Treasury (Bonds)': 'TLT',
    '🧈 Gold (Precious Metal)': 'GLD',
    '🏠 Real Estate (REITs)': 'VNQ',
    '₿ Bitcoin (Crypto)': 'BTC-USD',
    '🛢️ Oil (Energy)': 'CL=F'
}

# Macro Indicators for Regime Detection
REGIME_INDICATORS = {
    'Growth (Stocks)': 'SPY',
    'Growth (Copper)': 'HG=F',
    'Inflation (Oil)': 'CL=F',
    'Inflation (Yields)': '^TNX'
}

# Breadth & Internal Health
BREADTH_ASSETS = {
    'S&P 500 (Cap Weighted)': 'SPY',
    'S&P 500 (Equal Weighted)': 'RSP',
    'High Beta (Risk)': 'SPHB',
    'Low Volatility (Safety)': 'SPLV'
}

# Sectors
SECTORS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Healthcare': 'XLV',
    'Energy': 'XLE', 'Staples': 'XLP', 'Utilities': 'XLU'
}

# --- 4. Core Calculation Engines ---

@st.cache_data
def fetch_history(tickers_dict, start_date):
    """Robust data fetcher."""
    tickers = list(tickers_dict.values())
    try:
        data = yf.download(tickers, start=start_date, progress=False)['Close']
        # Handle single ticker returning Series instead of DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
        
        # Rename cols
        rev_map = {v: k for k, v in tickers_dict.items()}
        data.rename(columns=rev_map, inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()

def determine_economic_regime():
    """
    Calculates the Macro Regime quadrant based on 6-month trends 
    of Growth Assets vs Inflation Assets.
    """
    try:
        # Download data for calculations
        tickers = list(REGIME_INDICATORS.values())
        df = yf.download(tickers, period="1y", progress=False)['Close']
        
        # Calculate 6-month (126 trading days) return
        ret = df.pct_change(126).iloc[-1]
        
        # 1. Growth Score (Stocks + Copper)
        growth_score = (ret[REGIME_INDICATORS['Growth (Stocks)']] + ret[REGIME_INDICATORS['Growth (Copper)']]) / 2
        
        # 2. Inflation Score (Oil + 10Y Yields)
        # Note: Rising yields usually imply inflation expectations
        inflation_score = (ret[REGIME_INDICATORS['Inflation (Oil)']] + ret[REGIME_INDICATORS['Inflation (Yields)']]) / 2
        
        # 3. Determine Quadrant
        if growth_score > 0 and inflation_score > 0:
            regime = "INFLATIONARY BOOM (Reflation) 🔥"
            desc = "Strong Growth + Rising Prices. (Favors: Commodities, Value Stocks, Real Estate)"
            quadrant = 1
        elif growth_score < 0 and inflation_score > 0:
            regime = "STAGFLATION (Danger Zone) ⚠️"
            desc = "Falling Growth + High Inflation. Hardest environment. (Favors: Gold, Cash, Energy)"
            quadrant = 2
        elif growth_score < 0 and inflation_score < 0:
            regime = "DEFLATIONARY CRISIS (Recession) ❄️"
            desc = "Everything falls. (Favors: Govt Bonds, USD, Cash)"
            quadrant = 3
        else: # Growth > 0, Inflation < 0
            regime = "DISINFLATIONARY BOOM (Goldilocks) ☀️"
            desc = "Ideal scenario. Growth without price pressure. (Favors: Tech, Growth Stocks, Crypto)"
            quadrant = 4
            
        return {
            "Regime": regime,
            "Description": desc,
            "Growth_Val": growth_score * 100,
            "Inflation_Val": inflation_score * 100,
            "Quadrant": quadrant
        }
    except Exception as e:
        return None

def analyze_asset_technical(ticker, asset_name):
    """
    Generates a Multi-Factor Score (0-100) based on Trend, RSI, and MACD.
    """
    try:
        # Need enough data for 200 SMA
        df = yf.download(ticker, start=start_date - timedelta(days=300), progress=False)
        if df.empty: return None
        
        # Standardize to 1D Series
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        # Indicators
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Get latest values using .iloc[-1] and float() for safety
        current_price = float(close.iloc[-1])
        val_sma50 = float(sma50.iloc[-1])
        val_sma200 = float(sma200.iloc[-1])
        val_rsi = float(rsi.iloc[-1])
        val_macd = float(macd.iloc[-1])
        val_sig = float(signal.iloc[-1])
        
        # SCORING LOGIC
        score = 50 # Neutral Start
        reasons = []
        
        # Trend (40pts)
        if current_price > val_sma200:
            score += 20
            reasons.append("Bullish Trend (>200SMA)")
        else:
            score -= 20
            reasons.append("Bearish Trend (<200SMA)")
            
        if current_price > val_sma50: score += 10
        else: score -= 10
            
        # Momentum RSI (20pts)
        if val_rsi < 30: 
            score += 10
            reasons.append("Oversold")
        elif val_rsi > 70: 
            score -= 10
            reasons.append("Overbought")
            
        # MACD (20pts)
        if val_macd > val_sig:
            score += 10
            reasons.append("MACD Bullish")
        else:
            score -= 10
            
        # Score Cap
        score = max(0, min(100, score))
        
        # Verdict
        if score >= 75: verdict = "STRONG BUY 🚀"
        elif score >= 60: verdict = "ACCUMULATE 🟢"
        elif score >= 40: verdict = "HOLD 🟡"
        elif score >= 25: verdict = "REDUCE 🔴"
        else: verdict = "STRONG SELL 📉"
        
        return {
            "Asset": asset_name,
            "Price": current_price,
            "Score": score,
            "Verdict": verdict,
            "RSI": val_rsi,
            "Details": ", ".join(reasons)
        }
    except Exception as e:
        return None

# --- 5. Main Layout ---

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧭 Economic Regime",
    "🚦 Asset Signals",
    "🌊 Market Breadth",
    "🛢️ Macro & FX",
    "📈 Sectors & Yields"
])

# --- TAB 1: ECONOMIC REGIME (The "Big Picture") ---
with tab1:
    st.header("Global Economic Cycle Diagnosis")
    st.caption("We cross-reference Growth assets (SPY, Copper) against Inflation assets (Oil, Yields) to find our position in the cycle.")
    
    regime = determine_economic_regime()
    
    if regime:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(f"### CURRENT STATE: \n# {regime['Regime']}")
            st.write(f"**Interpretation:** {regime['Description']}")
            st.metric("Growth Score", f"{regime['Growth_Val']:.2f}", delta="Expansion" if regime['Growth_Val']>0 else "Contraction")
            st.metric("Inflation Score", f"{regime['Inflation_Val']:.2f}", delta="Rising Prices" if regime['Inflation_Val']>0 else "Falling Prices")

        with col2:
            # QUADRANT CHART
            fig = go.Figure()
            
            # Draw Quadrants
            fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="rgba(0,255,0,0.1)", line=dict(width=0)) # Top Right (Reflation)
            fig.add_shape(type="rect", x0=-50, y0=0, x1=0, y1=50, fillcolor="rgba(255,0,0,0.1)", line=dict(width=0)) # Top Left (Stagflation)
            fig.add_shape(type="rect", x0=-50, y0=-50, x1=0, y1=0, fillcolor="rgba(0,0,255,0.1)", line=dict(width=0)) # Bottom Left (Deflation)
            fig.add_shape(type="rect", x0=0, y0=-50, x1=50, y1=0, fillcolor="rgba(255,215,0,0.1)", line=dict(width=0)) # Bottom Right (Goldilocks)

            # Plot Current Position
            fig.add_trace(go.Scatter(
                x=[regime['Growth_Val']], y=[regime['Inflation_Val']],
                mode='markers+text', marker=dict(size=25, color='black'),
                text=['YOU ARE HERE'], textposition="top center"
            ))
            
            # Labels
            fig.add_annotation(x=25, y=25, text="REFLATION (Boom)", showarrow=False, font=dict(color="green", size=14))
            fig.add_annotation(x=-25, y=25, text="STAGFLATION (Risk)", showarrow=False, font=dict(color="red", size=14))
            fig.add_annotation(x=-25, y=-25, text="DEFLATION (Crash)", showarrow=False, font=dict(color="blue", size=14))
            fig.add_annotation(x=25, y=-25, text="GOLDILOCKS (Ideal)", showarrow=False, font=dict(color="orange", size=14))
            
            fig.update_layout(
                title="Economic Regime Map",
                xaxis_title="Growth Impulse", yaxis_title="Inflation Impulse",
                xaxis=dict(range=[-50, 50], zeroline=True), 
                yaxis=dict(range=[-50, 50], zeroline=True),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SIGNALS ---
with tab2:
    st.header("🤖 AI Asset Scoring Matrix")
    st.caption("Scores (0-100) are generated using Trend, RSI, and MACD logic.")
    
    results = []
    progress = st.progress(0)
    for i, (name, ticker) in enumerate(DECISION_ASSETS.items()):
        res = analyze_asset_technical(ticker, name)
        if res: results.append(res)
        progress.progress((i+1)/len(DECISION_ASSETS))
    progress.empty()
    
    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
        
        # Color Logic
        def color_verdict(val):
            color = 'white'
            if 'BUY' in val: color = '#d4edda; color: green'
            elif 'SELL' in val or 'REDUCE' in val: color = '#f8d7da; color: red'
            elif 'HOLD' in val: color = '#fff3cd; color: #856404'
            return f'background-color: {color}; font-weight: bold'

        st.dataframe(
            df_res[['Asset', 'Price', 'Score', 'Verdict', 'RSI', 'Details']].style.applymap(color_verdict, subset=['Verdict']),
            use_container_width=True, height=400
        )

# --- TAB 3: BREADTH ---
with tab3:
    st.header("Internal Market Health")
    
    # Fetch Data
    breadth_df = fetch_history(BREADTH_ASSETS, start_date)
    
    if not breadth_df.empty:
        # 1. Breadth Ratio (RSP vs SPY)
        breadth_df['Breadth Ratio'] = breadth_df['S&P 500 (Equal Weighted)'] / breadth_df['S&P 500 (Cap Weighted)']
        
        # MA for Breadth
        breadth_df['Breadth_MA'] = breadth_df['Breadth Ratio'].rolling(50).mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Breadth (RSP/SPY)")
            st.caption("Rising = Broad participation (Healthy). Falling = Narrow participation (Risk).")
            
            curr_b = breadth_df['Breadth Ratio'].iloc[-1]
            ma_b = breadth_df['Breadth_MA'].iloc[-1]
            
            if curr_b > ma_b:
                st.success("✅ Market Breadth is Healthy (Small caps participating)")
            else:
                st.error("❌ Market Breadth is Weak (Rally driven by few stocks)")
                
            fig_b = px.line(breadth_df[['Breadth Ratio', 'Breadth_MA']], title="Breadth Ratio Trend")
            st.plotly_chart(fig_b, use_container_width=True)
            
        with col2:
            st.subheader("Risk Appetite (High Beta vs Low Vol)")
            breadth_df['Risk Ratio'] = breadth_df['High Beta (Risk)'] / breadth_df['Low Volatility (Safety)']
            st.caption("Rising = Investors are chasing risk. Falling = Investors are defensive.")
            fig_r = px.line(breadth_df['Risk Ratio'], title="Risk Appetite Ratio")
            st.plotly_chart(fig_r, use_container_width=True)

# --- TAB 4: MACRO & FX ---
with tab4:
    st.header("Commodities & Currency Analysis")
    
    # Custom Fetch for macro
    macro_tickers = {'Oil (WTI)': 'CL=F', 'US Dollar (DXY)': 'DX-Y.NYB', 'Copper': 'HG=F', 'Gold': 'GC=F'}
    macro_df = fetch_history(macro_tickers, start_date)
    
    if not macro_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛢️ Oil Trend (Inflationary Force)")
            # Simple Trend Check
            last_oil = macro_df['Oil (WTI)'].iloc[-1]
            oil_ma = macro_df['Oil (WTI)'].rolling(200).mean().iloc[-1]
            
            if last_oil > oil_ma:
                st.warning(f"Oil is in an Uptrend (${last_oil:.2f}). Inflationary pressure.")
            else:
                st.success(f"Oil is Contained (${last_oil:.2f}). Disinflationary.")
            st.plotly_chart(px.line(macro_df['Oil (WTI)']), use_container_width=True)
            
        with col2:
            st.subheader("💵 US Dollar Strength")
            last_dxy = macro_df['US Dollar (DXY)'].iloc[-1]
            if last_dxy > 105:
                st.error(f"Dollar is Strong ({last_dxy:.2f}). Headwind for assets.")
            else:
                st.info(f"Dollar is Neutral/Weak ({last_dxy:.2f}). Tailwind for assets.")
            st.plotly_chart(px.line(macro_df['US Dollar (DXY)']), use_container_width=True)

        st.divider()
        st.subheader("Dr. Copper vs Gold Ratio")
        st.caption("Rising = Economic Expansion. Falling = Economic Fear.")
        macro_df['Copper/Gold'] = macro_df['Copper'] / macro_df['Gold']
        st.plotly_chart(px.line(macro_df['Copper/Gold']), use_container_width=True)

# --- TAB 5: SECTORS ---
with tab5:
    st.header("Sector Rotation & Yields")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sector Performance (Rebased)")
        sec_df = fetch_history(SECTORS, start_date)
        if not sec_df.empty:
            norm_sec = sec_df / sec_df.iloc[0] * 100
            st.plotly_chart(px.line(norm_sec), use_container_width=True)
            
    with col2:
        st.subheader("Yield Curve (Proxy)")
        try:
            # Quick yield snapshot
            y_tickers = ['^IRX', '^FVX', '^TNX', '^TYX']
            y_data = yf.download(y_tickers, period="5d", progress=False)['Close'].iloc[-1]
            
            # Map Series to proper labels
            yields = pd.Series({
                '13 Week': float(y_data['^IRX']),
                '5 Year': float(y_data['^FVX']),
                '10 Year': float(y_data['^TNX']),
                '30 Year': float(y_data['^TYX'])
            })
            st.plotly_chart(px.line(x=yields.index, y=yields.values, markers=True, title="Current US Treasury Yields"), use_container_width=True)
        except:
            st.write("Yield Data Unavailable")
