import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. Page Configuration ---
st.set_page_config(page_title="Global Macro Hedge Dashboard", layout="wide", page_icon="🏦")

st.title("🏦 Global Macro Hedge Dashboard (Pro Edition)")
st.markdown("""
**Institutional Grade Analysis.** Combines **Economic Regime** (Stagflation/Reflation), **Recession Risk** (GDP/Labor Proxies), and **AI Asset Scoring** into one unified view.
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

**Recession Watch:**
* **GDP:** Monitored via Transports (IYT).
* **Work/Labor:** Monitored via Staffing Stocks (RHI).
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

# Recession Proxies (GDP & Labor)
RECESSION_PROXIES = {
    'GDP (Transports)': 'IYT',       # Dow Theory: Goods must move for GDP to grow
    'GDP (Small Caps)': 'IWM',       # Domestic US Economy
    'Labor (Staffing)': 'RHI',       # Robert Half: Leading indicator for hiring
    'Labor (Manpower)': 'MAN',       # ManpowerGroup: Global hiring trends
    'Consumer (Discretionary)': 'XLY', # Confidence
    'Consumer (Staples)': 'XLP'      # Fear
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
    """Calculates the Macro Regime quadrant."""
    try:
        tickers = list(REGIME_INDICATORS.values())
        df = yf.download(tickers, period="1y", progress=False)['Close']
        ret = df.pct_change(126).iloc[-1]
        
        growth_score = (ret[REGIME_INDICATORS['Growth (Stocks)']] + ret[REGIME_INDICATORS['Growth (Copper)']]) / 2
        inflation_score = (ret[REGIME_INDICATORS['Inflation (Oil)']] + ret[REGIME_INDICATORS['Inflation (Yields)']]) / 2
        
        if growth_score > 0 and inflation_score > 0:
            regime = "INFLATIONARY BOOM (Reflation) 🔥"
            desc = "Strong Growth + Rising Prices. (Favors: Commodities, Value Stocks)"
        elif growth_score < 0 and inflation_score > 0:
            regime = "STAGFLATION (Danger Zone) ⚠️"
            desc = "Falling Growth + High Inflation. (Favors: Gold, Cash, Energy)"
        elif growth_score < 0 and inflation_score < 0:
            regime = "DEFLATIONARY CRISIS (Recession) ❄️"
            desc = "Everything falls. (Favors: Govt Bonds, USD)"
        else: 
            regime = "DISINFLATIONARY BOOM (Goldilocks) ☀️"
            desc = "Ideal scenario. Growth without price pressure. (Favors: Tech)"
            
        return {"Regime": regime, "Description": desc, "Growth_Val": growth_score*100, "Inflation_Val": inflation_score*100}
    except Exception as e:
        return None

def analyze_recession_risk():
    """
    Calculates recession risk using Market Proxies for GDP (Transports) and Labor (Staffing).
    """
    try:
        # 1. Fetch Data
        proxies = list(RECESSION_PROXIES.values())
        yields = ['^TNX', '^FVX'] # 10Y and 5Y (Using 5Y as proxy for 2Y if 2Y unavail)
        
        all_tickers = proxies + yields
        df = yf.download(all_tickers, period="1y", progress=False)['Close']
        
        current = df.iloc[-1]
        ma200 = df.rolling(200).mean().iloc[-1]
        
        risks = []
        score = 0 # Higher score = Higher Recession Risk
        
        # --- A. YIELD CURVE (The Classic) ---
        # Note: True inversion is 10Y - 2Y. We use 5Y if 2Y not reliable on YF free tier.
        curve = current['^TNX'] - current['^FVX']
        if curve < 0:
            score += 40
            risks.append(f"Yield Curve Inverted ({curve:.2f}%) - Banking Stress Warning")
        
        # --- B. LABOR MARKET (Staffing Agencies) ---
        # Logic: If RHI is below 200SMA, companies are firing temps.
        labor_ticker = RECESSION_PROXIES['Labor (Staffing)']
        if current[labor_ticker] < ma200[labor_ticker]:
            score += 20
            risks.append("Labor Market Weakness: Staffing Stocks (RHI) in Downtrend")
            
        # --- C. GDP (Transports) ---
        # Logic: Dow Theory - If goods aren't moving, GDP isn't growing.
        gdp_ticker = RECESSION_PROXIES['GDP (Transports)']
        if current[gdp_ticker] < ma200[gdp_ticker]:
            score += 20
            risks.append("GDP Warning: Transportation Sector (IYT) in Downtrend")
            
        # --- D. CONSUMER CONFIDENCE (Discretionary vs Staples) ---
        ratio = current[RECESSION_PROXIES['Consumer (Discretionary)']] / current[RECESSION_PROXIES['Consumer (Staples)']]
        # Compare to 6 months ago
        past_ratio = df.iloc[-126][RECESSION_PROXIES['Consumer (Discretionary)']] / df.iloc[-126][RECESSION_PROXIES['Consumer (Staples)']]
        
        if ratio < past_ratio * 0.9: # Dropped 10%
            score += 20
            risks.append("Consumer Spending Crunch: Staples outperforming Luxury")
            
        # Risk Level
        if score >= 60: level = "HIGH RECESSION RISK 🚨"
        elif score >= 40: level = "MODERATE RISK ⚠️"
        else: level = "LOW RISK ✅"
        
        return level, score, risks, df
        
    except Exception as e:
        return "Error", 0, [str(e)], None

def analyze_asset_technical(ticker, asset_name):
    """Standard technical analysis (RSI/MACD/Trend)."""
    try:
        df = yf.download(ticker, start=start_date - timedelta(days=300), progress=False)
        if df.empty: return None
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain/loss)))
        
        exp12 = close.ewm(span=12, adjust=False).mean(); exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26; signal = macd.ewm(span=9, adjust=False).mean()
        
        cur = float(close.iloc[-1]); s200 = float(sma200.iloc[-1]); r_val = float(rsi.iloc[-1])
        
        score = 50
        reasons = []
        if cur > s200: score += 20; reasons.append("Bull Trend")
        else: score -= 20; reasons.append("Bear Trend")
        if r_val < 30: score += 10; reasons.append("Oversold")
        elif r_val > 70: score -= 10; reasons.append("Overbought")
        
        verdict = "BUY 🟢" if score > 60 else ("SELL 🔴" if score < 40 else "HOLD 🟡")
        
        return {"Asset": asset_name, "Price": cur, "Score": score, "Verdict": verdict, "RSI": r_val, "Details": ", ".join(reasons)}
    except: return None

# --- 5. Main Layout ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚨 Recession Watch",
    "🧭 Economic Regime",
    "🚦 Asset Signals",
    "🌊 Market Breadth",
    "🛢️ Macro & FX",
    "📈 Sectors & Yields"
])

# --- TAB 1: RECESSION WATCH (NEW) ---
with tab1:
    st.header("🚨 Recession Risk Monitor")
    st.caption("Proxies Used: **Labor** (Staffing Stocks), **GDP** (Transports), **Sentiment** (Consumer Discretionary), **Credit** (Yield Curve).")
    
    risk_level, risk_score, risk_details, risk_df = analyze_recession_risk()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Recession Probability Score", f"{risk_score}%")
        if "HIGH" in risk_level:
            st.error(f"### {risk_level}")
        elif "MODERATE" in risk_level:
            st.warning(f"### {risk_level}")
        else:
            st.success(f"### {risk_level}")
            
        st.write("#### Risk Factors Detected:")
        if risk_details:
            for r in risk_details:
                st.write(f"- {r}")
        else:
            st.write("- No major warning signs detected.")

    with col2:
        if risk_df is not None:
            # Normalize Data for comparison
            norm_risk = risk_df / risk_df.iloc[0] * 100
            
            st.subheader("Labor & GDP Proxies (Trend)")
            st.write("Watch **RHI (Labor)** and **IYT (GDP)**. If these crash, recession is likely.")
            
            # Select specific cols for cleanliness
            plot_cols = [RECESSION_PROXIES['Labor (Staffing)'], RECESSION_PROXIES['GDP (Transports)'], RECESSION_PROXIES['GDP (Small Caps)']]
            # Map back to symbols if needed, but plotting symbols is fine
            st.plotly_chart(px.line(norm_risk[plot_cols], title="Real-Time Economic Proxies (Rebased to 100)"), use_container_width=True)

# --- TAB 2: ECONOMIC REGIME ---
with tab2:
    st.header("Global Economic Cycle Diagnosis")
    regime = determine_economic_regime()
    if regime:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"### CURRENT: {regime['Regime']}")
            st.write(f"{regime['Description']}")
        with col2:
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="rgba(0,255,0,0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=-50, y0=0, x1=0, y1=50, fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=-50, y0=-50, x1=0, y1=0, fillcolor="rgba(0,0,255,0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=0, y0=-50, x1=50, y1=0, fillcolor="rgba(255,215,0,0.1)", line=dict(width=0))
            fig.add_trace(go.Scatter(x=[regime['Growth_Val']], y=[regime['Inflation_Val']], mode='markers+text', marker=dict(size=25, color='black'), text=['YOU ARE HERE'], textposition="top center"))
            fig.add_annotation(x=25, y=25, text="REFLATION", showarrow=False, font=dict(color="green"))
            fig.add_annotation(x=-25, y=25, text="STAGFLATION", showarrow=False, font=dict(color="red"))
            fig.add_annotation(x=-25, y=-25, text="DEFLATION", showarrow=False, font=dict(color="blue"))
            fig.add_annotation(x=25, y=-25, text="GOLDILOCKS", showarrow=False, font=dict(color="orange"))
            fig.update_layout(title="Economic Regime Map", xaxis_title="Growth", yaxis_title="Inflation", xaxis=dict(range=[-50, 50]), yaxis=dict(range=[-50, 50]), height=400)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: SIGNALS ---
with tab3:
    st.header("🤖 AI Asset Scoring")
    results = []
    for i, (name, ticker) in enumerate(DECISION_ASSETS.items()):
        res = analyze_asset_technical(ticker, name)
        if res: results.append(res)
    
    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
        def color_verdict(val):
            color = '#d4edda' if 'BUY' in val else ('#f8d7da' if 'SELL' in val else '#fff3cd')
            return f'background-color: {color}; color: black'
        st.dataframe(df_res[['Asset', 'Price', 'Score', 'Verdict', 'RSI', 'Details']].style.applymap(color_verdict, subset=['Verdict']), use_container_width=True)

# --- TAB 4: BREADTH ---
with tab4:
    st.header("Internal Market Health")
    breadth_df = fetch_history(BREADTH_ASSETS, start_date)
    if not breadth_df.empty:
        breadth_df['Breadth Ratio'] = breadth_df['S&P 500 (Equal Weighted)'] / breadth_df['S&P 500 (Cap Weighted)']
        breadth_df['Breadth_MA'] = breadth_df['Breadth Ratio'].rolling(50).mean()
        curr_b = breadth_df['Breadth Ratio'].iloc[-1]; ma_b = breadth_df['Breadth_MA'].iloc[-1]
        
        st.metric("Market Breadth Health", "Healthy" if curr_b > ma_b else "Weak", delta=f"Ratio: {curr_b:.4f}")
        st.plotly_chart(px.line(breadth_df[['Breadth Ratio', 'Breadth_MA']], title="RSP/SPY Ratio (Rising = Healthy)"), use_container_width=True)

# --- TAB 5: MACRO & FX ---
with tab5:
    st.header("Commodities & Currency")
    macro_tickers = {'Oil (WTI)': 'CL=F', 'US Dollar (DXY)': 'DX-Y.NYB', 'Copper': 'HG=F', 'Gold': 'GC=F'}
    macro_df = fetch_history(macro_tickers, start_date)
    if not macro_df.empty:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.line(macro_df['Oil (WTI)'], title="Oil (Inflation Input)"), use_container_width=True)
        with col2: st.plotly_chart(px.line(macro_df['US Dollar (DXY)'], title="US Dollar (Global Stress)"), use_container_width=True)

# --- TAB 6: SECTORS ---
with tab6:
    st.header("Sector Rotation")
    sec_df = fetch_history(SECTORS, start_date)
    if not sec_df.empty:
        st.plotly_chart(px.line(sec_df / sec_df.iloc[0] * 100, title="Sector Performance (Normalized)"), use_container_width=True)

