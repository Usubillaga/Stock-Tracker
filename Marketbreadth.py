import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. Page Configuration ---
st.set_page_config(page_title="Macro Hedge Pro", layout="wide", page_icon="🏛️")

st.title("🏛️ Institutional Macro-Hedge Dashboard")
st.markdown("""
**Professional Grade Market Analysis.**
This dashboard does not summarize. It calculates raw macroeconomic scores based on asset performance to determine:
1.  **The Economic Regime:** (Inflation vs Growth)
2.  **Recession Probability:** (Labor vs Transport data)
3.  **Asset Allocation Signals:** (Technical Momentum + Macro Penalty)
""")

# --- 2. Sidebar Settings ---
st.sidebar.header("⚙️ Model Settings")
lookback_years = st.sidebar.slider("Analysis Lookback (Years)", 1, 5, 2)
start_date = datetime.now() - timedelta(days=lookback_years*365)

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Sources & Geography:**
* **Stocks (SPY):** US Large Cap (S&P 500)
* **Bonds (TLT/TNX):** US Treasury Market
* **Commodities (Oil/Copper):** Global Markets (Priced in USD)
* **Labor/GDP:** US Domestic Proxies (Transports/Staffing)

*All analysis is US-Economy Centric.*
""")

# --- 3. Asset Universes ---

# Core Portfolio for Scoring
DECISION_ASSETS = {
    '🇺🇸 US Equities (SPY)': 'SPY',
    '🌍 Developed Mkts (EFA)': 'EFA',
    '🏛 US 20Y Bonds (TLT)': 'TLT',
    '🧈 Gold (Bullion)': 'GLD',
    '🏠 US Real Estate (VNQ)': 'VNQ',
    '₿ Bitcoin (USD)': 'BTC-USD',
    '🛢️ Crude Oil (WTI)': 'CL=F'
}

# Macro Indicators for Regime Detection
REGIME_INDICATORS = {
    'Growth_Equity': 'SPY',     # Proxy: US Corporate Earnings Expectations
    'Growth_Industrial': 'HG=F',# Proxy: Global Manufacturing Demand (Copper)
    'Inflation_Energy': 'CL=F', # Proxy: Cost Push Inflation (Oil)
    'Inflation_Rates': '^TNX'   # Proxy: Market Inflation Expectations (10Y Yield)
}

# Breadth & Internal Health
BREADTH_ASSETS = {
    'S&P 500 (Cap Weighted)': 'SPY',
    'S&P 500 (Equal Weighted)': 'RSP',
    'High Beta (Risk On)': 'SPHB',
    'Low Volatility (Risk Off)': 'SPLV'
}

# Sectors
SECTORS = {
    'Technology (Growth)': 'XLK', 'Financials (Cyclical)': 'XLF', 'Healthcare (Defensive)': 'XLV',
    'Energy (Inflation)': 'XLE', 'Cons. Staples (Defensive)': 'XLP', 'Utilities (Bond-Proxy)': 'XLU'
}

# Recession Proxies (GDP & Labor)
RECESSION_PROXIES = {
    'GDP_Transports': 'IYT',       # Moving Goods = Real Economy
    'GDP_SmallCaps': 'IWM',        # Domestic Economy Sensitivity
    'Labor_Staffing': 'RHI',       # Robert Half (Temp Agencies lead hiring/firing)
    'Cons_Discretionary': 'XLY',   # Consumer Confidence
    'Cons_Staples': 'XLP'          # Consumer Fear
}

# --- 4. Core Calculation Engines ---

@st.cache_data
def fetch_history(tickers_dict, start_date):
    """Robust data fetcher."""
    tickers = list(tickers_dict.values())
    try:
        data = yf.download(tickers, start=start_date, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
        rev_map = {v: k for k, v in tickers_dict.items()}
        data.rename(columns=rev_map, inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()

def determine_economic_regime_detailed():
    """
    Detailed calculation of the Macro Regime.
    Returns the specific contribution of each asset to the score.
    """
    try:
        tickers = list(REGIME_INDICATORS.values())
        # We need 1 year of data to calculate the 6-month (126 day) ROC smoothing
        df = yf.download(tickers, period="1y", progress=False)['Close']
        
        # Calculate 6-month Rate of Change (ROC)
        # This filters out short-term noise and captures the 'Cycle' trend
        roc = df.pct_change(126).iloc[-1]
        
        # 1. Growth Component (50% US Equities, 50% Global Copper)
        g_equity_contrib = roc[REGIME_INDICATORS['Growth_Equity']]
        g_metal_contrib = roc[REGIME_INDICATORS['Growth_Industrial']]
        growth_score = (g_equity_contrib + g_metal_contrib) / 2
        
        # 2. Inflation Component (50% Oil Cost, 50% Bond Yields)
        i_energy_contrib = roc[REGIME_INDICATORS['Inflation_Energy']]
        i_yield_contrib = roc[REGIME_INDICATORS['Inflation_Rates']]
        inflation_score = (i_energy_contrib + i_yield_contrib) / 2
        
        # 3. Regime Identification & Deep Interpretation
        if growth_score > 0 and inflation_score > 0:
            regime = "REFLATIONARY BOOM (Inflationary Growth) 🔥"
            desc = """
            **Economic Logic:** Demand is expanding faster than supply. Companies are growing earnings, but input costs (Oil/Labor) are rising.
            \n**Implication:** Bond yields rise to fight inflation, hurting long-duration assets (Tech/Bonds). Value stocks and Real Assets outperform.
            \n**Key Trades:** Short Bonds, Long Commodities, Long Industrials/Financials.
            """
        elif growth_score < 0 and inflation_score > 0:
            regime = "STAGFLATION (The Danger Zone) ⚠️"
            desc = """
            **Economic Logic:** The worst case. Growth is slowing (Recession risk) but prices remain high. Central Banks are trapped; they cannot cut rates without worsening inflation.
            \n**Implication:** Stock/Bond correlations turn positive (both fall together). Corporate margins collapse due to high costs and low volume.
            \n**Key Trades:** Cash (King), Gold (Hedge), Energy (Source of the problem). Avoid Tech and Consumer Discretionary.
            """
        elif growth_score < 0 and inflation_score < 0:
            regime = "DEFLATIONARY RECESSION (Bust) ❄️"
            desc = """
            **Economic Logic:** Demand collapse. Liquidity dries up. Earnings crash. Unemployment rises, forcing prices down.
            \n**Implication:** Interest rates collapse as Central Banks pivot to easy money. Safe havens are bid up.
            \n**Key Trades:** Long US Treasuries (TLT), Long USD (DXY). Avoid Commodities and Leverage.
            """
        else: 
            regime = "DISINFLATIONARY BOOM (Goldilocks) ☀️"
            desc = """
            **Economic Logic:** Productivity boom. Growth is positive, but inflation is falling (supply chains healing).
            \n**Implication:** Real Yields stabilize. Central Banks can stay neutral or cut rates. Valuation multiples expand.
            \n**Key Trades:** Long Tech/Growth Stocks (Duration assets), Bitcoin. Long Corporate Bonds.
            """
            
        return {
            "Regime": regime,
            "Description": desc,
            "Total_Growth": growth_score * 100,
            "Total_Inflation": inflation_score * 100,
            "Details": {
                "US Equities (SPY)": g_equity_contrib * 100,
                "Global Copper (HG)": g_metal_contrib * 100,
                "Crude Oil (WTI)": i_energy_contrib * 100,
                "US 10Y Yields": i_yield_contrib * 100
            }
        }
    except Exception as e:
        return None

def analyze_recession_risk_detailed():
    """
    Detailed Recession Risk using Dow Theory and Labor Leading Indicators.
    """
    try:
        # Fetch Data
        proxies = list(RECESSION_PROXIES.values())
        yields = ['^TNX', '^FVX'] 
        all_tickers = proxies + yields
        
        df = yf.download(all_tickers, period="2y", progress=False)['Close']
        current = df.iloc[-1]
        ma200 = df.rolling(200).mean().iloc[-1]
        
        risks = []
        score = 0
        
        # A. Yield Curve (Banking Stress)
        curve_val = current['^TNX'] - current['^FVX']
        if curve_val < 0:
            score += 35
            risks.append(f"**Banking Credit Stress:** Yield Curve Inverted ({curve_val:.2f}%). Banks reduce lending.")
            
        # B. Labor Market (Leading Indicators)
        # RHI (Staffing) leads Non-Farm Payrolls by ~3-6 months.
        rhi_trend = "Bearish" if current[RECESSION_PROXIES['Labor_Staffing']] < ma200[RECESSION_PROXIES['Labor_Staffing']] else "Bullish"
        if rhi_trend == "Bearish":
            score += 25
            risks.append(f"**Labor Weakness:** Staffing stocks (RHI) are in a downtrend. Temporary hiring is slowing.")
            
        # C. GDP / Dow Theory
        # IYT (Transports) must confirm IWM (Small Caps)
        iyt_trend = "Bearish" if current[RECESSION_PROXIES['GDP_Transports']] < ma200[RECESSION_PROXIES['GDP_Transports']] else "Bullish"
        if iyt_trend == "Bearish":
            score += 25
            risks.append(f"**Physical GDP Contraction:** Transport stocks (IYT) are in a downtrend. Goods volume is dropping.")
            
        # D. Consumer Strength
        # Discretionary (XLY) vs Staples (XLP)
        ratio_curr = current[RECESSION_PROXIES['Cons_Discretionary']] / current[RECESSION_PROXIES['Cons_Staples']]
        ratio_ma = (df[RECESSION_PROXIES['Cons_Discretionary']] / df[RECESSION_PROXIES['Cons_Staples']]).rolling(50).mean().iloc[-1]
        
        if ratio_curr < ratio_ma:
            score += 15
            risks.append("**Consumer Fear:** Defensive Staples (XLP) are outperforming Discretionary (XLY).")
            
        # Risk Interpretation
        if score >= 65: level = "HIGH RECESSION RISK (Defensive Positioning Required) 🚨"
        elif score >= 35: level = "ELEVATED RISK (Caution) ⚠️"
        else: level = "LOW RISK (Expansionary) ✅"
        
        return level, score, risks, df
        
    except Exception as e:
        return "Error", 0, [str(e)], None

def analyze_asset_technical(ticker, asset_name):
    """
    Standard Technical Scoring.
    """
    try:
        df = yf.download(ticker, start=start_date - timedelta(days=300), progress=False)
        if df.empty: return None
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        # Metrics
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain/loss)))
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Current Values (Scalar)
        cur = float(close.iloc[-1])
        s200 = float(sma200.iloc[-1])
        r_val = float(rsi.iloc[-1])
        m_val = float(macd.iloc[-1])
        s_val = float(signal.iloc[-1])
        
        score = 50
        reasons = []
        
        # Trend
        if cur > s200: score += 20; reasons.append("Bull Trend (>200SMA)")
        else: score -= 20; reasons.append("Bear Trend (<200SMA)")
        
        # Momentum
        if r_val < 30: score += 10; reasons.append("Oversold (RSI<30)")
        elif r_val > 70: score -= 10; reasons.append("Overbought (RSI>70)")
        
        # Momentum 2
        if m_val > s_val: score += 10; reasons.append("MACD Bullish")
        else: score -= 10
        
        verdict = "BUY 🟢" if score > 60 else ("SELL 🔴" if score < 40 else "HOLD 🟡")
        
        return {"Asset": asset_name, "Price": cur, "Score": score, "Verdict": verdict, "RSI": r_val, "Details": ", ".join(reasons)}
    except: return None

# --- 5. Main Layout ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚨 Recession & Labor",
    "🧭 Economic Regime (Deep Dive)",
    "🚦 Asset Signal Matrix",
    "🌊 Market Breadth",
    "🛢️ Commodities & FX",
    "📈 Sector Rotation"
])

# --- TAB 1: RECESSION WATCH ---
with tab1:
    st.header("🚨 US Recession Risk Monitor")
    st.caption("This module triangulates recession risk using **Financial Data** (Yield Curve), **Labor Data** (Staffing Stocks), and **Physical GDP** (Transports).")
    
    risk_level, risk_score, risk_details, risk_df = analyze_recession_risk_detailed()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Recession Probability Score", f"{risk_score}/100")
        if "HIGH" in risk_level: st.error(f"### {risk_level}")
        elif "ELEVATED" in risk_level: st.warning(f"### {risk_level}")
        else: st.success(f"### {risk_level}")
            
        st.markdown("#### 🔍 Detected Risk Factors:")
        if risk_details:
            for r in risk_details:
                st.write(f"- {r}")
        else:
            st.success("No structural economic warnings detected.")

    with col2:
        if risk_df is not None:
            # Normalize Data for comparison
            norm_risk = risk_df / risk_df.iloc[0] * 100
            st.subheader("Real-Time Economic Proxies")
            st.write("Visualizing the health of **Physical GDP (IYT)** vs **Labor Demand (RHI)**.")
            
            plot_cols = [RECESSION_PROXIES['Labor_Staffing'], RECESSION_PROXIES['GDP_Transports'], RECESSION_PROXIES['GDP_SmallCaps']]
            st.plotly_chart(px.line(norm_risk[plot_cols], title="Labor & Transport Trends (Normalized)"), use_container_width=True)

# --- TAB 2: ECONOMIC REGIME (DETAILED) ---
with tab2:
    st.header("🧭 Global Economic Regime (The 4 Quadrants)")
    
    regime = determine_economic_regime_detailed()
    
    if regime:
        # Top Section: The Result
        st.info(f"### CURRENT REGIME: {regime['Regime']}")
        
        col_main, col_detail = st.columns([2, 1])
        
        with col_main:
            # Quadrant Plot
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="rgba(0,255,0,0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=-50, y0=0, x1=0, y1=50, fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=-50, y0=-50, x1=0, y1=0, fillcolor="rgba(0,0,255,0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=0, y0=-50, x1=50, y1=0, fillcolor="rgba(255,215,0,0.1)", line=dict(width=0))
            fig.add_trace(go.Scatter(x=[regime['Total_Growth']], y=[regime['Total_Inflation']], mode='markers+text', marker=dict(size=25, color='black'), text=['CURRENT'], textposition="top center"))
            fig.add_annotation(x=25, y=25, text="REFLATION", showarrow=False, font=dict(color="green"))
            fig.add_annotation(x=-25, y=25, text="STAGFLATION", showarrow=False, font=dict(color="red"))
            fig.add_annotation(x=-25, y=-25, text="DEFLATION", showarrow=False, font=dict(color="blue"))
            fig.add_annotation(x=25, y=-25, text="GOLDILOCKS", showarrow=False, font=dict(color="orange"))
            fig.update_layout(title="Economic Cycle Map", xaxis_title="Growth Impulse (Stocks + Copper)", yaxis_title="Inflation Impulse (Oil + Yields)", xaxis=dict(range=[-50, 50]), yaxis=dict(range=[-50, 50]), height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 Professional Interpretation")
            st.markdown(regime['Description'])
            
        with col_detail:
            st.subheader("📊 Score Breakdown")
            st.caption("Where did these numbers come from? (6-month Rate of Change)")
            
            st.write("**Growth Inputs:**")
            st.metric("US Equities (SPY)", f"{regime['Details']['US Equities (SPY)']:.2f}%")
            st.metric("Global Copper (HG)", f"{regime['Details']['Global Copper (HG)']:.2f}%")
            st.divider()
            st.write("**Inflation Inputs:**")
            st.metric("Crude Oil (WTI)", f"{regime['Details']['Crude Oil (WTI)']:.2f}%")
            st.metric("US 10Y Yields", f"{regime['Details']['US 10Y Yields']:.2f}%")

# --- TAB 3: SIGNALS ---
with tab3:
    st.header("🚦 Asset Signal Matrix")
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
