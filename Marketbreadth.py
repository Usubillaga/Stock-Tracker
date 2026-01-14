import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Global Macro & Signal Engine", layout="wide", page_icon="🌍")

st.title("🌍 Global Macro & Investment Signal Engine")
st.markdown("""
**One Dashboard to Rule Them All.** * **Tab 1:** AI Decision Signals (Buy/Sell indicators for your core assets).
* **Tabs 2-6:** Deep Macro Data (Sectors, Yields, Debt, Indicators, Indexes).
""")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
days_lookback = st.sidebar.slider("Lookback Period (Days)", min_value=180, max_value=365*5, value=730)
start_date = datetime.now() - timedelta(days=days_lookback)

st.sidebar.markdown("---")
st.sidebar.info("**Core Assets for Decision:**\n\n* 🏠 Real Estate (VNQ)\n* 🏛 Bonds (TLT)\n* 🪙 Bitcoin (BTC)\n* 🧈 Gold (GLD)\n* 📈 Stocks (SPY)")

# --- Data Dictionaries ---

# 1. CORE ASSETS (For Decision Making)
DECISION_ASSETS = {
    'Equities (S&P 500)': 'SPY',
    'Bonds (20Y Treasury)': 'TLT',
    'Gold': 'GLD',
    'Real Estate (Immobilien)': 'VNQ',
    'Bitcoin': 'BTC-USD'
}

# 2. SECTORS
SECTORS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Healthcare': 'XLV',
    'Energy': 'XLE', 'Consumer Disc.': 'XLY', 'Consumer Staples': 'XLP',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Industrials': 'XLI',
    'Real Estate': 'XLRE', 'Comm. Services': 'XLC'
}

# 3. YIELDS
YIELDS = {
    '13 Week': '^IRX', '5 Year': '^FVX', '10 Year': '^TNX', '30 Year': '^TYX'
}

# 4. DEBT PROXIES
DEBT_PROXIES = {
    'US Treasury Bond ETF': 'GOVT',
    'Intl Treasury Bond ETF': 'BWX',
    'Emerging Markets Debt': 'EMB',
    'High Yield Corporate': 'HYG'
}

# 5. LEADING INDICATORS
LEADING_INDICATORS = {
    'Copper (Economy)': 'HG=F',
    'Gold (Fear)': 'GC=F',
    'Semiconductors (Tech)': 'SMH',
    'Homebuilders (Early Cycle)': 'XHB'
}

# 6. GLOBAL INDEXES
INDEXES = {
    'S&P 500 (USA)': '^GSPC', 'Nasdaq 100 (USA)': '^NDX', 
    'DAX (Germany)': '^GDAXI', 'FTSE 100 (UK)': '^FTSE', 
    'Nikkei 225 (Japan)': '^N225', 'Shanghai Comp (China)': '000001.SS'
}

# --- Helper Functions ---

@st.cache_data
def fetch_data(tickers_dict, start_date):
    """Fetches historical closing prices for a dictionary of tickers."""
    tickers_list = list(tickers_dict.values())
    if not tickers_list: return pd.DataFrame()
    try:
        data = yf.download(tickers_list, start=start_date, progress=False)['Close']
        # Handle single ticker result vs multiple
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers_list
            
        reverse_map = {v: k for k, v in tickers_dict.items()}
        data.rename(columns=reverse_map, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_technical_signals(ticker_symbol, asset_name):
    """Calculates SMA Golden Cross / Death Cross signals."""
    try:
        df = yf.download(ticker_symbol, period="2y", progress=False)
        if df.empty: return None
        
        # Calculate Indicators
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        latest = df.iloc[-1]
        price = latest['Close']
        sma50 = latest['SMA_50']
        sma200 = latest['SMA_200']
        
        # Logic
        if price > sma50 and price > sma200:
            trend = "Strong Uptrend 🟢"
            action = "BUY / ACCUMULATE"
            score = 2
        elif price < sma50 and price < sma200:
            trend = "Downtrend 🔴"
            action = "SELL / AVOID"
            score = -2
        elif price < sma50 and price > sma200:
            trend = "Pullback (Watch) 🟠"
            action = "HOLD / WATCH"
            score = 0
        elif price > sma50 and price < sma200:
            trend = "Recovery Attempt 🟡"
            action = "SPECULATIVE BUY"
            score = 1
        else:
            trend = "Neutral ⚪"
            action = "WAIT"
            score = 0
            
        return {
            "Asset": asset_name,
            "Price": f"{price:,.2f}",
            "Trend": trend,
            "Signal": action,
            "SMA 50": f"{sma50:,.2f}",
            "SMA 200": f"{sma200:,.2f}",
            "_score": score
        }
    except Exception as e:
        return None

@st.cache_data
def fetch_current_yields():
    """Fetches latest yields for the curve plot."""
    tickers = list(YIELDS.values())
    try:
        data = yf.download(tickers, period="5d", progress=False)['Close']
        latest = data.iloc[-1]
        reverse_map = {v: k for k, v in YIELDS.items()}
        latest = latest.rename(index=reverse_map)
        order = ['13 Week', '5 Year', '10 Year', '30 Year']
        return latest.reindex(order)
    except:
        return pd.Series()

# --- Main Layout ---

# Define the tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚦 Signals (Decision)", 
    "📈 Market Sectors", 
    "💸 Yield Curves", 
    "🏛️ Debt & Bonds", 
    "🔮 Leading Indicators", 
    "🌍 Global Indexes"
])

# --- TAB 1: DECISION SIGNALS ---
with tab1:
    st.header("🤖 AI Technical Analysis (Buy/Sell Logic)")
    st.caption("Based on SMA 50/200 Crossovers. 'Green' signals imply strong momentum.")
    
    # Calculate Signals
    results = []
    progress_bar = st.progress(0)
    for i, (asset_name, ticker) in enumerate(DECISION_ASSETS.items()):
        res = calculate_technical_signals(ticker, asset_name)
        if res: results.append(res)
        progress_bar.progress((i + 1) / len(DECISION_ASSETS))
    progress_bar.empty()

    # Display Table
    if results:
        sig_df = pd.DataFrame(results)
        sig_df.set_index('Asset', inplace=True)
        sig_df = sig_df.sort_values(by="_score", ascending=False)
        
        # Coloring function
        def color_signal(val):
            color = ''
            if 'BUY' in val: color = 'background-color: #d4edda; color: green' # Green
            elif 'SELL' in val: color = 'background-color: #f8d7da; color: red' # Red
            elif 'WATCH' in val or 'HOLD' in val: color = 'background-color: #fff3cd; color: #856404' # Yellow
            return color

        display_cols = ['Price', 'Trend', 'Signal', 'SMA 50', 'SMA 200']
        st.dataframe(sig_df[display_cols].style.applymap(color_signal, subset=['Signal']), use_container_width=True)
    
    # Charting the Decision Assets
    st.subheader("Performance of Decision Assets")
    decision_data = fetch_data(DECISION_ASSETS, start_date)
    if not decision_data.empty:
        norm_decision = decision_data / decision_data.iloc[0] * 100
        fig_dec = px.line(norm_decision, title="Relative Performance (Rebased to 100)")
        st.plotly_chart(fig_dec, use_container_width=True)

# --- TAB 2: SECTORS ---
with tab2:
    st.subheader("Sector Rotation")
    st.write("Compare defensive sectors (Utilities, Staples) vs Growth (Tech, Discretionary).")
    sector_data = fetch_data(SECTORS, start_date)
    if not sector_data.empty:
        norm_sectors = sector_data / sector_data.iloc[0] * 100
        fig_sec = px.line(norm_sectors, title="US Sector Performance (Normalized)")
        st.plotly_chart(fig_sec, use_container_width=True)

# --- TAB 3: YIELDS ---
with tab3:
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        st.subheader("Yield Curve Shape")
        curr_yields = fetch_current_yields()
        if not curr_yields.empty:
            fig_curve = px.line(x=curr_yields.index, y=curr_yields.values, markers=True, 
                                labels={'x':'Maturity','y':'Yield %'}, title="Current US Treasury Curve")
            st.plotly_chart(fig_curve, use_container_width=True)
            # Inversion warning
            slope = curr_yields.get('10 Year', 0) - curr_yields.get('5 Year', 0)
            if slope < 0:
                st.error(f"⚠️ Curve Inverted (10Y-5Y): {slope:.2f}%. Recession signal.")
            else:
                st.success(f"✅ Normal Curve (10Y-5Y): +{slope:.2f}%.")

    with col_y2:
        st.subheader("10-Year Yield History")
        ten_y = fetch_data({'10 Year': '^TNX'}, start_date)
        if not ten_y.empty:
            st.plotly_chart(px.area(ten_y, title="10-Year Treasury Yield Trend"), use_container_width=True)

# --- TAB 4: DEBT ---
with tab4:
    st.subheader("Sovereign Debt Health (via Bond ETFs)")
    st.caption("Lower ETF prices = Higher Yields = Higher Cost of Debt for Govts.")
    debt_data = fetch_data(DEBT_PROXIES, start_date)
    if not debt_data.empty:
        norm_debt = debt_data / debt_data.iloc[0] * 100
        st.plotly_chart(px.line(norm_debt, title="Bond Market Performance"), use_container_width=True)
    
    st.markdown("### 📊 Debt-to-GDP Context (Approx Static Data)")
    debt_static = pd.DataFrame({
        'Country': ['Japan', 'Italy', 'USA', 'France', 'UK', 'China', 'Germany'],
        'Debt/GDP (%)': [263, 144, 123, 110, 104, 83, 66]
    }).sort_values('Debt/GDP (%)', ascending=False)
    st.plotly_chart(px.bar(debt_static, x='Country', y='Debt/GDP (%)', color='Debt/GDP (%)'), use_container_width=True)

# --- TAB 5: LEADING INDICATORS ---
with tab5:
    st.subheader("Economic Crystal Ball")
    raw_leads = fetch_data(LEADING_INDICATORS, start_date)
    
    if not raw_leads.empty:
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.write("**Dr. Copper vs Gold**")
            # Creating Ratio
            if 'Copper (Economy)' in raw_leads.columns and 'Gold (Fear)' in raw_leads.columns:
                raw_leads['Copper/Gold'] = raw_leads['Copper (Economy)'] / raw_leads['Gold (Fear)']
                st.plotly_chart(px.line(raw_leads['Copper/Gold'], title="Copper/Gold Ratio (Rising = Growth)"), use_container_width=True)
        with col_l2:
            st.write("**Tech vs Staples (Risk On/Off)**")
            risk_data = fetch_data({'Semis': 'SMH', 'Staples': 'XLP'}, start_date)
            if not risk_data.empty:
                risk_data['Ratio'] = risk_data['Semis'] / risk_data['Staples']
                st.plotly_chart(px.line(risk_data['Ratio'], title="Semis/Staples Ratio (Rising = Risk On)"), use_container_width=True)

# --- TAB 6: GLOBAL INDEXES ---
with tab6:
    st.subheader("Global Equity Markets")
    idx_data = fetch_data(INDEXES, start_date)
    if not idx_data.empty:
        norm_idx = idx_data / idx_data.iloc[0] * 100
        st.plotly_chart(px.line(norm_idx, title="Global Indexes (Normalized)"), use_container_width=True)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Data source: Yahoo Finance. 'Signals' are based on Simple Moving Averages and should not be taken as financial advice.")
