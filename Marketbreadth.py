import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Global Market Macro Dashboard", layout="wide")
st.title("📊 Global Market & Macro Dashboard")
st.markdown("""
This dashboard compares financial sectors, yield curves, sovereign debt proxies, 
leading indicators, and major global indexes using real-time data from Yahoo Finance.
""")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
days_lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=365*5, value=365)
start_date = datetime.now() - timedelta(days=days_lookback)

# --- Data Dictionaries (Tickers) ---
SECTORS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Healthcare': 'XLV',
    'Energy': 'XLE', 'Consumer Disc.': 'XLY', 'Consumer Staples': 'XLP',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Industrials': 'XLI',
    'Real Estate': 'XLRE', 'Comm. Services': 'XLC'
}

# Treasury Yield Tickers (CBOE Interest Rate 10-Year T-Note, etc.)
# Note: Yahoo uses ^IRX (13 wk), ^FVX (5 yr), ^TNX (10 yr), ^TYX (30 yr)
YIELDS = {
    '13 Week': '^IRX', 
    '5 Year': '^FVX', 
    '10 Year': '^TNX', 
    '30 Year': '^TYX'
}

# Major Country Indexes
INDEXES = {
    'S&P 500 (USA)': '^GSPC', 
    'Nasdaq 100 (USA)': '^NDX', 
    'DAX (Germany)': '^GDAXI', 
    'FTSE 100 (UK)': '^FTSE', 
    'Nikkei 225 (Japan)': '^N225', 
    'Shanghai Comp (China)': '000001.SS'
}

# Proxies for "Debt" and "Leading Indicators"
# Since raw Debt $ amounts aren't on YF, we use Sovereign Bond ETFs to track debt market health.
DEBT_PROXIES = {
    'US Treasury Bond ETF': 'GOVT',
    'Intl Treasury Bond ETF': 'BWX',
    'Emerging Markets Bond ETF': 'EMB',
    'German Bunds (Proxy)': 'BUNL', # sometimes tricky, sticking to ETFs
    'Japan Govt Bonds (Proxy)': 'ISJG.L' # often delayed, using broad ETFs is safer
}

# Market-based Leading Indicators
# Copper is Dr. Copper (Economic health), Gold (Fear), Semis (Tech Cycle)
LEADING_INDICATORS = {
    'Copper (Economic Activity)': 'HG=F',
    'Gold (Safe Haven)': 'GC=F',
    'Baltic Dry Index (Shipping)': 'BDI', # May not fetch on YF free tier sometimes
    'Semiconductors (Tech Cycle)': 'SMH',
    'Homebuilders (Early Cycle)': 'XHB'
}

# --- Helper Functions ---
@st.cache_data
def fetch_data(tickers_dict, start_date):
    """
    Fetches historical closing prices for a dictionary of tickers.
    """
    tickers_list = list(tickers_dict.values())
    try:
        data = yf.download(tickers_list, start=start_date, progress=False)['Close']
        # Rename columns to friendly names
        reverse_map = {v: k for k, v in tickers_dict.items()}
        data.rename(columns=reverse_map, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_current_yields():
    """
    Fetches the most recent yield data for the curve.
    """
    tickers = list(YIELDS.values())
    try:
        # Fetch last 5 days to ensure we get a valid close
        data = yf.download(tickers, period="5d", progress=False)['Close']
        latest = data.iloc[-1]
        
        # Map back to readable names
        reverse_map = {v: k for k, v in YIELDS.items()}
        latest = latest.rename(index=reverse_map)
        
        # Sort by duration for the curve plot
        order = ['13 Week', '5 Year', '10 Year', '30 Year']
        latest = latest.reindex(order)
        return latest
    except Exception as e:
        st.error(f"Error fetching yields: {e}")
        return pd.Series()

# --- Main App Logic ---

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Sectors", 
    "💸 Yield Curves", 
    "🏛️ National Debt (Proxies)", 
    "🔮 Leading Indicators", 
    "🌍 Global Indexes"
])

# 1. Market Sectors
with tab1:
    st.subheader("US Sector Performance (Normalized)")
    st.caption("Comparing how different sectors perform relative to each other (Rebased to 100).")
    
    sector_data = fetch_data(SECTORS, start_date)
    if not sector_data.empty:
        # Normalize data to start at 100
        normalized_sectors = sector_data / sector_data.iloc[0] * 100
        fig_sectors = px.line(normalized_sectors, title="Sector Rotation")
        st.plotly_chart(fig_sectors, use_container_width=True)
    else:
        st.write("No sector data available.")

# 2. Yield Curves
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current US Treasury Yield Curve")
        current_yields = fetch_current_yields()
        if not current_yields.empty:
            fig_curve = px.line(x=current_yields.index, y=current_yields.values, markers=True, 
                                labels={'x': 'Maturity', 'y': 'Yield (%)'}, title="Yield Curve Shape")
            st.plotly_chart(fig_curve, use_container_width=True)
            
            # Curve Inversion Check
            diff_10_2 = current_yields.get('10 Year', 0) - current_yields.get('5 Year', 0) # Using 5y as proxy if 2y not available
            st.info(f"Curve Slope (10Y - 5Y): {diff_10_2:.2f}% (Negative often signals recession)")

    with col2:
        st.subheader("Historical 10-Year Yield")
        yield_hist = fetch_data({'10 Year Treasury': '^TNX'}, start_date)
        if not yield_hist.empty:
            fig_10y = px.area(yield_hist, title="10-Year Yield Trend")
            st.plotly_chart(fig_10y, use_container_width=True)

# 3. National Debt
with tab3:
    st.subheader("Sovereign Debt Market Performance")
    st.caption("Since real-time debt totals aren't traded, we track **Government Bond ETFs**. Falling prices often indicate rising yields/inflation or lower confidence in debt.")
    
    debt_data = fetch_data(DEBT_PROXIES, start_date)
    if not debt_data.empty:
        norm_debt = debt_data / debt_data.iloc[0] * 100
        fig_debt = px.line(norm_debt, title="Sovereign Bond ETFs (Normalized)")
        st.plotly_chart(fig_debt, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Static Snapshot: Debt-to-GDP Ratios (Ref only)")
    # Hardcoded recent approximate data for context (since APIs for this are complex)
    debt_gdp = pd.DataFrame({
        'Country': ['Japan', 'USA', 'Italy', 'France', 'UK', 'China', 'Germany'],
        'Debt-to-GDP (%)': [260, 129, 140, 110, 100, 77, 66]
    }).sort_values('Debt-to-GDP (%)', ascending=False)
    
    fig_static_debt = px.bar(debt_gdp, x='Country', y='Debt-to-GDP (%)', color='Debt-to-GDP (%)', 
                             title="Major Economies Debt-to-GDP (Approx. Latest)")
    st.plotly_chart(fig_static_debt, use_container_width=True)

# 4. Leading Indicators
with tab4:
    st.subheader("Market-Based Leading Indicators")
    st.caption("Comparison of assets often used to predict economic direction.")
    
    # 1. Copper vs Gold Ratio (Risk Sentiment)
    raw_leads = fetch_data(LEADING_INDICATORS, start_date)
    
    if not raw_leads.empty:
        col_lead1, col_lead2 = st.columns(2)
        
        with col_lead1:
            st.write("### Dr. Copper vs Gold")
            # Calculate Ratio
            if 'Copper (Economic Activity)' in raw_leads.columns and 'Gold (Safe Haven)' in raw_leads.columns:
                raw_leads['Copper/Gold Ratio'] = raw_leads['Copper (Economic Activity)'] / raw_leads['Gold (Safe Haven)']
                fig_ratio = px.line(raw_leads['Copper/Gold Ratio'], title="Copper/Gold Ratio (Rising = Growth, Falling = Fear)")
                st.plotly_chart(fig_ratio, use_container_width=True)
        
        with col_lead2:
            st.write("### Semiconductors vs Staples")
            # Fetch Semis and Staples specifically if not in raw_leads
            semi_staple = fetch_data({'Semis': 'SMH', 'Staples': 'XLP'}, start_date)
            if not semi_staple.empty:
                semi_staple['Risk Appetite'] = semi_staple['Semis'] / semi_staple['Staples']
                fig_risk = px.line(semi_staple['Risk Appetite'], title="Tech vs Staples Ratio (Risk Appetite)")
                st.plotly_chart(fig_risk, use_container_width=True)

# 5. Global Indexes
with tab5:
    st.subheader("Major Global Equity Indices")
    index_data = fetch_data(INDEXES, start_date)
    if not index_data.empty:
        norm_indexes = index_data / index_data.iloc[0] * 100
        fig_indexes = px.line(norm_indexes, title="Global Market Performance (Normalized)")
        st.plotly_chart(fig_indexes, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Data provided by Yahoo Finance. Note that 'Debt' is visualized via Bond ETF performance proxies.")
