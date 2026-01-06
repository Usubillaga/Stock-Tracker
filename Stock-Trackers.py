import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import requests
from io import StringIO

# --- Page Config ---
st.set_page_config(page_title="Pro Market Scanner", layout="wide")

st.title("🌍 Pro Market Scanner (US, UK & Europe)")
st.markdown("""
**New Capabilities:**
* **Volume Analysis:** Detects buying pressure (Volume Spikes).
* **Multi-Market:** Scans S&P 500, Russell 2000 (Small Caps), FTSE (London), and DAX (Germany).
* **Extended Range:** Scan up to 150 stocks at a time.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("1. Market Selection")
market_choice = st.sidebar.selectbox(
    "Select Market / Index:",
    ["S&P 500 (US Large Cap)", "S&P 600 (US Small Cap/Russell Proxy)", "FTSE 100 (UK/London)", "DAX 40 (Germany)"]
)

st.sidebar.header("2. Scanner Settings")
# INCREASED LIMIT TO 150
batch_size = st.sidebar.slider("Stocks to Scan", 10, 150, 50, 10)
pe_threshold = st.sidebar.number_input("Max P/E Ratio", value=50, step=5)
peg_threshold = st.sidebar.slider("Max PEG Ratio", 0.5, 5.0, 1.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("3. Criteria")
upside_threshold = st.sidebar.slider("Min Analyst Upside (%)", 0, 50, 5, 5)

# --- Helper Functions ---

@st.cache_data(ttl=3600*12)
def get_tickers(market_name):
    """Scrapes Wikipedia for ticker lists based on the selected market."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    tickers = []
    
    try:
        if "S&P 500" in market_name:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
            tickers = df['Symbol'].tolist()
            
        elif "S&P 600" in market_name: # Proxy for Russell 2000 (Small Caps)
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
            df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
            tickers = df['Symbol'].tolist()
            
        elif "FTSE" in market_name: # London Stock Exchange
            url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
            df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[4] 
            tickers = [t + ".L" for t in df['Ticker'].tolist()]
            
        elif "DAX" in market_name: # Germany
            url = 'https://en.wikipedia.org/wiki/DAX'
            df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[4]
            tickers = [t + ".DE" for t in df['Ticker'].tolist()]
            
    except Exception as e:
        st.error(f"Could not load tickers for {market_name}. Using fallback list. Error: {e}")
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    # Clean tickers (Change BRK.B to BRK-B for US stocks)
    return [t.replace('.', '-') if "S&P" in market_name else t for t in tickers]

def fetch_fundamentals(ticker_list):
    data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker_symbol in enumerate(ticker_list):
        status_text.text(f"Scanning {i+1}/{len(ticker_list)}: {ticker_symbol}...")
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # Extract Data
            company_name = info.get('longName', info.get('shortName', ticker_symbol))
            current_price = info.get('currentPrice', 0)
            target_price = info.get('targetMeanPrice', current_price)
            pe_ratio = info.get('trailingPE')
            earnings_growth = info.get('earningsGrowth', None)
            peg_ratio = info.get('pegRatio', None)
            avg_volume = info.get('averageVolume', 0)
            current_volume = info.get('volume', 0)

            # Manual PEG Calculation
            if peg_ratio is None and pe_ratio is not None and earnings_growth is not None:
                if earnings_growth > 0:
                    peg_ratio = pe_ratio / (earnings_growth * 100)
                else:
                    peg_ratio = 999
            
            # Fill Defaults
            if peg_ratio is None: peg_ratio = 999
            if pe_ratio is None: pe_ratio = 999

            # Upside Calc
            if current_price and current_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
            else:
                upside = 0
            
            # Volume Spike Metric
            vol_spike = (current_volume / avg_volume) if avg_volume and avg_volume > 0 else 1.0

            if current_price > 0:
                data.append({
                    'Symbol': ticker_symbol,
                    'Name': company_name, # Added Name Column
                    'Price': current_price,
                    'P/E': round(pe_ratio, 2),
                    'PEG': round(peg_ratio, 2),
                    'Upside (%)': round(upside, 2),
                    'Vol Spike': round(vol_spike, 2), 
                    'Sector': info.get('sector', 'N/A')
                })
                
        except Exception:
            continue
        progress_bar.progress((i + 1) / len(ticker_list))
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(data)

def calculate_technicals(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volume MA (20 day)
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    return df

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🎲 Market Scanner", "📊 Deep Dive (Vol & Charts)"])

# --- TAB 1: SCANNER ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button('🎲 Shuffle & Scan', type="primary"):
            st.session_state['scan_run'] = True
            all_tickers = get_tickers(market_choice)
            # Safe sample handling
            sample_size = min(batch_size, len(all_tickers))
            st.session_state['current_batch'] = random.sample(all_tickers, sample_size)
    
    with col2:
        st.info(f"Scanning a random sample of **{batch_size}** stocks from **{market_choice}**.")

    if st.session_state.get('scan_run') and st.session_state.get('current_batch'):
        with st.spinner("Fetching Data..."):
            df_fund = fetch_fundamentals(st.session_state['current_batch'])
            
            if not df_fund.empty:
                matches = df_fund[
                    (df_fund['PEG'] < peg_threshold) & 
                    (df_fund['P/E'] < pe_threshold) &
                    (df_fund['Upside (%)'] >= upside_threshold)
                ]
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Scanned", len(df_fund))
                c2.metric("Opportunities", len(matches))
                c3.metric("Avg Upside", f"{df_fund['Upside (%)'].mean():.1f}%")

                if not matches.empty:
                    st.success(f"Found {len(matches)} stocks!")
                    
                    # Display Table with Names
                    st.dataframe(
                        matches.set_index('Symbol').style
                        .format({
                            'Price': "{:.2f}", 
                            'P/E': "{:.2f}", 
                            'PEG': "{:.2f}", 
                            'Upside (%)': "{:.2f}", 
                            'Vol Spike': "{:.2f}"
                        })
                        .background_gradient(subset=['Upside (%)'], cmap='Greens')
                        .background_gradient(subset=['Vol Spike'], cmap='Blues'),
                        use_container_width=True
                    )
                    st.session_state['valid_tickers'] = matches['Symbol'].tolist()
                else:
                    st.warning("No perfect matches. Showing all scanned data below.")
                    st.session_state['valid_tickers'] = df_fund['Symbol'].tolist()
                
                with st.expander("Raw Data"):
                    st.dataframe(df_fund)

# --- TAB 2: TECHNICAL DEEP DIVE ---
with tab2:
    st.subheader("Technical & Volume Analysis")
    
    available_tickers = st.session_state.get('valid_tickers', ["AAPL", "MSFT"])
    selected_ticker = st.selectbox("Select Stock:", available_tickers)
    
    if selected_ticker and st.button("Analyze Charts"):
        stock = yf.Ticker(selected_ticker)
        df_hist = stock.history(period="1y")
        
        if not df_hist.empty:
            df_hist = calculate_technicals(df_hist)
            latest = df_hist.iloc[-1]
            
            # --- 3-Row Chart (Price, Volume, RSI) ---
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f'{selected_ticker} Price', 'Volume Analysis', 'RSI (Momentum)')
            )

            # 1. Price & SMA
            fig.add_trace(go.Candlestick(x=df_hist.index, open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_50'], line=dict(color='orange'), name='50 SMA'), row=1, col=1)

            # 2. Volume Bar Chart
            colors = ['green' if c >= o else 'red' for c, o in zip(df_hist['Close'], df_hist['Open'])]
            fig.add_trace(go.Bar(x=df_hist.index, y=df_hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Vol_SMA_20'], line=dict(color='blue', width=1), name='Vol SMA (20)'), row=2, col=1)

            # 3. RSI
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Analysis Text ---
            vol_status = "High" if latest['Volume'] > latest['Vol_SMA_20'] else "Normal"
            st.info(f"""
            **Volume Analysis:**
            * **Volume Today:** {latest['Volume']:,} (vs Avg: {int(latest['Vol_SMA_20']):,})
            * **Status:** {vol_status} Activity. (High volume on Green days = Bullish).
            """)

# --- User Guide ---
st.markdown("---")
with st.expander("📖 Guide: Volume & New Markets"):
    st.markdown("""
    * **Markets:** Switch between **S&P 500** (Large Cap), **S&P 600** (Small Cap/Russell proxy), **FTSE** (UK), and **DAX** (Germany) in the sidebar.
    * **Vol Spike:** This metric in the table shows if trading volume is higher than usual. A value > 1.5 means volume is 50% higher than average (Institutional interest).
    * **Charts:** The Deep Dive now includes a Volume Bar chart. Look for tall Green bars (buying pressure).
    """)
