import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# --- Page Config ---
st.set_page_config(page_title="Random Market Scanner", layout="wide")

st.title("🎲 Random Market Scanner (NYSE/NASDAQ)")
st.markdown("""
**New Features:**
* **Dynamic Scanning:** Fetches the full S&P 500 list and picks a random batch every time.
* **Manual PEG Calc:** Calculates PEG manually if the data source is missing it.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("1. Scanner Settings")
batch_size = st.sidebar.slider("Number of Random Stocks to Scan", 10, 100, 30, 5)
pe_threshold = st.sidebar.number_input("Max P/E Ratio", value=50, step=5)
peg_threshold = st.sidebar.slider("Max PEG Ratio", 0.5, 5.0, 1.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("2. Analysis Criteria")
upside_threshold = st.sidebar.slider("Min Analyst Upside (%)", 0, 50, 5, 5)

# --- Helper Functions ---

@st.cache_data(ttl=3600*24) # Cache the S&P 500 list for 24 hours
def get_sp500_tickers():
    try:
        # Pulls the table of S&P 500 companies from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"Error fetching ticker list: {e}")
        # Fallback list if Wikipedia fails
        return ["GOOGL", "AMZN", "MSFT", "AAPL", "NVDA", "TSLA", "META", "AMD", "PLTR", "SOFI", "INTC"]

def fetch_fundamentals(ticker_list):
    data = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker_symbol in enumerate(ticker_list):
        status_text.text(f"Analyzing {ticker_symbol}...")
        try:
            # Clean ticker (replace . with - for BRK.B)
            ticker_symbol = ticker_symbol.replace('.', '-')
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # --- Data Extraction ---
            current_price = info.get('currentPrice', 0)
            target_price = info.get('targetMeanPrice', current_price)
            pe_ratio = info.get('trailingPE')
            
            # Handle Missing Growth Data for PEG Calc
            earnings_growth = info.get('earningsGrowth', None) # Returned as decimal (0.15 = 15%)
            peg_ratio = info.get('pegRatio', None)
            
            # --- 1. Fix PEG Calculation ---
            # If PEG is missing but we have P/E and Growth, calculate it manually
            if peg_ratio is None and pe_ratio is not None and earnings_growth is not None:
                if earnings_growth > 0:
                    peg_ratio = pe_ratio / (earnings_growth * 100)
                else:
                    peg_ratio = 999 # Negative growth = Bad PEG
            
            # If still None, set to default high value to fail filter
            if peg_ratio is None:
                peg_ratio = 999 
                
            if pe_ratio is None:
                pe_ratio = 999

            # --- 2. Calculate Upside ---
            if current_price and current_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
            else:
                upside = 0
            
            # Only add if data is valid (Price > 0)
            if current_price > 0:
                data.append({
                    'Symbol': ticker_symbol,
                    'Price': current_price,
                    'P/E': round(pe_ratio, 2),
                    'PEG': round(peg_ratio, 2),
                    'Growth (%)': round(earnings_growth * 100, 2) if earnings_growth else 0,
                    'Upside (%)': round(upside, 2),
                    'Sector': info.get('sector', 'N/A')
                })
                
        except Exception:
            continue
            
        progress_bar.progress((i + 1) / len(ticker_list))
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(data)

def calculate_technicals(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🎲 Random Scanner", "📈 Technical Deep Dive"])

# --- TAB 1: RANDOM SCANNER ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        # Button to trigger new batch
        if st.button('🎲 Shuffle & Scan New Batch', type="primary"):
            st.session_state['scan_run'] = True
            
            # Fetch full list and pick random sample
            all_tickers = get_sp500_tickers()
            random_batch = random.sample(all_tickers, min(batch_size, len(all_tickers)))
            st.session_state['current_batch'] = random_batch
    
    with col2:
        st.info("Click the button to pick a random set of stocks from the S&P 500 and analyze them.")

    # Logic to run the scan
    if st.session_state.get('scan_run') and st.session_state.get('current_batch'):
        
        with st.spinner(f"Scanning {len(st.session_state['current_batch'])} random stocks..."):
            df_fund = fetch_fundamentals(st.session_state['current_batch'])
            
            if not df_fund.empty:
                # Filter Logic
                matches = df_fund[
                    (df_fund['PEG'] < peg_threshold) & 
                    (df_fund['PEG'] > 0) &
                    (df_fund['P/E'] < pe_threshold) &
                    (df_fund['Upside (%)'] >= upside_threshold)
                ]
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Batch Size", len(df_fund))
                m2.metric("Gems Found", len(matches))
                m3.metric("Avg PEG (Batch)", f"{df_fund['PEG'].median():.2f}")
                
                if not matches.empty:
                    st.success(f"Found {len(matches)} Undervalued Growth stocks!")
                    st.dataframe(
                        matches.set_index('Symbol').style.format("{:.2f}").background_gradient(subset=['Upside (%)'], cmap='Greens'),
                        use_container_width=True
                    )
                    
                    # Store matches for Tab 2
                    st.session_state['valid_tickers'] = matches['Symbol'].tolist()
                else:
                    st.warning("No matches found in this random batch. Try clicking 'Shuffle' again!")
                    st.session_state['valid_tickers'] = df_fund['Symbol'].tolist() # Fallback to all scanned
                
                with st.expander("View Full Batch Data"):
                    st.dataframe(df_fund)
            else:
                st.error("Failed to fetch data. Check your internet connection.")

# --- TAB 2: TECHNICAL ANALYSIS ---
with tab2:
    st.subheader("Technical Analysis")
    
    # Get tickers from session state or default
    available_tickers = st.session_state.get('valid_tickers', ["AAPL", "MSFT", "GOOGL"])
    selected_ticker = st.selectbox("Select a Stock:", available_tickers)
    
    if selected_ticker and st.button("Analyze Charts"):
        stock = yf.Ticker(selected_ticker)
        df_hist = stock.history(period="1y")
        
        if not df_hist.empty:
            df_hist = calculate_technicals(df_hist)
            latest = df_hist.iloc[-1]
            
            # Simple Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7])
            fig.add_trace(go.Candlestick(x=df_hist.index, open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_50'], line=dict(color='orange'), name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            st.write(f"**RSI:** {latest['RSI']:.2f}")
            if latest['RSI'] < 30: st.success("✅ OVERSOLD - Potential Buy Signal")
            elif latest['RSI'] > 70: st.error("❌ OVERBOUGHT - Potential Sell Signal")
            else: st.info("Neutral Zone")
