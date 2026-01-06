import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(page_title="Undervalued Growth + Tech Analysis", layout="wide")

st.title("🚀 Undervalued Growth & Technical Tracker")
st.markdown("""
**Strategy:**
1. **Find Value:** Use the *Scanner* to find stocks with low PEG ratios (< 1.0) and high growth.
2. **Time Entry:** Use the *Deep Dive* tab to check if the stock is "Oversold" (RSI < 30) or trending up.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("1. Scanner Criteria")
pe_threshold = st.sidebar.number_input("Max P/E Ratio", value=30, step=5)
peg_threshold = st.sidebar.slider("Max PEG Ratio", 0.5, 3.0, 1.2, 0.1)
upside_threshold = st.sidebar.slider("Min Analyst Upside (%)", 0, 50, 10, 5)

st.sidebar.markdown("---")
st.sidebar.header("2. Ticker List")
default_tickers = "GOOGL, AMZN, MSFT, TSLA, NVDA, META, AMD, INTC, PYPL, AAPL, NFLX, JPM, V, PG, KO, DIS, PLTR, SOFI"
user_tickers = st.sidebar.text_area("Enter Tickers (comma separated)", default_tickers)

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker_list):
    data = []
    tickers = [t.strip().upper() for t in ticker_list.split(',')]
    
    # Progress bar for user feedback
    progress_bar = st.progress(0)
    
    for i, ticker_symbol in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            current_price = info.get('currentPrice', 0)
            target_price = info.get('targetMeanPrice', current_price)
            
            if current_price and current_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
            else:
                upside = 0
                
            data.append({
                'Symbol': ticker_symbol,
                'Price': current_price,
                'P/E': info.get('trailingPE', 0),
                'PEG': info.get('pegRatio', 0),
                'Rev Growth (%)': round(info.get('revenueGrowth', 0) * 100, 2) if info.get('revenueGrowth') else 0,
                'Analyst Target': target_price,
                'Upside (%)': round(upside, 2)
            })
        except Exception:
            continue
        progress_bar.progress((i + 1) / len(tickers))
        
    progress_bar.empty()
    return pd.DataFrame(data)

def calculate_technicals(df):
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    return df

# --- Main Tabs ---
tab1, tab2 = st.tabs(["📊 Market Scanner", "📈 Technical Deep Dive"])

# --- TAB 1: FUNDAMENTAL SCANNER ---
with tab1:
    if st.button('Run Scanner', key='scan_btn'):
        with st.spinner('Scanning market data...'):
            df_fund = fetch_fundamentals(user_tickers)
            
            if not df_fund.empty:
                # Filter Logic
                matches = df_fund[
                    (df_fund['PEG'] < peg_threshold) & 
                    (df_fund['PEG'] > 0) & 
                    (df_fund['Upside (%)'] >= upside_threshold)
                ]
                
                col1, col2 = st.columns(2)
                col1.metric("Stocks Scanned", len(df_fund))
                col2.metric("Undervalued Gems", len(matches))
                
                if not matches.empty:
                    st.success(f"Found {len(matches)} stocks matching your criteria!")
                    st.dataframe(
                        matches.set_index('Symbol').style.format("{:.2f}").background_gradient(subset=['Upside (%)'], cmap='Greens'),
                        use_container_width=True
                    )
                else:
                    st.warning("No stocks matched perfectly. Try raising the PEG threshold.")
                
                with st.expander("View All Data"):
                    st.dataframe(df_fund)

# --- TAB 2: TECHNICAL ANALYSIS ---
with tab2:
    st.subheader("Technical Analysis & Entry Points")
    
    # Dropdown to select a stock
    ticker_options = [t.strip().upper() for t in user_tickers.split(',')]
    selected_ticker = st.selectbox("Select a Stock to Analyze:", ticker_options)
    
    if selected_ticker:
        # Fetch History
        stock = yf.Ticker(selected_ticker)
        # We need roughly 1 year of data to calculate 200 SMA accurately
        df_hist = stock.history(period="1y")
        
        if not df_hist.empty:
            df_hist = calculate_technicals(df_hist)
            
            # Get latest values for metrics
            latest = df_hist.iloc[-1]
            rsi = latest['RSI']
            price = latest['Close']
            sma_50 = latest['SMA_50']
            
            # --- Technical Metrics Row ---
            m1, m2, m3 = st.columns(3)
            
            # RSI Metric
            rsi_color = "normal"
            if rsi < 30: rsi_color = "inverse" # Oversold (Green/Good)
            elif rsi > 70: rsi_color = "off"   # Overbought (Red/Bad)
            
            m1.metric("Current RSI (14)", f"{rsi:.1f}", delta="< 30 is Buy Signal" if rsi < 30 else "Neutral", delta_color=rsi_color)
            
            # Trend Metric
            trend = "Uptrend" if price > sma_50 else "Downtrend"
            m2.metric("Trend (vs 50 SMA)", trend, delta=f"{price - sma_50:.2f}")
            
            m3.metric("Current Price", f"${price:.2f}")

            # --- Interactive Plotly Chart ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f'{selected_ticker} Price & SMA', 'RSI (14)'),
                                row_width=[0.2, 0.7])

            # Candlestick
            fig.add_trace(go.Candlestick(x=df_hist.index,
                                         open=df_hist['Open'], high=df_hist['High'],
                                         low=df_hist['Low'], close=df_hist['Close'], name='OHLC'), 
                          row=1, col=1)
            
            # Moving Averages
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_50'], line=dict(color='orange', width=1), name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_200'], line=dict(color='blue', width=1), name='200 SMA'), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
            
            # RSI Levels (30 and 70)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # Layout updates
            fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Could not retrieve historical data for this ticker.")

# --- FOOTER: USER GUIDE ---
st.markdown("---")
with st.expander("📖 User Guide & Methodology (Click to Open)"):
    st.markdown("""
    ### **How to Use This Tracker**
    
    #### **1. Configure Your Strategy (Sidebar)**
    * **Max PEG Ratio:** The most important filter. A PEG < 1.0 implies the stock is undervalued relative to its growth rate.
    * **Min Analyst Upside:** Filters stocks where the Wall St. consensus target is higher than the current price.
    * **Ticker List:** Add or remove stock symbols (comma-separated) to scan your favorite sectors.

    #### **2. Scan for Value (Tab 1)**
    * Click **Run Scanner**.
    * Look for rows highlighted in Green. These are your candidates.
    * *Note:* If the list is empty, try increasing the PEG threshold (e.g., to 1.5).

    #### **3. Check the Charts (Tab 2)**
    * Select a stock from the dropdown.
    * **RSI Indicator:**
        * **< 30 (Green Line):** Oversold. Price may be due for a bounce. (Potential Buy)
        * **> 70 (Red Line):** Overbought. Price may be due for a drop. (Wait)
    * **Trend (SMA 50):** * If the candles are **above** the Orange Line, the trend is UP.
    
    #### **Methodology**
    * **Data Source:** Yahoo Finance (Real-time delayed).
    * **RSI Formula:** 14-period Relative Strength Index.
    * **Upside:** Calculated as `(Analyst Target - Current Price) / Current Price`.
    """)
