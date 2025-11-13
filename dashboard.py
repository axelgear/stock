"""
Stock Trading Dashboard
Interactive Streamlit dashboard for stock analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import glob

# Page configuration
st.set_page_config(
    page_title="NSE Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data(stock_file):
    """Load stock data from CSV file"""
    try:
        data = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_available_stocks(eod_dir="EOD"):
    """Get list of available stocks"""
    if not os.path.exists(eod_dir):
        return []
    
    stock_files = glob.glob(os.path.join(eod_dir, "*.csv"))
    stocks = [os.path.basename(f).replace('.csv', '') for f in stock_files]
    return sorted(stocks)

def calculate_metrics(data):
    """Calculate key metrics"""
    latest_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    change = latest_price - prev_close
    change_pct = (change / prev_close) * 100
    
    # Returns
    returns = data['Close'].pct_change()
    daily_volatility = returns.std()
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Price stats
    high_52w = data['High'].tail(252).max()
    low_52w = data['Low'].tail(252).min()
    
    # Volume
    avg_volume = data['Volume'].mean()
    latest_volume = data['Volume'].iloc[-1]
    
    return {
        'latest_price': latest_price,
        'change': change,
        'change_pct': change_pct,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'daily_volatility': daily_volatility,
        'annual_volatility': annual_volatility,
        'avg_volume': avg_volume,
        'latest_volume': latest_volume
    }

def plot_candlestick(data, stock_name, show_volume=True):
    """Create candlestick chart with volume"""
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{stock_name} Price', 'Volume')
        )
    else:
        fig = go.Figure()
    
    # Candlestick
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
        
        # Volume bars
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        volume_bars = go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        )
        fig.add_trace(volume_bars, row=2, col=1)
    else:
        fig.add_trace(candlestick)
    
    fig.update_layout(
        title=f'{stock_name} Stock Chart',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )
    
    return fig

def plot_technical_indicators(data, stock_name):
    """Plot price with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{stock_name} Price & Moving Averages', 'RSI', 'MACD')
    )
    
    # Price and MAs
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                            name='Close', line=dict(color='blue')), row=1, col=1)
    
    # Calculate MAs
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                            name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                            name='SMA 50', line=dict(color='red')), row=1, col=1)
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(go.Scatter(x=data.index, y=rsi, 
                            name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    fig.add_trace(go.Scatter(x=data.index, y=macd, 
                            name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=signal, 
                            name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=900, hovermode='x unified')
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def plot_returns_distribution(data):
    """Plot returns distribution"""
    returns = data['Close'].pct_change().dropna() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Daily Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">📈 NSE Stock Trading Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Stock selection
        stocks = get_available_stocks()
        
        if not stocks:
            st.error("No stock data found! Please run fetch_stocks.py first.")
            st.stop()
        
        selected_stock = st.selectbox(
            "Select Stock",
            stocks,
            index=stocks.index('RELIANCE') if 'RELIANCE' in stocks else 0
        )
        
        # Date range
        st.subheader("Date Range")
        date_range = st.selectbox(
            "Select Period",
            ['1M', '3M', '6M', '1Y', '2Y', '5Y', 'All'],
            index=3
        )
        
        # Chart options
        st.subheader("Chart Options")
        show_volume = st.checkbox("Show Volume", value=True)
        show_technical = st.checkbox("Show Technical Indicators", value=False)
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This dashboard provides real-time analysis of NSE stocks with technical indicators and historical data.")
    
    # Load data
    stock_file = f"EOD/{selected_stock}.csv"
    data = load_stock_data(stock_file)
    
    if data is None:
        st.error("Failed to load stock data!")
        st.stop()
    
    # Filter by date range
    if date_range != 'All':
        date_map = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365,
            '2Y': 730,
            '5Y': 1825
        }
        days = date_map[date_range]
        data = data.tail(days)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{metrics['latest_price']:.2f}",
            f"{metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)"
        )
    
    with col2:
        st.metric(
            "52W High",
            f"₹{metrics['high_52w']:.2f}"
        )
    
    with col3:
        st.metric(
            "52W Low",
            f"₹{metrics['low_52w']:.2f}"
        )
    
    with col4:
        st.metric(
            "Volatility (Annual)",
            f"{metrics['annual_volatility']*100:.2f}%"
        )
    
    with col5:
        st.metric(
            "Volume",
            f"{metrics['latest_volume']:,.0f}",
            f"{(metrics['latest_volume']/metrics['avg_volume']-1)*100:+.1f}% vs avg"
        )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📈 Technical Analysis", "📋 Data", "📉 Statistics"])
    
    with tab1:
        st.subheader(f"{selected_stock} Stock Chart")
        
        if show_technical:
            fig = plot_technical_indicators(data, selected_stock)
        else:
            fig = plot_candlestick(data, selected_stock, show_volume)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            st.metric("RSI (14)", f"{current_rsi:.2f}")
            
            if current_rsi > 70:
                st.warning("⚠️ Overbought territory")
            elif current_rsi < 30:
                st.success("✅ Oversold territory")
            else:
                st.info("ℹ️ Neutral")
        
        with col2:
            # Moving averages
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            
            st.metric("SMA 20", f"₹{sma_20:.2f}")
            st.metric("SMA 50", f"₹{sma_50:.2f}")
            
            if data['Close'].iloc[-1] > sma_20 > sma_50:
                st.success("✅ Bullish trend")
            elif data['Close'].iloc[-1] < sma_20 < sma_50:
                st.error("⚠️ Bearish trend")
        
        # Returns distribution
        st.subheader("Returns Distribution")
        fig_returns = plot_returns_distribution(data)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab3:
        st.subheader("Historical Data")
        
        # Display options
        col1, col2 = st.columns([1, 3])
        with col1:
            rows_to_show = st.selectbox("Rows to display", [10, 25, 50, 100, 'All'])
        
        # Show data
        display_data = data if rows_to_show == 'All' else data.tail(rows_to_show)
        st.dataframe(display_data.sort_index(ascending=False), use_container_width=True)
        
        # Download button
        csv = data.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_stock}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Statistics")
            stats = data['Close'].describe()
            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [f"{stats['count']:.0f}", f"₹{stats['mean']:.2f}", 
                         f"₹{stats['std']:.2f}", f"₹{stats['min']:.2f}",
                         f"₹{stats['25%']:.2f}", f"₹{stats['50%']:.2f}",
                         f"₹{stats['75%']:.2f}", f"₹{stats['max']:.2f}"]
            })
            st.table(stats_df)
        
        with col2:
            st.markdown("#### Returns Statistics")
            returns = data['Close'].pct_change().dropna() * 100
            st.write(f"**Mean Daily Return:** {returns.mean():.3f}%")
            st.write(f"**Median Daily Return:** {returns.median():.3f}%")
            st.write(f"**Std Deviation:** {returns.std():.3f}%")
            st.write(f"**Skewness:** {returns.skew():.3f}")
            st.write(f"**Kurtosis:** {returns.kurtosis():.3f}")
            st.write(f"**Best Day:** {returns.max():.2f}%")
            st.write(f"**Worst Day:** {returns.min():.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"Last Updated: {data.index[-1].strftime('%Y-%m-%d')} | "
        f"Data Points: {len(data):,} | "
        f"Powered by yfinance & Streamlit"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

