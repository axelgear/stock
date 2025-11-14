# 🚀 Quick Usage Guide

## Step-by-Step Instructions

### Option 1: Automated Setup (Recommended)

```bash
cd /www/wwwroot/axel/TRADING
./quick_start.sh
```

Follow the interactive menu to:
1. Install dependencies
2. Download stock data
3. Run analysis tools
4. Launch dashboard

### Option 2: Manual Setup

#### 1️⃣ Install Dependencies

```bash
cd /www/wwwroot/axel/TRADING
pip3 install -r requirements.txt
```

#### 2️⃣ Download Stock Data

**Important**: This step downloads 20 years of data for 2,191 NSE stocks and may take 1-2 hours!

```bash
python3 fetch_stocks.py
```

Expected output:
```
Loading equity symbols from EQUITY_L.csv...
Found 2191 stocks to fetch
Fetching 20y of historical data...
----------------------------------------------------------------------
[1/2191] Fetching 20MICRONS... ✓ Saved 4523 records
[2/2191] Fetching 21STCENMGM... ✓ Saved 3245 records
...
```

This creates an `EOD/` directory with individual CSV files for each stock.

#### 3️⃣ Launch Interactive Dashboard

```bash
streamlit run dashboard.py
```

Then open your browser to: **http://localhost:8501**

Dashboard features:
- 📊 Interactive candlestick charts
- 📈 Technical indicators (RSI, MACD, Moving Averages)
- 📉 Statistical analysis
- 💾 Download data as CSV

#### 4️⃣ Run Backtesting

```bash
# Test Moving Average Crossover strategy on RELIANCE
python3 backtest.py
```

Example output:
```
==================================================================
Running Backtest: RELIANCE.csv
Strategy: MA_CROSSOVER
==================================================================

Performance Metrics:
----------------------------------------------------------------------
Initial Capital............................................ 100,000.00
Final Portfolio Value...................................... 156,234.50
Total Return (%)........................................... 56.23
Sharpe Ratio............................................... 1.45
Max Drawdown (%).......................................... -18.34
Win Rate (%)............................................... 54.23
Total Trades............................................... 127

📊 Chart saved as: backtest_results.png
```

Custom backtesting:
```python
from backtest import run_backtest_example

# Test different strategies
run_backtest_example("EOD/TCS.csv", strategy='ma_crossover')
run_backtest_example("EOD/INFY.csv", strategy='rsi')
```

#### 5️⃣ Train ML Models

```bash
# Train Random Forest model for RELIANCE
python3 ml_forecasting.py
```

Example output:
```
==================================================================
Training ML Model: RELIANCE.csv
Model Type: RANDOM_FOREST
==================================================================

Creating features...
✓ Created 45 features

==================================================================
Training Results:
==================================================================

📈 Training Set Metrics:
   MSE.................... 245.3421
   RMSE................... 15.6625
   MAE.................... 10.2341
   R2..................... 0.9234
   MAPE................... 2.3456

📊 Test Set Metrics:
   MSE.................... 423.5432
   RMSE................... 20.5804
   MAE.................... 14.5678
   R2..................... 0.8567
   MAPE................... 3.2345

🎯 Top 10 Important Features:
      Feature  Importance
0  Close_Lag_1    0.234567
1       SMA_20    0.156789
2          RSI    0.098765
...
```

Train different models:
```python
from ml_forecasting import train_forecasting_model

# Random Forest (default)
train_forecasting_model("EOD/INFY.csv", model_type='random_forest')

# Gradient Boosting
train_forecasting_model("EOD/TCS.csv", model_type='gradient_boost')

# Linear Regression
train_forecasting_model("EOD/WIPRO.csv", model_type='linear')
```

Models are saved in `models/` directory.

#### 6️⃣ Data Processing & Feature Engineering

```bash
# Process and clean RELIANCE data
python3 data_engineering.py
```

Example output:
```
==================================================================
Processing: RELIANCE.csv
==================================================================

Cleaning data...
   ✓ Removed 5 duplicate rows
   ✓ Filled missing values
   ✓ Removed 12 outlier rows
   ✓ Data cleaned: 5000 → 4983 rows

Adding technical indicators...
   ✓ Added 52 technical indicators

Adding time features...
   ✓ Added time-based features

Adding lag features...
   ✓ Added lag features for 2 columns

==================================================================
DATA QUALITY REPORT
==================================================================

📊 Dataset Overview:
   Total rows: 4,983
   Total columns: 78
   Date range: 2004-01-01 to 2024-11-12
   Duration: 7621 days

💹 Price Statistics:
   Highest Close: ₹3,024.50
   Lowest Close: ₹142.75
   Average Close: ₹1,234.56
   Latest Close: ₹2,845.30

📈 Returns:
   Daily Avg Return: 0.08%
   Daily Volatility: 1.95%
   Max Daily Gain: 15.23%
   Max Daily Loss: -13.45%

✅ Data Quality:
   Missing values: 0.00%
   Data completeness: 100.00%
==================================================================

✓ Processed data saved to: processed/RELIANCE_processed.csv
```

## 📊 Example Workflows

### Workflow 1: Quick Stock Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load stock data
data = pd.read_csv("EOD/RELIANCE.csv", index_col='Date', parse_dates=True)

# Plot closing prices
data['Close'].plot(figsize=(12, 6), title='RELIANCE Stock Price')
plt.ylabel('Price (₹)')
plt.grid(True)
plt.show()

# Calculate returns
returns = data['Close'].pct_change()
print(f"Average Daily Return: {returns.mean()*100:.2f}%")
print(f"Daily Volatility: {returns.std()*100:.2f}%")
```

### Workflow 2: Compare Multiple Stocks

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple stocks
stocks = ['RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HDFCBANK']
data_dict = {}

for stock in stocks:
    df = pd.read_csv(f"EOD/{stock}.csv", index_col='Date', parse_dates=True)
    data_dict[stock] = df['Close']

# Combine into one DataFrame
combined = pd.DataFrame(data_dict)

# Normalize to 100 for comparison
normalized = (combined / combined.iloc[0]) * 100

# Plot
normalized.plot(figsize=(14, 7), title='Stock Performance Comparison (Normalized)')
plt.ylabel('Normalized Price (Base=100)')
plt.legend()
plt.grid(True)
plt.show()
```

### Workflow 3: Screen Stocks by Criteria

```python
import pandas as pd
import glob

# Load all stocks
stocks = []
for file in glob.glob("EOD/*.csv"):
    stock_name = file.split('/')[-1].replace('.csv', '')
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    
    if len(df) > 0:
        latest = df.iloc[-1]
        # Calculate 52-week high/low
        high_52w = df['High'].tail(252).max()
        low_52w = df['Low'].tail(252).min()
        
        stocks.append({
            'Stock': stock_name,
            'Price': latest['Close'],
            '52W_High': high_52w,
            '52W_Low': low_52w,
            'Distance_from_High': ((latest['Close'] - high_52w) / high_52w) * 100
        })

# Create DataFrame
screening = pd.DataFrame(stocks)

# Find stocks near 52-week high (within 5%)
near_high = screening[screening['Distance_from_High'] > -5]
print("\nStocks near 52-week high:")
print(near_high.sort_values('Distance_from_High', ascending=False).head(10))
```

### Workflow 4: Portfolio Backtesting

```python
from backtest import Backtester
import pandas as pd

# Equal-weighted portfolio
stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
capital_per_stock = 100000 / len(stocks)

portfolio_results = []

for stock in stocks:
    data = pd.read_csv(f"EOD/{stock}.csv", index_col='Date', parse_dates=True)
    
    backtester = Backtester(data, initial_capital=capital_per_stock)
    backtester.moving_average_crossover()
    backtester.calculate_returns()
    
    metrics = backtester.performance_metrics()
    metrics['Stock'] = stock
    portfolio_results.append(metrics)

# Portfolio summary
results_df = pd.DataFrame(portfolio_results)
print("\nPortfolio Performance:")
print(results_df[['Stock', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']].to_string(index=False))
print(f"\nPortfolio Total Return: {results_df['Total Return (%)'].mean():.2f}%")
```

## 🎯 Pro Tips

### 1. Incremental Downloads
If download interrupts, the script will skip already downloaded stocks:

```bash
# Downloads will resume from where it left off
python3 fetch_stocks.py
```

### 2. Process Stocks in Batches
For better performance, process stocks in batches:

```python
import glob

files = glob.glob("EOD/*.csv")
batch_size = 50

for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    # Process batch...
```

### 3. Save Dashboard Configuration
In Streamlit, you can bookmark specific views:
- Set your favorite stock
- Choose your preferred indicators
- Bookmark the URL

### 4. Schedule Regular Updates
Create a cron job to update data daily:

```bash
# Add to crontab (crontab -e)
0 18 * * * cd /www/wwwroot/axel/TRADING && python3 fetch_stocks.py
```

### 5. Export for Excel Analysis
```python
data = pd.read_csv("EOD/RELIANCE.csv")
data.to_excel("RELIANCE_analysis.xlsx", index=False)
```

## 🔍 Troubleshooting

### Problem: "No module named 'yfinance'"
**Solution**: 
```bash
pip3 install yfinance
```

### Problem: Dashboard shows "No stock data found"
**Solution**: Run fetch_stocks.py first to download data

### Problem: Memory error during batch processing
**Solution**: Process fewer stocks at once or increase swap space

### Problem: Some stocks show no data
**Solution**: Normal - some stocks may be delisted or have limited history

## 📚 Next Steps

1. **Learn More**: Read the full README.md
2. **Customize**: Modify strategies in backtest.py
3. **Experiment**: Try different ML models and features
4. **Scale**: Deploy dashboard on a server
5. **Automate**: Set up scheduled data updates

## 💡 Ideas for Extension

- Add more technical indicators
- Implement portfolio optimization
- Add fundamental analysis data
- Create alert system for trading signals
- Integrate with broker APIs for live trading
- Add sentiment analysis from news
- Implement risk management tools

---

**Happy Analyzing! 📊**

