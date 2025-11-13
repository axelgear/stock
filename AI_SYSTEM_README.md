# 🤖 AI-Powered Stock Trading System

## Complete AI System with Auto-Validation and Custom Dashboard

This advanced system includes:
- ✅ AI Predictions (Ensemble ML Models + LSTM + Prophet)
- ✅ Auto-Validation & Performance Tracking
- ✅ Strategy Auto-Adjustment based on Results
- ✅ Custom Dashboard (No Streamlit!)
- ✅ REST API Backend (Flask)
- ✅ Advanced Filtering & Sorting
- ✅ Intelligent Recommendations

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /www/wwwroot/axel/TRADING
source venv/bin/activate
pip install flask flask-cors
```

### 2. Fix Existing Data (If Already Downloaded)

```bash
# Re-download a few stocks to fix Date column issue
source venv/bin/activate
python fetch_stocks.py
```

### 3. Start the Backend API

```bash
source venv/bin/activate
python backend/app.py
```

The API will run on: **http://localhost:5000**

### 4. Open the Dashboard

Open your browser and go to: **http://localhost:5000**

---

## 📂 New File Structure

```
TRADING/
│
├── ai_predictor.py              # AI prediction engine
├── auto_validator.py            # Auto-validation system
│
├── backend/
│   └── app.py                   # Flask REST API
│
├── frontend/
│   ├── index.html               # Custom dashboard UI
│   └── app.js                   # Frontend JavaScript
│
├── predictions.json             # Prediction history
├── strategies.json              # Trading strategies
│
└── (other existing files...)
```

---

## 🤖 AI Prediction Engine

### Features

- **Ensemble Models**: Random Forest + Gradient Boosting + Ridge Regression
- **LSTM** (Optional): Deep learning for time series
- **Prophet** (Optional): Facebook's forecasting model
- **Auto-Feature Engineering**: 45+ technical indicators
- **Confidence Scores**: Know how reliable predictions are

### Usage

```python
from ai_predictor import train_and_predict

# Generate prediction
predictor, prediction, signal = train_and_predict("EOD/RELIANCE.csv", "RELIANCE")

# Results:
# - Current Price: ₹2,500
# - Predicted Price: ₹2,575 (+3.0%)
# - Confidence: 82.5%
# - Signal: BUY
```

### Command Line

```bash
source venv/bin/activate
python ai_predictor.py
```

---

## ✅ Auto-Validation System

### Features

- Automatically validates all predictions next day
- Tracks accuracy over time
- Auto-adjusts strategies based on performance
- Generates performance reports

### How It Works

1. **Make Prediction**: System records prediction with timestamp
2. **Next Day**: System automatically validates when new data is available
3. **Calculate Accuracy**: Compares predicted vs actual prices
4. **Adjust Strategies**: If accuracy < 60%, makes strategies more conservative
5. **If accuracy > 80%**: Makes strategies more aggressive

### Usage

```python
from auto_validator import run_auto_validation

# Validate predictions
validator = run_auto_validation("EOD/RELIANCE.csv", "RELIANCE")

# Results:
# - Validated: 5 predictions
# - Average Accuracy: 87.3%
# - Accurate: 4, Moderate: 1, Poor: 0
# - Strategy auto-adjusted to be more aggressive
```

### Command Line

```bash
source venv/bin/activate
python auto_validator.py
```

---

## 🌐 REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks` | List all available stocks |
| GET | `/api/stock/<name>?period=1y` | Get stock data with filtering |
| GET | `/api/predict/<name>` | Generate AI prediction |
| GET | `/api/validate/<name>` | Validate previous predictions |
| POST | `/api/filter` | Filter stocks by criteria |
| GET | `/api/strategies` | Get current strategies |
| PUT | `/api/strategies` | Update strategies |

### Example API Calls

```bash
# Get all stocks
curl http://localhost:5000/api/stocks

# Get stock data
curl "http://localhost:5000/api/stock/RELIANCE?period=1y"

# Get AI prediction
curl http://localhost:5000/api/predict/RELIANCE

# Filter stocks
curl -X POST http://localhost:5000/api/filter \
  -H "Content-Type: application/json" \
  -d '{"price_min": 100, "returns_1d_min": 2, "sort_by": "returns_1d"}'
```

---

## 💻 Custom Dashboard

### Features

#### Analysis Tab
- **Real-time Price Charts**: Interactive Chart.js visualizations
- **Current Metrics**: Price, change, 52W high/low
- **AI Predictions**: Next-day price prediction
- **Trading Signals**: BUY/SELL/HOLD with confidence
- **Recommendations**: Multiple strategy suggestions

#### Filter Tab
- **Advanced Filtering**: Price range, returns, volume
- **Sorting**: By any metric
- **Click to View**: Click any stock to analyze

#### Strategies Tab
- **View All Strategies**: Aggressive, Moderate, Conservative
- **Auto-Adjusting**: Strategies update based on performance
- **Live Status**: See which strategies are active

### Screenshots

The dashboard includes:
- Beautiful gradient design
- Responsive layout
- Real-time data updates
- Interactive charts
- Color-coded metrics (green for gains, red for losses)

---

## 🔍 Advanced Features

### 1. Stock Filtering

Filter stocks by multiple criteria:

```javascript
{
  "price_min": 100,
  "price_max": 5000,
  "returns_1d_min": 2.0,
  "volume_min": 100000,
  "sort_by": "returns_1d",
  "sort_order": "desc",
  "limit": 50
}
```

### 2. Strategy Types

**Aggressive Strategy**
- Buy Threshold: 1.0%
- Confidence Min: 60%
- Position Size: 30%
- Stop Loss: -5%
- Take Profit: 10%

**Moderate Strategy**
- Buy Threshold: 0.5%
- Confidence Min: 70%
- Position Size: 20%
- Stop Loss: -3%
- Take Profit: 6%

**Conservative Strategy**
- Buy Threshold: 2.0%
- Confidence Min: 80%
- Position Size: 10%
- Stop Loss: -2%
- Take Profit: 4%

### 3. Trading Signals

System generates 5 signal types:
- **STRONG BUY**: Predicted +2%+, Confidence 70%+
- **BUY**: Predicted +0.5%+, Confidence 60%+
- **HOLD**: Between thresholds
- **SELL**: Predicted -0.5%+, Confidence 60%+
- **STRONG SELL**: Predicted -2%+, Confidence 70%+

---

## 🎯 Complete Workflow

### Daily Trading Workflow

```bash
# Morning: Start the system
source venv/bin/activate
python backend/app.py

# Open dashboard in browser
# http://localhost:5000

# 1. Check top performers (Filter tab)
#    - Filter: returns_1d_min = 2
#    - Sort by: returns_1d
#    - Review top 10 stocks

# 2. Analyze selected stock (Analysis tab)
#    - View price chart and trends
#    - Check 52W high/low

# 3. Get AI prediction
#    - Click "Get AI Prediction"
#    - Review predicted price & confidence
#    - Check trading signal

# 4. Review recommendations
#    - See suggested strategies
#    - Check position size, stop loss, take profit

# 5. Execute trade (external broker)
#    - Use recommendations as guidance

# Next Day: Validate
# - Click "Validate" to check yesterday's prediction
# - System auto-adjusts strategies based on accuracy
```

---

## 📊 Performance Metrics

The system tracks:

- **Prediction Accuracy**: How close predictions are to actual
- **Win Rate**: % of profitable trades
- **Average Return**: Mean return per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline

---

## 🔧 Advanced Configuration

### Enable LSTM (Deep Learning)

```bash
# Install TensorFlow
pip install tensorflow

# LSTM will automatically be used if available
# Requires more time to train but often more accurate
```

### Enable Prophet (Facebook's Model)

```bash
# Install Prophet
pip install prophet

# Prophet will automatically be used if available
# Excellent for seasonality and trends
```

### Adjust Strategy Parameters

Edit `strategies.json`:

```json
{
  "aggressive": {
    "buy_threshold": 1.0,
    "sell_threshold": -1.0,
    "confidence_min": 60,
    "position_size": 0.3,
    "stop_loss": -5.0,
    "take_profit": 10.0,
    "enabled": true
  }
}
```

---

## 🐛 Troubleshooting

### Issue: "Date column not found"

**Solution**: Re-download stock data with fixed script:
```bash
source venv/bin/activate
python fetch_stocks.py
```

### Issue: API not accessible

**Solution**: Make sure backend is running:
```bash
source venv/bin/activate
python backend/app.py
```

### Issue: No predictions available

**Solution**: Click "Get AI Prediction" first. System needs to generate predictions before validation.

### Issue: CORS errors in browser

**Solution**: Access dashboard through Flask server (http://localhost:5000), not file:// protocol.

---

## 📈 Example Outputs

### AI Prediction Output

```
==================================================================
AI Prediction System: RELIANCE
==================================================================

Training AI models...
✓ Ensemble Model Accuracy: 84.27%

Generating next-day prediction...

📊 Prediction Results:
   Current Price: ₹2,500.00
   Predicted Price: ₹2,575.00
   Expected Change: +3.00%
   Confidence: 84.27%

🎯 Trading Signal:
   Signal: STRONG BUY
   Action: Buy aggressively
   Expected Return: +3.00%
   Confidence: 84.27%

==================================================================
```

### Validation Output

```
==================================================================
Auto-Validation System: RELIANCE
==================================================================

Validating previous predictions...
✓ Validated 5 predictions

Validation Results:
   ✅ 2024-11-10: Accuracy 94.23% (Error: 5.77%)
   ✅ 2024-11-11: Accuracy 87.15% (Error: 12.85%)
   ✅ 2024-11-12: Accuracy 91.50% (Error: 8.50%)
   ⚠️  2024-11-13: Accuracy 76.32% (Error: 23.68%)
   ✅ 2024-11-14: Accuracy 89.67% (Error: 10.33%)

🔧 Auto-Adjusted Strategies:
   • aggressive: Made more aggressive (high accuracy)
   • moderate: Made more aggressive (high accuracy)

📊 Performance Report:
   Total Predictions: 5
   Validated: 5
   Average Accuracy: 87.77%
   Accurate: 4, Moderate: 1, Poor: 0

==================================================================
```

---

## 🎓 Learning Resources

### Understanding the Models

- **Random Forest**: Ensemble of decision trees, robust to overfitting
- **Gradient Boosting**: Sequential trees, each correcting previous errors
- **Ridge Regression**: Linear model with L2 regularization
- **LSTM**: Recurrent neural network for sequential data
- **Prophet**: Additive model for seasonality and trends

### Key Concepts

- **Ensemble**: Combining multiple models for better predictions
- **Confidence**: Based on historical accuracy of model
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Backtesting**: Testing strategy on historical data
- **Validation**: Checking prediction accuracy after the fact

---

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This is an educational tool
2. **Past Performance ≠ Future Results**: Models may not always be accurate
3. **Use at Your Own Risk**: Always do your own research
4. **Test Thoroughly**: Backtest strategies before live trading
5. **Risk Management**: Never invest more than you can afford to lose

---

## 🚀 Next Steps

1. ✅ Start backend API
2. ✅ Open dashboard
3. ✅ Filter and analyze stocks
4. ✅ Generate predictions
5. ✅ Review recommendations
6. ✅ Validate predictions daily
7. ✅ Monitor strategy adjustments
8. 📈 Profit responsibly!

---

**Happy Trading! 🤖📈💰**

