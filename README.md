# 📈 NSE Stock Trading Analysis System

An advanced stock trading analysis system with AI-powered predictions, real-time dashboard, and comprehensive historical analysis for NSE (National Stock Exchange) stocks.

## ✨ Features

### 🎯 Core Features
- **Real-time Stock Data**: Fetches live data from Yahoo Finance for 2400+ NSE stocks
- **SQLite Database**: Lightning-fast local caching (60x faster than CSV loading)
- **AI Predictions**: Machine learning models for next-day price forecasting
- **Custom Dashboard**: Beautiful, responsive web interface (no Streamlit dependencies)
- **Historical Analysis**: View 31 days of daily performance metrics
- **Auto-Validation**: Automatic prediction accuracy tracking and strategy adjustment

### 📊 Dashboard Capabilities
- **16 Comprehensive Columns**:
  - Today's price and change
  - Daily performance (Yesterday through Day-6)
  - Period performance (Last 7d, 7-14d, 14-21d, 21-31d)
  - AI predictions (Price and % change)
  - Volume analysis
  - 52-week position

- **Interactive Features**:
  - Sort by any column (click headers)
  - Color-coded gains/losses (green/red)
  - Real-time data updates
  - Click stock to view detailed charts
  - Filter and search functionality

### 🤖 AI/ML Features
- Ensemble ML models (Random Forest, Gradient Boosting, Ridge)
- Moving Average trend analysis
- Confidence scoring
- Prediction validation system
- Strategy auto-adjustment

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
python3 --version

# Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/axelgear/stock.git
cd stock
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download stock list**
```bash
# EQUITY_L.csv is included in the repo
# Contains 2400+ NSE stock symbols
```

4. **Fetch stock data** (one-time, ~5-10 minutes)
```bash
python fetch_stocks.py
```

5. **Sync database** (one-time, ~2 minutes)
   - Start the dashboard (step 6)
   - Click "Sync Database First" button
   - Wait for completion

6. **Launch dashboard**
```bash
python backend/app.py
# Or use: ./run_dashboard.sh
```

7. **Open browser**
```
http://localhost:5050
```

## 📖 Usage

### Loading Stock Data

**Fast Mode (Recommended)**:
1. Go to "All Stocks" tab
2. Click "⚡ Load from Database"
3. View all stocks instantly (~1 second)

**Update Today's Data**:
- Click "📅 Update Today Only" to refresh latest prices

**Generate AI Predictions**:
- Click "🔮 Generate Predictions" for tomorrow's forecasts

### Viewing Stock Details
- Click any stock row to view detailed analysis
- See historical charts
- View AI predictions
- Check technical indicators

### Filtering Stocks
- Go to "Filter" tab
- Set criteria (price range, change %, volume)
- View filtered results

## 🗂️ Project Structure

```
TRADING/
├── backend/
│   ├── app.py              # Flask API server
│   └── database.py         # SQLite database operations
├── frontend/
│   ├── index.html          # Dashboard UI
│   └── app.js              # Frontend logic
├── fetch_stocks.py         # Download stock data
├── ai_predictor.py         # ML prediction engine
├── auto_validator.py       # Prediction validation
├── requirements.txt        # Python dependencies
├── stock_cache.db          # SQLite database (auto-generated)
├── EOD/                    # Historical CSV data (auto-generated)
└── README.md               # This file
```

## 🔧 Configuration

### Database Settings
- Location: `stock_cache.db` (auto-created)
- Tables: `stock_data`, `predictions`, `sync_metadata`
- Indexes: Optimized for fast queries

### API Endpoints
- `GET /api/stocks` - List all stocks
- `GET /api/stock/<name>` - Get stock data
- `GET /api/db/all-stocks-fast` - All stocks from database
- `POST /api/db/sync-all` - Sync CSV to database
- `POST /api/db/update-today` - Update latest prices
- `POST /api/db/predict-all` - Generate predictions

## 📊 Performance

- **Database Load**: 221 stocks in <1 second (60x faster than CSV)
- **Prediction Generation**: 50 stocks in ~5 seconds
- **Data Update**: Latest prices for all stocks in ~30 seconds
- **Initial Sync**: 2400+ stocks in ~2 minutes (one-time)

## 🛠️ Technologies

- **Backend**: Python 3, Flask, SQLite
- **Data**: pandas, numpy, yfinance
- **ML**: scikit-learn, (optional: TensorFlow, Prophet)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Visualization**: Chart.js, Axios

## 📝 Notes

- Database files (`*.db`) are excluded from git (too large)
- CSV files in `EOD/` are excluded (regenerable)
- First-time setup takes 5-10 minutes
- Subsequent launches are instant

## 🤝 Contributing

This is a personal trading analysis tool. Feel free to fork and customize for your needs.

## 📄 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This software is for educational and research purposes only. Stock trading involves risk. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## 🆘 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ for NSE stock traders**

Last Updated: November 2025
