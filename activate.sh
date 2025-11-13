#!/bin/bash
# Quick activation script for TRADING project

echo "🚀 Activating NSE Trading Analysis System..."
source /www/wwwroot/axel/TRADING/venv/bin/activate
echo "✅ Virtual environment activated!"
echo ""
echo "Available commands:"
echo "  python fetch_stocks.py      - Download stock data"
echo "  python backtest.py          - Run backtesting"
echo "  python ml_forecasting.py    - Train ML models"
echo "  streamlit run dashboard.py  - Launch dashboard"
echo ""
echo "Type 'deactivate' to exit the virtual environment"
