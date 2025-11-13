#!/bin/bash
# Test AI Prediction System

cd /www/wwwroot/axel/TRADING
source venv/bin/activate

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║          🧪 TESTING AI PREDICTION SYSTEM                      ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if data exists
if [ ! -d "EOD" ] || [ -z "$(ls -A EOD 2>/dev/null)" ]; then
    echo "❌ No stock data found!"
    echo "Please run: source venv/bin/activate && python fetch_stocks.py"
    exit 1
fi

# Test AI Predictor
echo "1️⃣  Testing AI Prediction Engine..."
echo "═══════════════════════════════════════════════════════════════"
python ai_predictor.py
echo ""

# Test Auto Validator
echo "2️⃣  Testing Auto-Validation System..."
echo "═══════════════════════════════════════════════════════════════"
python auto_validator.py
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║          ✅ ALL TESTS COMPLETED                               ║"
echo "║                                                                ║"
echo "║  Ready to launch dashboard: ./run_dashboard.sh                ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"

