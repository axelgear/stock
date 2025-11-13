#!/bin/bash
# Launch AI Stock Trading Dashboard

cd /www/wwwroot/axel/TRADING

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     🤖 AI STOCK TRADING DASHBOARD - STARTING...               ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
source venv/bin/activate

# Start backend API
echo "🚀 Starting Flask Backend API..."
echo "📡 API Server: http://localhost:5050/api"
echo "🌐 Dashboard: http://localhost:5050"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

python backend/app.py

