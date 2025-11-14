#!/usr/bin/env python3
"""
Quick test script to verify dashboard setup
"""

import os
import sys

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║           🧪 TESTING DASHBOARD SETUP                             ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# Test 1: Check files exist
print("1️⃣  Checking files...")
files_to_check = [
    'frontend/index.html',
    'frontend/app.js',
    'backend/app.py',
    'ai_predictor.py',
    'auto_validator.py',
    'EQUITY_L.csv'
]

all_ok = True
for file in files_to_check:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
    if not exists:
        all_ok = False

print()

# Test 2: Check virtual environment
print("2️⃣  Checking Python environment...")
python_path = sys.executable
print(f"   Python: {python_path}")
if 'venv' in python_path:
    print("   ✅ Virtual environment active")
else:
    print("   ⚠️  Not using virtual environment")

print()

# Test 3: Check imports
print("3️⃣  Checking dependencies...")
try:
    import flask
    print(f"   ✅ Flask {flask.__version__}")
except ImportError:
    print("   ❌ Flask not installed")
    all_ok = False

try:
    import flask_cors
    print("   ✅ Flask-CORS installed")
except ImportError:
    print("   ❌ Flask-CORS not installed")
    all_ok = False

try:
    import pandas
    print(f"   ✅ Pandas {pandas.__version__}")
except ImportError:
    print("   ❌ Pandas not installed")
    all_ok = False

try:
    import sklearn
    print(f"   ✅ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("   ❌ Scikit-learn not installed")
    all_ok = False

print()

# Test 4: Check EOD directory
print("4️⃣  Checking stock data...")
if os.path.exists('EOD'):
    stock_count = len([f for f in os.listdir('EOD') if f.endswith('.csv')])
    print(f"   ✅ EOD directory exists with {stock_count} stocks")
    if stock_count == 0:
        print("   ⚠️  No stock data downloaded yet")
        print("   Run: python fetch_stocks.py")
else:
    print("   ❌ EOD directory not found")
    print("   Run: python fetch_stocks.py")

print()

# Test 5: Try importing backend
print("5️⃣  Testing backend import...")
try:
    sys.path.insert(0, os.getcwd())
    from backend.app import app
    print("   ✅ Backend imports successfully")
except Exception as e:
    print(f"   ❌ Backend import failed: {e}")
    all_ok = False

print()

# Final result
print("═" * 70)
if all_ok:
    print("✅ ALL TESTS PASSED!")
    print()
    print("🚀 Ready to launch dashboard:")
    print("   ./run_dashboard.sh")
    print()
    print("   Then open: http://localhost:5050")
else:
    print("❌ SOME TESTS FAILED")
    print()
    print("Fix issues above before launching dashboard")

print("═" * 70)

