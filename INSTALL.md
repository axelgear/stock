# 🚀 Installation Guide

## Quick Install (Recommended)

### Using the automated script:

```bash
cd /www/wwwroot/axel/TRADING
./quick_start.sh
```

The script will automatically:
- Create a virtual environment
- Install all dependencies
- Guide you through the setup

---

## Manual Installation

### Step 1: Create Virtual Environment

```bash
cd /www/wwwroot/axel/TRADING
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, yfinance, sklearn, streamlit; print('✅ All packages installed successfully!')"
```

---

## Running Scripts with Virtual Environment

### Always activate the virtual environment first:

```bash
# Activate venv
source venv/bin/activate

# Now run any script
python fetch_stocks.py
python backtest.py
streamlit run dashboard.py
```

### One-line commands (no need to activate separately):

```bash
# Download stock data
source venv/bin/activate && python fetch_stocks.py

# Run backtest
source venv/bin/activate && python backtest.py

# Launch dashboard
source venv/bin/activate && streamlit run dashboard.py

# Train ML model
source venv/bin/activate && python ml_forecasting.py

# Process data
source venv/bin/activate && python data_engineering.py
```

---

## Helper Aliases (Optional)

Add these to your `~/.bashrc` or `~/.bash_aliases` for convenience:

```bash
# Navigate to project
alias trading='cd /www/wwwroot/axel/TRADING'

# Activate venv
alias vact='source /www/wwwroot/axel/TRADING/venv/bin/activate'

# Quick commands with auto-activation
alias trading-fetch='cd /www/wwwroot/axel/TRADING && source venv/bin/activate && python fetch_stocks.py'
alias trading-backtest='cd /www/wwwroot/axel/TRADING && source venv/bin/activate && python backtest.py'
alias trading-dashboard='cd /www/wwwroot/axel/TRADING && source venv/bin/activate && streamlit run dashboard.py'
alias trading-ml='cd /www/wwwroot/axel/TRADING && source venv/bin/activate && python ml_forecasting.py'
```

After adding, reload your shell:
```bash
source ~/.bashrc
```

Now you can simply type:
```bash
trading-dashboard  # Launches dashboard instantly!
```

---

## Troubleshooting

### Problem: "externally-managed-environment" error
**Solution**: You're trying to install packages system-wide. Always use the virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: Virtual environment not found
**Solution**: Create it first:
```bash
python3 -m venv venv
```

### Problem: "python: command not found"
**Solution**: Use `python3` instead:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Problem: Permission denied on quick_start.sh
**Solution**: Make it executable:
```bash
chmod +x quick_start.sh
```

### Problem: Old packages or conflicts
**Solution**: Recreate the virtual environment:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

---

## Installed Packages

After installation, you'll have:

### Data & Analysis
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Stock data download

### Machine Learning
- `scikit-learn` - ML models
- `joblib` - Model serialization

### Visualization
- `matplotlib` - Static plots
- `plotly` - Interactive charts
- `streamlit` - Web dashboard

### Utilities
- `requests` - HTTP requests
- `urllib3` - URL handling
- `python-dateutil` - Date parsing
- `pytz` - Timezone handling

---

## System Requirements

- **OS**: Linux (WSL2, Ubuntu, Debian, etc.)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for dependencies + data
- **Internet**: Required for downloading stock data

---

## Next Steps

After installation:

1. **Download stock data**:
   ```bash
   source venv/bin/activate && python fetch_stocks.py
   ```

2. **Launch dashboard**:
   ```bash
   source venv/bin/activate && streamlit run dashboard.py
   ```

3. **Read the documentation**:
   ```bash
   cat README.md
   cat USAGE_GUIDE.md
   ```

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate venv | `source venv/bin/activate` |
| Install packages | `pip install -r requirements.txt` |
| Download data | `python fetch_stocks.py` |
| Run backtest | `python backtest.py` |
| Launch dashboard | `streamlit run dashboard.py` |
| Train ML model | `python ml_forecasting.py` |
| Process data | `python data_engineering.py` |
| Deactivate venv | `deactivate` |

---

**You're all set! Happy trading! 📈**

