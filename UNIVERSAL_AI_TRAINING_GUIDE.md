# 🌍 Universal AI Training - Command Line Guide

## Quick Start

### 1️⃣ **Train on ALL Stocks (Recommended)**
```bash
cd /www/wwwroot/axel/TRADING
python universal_ai_trainer.py --all
```

This will:
- ✅ Load ALL stocks from `EOD/` directory
- ✅ Train Random Forest & Gradient Boosting models
- ✅ Save models to `universal_models/` directory
- ✅ Display training metrics (accuracy, R², MAE)

**Time:** 10-30 minutes depending on dataset size

---

### 2️⃣ **Train on Limited Stocks (Testing)**
```bash
python universal_ai_trainer.py --max-stocks 100
```

Good for testing the system quickly.

---

### 3️⃣ **Train and Test Prediction**
```bash
python universal_ai_trainer.py --all --predict RELIANCE
```

This will train the models and immediately test prediction on RELIANCE stock.

---

## Command-Line Options

### Basic Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--all` | Train on ALL stocks | - | `--all` |
| `--max-stocks N` | Limit to N stocks | 100 | `--max-stocks 200` |
| `--min-data-points N` | Min rows per stock | 200 | `--min-data-points 300` |
| `--predict STOCK` | Test prediction after training | - | `--predict TATAMOTORS` |
| `--no-save` | Skip saving models | - | `--no-save` |

### Advanced Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--eod-dir PATH` | Path to EOD data | `EOD` | `--eod-dir /path/to/data` |
| `--models-dir PATH` | Where to save models | `universal_models` | `--models-dir my_models` |

---

## Usage Examples

### Example 1: Quick Test (100 stocks)
```bash
cd /www/wwwroot/axel/TRADING
python universal_ai_trainer.py --max-stocks 100 --predict TCS
```

**Output:**
```
🌍 Training on 100 stocks
======================================================================
📊 LOADING ALL STOCKS DATA
======================================================================
📈 Found 100 stock files
   ✓ Processed 50/100 stocks...
   ✓ Processed 100/100 stocks...

✅ Successfully loaded 95 stocks
⏭️ Skipped 5 stocks (insufficient data)

💾 Combining 95 stock dataframes...
📊 Combined dataset: 450,000 records from 95 stocks
...
🎉 TRAINING COMPLETE!
   Random Forest: 75.3% accuracy
   Gradient Boosting: 76.8% accuracy
   
💾 Saving models to universal_models/
✅ Models saved successfully!

======================================================================
🔮 TESTING PREDICTION: TCS
======================================================================

📊 Prediction for TCS:
   Current Price: ₹3,245.50
   Predicted Price: ₹3,289.75
   Expected Change: +1.36%
   Confidence: 78.5%
   Direction: UP
   Model: ensemble
```

---

### Example 2: Full Training (ALL stocks)
```bash
python universal_ai_trainer.py --all
```

**This is what you want for production use!**

---

### Example 3: Custom Paths
```bash
python universal_ai_trainer.py \
  --all \
  --eod-dir /custom/path/to/EOD \
  --models-dir /custom/models/location
```

---

### Example 4: Train Without Saving
```bash
python universal_ai_trainer.py --all --no-save
```

Useful for testing or benchmarking.

---

## Help Command

```bash
python universal_ai_trainer.py --help
```

**Output:**
```
usage: universal_ai_trainer.py [-h] [--all] [--max-stocks MAX_STOCKS]
                               [--min-data-points MIN_DATA_POINTS]
                               [--eod-dir EOD_DIR] [--models-dir MODELS_DIR]
                               [--predict PREDICT] [--no-save]

🌍 Universal Stock AI Trainer - Learn from ALL stocks

optional arguments:
  -h, --help            show this help message and exit
  --all                 Train on ALL stocks (no limit)
  --max-stocks MAX_STOCKS
                        Maximum number of stocks to train on (default: all)
  --min-data-points MIN_DATA_POINTS
                        Minimum data points required per stock (default: 200)
  --eod-dir EOD_DIR     Path to EOD data directory (default: EOD)
  --models-dir MODELS_DIR
                        Path to save trained models (default: universal_models)
  --predict PREDICT     Stock name to predict after training (e.g., RELIANCE)
  --no-save             Skip saving models to disk

Examples:
  # Train on ALL stocks (recommended)
  python universal_ai_trainer.py --all
  
  # Train on first 100 stocks (for testing)
  python universal_ai_trainer.py --max-stocks 100
  
  # Train and predict specific stock
  python universal_ai_trainer.py --all --predict RELIANCE
  
  # Train with custom paths
  python universal_ai_trainer.py --eod-dir /path/to/EOD --models-dir /path/to/models
```

---

## Monitoring During Training

The script will show progress:

```
📊 LOADING ALL STOCKS DATA
📈 Found 500 stock files
   ✓ Processed 50/500 stocks...
   ✓ Processed 100/500 stocks...
   ...

🔧 Engineering universal features...

📊 PREPARING TRAINING DATA
   ✓ Features: 45
   ✓ Training samples: 380,000
   ✓ Test samples: 95,000

🌲 Training Random Forest on combined data...
   Progress: ████████████████████ 100%
   ✓ Train R²: 0.8234
   ✓ Test R²: 0.7156
   ✓ Direction Accuracy: 75.34%

📈 Training Gradient Boosting on combined data...
   Progress: ████████████████████ 100%
   ✓ Train R²: 0.8567
   ✓ Test R²: 0.7489
   ✓ Direction Accuracy: 76.82%
```

---

## Output Files

After training, you'll have:

```
TRADING/
├── universal_models/
│   ├── random_forest.pkl          # Random Forest model
│   ├── gradient_boosting.pkl      # Gradient Boosting model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder.pkl          # Stock name encoder
│   └── feature_columns.pkl        # Feature list
```

These files are used by:
- ✅ Backend API (`/api/db/predict-universal/<stock>`)
- ✅ Frontend Dashboard ("Universal Predictions" button)

---

## Troubleshooting

### Error: "No valid stock data loaded!"
**Cause:** CSV files are not in the expected format or directory is wrong.

**Fix:**
```bash
# Check if EOD directory exists
ls EOD/

# Verify CSV format
head -5 EOD/RELIANCE.csv

# Use correct path
python universal_ai_trainer.py --all --eod-dir /correct/path/to/EOD
```

---

### Error: "ModuleNotFoundError: No module named 'sklearn'"
**Cause:** Missing dependencies.

**Fix:**
```bash
pip install scikit-learn pandas numpy joblib
```

---

### Training is too slow
**Solutions:**

1. **Use fewer stocks for testing:**
   ```bash
   python universal_ai_trainer.py --max-stocks 50
   ```

2. **Check RAM usage:**
   ```bash
   htop  # Monitor RAM
   ```

3. **Reduce min-data-points:**
   ```bash
   python universal_ai_trainer.py --all --min-data-points 100
   ```

---

## Integration with Backend

After training, the models are automatically available in your Flask backend:

```python
# Backend will use these models
POST /api/db/train-universal       # Trains the models
GET  /api/db/predict-universal/<stock>  # Predicts using universal model
POST /api/db/predict-all-universal # Bulk predictions
```

---

## Best Practices

### ✅ **DO:**
- Use `--all` for production training
- Train regularly (weekly/monthly) with new data
- Test predictions with `--predict` after training
- Monitor training metrics (accuracy > 70% is good)

### ❌ **DON'T:**
- Don't use `--max-stocks` in production (testing only)
- Don't skip saving models (`--no-save`) in production
- Don't train on very old data only
- Don't ignore low accuracy warnings

---

## Performance Expectations

| Dataset Size | Training Time | Expected Accuracy |
|--------------|---------------|-------------------|
| 50 stocks | 2-5 minutes | 70-75% |
| 100 stocks | 5-10 minutes | 73-77% |
| 200 stocks | 10-20 minutes | 75-79% |
| 500+ stocks | 20-40 minutes | 76-82% |

**Note:** More stocks = better generalization!

---

## Cron Job (Automatic Training)

To train automatically every week:

```bash
# Edit crontab
crontab -e

# Add this line (trains every Sunday at 2 AM)
0 2 * * 0 cd /www/wwwroot/axel/TRADING && python universal_ai_trainer.py --all >> training.log 2>&1
```

---

## Summary

### Most Common Commands:

```bash
# Full production training
python universal_ai_trainer.py --all

# Quick test
python universal_ai_trainer.py --max-stocks 100

# Train + test prediction
python universal_ai_trainer.py --all --predict RELIANCE
```

That's it! 🚀

