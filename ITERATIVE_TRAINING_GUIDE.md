# 🎲 Iterative Training with Random Sampling

## 🎯 What's New?

**Random Stock Sampling** - Train on different random stocks each run to build a more robust model through multiple iterations!

---

## 🔥 The Concept

### **Problem with Sequential Training:**
```bash
# Old way (always same stocks):
python universal_ai_trainer.py --max-stocks 10
# Trains on: Stock1, Stock2, Stock3, ... Stock10

python universal_ai_trainer.py --max-stocks 10
# Trains on: Stock1, Stock2, Stock3, ... Stock10 (SAME!)
```

### **New Way - Random Sampling:**
```bash
# First run:
python universal_ai_trainer.py --max-stocks 10
# Trains on: RELIANCE, TCS, INFY, WIPRO, HDFC, ...

# Second run:
python universal_ai_trainer.py --max-stocks 10
# Trains on: TATAMOTORS, ITC, SBIN, BHARTIARTL, ... (DIFFERENT!)

# Third run:
python universal_ai_trainer.py --max-stocks 10
# Trains on: HCLTECH, LT, AXISBANK, ... (DIFFERENT AGAIN!)
```

---

## 🚀 How It Works

### **Automatic Random Sampling:**
By default, when you use `--max-stocks`, the trainer will:
1. ✅ Find all 2000 stock files in your `EOD/` directory
2. ✅ **Randomly shuffle** the list
3. ✅ Select the first N stocks
4. ✅ Train on those stocks

Each run = **different random stocks** = **better learning across market!**

---

## 💡 Use Cases

### **1. Quick Iterative Training**
Train multiple times on small batches to explore different market segments:

```bash
# Run 1: Random 50 stocks
python universal_ai_trainer.py --max-stocks 50

# Run 2: Different random 50 stocks
python universal_ai_trainer.py --max-stocks 50

# Run 3: Another different random 50 stocks
python universal_ai_trainer.py --max-stocks 50
```

**Result:** You've now tested the model on 150 stocks total (with potential overlap), exploring different sectors and patterns!

### **2. Resource-Constrained Training**
Train on different subsets when you can't train on all stocks at once:

```bash
# Limited time? Train on 200 random stocks (30 min)
python universal_ai_trainer.py --max-stocks 200

# Next day: Train on another 200 random stocks
python universal_ai_trainer.py --max-stocks 200

# Next week: Train on another 200 random stocks
python universal_ai_trainer.py --max-stocks 200
```

### **3. Model Robustness Testing**
Check if your model is stable across different stock combinations:

```bash
# Test 1:
python universal_ai_trainer.py --max-stocks 100
# Result: 75% accuracy

# Test 2:
python universal_ai_trainer.py --max-stocks 100
# Result: 74% accuracy

# Test 3:
python universal_ai_trainer.py --max-stocks 100
# Result: 76% accuracy

# Conclusion: Model is stable! (75% ± 1%)
```

---

## 🎮 Commands

### **Random Sampling (Default):**
```bash
# These all use random sampling:
python universal_ai_trainer.py --max-stocks 10
python universal_ai_trainer.py --max-stocks 50
python universal_ai_trainer.py --max-stocks 100
```

### **Disable Random Sampling:**
```bash
# Use first N stocks (deterministic):
python universal_ai_trainer.py --max-stocks 10 --no-random

# Will always use the same 10 stocks
```

### **Train on ALL Stocks:**
```bash
# No random sampling needed (uses all stocks):
python universal_ai_trainer.py --all
```

---

## 📊 What You'll See

### **With Random Sampling (Default):**
```
======================================================================
📊 LOADING ALL STOCKS DATA
======================================================================
📈 Found 2190 stock files, randomly sampling 10 stocks
   🎲 Random sampling enabled - different stocks each run!
```

### **Without Random Sampling:**
```
======================================================================
📊 LOADING ALL STOCKS DATA
======================================================================
📈 Found 2190 stock files, loading first 10 stocks
```

---

## 🎯 Recommended Strategy

### **Strategy 1: Quick Exploration (5 runs × 2 min = 10 min)**
```bash
for i in {1..5}; do
  echo "=== Training iteration $i ==="
  python universal_ai_trainer.py --max-stocks 50
  echo ""
done
```

Each run trains on different 50 stocks, giving you a feel for model performance across 250+ different stocks!

### **Strategy 2: Daily Training (1 run/day)**
```bash
# Every day, train on 100 random stocks (10 min)
0 9 * * * cd /www/wwwroot/axel/TRADING && python universal_ai_trainer.py --max-stocks 100

# After 30 days: You've covered most stocks multiple times!
```

### **Strategy 3: Final Production Model (1 run)**
```bash
# When you're ready, train on ALL stocks:
python universal_ai_trainer.py --all

# This is your production model!
```

---

## 🔬 Advanced: Iterative Model Improvement

### **Can I Combine Multiple Runs?**

**No** - Each run **overwrites** the previous models. But you can:

1. **Keep best performing model:**
   ```bash
   # Run 1:
   python universal_ai_trainer.py --max-stocks 100
   # Accuracy: 73%
   
   # Run 2:
   python universal_ai_trainer.py --max-stocks 100
   # Accuracy: 76% ✅ Better!
   
   # Keep this one, it's better!
   ```

2. **Manual ensemble (advanced):**
   Save models to different directories:
   ```bash
   python universal_ai_trainer.py --max-stocks 100 --models-dir models_v1
   python universal_ai_trainer.py --max-stocks 100 --models-dir models_v2
   python universal_ai_trainer.py --max-stocks 100 --models-dir models_v3
   
   # Then load and ensemble them in your backend
   ```

3. **Incremental Learning (not supported yet):**
   This would require `warm_start=True` and loading previous model weights. Currently, each run trains from scratch.

---

## 📈 Expected Results

### **Single Run (100 stocks):**
```
Random Forest: 74% accuracy
Gradient Boosting: 75% accuracy
XGBoost: 76% accuracy
```

### **After 5 Random Runs:**
You'll observe:
- Average accuracy: ~75% ± 2%
- Some runs better (78%), some worse (72%)
- Stable = good model!
- Unstable = need more data or different features

### **Full Training (ALL stocks):**
```
Random Forest: 76% accuracy
Gradient Boosting: 78% accuracy
XGBoost: 79% accuracy
```

Best results always come from training on ALL stocks!

---

## 🔍 Debug: Which Stocks Were Used?

The trainer doesn't currently log which stocks were selected, but you can modify the code:

```python
# In load_all_stocks_data(), after line 90, add:
if max_stocks and random_sample:
    with open('training_stocks.txt', 'w') as f:
        for file in stock_files[:20]:  # Log first 20
            stock_name = os.path.basename(file).replace('.csv', '')
            f.write(f"{stock_name}\n")
    print(f"   📝 Logged selected stocks to training_stocks.txt")
```

---

## ⚠️ Important Notes

1. **Random sampling ONLY applies when `--max-stocks` is used**
   - `--all` always uses all stocks
   - No `--max-stocks` = uses all stocks

2. **Models are OVERWRITTEN each run**
   - Backup `universal_models/` if you want to keep old models
   - Or use `--models-dir custom_name` to save to different location

3. **Reproducibility:**
   - No seed control (truly random each run)
   - Use `--no-random` if you need deterministic training

4. **Minimum stocks for meaningful results:**
   - < 50 stocks: Poor results
   - 50-100 stocks: Decent for testing
   - 100-500 stocks: Good
   - 500+ stocks: Excellent
   - ALL stocks: Best!

---

## 🎯 Summary

### **What Changed:**
✅ Added `random_sample=True` parameter (default)  
✅ Automatically shuffles stock list before selecting  
✅ Different stocks each run with `--max-stocks`  
✅ Can disable with `--no-random`  

### **Quick Commands:**
```bash
# Random 10 stocks (different each time)
python universal_ai_trainer.py --max-stocks 10

# Random 100 stocks (different each time)
python universal_ai_trainer.py --max-stocks 100

# First 10 stocks (same each time)
python universal_ai_trainer.py --max-stocks 10 --no-random

# All stocks (best results)
python universal_ai_trainer.py --all
```

### **Best Practice:**
1. **Test:** Run 3-5 times with `--max-stocks 100` to check stability
2. **Production:** Train once with `--all` for final model
3. **Daily:** Automate with `--max-stocks 200` cron job

Now you can train iteratively on different stock combinations! 🎲🚀


