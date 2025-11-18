# Troubleshooting Guide - Update Pending Data

## Issue: "Nothing showing after Step 1"

### What Should Happen

When you click "📅 Update Pending Data", you should see:

```
======================================================================
🔄 UPDATING DATABASE WITH PENDING DATA
======================================================================

📥 Step 1: Downloading fresh CSV files using fetch_stocks.py...
   This will take 2-5 minutes depending on network speed...

📥 Downloading latest equity list from NSE...
   URL: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
✅ Downloaded latest equity list: EQUITY_L.csv
   Found 1890 stocks in equity list

📊 Loading equity symbols from EQUITY_L.csv...

Found 1890 stocks to fetch
Fetching 20y of historical data...

----------------------------------------------------------------------
[1/1890] Fetching RELIANCE... ✓ Updated 5024 records
[2/1890] Fetching TCS... ✓ Updated 4823 records
[3/1890] Fetching INFY... ✓ Updated 4981 records
...
```

### If You See Nothing After "Step 1"

This means the output buffering is preventing real-time display. Here are the solutions:

## Solution 1: Check Backend Console/Logs

The progress is being printed to the **Flask backend console**, not the browser.

**Where to look:**
- If running Flask manually: Check the terminal where you started the backend
- If running as a service: Check the service logs
- Linux: `journalctl -u your-service-name -f`
- Or check the log file if configured

## Solution 2: Test Real-Time Output

Run this test to verify output streaming works:

```bash
cd /www/wwwroot/axel/TRADING
python3 test_realtime_output.py
```

You should see real-time progress updates. If you do, the system is working correctly.

## Solution 3: Run fetch_stocks.py Manually

To see if it's working, run directly:

```bash
cd /www/wwwroot/axel/TRADING
python3 fetch_stocks.py
```

You should see:
1. Download of EQUITY_L.csv from NSE
2. Progress for each stock being fetched
3. Summary at the end

## Solution 4: Check Backend is Running

Make sure the Flask backend is running:

```bash
ps aux | grep app.py
```

If not running:

```bash
cd /www/wwwroot/axel/TRADING/backend
python3 app.py
```

## Solution 5: Use Backend API Directly

Test the API endpoint directly:

```bash
curl -X POST http://localhost:5050/api/db/update-today
```

This will show you the response and any errors.

## Common Issues

### Issue 1: Network Timeout

**Symptom**: Process hangs on EQUITY_L.csv download

**Solution**: 
- Check internet connection
- Verify NSE website is accessible: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
- Try accessing in browser first

**Fix**: If download fails, it uses existing EQUITY_L.csv and continues

### Issue 2: yfinance Rate Limiting

**Symptom**: Many "✗ No data available" errors

**Solution**:
- The script has 0.1 second delay between stocks
- If rate-limited, wait 15 minutes and try again
- Consider increasing delay in fetch_stocks.py (line 107)

### Issue 3: Timeout After 10 Minutes

**Symptom**: "CSV download timeout" error

**Solution**:
- Increase timeout in app.py (line 778): `process.wait(timeout=1200)` for 20 minutes
- Or run fetch_stocks.py manually first, then use "Load from Database"

### Issue 4: Browser Timeout

**Symptom**: Browser shows "Request timeout" or connection lost

**Solution**: This is normal for long operations. The process continues in the background.

**Check if still running**:
```bash
ps aux | grep fetch_stocks.py
```

**Monitor progress**:
```bash
# Watch the EOD directory
ls -lrt /www/wwwroot/axel/TRADING/EOD/ | tail -20

# Count files being updated
watch -n 5 'ls /www/wwwroot/axel/TRADING/EOD/ | wc -l'
```

## Verification

After the process completes, verify it worked:

### 1. Check CSV Files
```bash
cd /www/wwwroot/axel/TRADING/EOD
ls -lh | head -10  # Should show recent timestamps
```

### 2. Check Database Stats
Click "📊 DB Stats" button or:
```bash
sqlite3 /www/wwwroot/axel/TRADING/stock_cache.db "SELECT COUNT(*) FROM stock_data;"
```

### 3. Load from Database
Click "⚡ Load from Database" - should load instantly with latest data

## Expected Timeline

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| EQUITY_L.csv download | 2-5 sec | Downloading stock list from NSE |
| Stock data fetch | 2-5 min | Fetching ~1890 stocks from Yahoo Finance |
| Database sync | 30-60 sec | Syncing new records to database |
| **Total** | **3-7 min** | Complete update cycle |

## Best Practices

1. **Run during off-peak hours**: Less network congestion
2. **Monitor first run**: Watch backend console for any errors
3. **Don't close browser immediately**: Wait for success message
4. **Check logs regularly**: Identify pattern of any failures

## Alternative: Manual Update Process

If automated update keeps failing, do it manually:

```bash
# Step 1: Download fresh CSV files
cd /www/wwwroot/axel/TRADING
python3 fetch_stocks.py

# Step 2: Sync to database using Python
python3 -c "
from backend.database import sync_pending_data_to_db
from backend.app import load_stock_csv
import glob
import os

eod_dir = 'EOD'
for file_path in glob.glob(os.path.join(eod_dir, '*.csv')):
    stock_name = os.path.basename(file_path).replace('.csv', '')
    df = load_stock_csv(file_path)
    success, result = sync_pending_data_to_db(stock_name, df)
    if success and result > 0:
        print(f'{stock_name}: +{result} records')
"
```

## Still Having Issues?

Check these files for errors:

1. **Backend logs**: Where Flask is running
2. **fetch_stocks.py**: Line-by-line progress
3. **Network connectivity**: Can reach NSE and Yahoo Finance
4. **Disk space**: Enough space for CSV files (~500MB total)
5. **Permissions**: Can write to EOD directory and database file

## Contact/Debug Info

If reporting issues, provide:
- Backend console output
- Output from `python3 fetch_stocks.py`
- Database stats
- Any error messages
- OS and Python version

