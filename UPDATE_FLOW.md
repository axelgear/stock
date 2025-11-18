# Update Database Flow

## Overview
The "Update Pending Data" button performs a complete sync of the database with the latest stock prices from NSE.

## Process Flow

### Step 1: Download Latest EQUITY_L.csv
**Source**: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv

The system first downloads the latest equity list from NSE. This ensures:
- ✅ New stocks added to NSE are automatically included
- ✅ Delisted stocks are removed
- ✅ Stock symbol changes are reflected

**File**: `fetch_stocks.py` → `download_equity_list()` function

### Step 2: Fetch Fresh Stock Data
For each stock in EQUITY_L.csv:
- Downloads complete historical data (default: 20 years)
- **Overwrites existing CSV files** in the EOD directory
- Uses yfinance API to fetch from Yahoo Finance

**File**: `fetch_stocks.py` → `fetch_nse_stocks()` function

### Step 3: Sync Pending Data to Database
For each stock CSV file:
- Checks the latest date in the database
- **Only inserts NEW records** (dates after the last database entry)
- Efficient incremental update (no full reload needed)
- Updates metadata tracking

**File**: `database.py` → `sync_pending_data_to_db()` function

## Key Features

### 🔄 Overwrite Behavior
- **EQUITY_L.csv**: Always downloaded fresh from NSE
- **Stock CSV files**: Completely overwritten with latest data
- **Database**: Only NEW records are added (incremental)

### 📊 Smart Sync
```
Database has data up to: 2024-01-15
CSV file has data up to: 2024-01-18
Result: Only 3 new records (16th, 17th, 18th) are inserted
```

### 🚀 Benefits
1. **Always Up-to-Date**: Fresh equity list ensures new stocks are included
2. **Efficient**: Only pending data is synced to database
3. **Complete**: Full historical data maintained in CSV files
4. **Fast Queries**: Database enables instant loading of all stocks

## Usage

### Initial Setup (One-Time)
```
Click "🔄 Sync Database First"
```
- Downloads all CSV files
- Loads complete historical data to database
- Takes ~2-5 minutes

### Daily Updates
```
Click "📅 Update Pending Data"
```
- Downloads fresh EQUITY_L.csv from NSE
- Re-downloads all stock CSV files with latest data
- Syncs only new records to database
- Takes ~2-5 minutes

### Quick Access
```
Click "⚡ Load from Database"
```
- Instantly loads all stocks from database
- Shows today's prices and 7-day performance
- Takes <1 second

## Technical Details

### API Endpoint
```
POST /api/db/update-today
```

### Response Format
```json
{
  "success": true,
  "message": "Synced 1890 stocks with 5670 new records",
  "results": {
    "success": 1890,
    "new_records": 5670,
    "no_new_data": 0,
    "failed": 0,
    "errors": []
  }
}
```

### Database Schema
- **stock_data**: Main price data (date, open, high, low, close, volume)
- **sync_metadata**: Tracks last sync time and record counts
- **predictions**: AI predictions cache
- **validation_results**: Model accuracy tracking

## Error Handling

### If EQUITY_L.csv Download Fails
- System uses existing local copy
- Continues with current stock list
- Warning logged

### If Stock CSV Fails
- Stock is skipped
- Error logged in results
- Process continues with other stocks

### If Database Sync Fails
- Transaction rolled back
- Detailed error message returned
- Database remains consistent

## Maintenance

### Database Optimization
```
Click "📊 DB Stats" to view:
- Total records
- Total stocks
- Database size
- Date range
```

### Clean Old Data
```python
# In database.py
clear_old_data(days_to_keep=730)  # Keep 2 years
```

## Files Modified

1. **fetch_stocks.py**: Added EQUITY_L.csv download, overwrite confirmation
2. **database.py**: Added `sync_pending_data_to_db()` for incremental updates
3. **app.py**: Updated `/api/db/update-today` endpoint
4. **app.js**: Enhanced UI feedback and confirmation
5. **index.html**: Updated button text and descriptions

## Dependencies

All required libraries are in `requirements.txt`:
- `requests>=2.31.0` - For EQUITY_L.csv download
- `yfinance>=0.2.28` - For stock data fetching
- `pandas>=2.0.0` - For data processing
- `flask>=3.0.0` - For backend API

## Best Practices

1. **Run Initial Sync First**: Use "Sync Database First" before first use
2. **Daily Updates**: Run "Update Pending Data" once daily after market close
3. **Monitor Results**: Check success/failure counts after updates
4. **Database Stats**: Periodically verify database health
5. **Backup**: Consider backing up `stock_cache.db` periodically

