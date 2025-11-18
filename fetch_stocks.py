"""
NSE Stock Data Fetcher
Downloads End-of-Day (EOD) data for all NSE equity stocks using yfinance
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import time
import requests

def download_equity_list(equity_file="EQUITY_L.csv"):
    """
    Download the latest EQUITY_L.csv from NSE
    This ensures we have the most up-to-date list of stocks
    """
    nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    
    print(f"📥 Downloading latest equity list from NSE...", flush=True)
    print(f"   URL: {nse_url}", flush=True)
    
    try:
        # Download with headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(nse_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save to file (overwrite existing)
        with open(equity_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded latest equity list: {equity_file}", flush=True)
        
        # Verify the file
        df = pd.read_csv(equity_file)
        print(f"   Found {len(df)} stocks in equity list", flush=True)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to download equity list: {e}", flush=True)
        print(f"   Will use existing {equity_file} if available", flush=True)
        return False

def fetch_nse_stocks(equity_file="EQUITY_L.csv", output_dir="EOD", period="20y"):
    """
    Fetch historical stock data for all NSE equities
    
    Args:
        equity_file: Path to EQUITY_L.csv file
        output_dir: Directory to save stock data
        period: Historical period (default 20y)
    """
    # Step 1: Download latest equity list from NSE
    download_equity_list(equity_file)
    
    # Create directory to store EOD files
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NSE equity symbols
    print(f"\n📊 Loading equity symbols from {equity_file}...", flush=True)
    equity_details = pd.read_csv(equity_file)
    
    total_stocks = len(equity_details)
    successful = 0
    failed = 0
    
    print(f"\nFound {total_stocks} stocks to fetch", flush=True)
    print(f"Fetching {period} of historical data...\n", flush=True)
    print("-" * 70, flush=True)
    
    # Loop through each symbol
    for idx, name in enumerate(equity_details.SYMBOL, 1):
        try:
            print(f"[{idx}/{total_stocks}] Fetching {name}...", end=" ", flush=True)
            
            # Add .NS suffix for NSE stocks on Yahoo Finance
            # auto_adjust=False to keep original OHLC values (not adjusted for splits/dividends)
            data = yf.download(f"{name}.NS", period=period, progress=False, auto_adjust=False)
            
            if not data.empty:
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Save to CSV (overwrites existing file)
                output_path = os.path.join(output_dir, f"{name}.csv")
                file_exists = os.path.exists(output_path)
                data.to_csv(output_path, index=False)
                successful += 1
                
                if file_exists:
                    print(f"✓ Updated {len(data)} records")
                else:
                    print(f"✓ Saved {len(data)} records")
            else:
                print(f"✗ No data available")
                failed += 1
                
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Summary
    print("-" * 70, flush=True)
    print(f"\n📊 Summary:", flush=True)
    print(f"   Total stocks: {total_stocks}", flush=True)
    print(f"   ✓ Successful: {successful}", flush=True)
    print(f"   ✗ Failed: {failed}", flush=True)
    print(f"   📁 Data saved in: {output_dir}/", flush=True)
    print(f"\n✨ Fetch completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

if __name__ == "__main__":
    fetch_nse_stocks()

