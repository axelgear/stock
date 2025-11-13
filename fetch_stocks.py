"""
NSE Stock Data Fetcher
Downloads End-of-Day (EOD) data for all NSE equity stocks using yfinance
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import time

def fetch_nse_stocks(equity_file="EQUITY_L.csv", output_dir="EOD", period="20y"):
    """
    Fetch historical stock data for all NSE equities
    
    Args:
        equity_file: Path to EQUITY_L.csv file
        output_dir: Directory to save stock data
        period: Historical period (default 20y)
    """
    # Create directory to store EOD files
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NSE equity symbols
    print(f"Loading equity symbols from {equity_file}...")
    equity_details = pd.read_csv(equity_file)
    
    total_stocks = len(equity_details)
    successful = 0
    failed = 0
    
    print(f"\nFound {total_stocks} stocks to fetch")
    print(f"Fetching {period} of historical data...\n")
    print("-" * 70)
    
    # Loop through each symbol
    for idx, name in enumerate(equity_details.SYMBOL, 1):
        try:
            print(f"[{idx}/{total_stocks}] Fetching {name}...", end=" ")
            
            # Add .NS suffix for NSE stocks on Yahoo Finance
            data = yf.download(f"{name}.NS", period=period, progress=False)
            
            if not data.empty:
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Save to CSV
                output_path = os.path.join(output_dir, f"{name}.csv")
                data.to_csv(output_path, index=False)
                successful += 1
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
    print("-" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total stocks: {total_stocks}")
    print(f"   ✓ Successful: {successful}")
    print(f"   ✗ Failed: {failed}")
    print(f"   📁 Data saved in: {output_dir}/")
    print(f"\n✨ Fetch completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    fetch_nse_stocks()

