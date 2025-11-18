#!/usr/bin/env python3
"""
Test script to verify real-time output from fetch_stocks.py
This simulates what the backend does when you click "Update Pending Data"
"""

import subprocess
import sys
import os

def test_realtime_output():
    print("="*70)
    print("Testing Real-Time Output from fetch_stocks.py")
    print("="*70)
    print("\nThis test will fetch a few stocks to verify output streams correctly.")
    print("Watch for real-time progress updates below:\n")
    
    fetch_stocks_path = os.path.join(os.path.dirname(__file__), 'fetch_stocks.py')
    
    try:
        # Run with real-time output streaming
        process = subprocess.Popen(
            [sys.executable, fetch_stocks_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        print("🔄 Process started, streaming output:\n")
        
        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Test completed successfully!")
            print("Real-time output is working correctly.")
        else:
            print(f"\n⚠️ Process exited with code {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_realtime_output()

