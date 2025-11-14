"""
Data Cleaning and Feature Engineering Utilities
Prepare and transform stock data for analysis and ML
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataEngineer:
    """Data cleaning and feature engineering for stock data"""
    
    def __init__(self, data):
        """
        Initialize data engineer
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.original_rows = len(data)
        
    def clean_data(self):
        """
        Clean stock data:
        - Remove duplicates
        - Handle missing values
        - Fix data types
        - Remove outliers
        """
        print("Cleaning data...")
        
        # Remove duplicates
        before_dups = len(self.data)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        after_dups = len(self.data)
        print(f"   ✓ Removed {before_dups - after_dups} duplicate rows")
        
        # Sort by date
        self.data = self.data.sort_index()
        
        # Handle missing values
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            print(f"   ⚠ Found {missing_count} missing values")
            # Forward fill for price data
            self.data = self.data.fillna(method='ffill')
            # Backward fill for any remaining
            self.data = self.data.fillna(method='bfill')
            print(f"   ✓ Filled missing values")
        
        # Remove rows with zero or negative prices
        before_invalid = len(self.data)
        self.data = self.data[
            (self.data['Open'] > 0) & 
            (self.data['High'] > 0) & 
            (self.data['Low'] > 0) & 
            (self.data['Close'] > 0)
        ]
        after_invalid = len(self.data)
        if before_invalid > after_invalid:
            print(f"   ✓ Removed {before_invalid - after_invalid} rows with invalid prices")
        
        # Check for price anomalies (High < Low, etc.)
        anomalies = self.data[
            (self.data['High'] < self.data['Low']) |
            (self.data['High'] < self.data['Open']) |
            (self.data['High'] < self.data['Close']) |
            (self.data['Low'] > self.data['Open']) |
            (self.data['Low'] > self.data['Close'])
        ]
        if len(anomalies) > 0:
            print(f"   ⚠ Found {len(anomalies)} rows with price anomalies")
            self.data = self.data.drop(anomalies.index)
            print(f"   ✓ Removed anomalous rows")
        
        # Remove extreme outliers using IQR method
        before_outliers = len(self.data)
        self.data = self._remove_outliers(self.data, 'Close')
        after_outliers = len(self.data)
        if before_outliers > after_outliers:
            print(f"   ✓ Removed {before_outliers - after_outliers} outlier rows")
        
        print(f"   ✓ Data cleaned: {self.original_rows} → {len(self.data)} rows")
        
        return self.data
    
    def _remove_outliers(self, df, column, threshold=3):
        """Remove outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def add_technical_indicators(self):
        """Add comprehensive technical indicators"""
        print("Adding technical indicators...")
        
        df = self.data
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Rate of Change (ROC)
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Average True Range (ATR) - Volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Price channels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
        self.data = df
        print(f"   ✓ Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']])} technical indicators")
        
        return self.data
    
    def add_time_features(self):
        """Add time-based features"""
        print("Adding time features...")
        
        df = self.data
        
        # Extract date components
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        df['DayOfYear'] = df.index.dayofyear
        df['WeekOfYear'] = df.index.isocalendar().week
        
        # Cyclical encoding for day of week and month
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Is month start/end
        df['IsMonthStart'] = df.index.is_month_start.astype(int)
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsQuarterStart'] = df.index.is_quarter_start.astype(int)
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        self.data = df
        print(f"   ✓ Added time-based features")
        
        return self.data
    
    def add_lag_features(self, columns=['Close', 'Volume'], lags=[1, 2, 3, 5, 10]):
        """Add lagged features"""
        print(f"Adding lag features...")
        
        df = self.data
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        print(f"   ✓ Added lag features for {len(columns)} columns")
        
        self.data = df
        return self.data
    
    def generate_report(self):
        """Generate data quality report"""
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total rows: {len(self.data):,}")
        print(f"   Total columns: {len(self.data.columns):,}")
        print(f"   Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
        print(f"   Duration: {(self.data.index.max() - self.data.index.min()).days} days")
        
        print(f"\n💹 Price Statistics:")
        print(f"   Highest Close: ₹{self.data['Close'].max():,.2f}")
        print(f"   Lowest Close: ₹{self.data['Close'].min():,.2f}")
        print(f"   Average Close: ₹{self.data['Close'].mean():,.2f}")
        print(f"   Latest Close: ₹{self.data['Close'].iloc[-1]:,.2f}")
        
        print(f"\n📈 Returns:")
        returns = self.data['Close'].pct_change()
        print(f"   Daily Avg Return: {returns.mean()*100:.2f}%")
        print(f"   Daily Volatility: {returns.std()*100:.2f}%")
        print(f"   Max Daily Gain: {returns.max()*100:.2f}%")
        print(f"   Max Daily Loss: {returns.min()*100:.2f}%")
        
        print(f"\n✅ Data Quality:")
        missing_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        print(f"   Missing values: {missing_pct:.2f}%")
        print(f"   Data completeness: {100-missing_pct:.2f}%")
        
        print("="*70 + "\n")

def process_stock_data(stock_file, output_dir="processed"):
    """
    Complete data processing pipeline
    
    Args:
        stock_file: Path to stock CSV file
        output_dir: Directory to save processed data
    """
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(stock_file)}")
    print(f"{'='*70}\n")
    
    # Load data
    data = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
    
    # Initialize engineer
    engineer = DataEngineer(data)
    
    # Clean data
    engineer.clean_data()
    
    # Add features
    engineer.add_technical_indicators()
    engineer.add_time_features()
    engineer.add_lag_features()
    
    # Generate report
    engineer.generate_report()
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    stock_name = os.path.basename(stock_file).replace('.csv', '')
    output_file = os.path.join(output_dir, f"{stock_name}_processed.csv")
    engineer.data.to_csv(output_file)
    print(f"✓ Processed data saved to: {output_file}")
    
    return engineer.data

if __name__ == "__main__":
    # Example: Process RELIANCE stock data
    eod_dir = "EOD"
    
    if os.path.exists(f"{eod_dir}/RELIANCE.csv"):
        processed_data = process_stock_data(f"{eod_dir}/RELIANCE.csv")
    else:
        print("Please run fetch_stocks.py first to download stock data!")

