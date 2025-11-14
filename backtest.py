"""
Trading Strategy Backtesting Framework
Backtest various trading strategies using historical NSE stock data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

class Backtester:
    """Simple backtesting framework for trading strategies"""
    
    def __init__(self, data, initial_capital=100000):
        """
        Initialize backtester
        
        Args:
            data: DataFrame with OHLCV data
            initial_capital: Starting capital for backtesting
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        
    def moving_average_crossover(self, short_window=50, long_window=200):
        """
        Simple Moving Average Crossover Strategy
        Buy when short MA crosses above long MA, sell when it crosses below
        """
        # Calculate moving averages
        self.data['SMA_short'] = self.data['Close'].rolling(window=short_window).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['SMA_short'] > self.data['SMA_long'], 'Signal'] = 1
        self.data['Position'] = self.data['Signal'].diff()
        
        return self.data
    
    def rsi_strategy(self, period=14, oversold=30, overbought=70):
        """
        RSI (Relative Strength Index) Strategy
        Buy when RSI < oversold, sell when RSI > overbought
        """
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['RSI'] < oversold, 'Signal'] = 1  # Buy signal
        self.data.loc[self.data['RSI'] > overbought, 'Signal'] = -1  # Sell signal
        
        return self.data
    
    def calculate_returns(self):
        """Calculate strategy returns and performance metrics"""
        # Calculate daily returns
        self.data['Market_Return'] = self.data['Close'].pct_change()
        self.data['Strategy_Return'] = self.data['Market_Return'] * self.data['Signal'].shift(1)
        
        # Cumulative returns
        self.data['Cumulative_Market_Return'] = (1 + self.data['Market_Return']).cumprod()
        self.data['Cumulative_Strategy_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        
        # Portfolio value
        self.data['Portfolio_Value'] = self.initial_capital * self.data['Cumulative_Strategy_Return']
        
        return self.data
    
    def performance_metrics(self):
        """Calculate key performance metrics"""
        total_return = (self.data['Portfolio_Value'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = np.sqrt(252) * self.data['Strategy_Return'].mean() / self.data['Strategy_Return'].std()
        
        # Maximum Drawdown
        cumulative = self.data['Cumulative_Strategy_Return']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win Rate
        winning_trades = len(self.data[self.data['Strategy_Return'] > 0])
        total_trades = len(self.data[self.data['Strategy_Return'] != 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        metrics = {
            'Initial Capital': self.initial_capital,
            'Final Portfolio Value': self.data['Portfolio_Value'].iloc[-1],
            'Total Return (%)': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades
        }
        
        return metrics
    
    def plot_results(self, stock_name="Stock"):
        """Plot backtest results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot price and signals
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.7)
        
        if 'SMA_short' in self.data.columns:
            ax1.plot(self.data.index, self.data['SMA_short'], label='Short MA', alpha=0.7)
            ax1.plot(self.data.index, self.data['SMA_long'], label='Long MA', alpha=0.7)
        
        # Mark buy/sell signals
        buy_signals = self.data[self.data['Position'] == 1]
        sell_signals = self.data[self.data['Position'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', 
                   color='green', label='Buy Signal', s=100, zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', 
                   color='red', label='Sell Signal', s=100, zorder=5)
        
        ax1.set_title(f'{stock_name} - Price and Trading Signals')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot portfolio value
        ax2.plot(self.data.index, self.data['Portfolio_Value'], 
                label='Strategy Portfolio', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital', alpha=0.7)
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value (₹)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def run_backtest_example(stock_file, strategy='ma_crossover'):
    """
    Run a backtest example on a stock
    
    Args:
        stock_file: Path to stock CSV file
        strategy: Strategy to use ('ma_crossover' or 'rsi')
    """
    print(f"\n{'='*70}")
    print(f"Running Backtest: {os.path.basename(stock_file)}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    data = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
    
    # Initialize backtester
    backtester = Backtester(data, initial_capital=100000)
    
    # Apply strategy
    if strategy == 'ma_crossover':
        backtester.moving_average_crossover(short_window=50, long_window=200)
    elif strategy == 'rsi':
        backtester.rsi_strategy(period=14, oversold=30, overbought=70)
    
    # Calculate returns
    backtester.calculate_returns()
    
    # Get performance metrics
    metrics = backtester.performance_metrics()
    
    # Print results
    print("Performance Metrics:")
    print("-" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<50} {value:>15,.2f}")
        else:
            print(f"{key:.<50} {value:>15,}")
    
    return backtester, metrics

if __name__ == "__main__":
    # Example: Backtest RELIANCE stock with MA crossover strategy
    eod_dir = "EOD"
    
    if os.path.exists(f"{eod_dir}/RELIANCE.csv"):
        backtester, metrics = run_backtest_example(
            f"{eod_dir}/RELIANCE.csv", 
            strategy='ma_crossover'
        )
        
        # Save plot
        fig = backtester.plot_results("RELIANCE")
        plt.savefig("backtest_results.png", dpi=300, bbox_inches='tight')
        print(f"\n📊 Chart saved as: backtest_results.png")
    else:
        print("Please run fetch_stocks.py first to download stock data!")

