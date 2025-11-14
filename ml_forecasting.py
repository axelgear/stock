"""
Machine Learning Stock Forecasting
Train ML models to predict stock prices using historical data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class StockForecaster:
    """ML-based stock price forecasting"""
    
    def __init__(self, data):
        """
        Initialize forecaster
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_features(self, lookback_days=60):
        """
        Create technical indicators and features for ML
        
        Args:
            lookback_days: Number of days to use for lagged features
        """
        df = self.data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
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
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        self.data = df
        return df
    
    def prepare_data(self, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            test_size: Proportion of data for testing
        """
        # Select feature columns (exclude target and date-related columns)
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[self.feature_columns]
        y = self.data['Target']
        
        # Split data (time series split - no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, model_type='random_forest'):
        """
        Train ML model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'linear')
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred,
                'actual_train': y_train,
                'actual_test': y_test
            }
        }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def feature_importance(self, top_n=15):
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            return None
    
    def save_model(self, filename='stock_model.pkl'):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='stock_model.pkl'):
        """Load trained model"""
        saved_data = joblib.load(filename)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_columns = saved_data['feature_columns']
        print(f"Model loaded from {filename}")

def train_forecasting_model(stock_file, model_type='random_forest'):
    """
    Train a forecasting model for a stock
    
    Args:
        stock_file: Path to stock CSV file
        model_type: ML model type
    """
    print(f"\n{'='*70}")
    print(f"Training ML Model: {os.path.basename(stock_file)}")
    print(f"Model Type: {model_type.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    data = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
    
    # Initialize forecaster
    forecaster = StockForecaster(data)
    
    # Create features
    print("Creating features...")
    forecaster.create_features()
    print(f"✓ Created {len(forecaster.feature_columns)} features")
    
    # Train model
    results = forecaster.train_model(model_type)
    
    # Print results
    print("\n" + "="*70)
    print("Training Results:")
    print("="*70)
    
    print("\n📈 Training Set Metrics:")
    for metric, value in results['train'].items():
        print(f"   {metric:.<20} {value:>15.4f}")
    
    print("\n📊 Test Set Metrics:")
    for metric, value in results['test'].items():
        print(f"   {metric:.<20} {value:>15.4f}")
    
    # Feature importance
    importance = forecaster.feature_importance()
    if importance is not None:
        print("\n🎯 Top 10 Important Features:")
        print(importance.head(10).to_string(index=False))
    
    # Save model
    stock_name = os.path.basename(stock_file).replace('.csv', '')
    model_filename = f"models/{stock_name}_{model_type}_model.pkl"
    os.makedirs("models", exist_ok=True)
    forecaster.save_model(model_filename)
    
    return forecaster, results

if __name__ == "__main__":
    # Example: Train model for RELIANCE stock
    eod_dir = "EOD"
    
    if os.path.exists(f"{eod_dir}/RELIANCE.csv"):
        forecaster, results = train_forecasting_model(
            f"{eod_dir}/RELIANCE.csv",
            model_type='random_forest'
        )
    else:
        print("Please run fetch_stocks.py first to download stock data!")

