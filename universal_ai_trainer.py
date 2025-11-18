"""
Universal AI Stock Predictor - Train on ALL stocks combined
This predictor learns from patterns across the entire market rather than individual stocks
By combining data from all stocks, it can learn universal market patterns and trends
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM models disabled.")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Try to import XGBoost for GPU support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class UniversalStockPredictor:
    """
    Train a single model on data from ALL stocks to learn universal market patterns
    
    Benefits:
    1. Learns patterns that work across entire market
    2. Can identify sector trends and correlations
    3. More robust predictions with larger training dataset
    4. Can predict stocks with limited historical data
    """
    
    def __init__(self, eod_directory='EOD'):
        """
        Initialize universal predictor
        
        Args:
            eod_directory: Directory containing all stock CSV files
        """
        self.eod_directory = eod_directory
        self.models = {}
        self.metrics = {}  # Store training metrics
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = []
        self.combined_data = None
        
        print(f"🌍 Universal Stock Predictor initialized")
        print(f"📁 EOD Directory: {eod_directory}")
    
    def load_all_stocks_data(self, max_stocks=None, min_data_points=2, random_sample=True):
        """
        Load and combine data from all stocks - BACKEND CAN USE RAM
        
        Args:
            max_stocks: Maximum number of stocks to load (None for all)
            min_data_points: Minimum data points required per stock
            random_sample: If True, randomly sample stocks (default: True)
            
        Returns:
            DataFrame with combined data from all stocks
        """
        print(f"\n{'='*70}")
        print("📊 LOADING ALL STOCKS DATA")
        print(f"{'='*70}")
        
        stock_files = glob.glob(os.path.join(self.eod_directory, '*.csv'))
        total_available = len(stock_files)
        
        # RANDOM SAMPLING: Shuffle to get different stocks each run
        if max_stocks and random_sample:
            import random
            random.shuffle(stock_files)
            stock_files = stock_files[:max_stocks]
            print(f"📈 Found {total_available} stock files, randomly sampling {len(stock_files)} stocks")
            print(f"   🎲 Random sampling enabled - different stocks each run!")
        elif max_stocks:
            stock_files = stock_files[:max_stocks]
            print(f"📈 Found {total_available} stock files, loading first {len(stock_files)} stocks")
        else:
            print(f"📈 Found {len(stock_files)} stock files, loading ALL stocks")
        
        all_data = []
        success_count = 0
        skip_count = 0
        
        for i, file_path in enumerate(stock_files, 1):
            stock_name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                # Read CSV - skip metadata rows (rows 1-2)
                df = pd.read_csv(file_path, skiprows=[1, 2])
                
                # Debug first file
                if i == 1:
                    print(f"   📄 First file columns: {df.columns.tolist()}")
                    print(f"   📄 First few rows:")
                    print(df.head(2).to_string())
                
                # Rename 'Price' column to 'Date' (your CSVs have this)
                if 'Price' in df.columns and 'Date' not in df.columns:
                    df = df.rename(columns={'Price': 'Date'})
                
                # Standardize column names (case insensitive)
                df.columns = df.columns.str.strip()
                column_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'date' in col_lower or 'price' in col_lower:
                        column_map[col] = 'date'
                    elif 'open' in col_lower:
                        column_map[col] = 'open'
                    elif 'high' in col_lower:
                        column_map[col] = 'high'
                    elif 'low' in col_lower:
                        column_map[col] = 'low'
                    elif 'close' in col_lower:
                        column_map[col] = 'close'
                    elif 'volume' in col_lower:
                        column_map[col] = 'volume'
                
                df = df.rename(columns=column_map)
                
                # Skip if insufficient data
                if len(df) < min_data_points:
                    skip_count += 1
                    if i <= 5:
                        print(f"   ⚠️ {stock_name}: Too few rows ({len(df)} < {min_data_points})")
                    continue
                
                # Add stock identifier
                df['stock_name'] = stock_name
                
                # Check we have required columns
                required = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    skip_count += 1
                    if i <= 5:
                        print(f"   ⚠️ {stock_name}: Missing columns {missing}")
                    continue
                
                # Convert to proper types
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove rows with NaN values
                df = df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
                
                if len(df) >= min_data_points:
                    all_data.append(df)
                    success_count += 1
                    
                    if i % 50 == 0:
                        print(f"   ✓ Processed {i}/{len(stock_files)} stocks...")
                else:
                    skip_count += 1
                
            except Exception as e:
                skip_count += 1
                if i <= 5:
                    print(f"   ❌ {stock_name}: {str(e)}")
                continue
        
        print(f"\n✅ Successfully loaded {success_count} stocks")
        print(f"⏭️ Skipped {skip_count} stocks (insufficient data)")
        
        if not all_data:
            raise ValueError(f"No valid stock data loaded! All {len(stock_files)} files failed. Check CSV format.")
        
        # Combine all data
        print(f"💾 Combining {len(all_data)} stock dataframes...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        combined_df = combined_df.sort_values(['stock_name', 'date'])
        
        print(f"📊 Combined dataset: {len(combined_df):,} records from {success_count} stocks")
        print(f"📅 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        self.combined_data = combined_df
        return combined_df
    
    def engineer_universal_features(self, df):
        """
        Engineer features that work across all stocks
        
        These features are normalized and work for any stock
        """
        print("\n🔧 Engineering universal features...")
        
        engineered = []
        
        # Process each stock separately to maintain time series integrity
        for stock_name, stock_df in df.groupby('stock_name'):
            stock_df = stock_df.sort_values('date').copy()
            
            # Price-based features (normalized) - with safe division
            eps = 1e-10  # Small epsilon to prevent division by zero
            stock_df['returns'] = stock_df['close'].pct_change()
            stock_df['log_returns'] = np.log(stock_df['close'] / (stock_df['close'].shift(1) + eps))
            stock_df['high_low_ratio'] = stock_df['high'] / (stock_df['low'] + eps)
            stock_df['close_open_ratio'] = stock_df['close'] / (stock_df['open'] + eps)
            
            # Moving averages (normalized by current price)
            for window in [5, 10, 20, 50]:
                sma = stock_df['close'].rolling(window).mean()
                stock_df[f'sma_{window}_ratio'] = stock_df['close'] / (sma + eps)
                stock_df[f'sma_{window}_slope'] = sma.pct_change(3)
            
            # Volatility measures
            for window in [5, 10, 20]:
                stock_df[f'volatility_{window}'] = stock_df['returns'].rolling(window).std()
                stock_df[f'range_{window}'] = ((stock_df['high'] - stock_df['low']) / (stock_df['close'] + eps)).rolling(window).mean()
            
            # Volume indicators (normalized)
            stock_df['volume_ma_20'] = stock_df['volume'].rolling(20).mean()
            stock_df['volume_ratio'] = stock_df['volume'] / (stock_df['volume_ma_20'] + eps)
            stock_df['volume_change'] = stock_df['volume'].pct_change()
            
            # RSI
            for window in [14]:
                delta = stock_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / (loss + eps)
                stock_df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = stock_df['close'].ewm(span=12, adjust=False).mean()
            exp2 = stock_df['close'].ewm(span=26, adjust=False).mean()
            stock_df['macd'] = (exp1 - exp2) / (stock_df['close'] + eps)
            stock_df['macd_signal'] = stock_df['macd'].ewm(span=9, adjust=False).mean()
            stock_df['macd_hist'] = stock_df['macd'] - stock_df['macd_signal']
            
            # Bollinger Bands
            sma_20 = stock_df['close'].rolling(20).mean()
            std_20 = stock_df['close'].rolling(20).std()
            stock_df['bb_upper'] = sma_20 + (2 * std_20)
            stock_df['bb_lower'] = sma_20 - (2 * std_20)
            stock_df['bb_position'] = (stock_df['close'] - stock_df['bb_lower']) / ((stock_df['bb_upper'] - stock_df['bb_lower']) + eps)
            stock_df['bb_width'] = (stock_df['bb_upper'] - stock_df['bb_lower']) / (sma_20 + eps)
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                stock_df[f'returns_lag_{lag}'] = stock_df['returns'].shift(lag)
                stock_df[f'volume_ratio_lag_{lag}'] = stock_df['volume_ratio'].shift(lag)
            
            # Day of week and month (cyclical encoding)
            stock_df['day_of_week'] = stock_df['date'].dt.dayofweek
            stock_df['day_sin'] = np.sin(2 * np.pi * stock_df['day_of_week'] / 7)
            stock_df['day_cos'] = np.cos(2 * np.pi * stock_df['day_of_week'] / 7)
            stock_df['month'] = stock_df['date'].dt.month
            stock_df['month_sin'] = np.sin(2 * np.pi * stock_df['month'] / 12)
            stock_df['month_cos'] = np.cos(2 * np.pi * stock_df['month'] / 12)
            
            # Target: Next day's return (what we want to predict)
            stock_df['target_return'] = stock_df['returns'].shift(-1)
            stock_df['target_price'] = stock_df['close'].shift(-1)
            
            engineered.append(stock_df)
        
        result = pd.concat(engineered, ignore_index=True)
        
        print(f"   ✓ Created {len(result.columns) - len(df.columns)} new features")
        print(f"   ✓ Total features: {len(result.columns)}")
        
        return result
    
    def prepare_training_data(self, df, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with engineered features
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n📋 Preparing training data...")
        
        # Encode stock names as categorical feature
        self.label_encoder = LabelEncoder()
        df['stock_encoded'] = self.label_encoder.fit_transform(df['stock_name'])
        
        # Select feature columns (exclude target, date, and original price columns)
        exclude_cols = [
            'date', 'stock_name', 'target_return', 'target_price',
            'open', 'high', 'low', 'close', 'volume',
            'volume_ma_20', 'bb_upper', 'bb_lower', 'day_of_week', 'month'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_columns + ['target_return']].dropna()
        
        print(f"   ✓ Features: {len(self.feature_columns)}")
        print(f"   ✓ Training samples after dropna: {len(df_clean):,}")
        
        X = df_clean[self.feature_columns]
        y = df_clean['target_return']
        
        # CRITICAL: Replace infinite values with NaN, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN or inf
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"   ✓ Training samples after cleaning inf: {len(X):,}")
        
        # Time-series split (don't shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   ✓ Train samples: {len(X_train):,}")
        print(f"   ✓ Test samples: {len(X_test):,}")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X_train.index, X_test.index
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest on combined data...")
        print(f"   📊 Training on {len(X_train):,} samples with {X_train.shape[1]} features")
        print(f"   🔧 n_estimators=200, max_depth=15, n_jobs=-1 (all CPUs)")
        
        import time
        start_time = time.time()
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbose=1,  # Show progress
            warm_start=False
        )
        
        print("   🏃 Training started...")
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Calculate direction accuracy
        direction_accuracy = np.mean((test_pred > 0) == (y_test > 0)) * 100
        
        self.models['random_forest'] = {
            'model': model,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'direction_accuracy': direction_accuracy
        }
        
        print(f"   ✓ Train MSE: {train_mse:.6f}")
        print(f"   ✓ Test MSE: {test_mse:.6f}")
        print(f"   ✓ Test MAE: {test_mae:.6f}")
        print(f"   ✓ Test R²: {test_r2:.4f}")
        print(f"   ✓ Direction Accuracy: {direction_accuracy:.2f}%")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting model"""
        print("\n📈 Training Gradient Boosting on combined data...")
        print(f"   📊 Training on {len(X_train):,} samples with {X_train.shape[1]} features")
        print(f"   🔧 n_estimators=200, max_depth=8, learning_rate=0.1")
        
        import time
        start_time = time.time()
        
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=1  # Show progress
        )
        
        print("   🏃 Training started...")
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Evaluate
        test_pred = model.predict(X_test)
        
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        direction_accuracy = np.mean((test_pred > 0) == (y_test > 0)) * 100
        
        self.models['gradient_boosting'] = {
            'model': model,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'direction_accuracy': direction_accuracy
        }
        
        print(f"   ✓ Test MSE: {test_mse:.6f}")
        print(f"   ✓ Test MAE: {test_mae:.6f}")
        print(f"   ✓ Test R²: {test_r2:.4f}")
        print(f"   ✓ Direction Accuracy: {direction_accuracy:.2f}%")
        
        return model
    
    def train_xgboost_gpu(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model with GPU support (if available)"""
        if not XGBOOST_AVAILABLE:
            print("\n⚠️ XGBoost not available. Skipping GPU training.")
            print("   💡 Install with: pip install xgboost")
            return None
        
        print("\n🚀 Training XGBoost...")
        
        # Check for GPU support
        gpu_available = False
        tree_method = 'hist'  # Default to CPU
        
        try:
            # Try to use GPU - if it fails, we'll catch it
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                # NVIDIA GPU exists, try gpu_hist
                tree_method = 'gpu_hist'
                gpu_available = True
                print(f"   🎮 GPU detected! Attempting GPU acceleration")
            else:
                print(f"   💻 Using CPU (hist method)")
        except:
            print(f"   💻 Using CPU (hist method)")
        
        print(f"   📊 Training on {len(X_train):,} samples with {X_train.shape[1]} features")
        print(f"   🔧 n_estimators=200, max_depth=8, tree_method={tree_method}")
        
        import time
        start_time = time.time()
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,  # 'gpu_hist' or 'hist'
            predictor='auto',
            random_state=42,
            n_jobs=-1,
            verbosity=1  # Show progress
        )
        
        print("   🏃 Training started...")
        
        # Try to fit, fall back to CPU if GPU fails
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=10  # Print every 10 iterations
            )
        except Exception as e:
            if 'gpu_hist' in str(e).lower():
                print(f"   ⚠️ GPU training failed, falling back to CPU...")
                # Retry with CPU
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method='hist',  # Force CPU
                    predictor='auto',
                    random_state=42,
                    n_jobs=-1,
                    verbosity=1
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=10
                )
                gpu_available = False
            else:
                raise e
        
        elapsed = time.time() - start_time
        print(f"   ✅ Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Evaluate
        test_pred = model.predict(X_test)
        
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Direction accuracy
        direction_correct = ((test_pred > 0) == (y_test > 0)).sum()
        direction_accuracy = (direction_correct / len(y_test)) * 100
        
        self.metrics['xgboost'] = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'direction_accuracy': direction_accuracy,
            'gpu_used': gpu_available
        }
        
        print(f"   ✓ Test MSE: {test_mse:.6f}")
        print(f"   ✓ Test MAE: {test_mae:.6f}")
        print(f"   ✓ Test R²: {test_r2:.4f}")
        print(f"   ✓ Direction Accuracy: {direction_accuracy:.2f}%")
        if gpu_available:
            print(f"   🎮 GPU speedup achieved!")
        
        return model
    
    def train_all_models(self, max_stocks=None, min_data_points=200, random_sample=True):
        """
        Train all models on combined data from all stocks
        
        Args:
            max_stocks: Maximum number of stocks to use (None for all)
            min_data_points: Minimum data points required per stock (default: 200)
            random_sample: If True, randomly sample stocks each run (default: True)
        """
        print(f"\n{'='*70}")
        print("🚀 UNIVERSAL AI TRAINING - LEARNING FROM ALL STOCKS")
        print(f"{'='*70}")
        
        # Load all stocks (random sampling by default)
        df = self.load_all_stocks_data(
            max_stocks=max_stocks, 
            min_data_points=min_data_points,
            random_sample=random_sample
        )
        
        # Engineer features
        df_features = self.engineer_universal_features(df)
        
        # Prepare training data
        X_train, X_test, y_train, y_test, train_idx, test_idx = self.prepare_training_data(df_features)
        
        # Train models
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        # Train XGBoost with GPU support (if available)
        xgb_model = self.train_xgboost_gpu(X_train, y_train, X_test, y_test)
        if xgb_model is not None:
            self.models['xgboost'] = {'model': xgb_model, 'name': 'XGBoost'}
        
        print(f"\n{'='*70}")
        print("🎯 TRAINING COMPLETE")
        print(f"{'='*70}")
        
        # Show feature importance
        self.display_feature_importance()
        
        return self.models
    
    def display_feature_importance(self, top_n=15):
        """Display top N most important features"""
        if 'random_forest' in self.models:
            model = self.models['random_forest']['model']
            importances = model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 TOP {top_n} MOST IMPORTANT FEATURES:")
            print("-" * 60)
            for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    def predict_stock(self, stock_name, recent_data=None):
        """
        Predict next day's return for a specific stock
        
        Args:
            stock_name: Name of the stock
            recent_data: Recent data for the stock (if None, loads from file)
            
        Returns:
            dict with prediction details
        """
        if not self.models:
            raise ValueError("No models trained! Call train_all_models() first.")
        
        # Load recent data if not provided
        if recent_data is None:
            file_path = os.path.join(self.eod_directory, f'{stock_name}.csv')
            recent_data = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Engineer features for this stock
        df = self.engineer_universal_features(recent_data)
        
        # Get latest row with features
        latest = df[self.feature_columns].iloc[-1:].dropna(axis=1)
        
        # Make sure we have all required features
        missing_features = set(self.feature_columns) - set(latest.columns)
        for feat in missing_features:
            latest[feat] = 0
        
        latest = latest[self.feature_columns]
        
        # Scale features
        latest_scaled = self.scaler.transform(latest)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model_info in self.models.items():
            pred_return = model_info['model'].predict(latest_scaled)[0]
            predictions[model_name] = pred_return
        
        # Average prediction
        avg_predicted_return = np.mean(list(predictions.values()))
        
        # Calculate predicted price
        current_price = recent_data['Close'].iloc[-1]
        predicted_price = current_price * (1 + avg_predicted_return)
        predicted_change_pct = avg_predicted_return * 100
        
        # Calculate confidence based on model agreement
        pred_values = list(predictions.values())
        std_dev = np.std(pred_values)
        confidence = max(50, min(95, 85 - (std_dev * 1000)))
        
        return {
            'stock_name': stock_name,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': avg_predicted_return,
            'predicted_change_pct': predicted_change_pct,
            'confidence': confidence,
            'model': 'Universal_Ensemble',
            'individual_predictions': predictions
        }
    
    def save_models(self, directory='universal_models'):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for model_name, model_info in self.models.items():
            joblib.dump(model_info['model'], f"{directory}/{model_name}.pkl")
        
        # Save metadata
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_metrics': {name: {k: v for k, v in info.items() if k != 'model'} 
                            for name, info in self.models.items()}
        }
        joblib.dump(metadata, f"{directory}/metadata.pkl")
        
        print(f"✅ Models saved to {directory}/")
    
    def load_models(self, directory='universal_models'):
        """Load saved models"""
        # Load metadata
        metadata = joblib.load(f"{directory}/metadata.pkl")
        self.scaler = metadata['scaler']
        self.label_encoder = metadata['label_encoder']
        self.feature_columns = metadata['feature_columns']
        
        # Load models
        self.models = {}
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = f"{directory}/{model_name}.pkl"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.models[model_name] = {
                    'model': model,
                    **metadata['model_metrics'].get(model_name, {})
                }
        
        print(f"✅ Models loaded from {directory}/")


def main():
    """Command-line interface for Universal AI training"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🌍 Universal Stock AI Trainer - Learn from ALL stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train on ALL stocks (recommended)
  python universal_ai_trainer.py --all
  
  # Train on first 100 stocks (for testing)
  python universal_ai_trainer.py --max-stocks 100
  
  # Train and predict specific stock
  python universal_ai_trainer.py --all --predict RELIANCE
  
  # Train with custom paths
  python universal_ai_trainer.py --eod-dir /path/to/EOD --models-dir /path/to/models
        '''
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Train on ALL stocks (no limit)')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='Maximum number of stocks to train on (default: all)')
    parser.add_argument('--min-data-points', type=int, default=200,
                        help='Minimum data points required per stock (default: 200)')
    parser.add_argument('--no-random', action='store_true',
                        help='Disable random sampling (use first N stocks instead)')
    parser.add_argument('--eod-dir', type=str, default='EOD',
                        help='Path to EOD data directory (default: EOD)')
    parser.add_argument('--models-dir', type=str, default='universal_models',
                        help='Path to save trained models (default: universal_models)')
    parser.add_argument('--predict', type=str, default=None,
                        help='Stock name to predict after training (e.g., RELIANCE)')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving models to disk')
    
    args = parser.parse_args()
    
    # Determine max_stocks
    if args.all:
        max_stocks = None
        print("🌍 Training on ALL stocks (full dataset)")
    elif args.max_stocks:
        max_stocks = args.max_stocks
        print(f"🌍 Training on {max_stocks} stocks")
    else:
        max_stocks = 100
        print(f"🌍 Training on {max_stocks} stocks (default)")
        print("   💡 Tip: Use --all to train on ALL stocks")
    
    print("=" * 70)
    
    # Initialize predictor
    print(f"📁 EOD Directory: {args.eod_dir}")
    predictor = UniversalStockPredictor(eod_directory=args.eod_dir)
    
    # Train models
    models = predictor.train_all_models(
        max_stocks=max_stocks,
        min_data_points=args.min_data_points,
        random_sample=not args.no_random  # Random sampling by default
    )
    
    # Save models
    if not args.no_save:
        print(f"\n💾 Saving models to {args.models_dir}/")
        predictor.save_models(directory=args.models_dir)
        print("✅ Models saved successfully!")
    else:
        print("\n⚠️ Models not saved (--no-save flag)")
    
    # Test prediction if requested
    if args.predict:
        print("\n" + "=" * 70)
        print(f"🔮 TESTING PREDICTION: {args.predict}")
        print("=" * 70)
        
        try:
            prediction = predictor.predict_stock(args.predict)
            
            print(f"\n📊 Prediction for {args.predict}:")
            print(f"   Current Price: ₹{prediction['current_price']:.2f}")
            print(f"   Predicted Price: ₹{prediction['predicted_price']:.2f}")
            print(f"   Expected Change: {prediction['predicted_change_pct']:+.2f}%")
            print(f"   Confidence: {prediction['confidence']:.1f}%")
            print(f"   Direction: {prediction['direction']}")
            print(f"   Model: {prediction['model']}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()

