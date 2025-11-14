"""
Advanced AI Prediction Engine for Stock Trading
Includes LSTM, TrendSpider-inspired analysis, Polynomial Regression, and Ensemble methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models disabled.")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class AdvancedStockPredictor:
    """Advanced AI-powered stock predictor with multiple sophisticated models"""
    
    def __init__(self, stock_data):
        """
        Initialize predictor
        
        Args:
            stock_data: DataFrame with stock data
        """
        self.data = stock_data.copy()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.validation_results = {}
        
        # Ensure Date is properly formatted
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.set_index('Date')
        
        print(f"✓ Initialized predictor with {len(self.data)} data points")
        
    def engineer_trendspider_features(self):
        """Create TrendSpider-inspired technical features"""
        df = self.data.copy()
        
        print("🔧 Engineering TrendSpider-inspired features...")
        
        # Price action patterns
        df['Doji'] = ((df['Open'] - df['Close']).abs() / (df['High'] - df['Low'])) < 0.1
        df['Hammer'] = ((df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'])) < 0.1
        df['ShootingStar'] = ((df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'])) < 0.1
        
        # Support and Resistance levels
        for window in [20, 50]:
            df[f'Resistance_{window}'] = df['High'].rolling(window).max()
            df[f'Support_{window}'] = df['Low'].rolling(window).min()
            df[f'SupportResistanceRatio_{window}'] = (df['Close'] - df[f'Support_{window}']) / (df[f'Resistance_{window}'] - df[f'Support_{window}'])
        
        # Trend strength indicators
        for period in [14, 21, 50]:
            # Directional Movement Index (DMI)
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = np.maximum(df['High'] - df['Low'], 
                           np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                    abs(df['Low'] - df['Close'].shift(1))))
            
            atr = pd.Series(tr).rolling(period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
            
            df[f'ADX_{period}'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'DI_Diff_{period}'] = plus_di - minus_di
        
        # Volume Profile Analysis
        df['VolumeProfile'] = df['Volume'] / df['Volume'].rolling(50).mean()
        df['PriceVolumeCorr'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # Fibonacci retracement levels
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        diff = high_20 - low_20
        
        df['Fib_23.6'] = high_20 - 0.236 * diff
        df['Fib_38.2'] = high_20 - 0.382 * diff
        df['Fib_61.8'] = high_20 - 0.618 * diff
        df['FibPosition'] = (df['Close'] - low_20) / diff
        
        # Market structure
        df['HigherHighs'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['LowerLows'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        df['TrendStrength'] = df['HigherHighs'].rolling(10).sum() - df['LowerLows'].rolling(10).sum()
        
        print(f"   ✓ Created {len([c for c in df.columns if c not in self.data.columns])} TrendSpider features")
        return df
    
    def engineer_comprehensive_features(self):
        """Create comprehensive technical and statistical features"""
        df = self.engineer_trendspider_features()
        
        print("🔧 Engineering comprehensive features...")
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['GarmanKlass_Vol'] = np.sqrt(252) * np.sqrt(
            np.log(df['High']/df['Low']) * np.log(df['Close']/df['Low']) - 
            (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2
        )
        
        # Multiple timeframe analysis
        for window in [5, 10, 20, 50, 100, 200]:
            # Moving averages
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            df[f'WMA_{window}'] = df['Close'].rolling(window).apply(lambda x: np.average(x, weights=range(1, len(x)+1)))
            
            # Price position relative to MA
            df[f'Price_vs_SMA_{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
            df[f'Price_vs_EMA_{window}'] = df['Close'] / df[f'EMA_{window}'] - 1
            
            # MA slopes
            df[f'SMA_Slope_{window}'] = df[f'SMA_{window}'].pct_change(5)
            df[f'EMA_Slope_{window}'] = df[f'EMA_{window}'].pct_change(5)
        
        # Advanced volatility measures
        for window in [10, 20, 50]:
            df[f'RealizedVol_{window}'] = df['Returns'].rolling(window).std() * np.sqrt(252)
            df[f'VolatilityRatio_{window}'] = df[f'RealizedVol_{window}'] / df['RealizedVol_20']
            df[f'RangeVol_{window}'] = (df['High'] - df['Low']).rolling(window).std()
        
        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
            df[f'MACD_Hist_{fast}_{slow}_{signal}'] = macd - signal_line
            df[f'MACD_Slope_{fast}_{slow}'] = macd.pct_change(3)
        
        # RSI variations
        for window in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            df[f'RSI_Slope_{window}'] = df[f'RSI_{window}'].pct_change(3)
        
        # Bollinger Bands variations
        for window, std_mult in [(20, 2), (10, 1.5), (50, 2.5)]:
            middle = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            upper = middle + (std * std_mult)
            lower = middle - (std * std_mult)
            df[f'BB_Upper_{window}'] = upper
            df[f'BB_Lower_{window}'] = lower
            df[f'BB_Width_{window}'] = (upper - lower) / middle
            df[f'BB_Position_{window}'] = (df['Close'] - lower) / (upper - lower)
            df[f'BB_Squeeze_{window}'] = df[f'BB_Width_{window}'] < df[f'BB_Width_{window}'].rolling(20).quantile(0.1)
        
        # Stochastic variations
        for k_window, d_window in [(14, 3), (5, 3), (21, 5)]:
            low_min = df['Low'].rolling(k_window).min()
            high_max = df['High'].rolling(k_window).max()
            k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(d_window).mean()
            df[f'Stoch_K_{k_window}_{d_window}'] = k_percent
            df[f'Stoch_D_{k_window}_{d_window}'] = d_percent
            df[f'Stoch_Slope_{k_window}'] = k_percent.pct_change(2)
        
        # Williams %R
        for window in [14, 28]:
            high_max = df['High'].rolling(window).max()
            low_min = df['Low'].rolling(window).min()
            df[f'WilliamsR_{window}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        for window in [14, 20]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            ma_tp = tp.rolling(window).mean()
            md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df[f'CCI_{window}'] = (tp - ma_tp) / (0.015 * md)
        
        # Money Flow Index
        for window in [14, 28]:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            raw_money_flow = typical_price * df['Volume']
            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window).sum()
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
            money_ratio = positive_flow / negative_flow
            df[f'MFI_{window}'] = 100 - (100 / (1 + money_ratio))
        
        # Volume indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Price_vs_VWAP'] = df['Close'] / df['VWAP'] - 1
        
        # Lag features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_Skew_{window}'] = df['Close'].rolling(window).skew()
            df[f'Close_Kurt_{window}'] = df['Close'].rolling(window).kurt()
            df[f'Returns_Skew_{window}'] = df['Returns'].rolling(window).skew()
            df[f'Returns_Kurt_{window}'] = df['Returns'].rolling(window).kurt()
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        self.engineered_data = df.dropna()
        print(f"   ✓ Total features created: {len(df.columns) - len(self.data.columns)}")
        print(f"   ✓ Final dataset shape: {self.engineered_data.shape}")
        
        return self.engineered_data
    
    def prepare_lstm_data(self, sequence_length=60, test_size=0.2):
        """Prepare data for LSTM training"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available, skipping LSTM preparation")
            return None, None, None, None, None
        
        # Use only price-based features for LSTM
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = self.data[price_features].dropna()
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 3])  # Close price index
        
        X, y = np.array(X), np.array(y)
        
        # Split data (time series split)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.scalers['lstm'] = scaler
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_lstm_model(self, sequence_length=60, epochs=100):
        """Train LSTM neural network"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available, skipping LSTM training")
            return 0
        
        print("🧠 Training LSTM model...")
        
        X_train, X_test, y_train, y_test, scaler = self.prepare_lstm_data(sequence_length)
        if X_train is None:
            return 0
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )
        
        # Evaluate
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        train_pred_inv = scaler.inverse_transform(
            np.column_stack([np.zeros((len(train_pred), 3)), train_pred, np.zeros((len(train_pred), 1))])
        )[:, 3]
        test_pred_inv = scaler.inverse_transform(
            np.column_stack([np.zeros((len(test_pred), 3)), test_pred, np.zeros((len(test_pred), 1))])
        )[:, 3]
        
        y_train_inv = scaler.inverse_transform(
            np.column_stack([np.zeros((len(y_train), 3)), y_train, np.zeros((len(y_train), 1))])
        )[:, 3]
        y_test_inv = scaler.inverse_transform(
            np.column_stack([np.zeros((len(y_test), 3)), y_test, np.zeros((len(y_test), 1))])
        )[:, 3]
        
        # Calculate accuracy
        mape_test = np.mean(np.abs((y_test_inv - test_pred_inv) / y_test_inv)) * 100
        accuracy = max(0, 100 - mape_test)
        
        self.models['lstm'] = {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'accuracy': accuracy,
            'history': history.history
        }
        
        print(f"   ✓ LSTM accuracy: {accuracy:.2f}%")
        return accuracy
    
    def train_polynomial_regression(self, degree=3):
        """Train polynomial regression model"""
        print(f"📈 Training Polynomial Regression (degree={degree})...")
        
        # Prepare features
        df = self.engineer_comprehensive_features()
        
        # Select key features for polynomial regression to avoid overfitting
        key_features = [
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
            'SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26',
            'BB_Position_20', 'Stoch_K_14_3', 'Returns_Lag_1'
        ]
        
        available_features = [f for f in key_features if f in df.columns]
        X = df[available_features].dropna()
        y = df.loc[X.index, 'Target']
        
        # Remove samples with missing target
        mask = ~y.isna()
        X, y = X[mask], y[mask]
        
        if len(X) == 0:
            print("   ⚠️ No valid samples for polynomial regression")
            return 0
        
        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        
        # Use Ridge regression to handle multicollinearity
        model = Ridge(alpha=10.0)
        model.fit(X_train_poly, y_train)
        
        # Predictions
        test_pred = model.predict(X_test_poly)
        
        # Calculate accuracy
        mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        accuracy = max(0, 100 - mape)
        
        self.models['polynomial'] = {
            'model': model,
            'scaler': scaler,
            'poly_features': poly_features,
            'feature_columns': available_features,
            'accuracy': accuracy,
            'degree': degree
        }
        
        print(f"   ✓ Polynomial regression accuracy: {accuracy:.2f}%")
        return accuracy
    
    def train_advanced_ensemble(self):
        """Train advanced ensemble with multiple algorithms"""
        print("🎯 Training Advanced Ensemble...")
        
        # Prepare comprehensive features
        df = self.engineer_comprehensive_features()
        
        # Select features (exclude target and non-predictive columns)
        exclude_cols = [
            'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Adj Close', 'Doji', 'Hammer', 'ShootingStar', 'HigherHighs', 'LowerLows'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_0')]
        
        X = df[feature_cols]
        y = df['Target']
        
        # Remove samples with missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]
        
        if len(X) == 0:
            print("   ⚠️ No valid samples for ensemble training")
            return 0
        
        print(f"   📊 Training with {len(X)} samples and {len(feature_cols)} features")
        
        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'gbm': GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
            'ada': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                predictions[name] = pred
                
                # Calculate accuracy
                mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
                accuracy = max(0, 100 - mape)
                model_scores[name] = accuracy
                
                print(f"     ✓ {name.upper()} accuracy: {accuracy:.2f}%")
                
            except Exception as e:
                print(f"     ⚠️ {name} failed: {e}")
                continue
        
        if not predictions:
            print("   ❌ No models trained successfully")
            return 0
        
        # Weighted ensemble based on performance
        total_score = sum(model_scores.values())
        weights = {name: score/total_score for name, score in model_scores.items()}
        
        # Create ensemble prediction
        ensemble_pred = np.zeros(len(y_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Calculate ensemble accuracy
        ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        ensemble_accuracy = max(0, 100 - ensemble_mape)
        
        self.models['advanced_ensemble'] = {
            'models': models,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'weights': weights,
            'accuracy': ensemble_accuracy,
            'individual_scores': model_scores
        }
        
        print(f"   🎯 Ensemble accuracy: {ensemble_accuracy:.2f}%")
        return ensemble_accuracy
    
    def train_all_models(self):
        """Train all available models"""
        print(f"\n{'='*70}")
        print("🚀 ADVANCED AI MODEL TRAINING")
        print(f"{'='*70}")
        
        results = {}
        
        # Train LSTM
        results['lstm'] = self.train_lstm_model()
        
        # Train Polynomial Regression
        results['polynomial'] = self.train_polynomial_regression()
        
        # Train Advanced Ensemble
        results['advanced_ensemble'] = self.train_advanced_ensemble()
        
        print(f"\n{'='*70}")
        print("📊 TRAINING RESULTS SUMMARY")
        print(f"{'='*70}")
        
        for model_name, accuracy in results.items():
            status = "✅" if accuracy > 70 else "⚠️" if accuracy > 50 else "❌"
            print(f"   {status} {model_name.upper():<20} {accuracy:>7.2f}%")
        
        best_model = max(results, key=results.get) if results else None
        if best_model:
            print(f"\n🏆 Best model: {best_model.upper()} ({results[best_model]:.2f}%)")
        
        print(f"{'='*70}\n")
        
        return results
    
    def predict_next_day(self, model_type='advanced_ensemble'):
        """Predict next day's price using specified model"""
        if model_type not in self.models:
            print(f"❌ Model {model_type} not trained!")
            return None
        
        model_info = self.models[model_type]
        current_price = self.data['Close'].iloc[-1]
        
        try:
            if model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                # LSTM prediction
                sequence_length = model_info['sequence_length']
                scaler = model_info['scaler']
                model = model_info['model']
                
                # Prepare last sequence
                price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
                recent_data = self.data[price_features].tail(sequence_length)
                scaled_data = scaler.transform(recent_data)
                X_pred = np.array([scaled_data])
                
                # Predict
                pred_scaled = model.predict(X_pred, verbose=0)[0][0]
                
                # Inverse transform
                pred_inv = scaler.inverse_transform(
                    np.array([[0, 0, 0, pred_scaled, 0]])
                )[0, 3]
                
                predicted_price = float(pred_inv)
                
            elif model_type == 'polynomial':
                # Polynomial regression prediction
                df = self.engineer_comprehensive_features()
                latest_row = df.iloc[-2:-1]  # -2 because -1 has NaN target
                
                X_latest = latest_row[model_info['feature_columns']]
                X_latest_scaled = model_info['scaler'].transform(X_latest)
                X_latest_poly = model_info['poly_features'].transform(X_latest_scaled)
                
                predicted_price = float(model_info['model'].predict(X_latest_poly)[0])
                
            elif model_type == 'advanced_ensemble':
                # Advanced ensemble prediction
                df = self.engineer_comprehensive_features()
                latest_row = df.iloc[-2:-1]  # -2 because -1 has NaN target
                
                X_latest = latest_row[model_info['feature_columns']]
                X_latest_scaled = model_info['scaler'].transform(X_latest)
                
                # Predict with all models and weight
                ensemble_pred = 0
                for name, model in model_info['models'].items():
                    if name in model_info['weights']:
                        pred = model.predict(X_latest_scaled)[0]
                        ensemble_pred += model_info['weights'][name] * pred
                
                predicted_price = float(ensemble_pred)
                
            else:
                print(f"❌ Unknown model type: {model_type}")
                return None
            
            # Calculate metrics
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            confidence = model_info['accuracy']
            
            return {
                'prediction': predicted_price,
                'current_price': current_price,
                'change': change,
                'change_pct': change_pct,
                'confidence': confidence,
                'model': model_type
            }
            
        except Exception as e:
            print(f"❌ Prediction error for {model_type}: {e}")
            return None
    
    def generate_ensemble_prediction(self):
        """Generate prediction using all available models"""
        predictions = []
        total_confidence = 0
        
        for model_type in self.models:
            pred = self.predict_next_day(model_type)
            if pred:
                predictions.append(pred)
                total_confidence += pred['confidence']
        
        if not predictions:
            return None
        
        # Weighted average based on confidence
        weighted_price = sum(p['prediction'] * p['confidence'] for p in predictions) / total_confidence
        avg_confidence = total_confidence / len(predictions)
        
        current_price = predictions[0]['current_price']
        change = weighted_price - current_price
        change_pct = (change / current_price) * 100
        
        return {
            'prediction': weighted_price,
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'confidence': avg_confidence,
            'model': 'ensemble_meta',
            'individual_predictions': predictions
        }
    
    def save_models(self, directory='models'):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model_info in self.models.items():
            if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                # Save LSTM model
                model_info['model'].save(f"{directory}/lstm_model.h5")
                joblib.dump({k: v for k, v in model_info.items() if k != 'model'}, 
                          f"{directory}/lstm_metadata.pkl")
            else:
                # Save other models
                joblib.dump(model_info, f"{directory}/{model_name}_model.pkl")
        
        print(f"✅ Models saved to {directory}/")
        
    def load_models(self, directory='models'):
        """Load saved models"""
        import os
        
        if not os.path.exists(directory):
            print(f"❌ Directory {directory} not found")
            return
        
        # Load non-LSTM models
        for filename in os.listdir(directory):
            if filename.endswith('_model.pkl') and 'lstm' not in filename:
                model_name = filename.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(f"{directory}/{filename}")
        
        # Load LSTM model if available
        if TENSORFLOW_AVAILABLE and os.path.exists(f"{directory}/lstm_model.h5"):
            from tensorflow.keras.models import load_model
            lstm_metadata = joblib.load(f"{directory}/lstm_metadata.pkl")
            lstm_model = load_model(f"{directory}/lstm_model.h5")
            lstm_metadata['model'] = lstm_model
            self.models['lstm'] = lstm_metadata
        
        print(f"✅ Models loaded from {directory}/")
