"""
Advanced AI Prediction Engine
Uses LSTM, Prophet, and ensemble methods for stock prediction with auto-validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AIStockPredictor:
    """Advanced AI-powered stock predictor with multiple models"""
    
    def __init__(self, stock_data):
        """
        Initialize predictor
        
        Args:
            stock_data: DataFrame with stock data
        """
        self.data = stock_data.copy()
        self.predictions = {}
        self.validation_results = {}
        self.models = {}
        
    def prepare_lstm_data(self, lookback=60):
        """Prepare data for LSTM model"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data[['Close']].values)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def train_lstm(self, lookback=60, epochs=50):
        """Train LSTM model for predictions"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Prepare data
            X, y, scaler = self.prepare_lstm_data(lookback)
            
            # Split train/test
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train
            print("Training LSTM model...")
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            
            # Predict
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate accuracy
            mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
            accuracy = 100 - mape
            
            self.models['lstm'] = {
                'model': model,
                'scaler': scaler,
                'lookback': lookback,
                'accuracy': accuracy
            }
            
            return accuracy
            
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            return None
    
    def train_prophet(self):
        """Train Prophet model for predictions"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            df_prophet = self.data[['Date', 'Close']].copy()
            df_prophet.columns = ['ds', 'y']
            
            # Train model
            print("Training Prophet model...")
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df_prophet)
            
            # Make predictions
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Calculate accuracy on last 30 days
            actual = df_prophet['y'].tail(30).values
            predicted = forecast['yhat'].tail(30).values
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            accuracy = 100 - mape
            
            self.models['prophet'] = {
                'model': model,
                'forecast': forecast,
                'accuracy': accuracy
            }
            
            return accuracy
            
        except ImportError:
            print("Prophet not installed. Install with: pip install prophet")
            return None
    
    def train_ensemble(self):
        """Train ensemble of models"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        # Create features
        df = self.data.copy()
        
        # Technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Target
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        # Prepare train/test
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Target', 'Open', 'High', 'Low', 'Volume']]
        X = df[feature_cols]
        y = df['Target']
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        print("Training ensemble models...")
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        predictions = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            predictions[name] = pred
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Calculate accuracy
        mape = np.mean(np.abs((y_test.values - ensemble_pred) / y_test.values)) * 100
        accuracy = 100 - mape
        
        self.models['ensemble'] = {
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'predictions': predictions
        }
        
        return accuracy
    
    def predict_next_day(self, model_type='ensemble'):
        """Predict next day's closing price"""
        if model_type not in self.models:
            print(f"Model {model_type} not trained yet!")
            return None
        
        model_info = self.models[model_type]
        
        if model_type == 'ensemble':
            # Prepare latest data
            df = self.data.copy()
            df['Returns'] = df['Close'].pct_change()
            df['MA_5'] = df['Close'].rolling(5).mean()
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()
            df['Volatility'] = df['Returns'].rolling(20).std()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            for lag in [1, 2, 3, 5, 10]:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
            df = df.dropna()
            
            # Get latest features
            X_latest = df[model_info['feature_cols']].iloc[-1:].values
            X_latest_scaled = model_info['scaler'].transform(X_latest)
            
            # Predict with all models
            predictions = []
            for model in model_info['models'].values():
                pred = model.predict(X_latest_scaled)[0]
                predictions.append(pred)
            
            # Ensemble prediction
            prediction = np.mean(predictions)
            confidence = model_info['accuracy']
            
            return {
                'prediction': prediction,
                'current_price': self.data['Close'].iloc[-1],
                'change': prediction - self.data['Close'].iloc[-1],
                'change_pct': ((prediction - self.data['Close'].iloc[-1]) / self.data['Close'].iloc[-1]) * 100,
                'confidence': confidence,
                'model': model_type
            }
        
        return None
    
    def validate_prediction(self, prediction_date, predicted_price, actual_price):
        """Validate previous predictions"""
        error = abs(predicted_price - actual_price)
        error_pct = (error / actual_price) * 100
        accuracy = 100 - error_pct
        
        validation = {
            'date': prediction_date,
            'predicted': predicted_price,
            'actual': actual_price,
            'error': error,
            'error_pct': error_pct,
            'accuracy': accuracy,
            'status': 'accurate' if error_pct < 5 else 'moderate' if error_pct < 10 else 'poor'
        }
        
        self.validation_results[prediction_date] = validation
        return validation
    
    def generate_trading_signal(self):
        """Generate trading signals based on predictions"""
        if 'ensemble' not in self.models:
            return None
        
        prediction = self.predict_next_day('ensemble')
        if not prediction:
            return None
        
        current_price = prediction['current_price']
        predicted_price = prediction['prediction']
        change_pct = prediction['change_pct']
        confidence = prediction['confidence']
        
        # Generate signal
        if change_pct > 2 and confidence > 70:
            signal = 'STRONG BUY'
            action = 'Buy aggressively'
        elif change_pct > 0.5 and confidence > 60:
            signal = 'BUY'
            action = 'Consider buying'
        elif change_pct < -2 and confidence > 70:
            signal = 'STRONG SELL'
            action = 'Sell immediately'
        elif change_pct < -0.5 and confidence > 60:
            signal = 'SELL'
            action = 'Consider selling'
        else:
            signal = 'HOLD'
            action = 'Hold position'
        
        return {
            'signal': signal,
            'action': action,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_change': change_pct,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def load_stock_csv(file_path):
    """Load stock CSV handling yfinance multi-index format"""
    with open(file_path, 'r') as f:
        lines = [f.readline().strip() for _ in range(3)]
    
    if lines[0].startswith('Price,') and lines[1].startswith('Ticker,') and lines[2].startswith('Date,'):
        df = pd.read_csv(file_path, skiprows=3, header=None)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    else:
        df = pd.read_csv(file_path)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("Date column not found")
    
    return df

def train_and_predict(stock_file, stock_name):
    """Complete training and prediction pipeline"""
    print(f"\n{'='*70}")
    print(f"AI Prediction System: {stock_name}")
    print(f"{'='*70}\n")
    
    # Load data with proper format handling
    try:
        data = load_stock_csv(stock_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Initialize predictor
    predictor = AIStockPredictor(data)
    
    # Train models
    print("Training AI models...")
    ensemble_acc = predictor.train_ensemble()
    print(f"✓ Ensemble Model Accuracy: {ensemble_acc:.2f}%")
    
    # Try LSTM (optional, requires tensorflow)
    # lstm_acc = predictor.train_lstm(epochs=20)
    # if lstm_acc:
    #     print(f"✓ LSTM Model Accuracy: {lstm_acc:.2f}%")
    
    # Try Prophet (optional, requires prophet)
    # prophet_acc = predictor.train_prophet()
    # if prophet_acc:
    #     print(f"✓ Prophet Model Accuracy: {prophet_acc:.2f}%")
    
    # Generate prediction
    print("\nGenerating next-day prediction...")
    prediction = predictor.predict_next_day('ensemble')
    
    if prediction:
        print(f"\n📊 Prediction Results:")
        print(f"   Current Price: ₹{prediction['current_price']:.2f}")
        print(f"   Predicted Price: ₹{prediction['prediction']:.2f}")
        print(f"   Expected Change: {prediction['change_pct']:+.2f}%")
        print(f"   Confidence: {prediction['confidence']:.2f}%")
    
    # Generate trading signal
    signal = predictor.generate_trading_signal()
    if signal:
        print(f"\n🎯 Trading Signal:")
        print(f"   Signal: {signal['signal']}")
        print(f"   Action: {signal['action']}")
        print(f"   Expected Return: {signal['expected_change']:+.2f}%")
        print(f"   Confidence: {signal['confidence']:.2f}%")
    
    print(f"\n{'='*70}\n")
    
    return predictor, prediction, signal

if __name__ == "__main__":
    # Example: Predict RELIANCE stock
    import os
    
    if os.path.exists("EOD/RELIANCE.csv"):
        predictor, prediction, signal = train_and_predict("EOD/RELIANCE.csv", "RELIANCE")
    else:
        print("Please run fetch_stocks.py first!")

