"""
Auto-Validation and Strategy Adjustment System
Automatically validates predictions and adjusts strategies based on performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class AutoValidator:
    """Automatic validation and strategy adjustment"""
    
    def __init__(self, predictions_file='predictions.json', strategies_file='strategies.json'):
        self.predictions_file = predictions_file
        self.strategies_file = strategies_file
        self.predictions_history = self.load_predictions()
        self.strategies = self.load_strategies()
        
    def load_predictions(self):
        """Load prediction history"""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_predictions(self):
        """Save prediction history"""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions_history, f, indent=2, default=str)
    
    def load_strategies(self):
        """Load trading strategies"""
        if os.path.exists(self.strategies_file):
            with open(self.strategies_file, 'r') as f:
                return json.load(f)
        return self.get_default_strategies()
    
    def save_strategies(self):
        """Save strategies"""
        with open(self.strategies_file, 'w') as f:
            json.dump(self.strategies, f, indent=2)
    
    def get_default_strategies(self):
        """Get default trading strategies"""
        return {
            'aggressive': {
                'buy_threshold': 1.0,  # Buy if predicted increase > 1%
                'sell_threshold': -1.0,  # Sell if predicted decrease > 1%
                'confidence_min': 60,
                'position_size': 0.3,  # 30% of capital
                'stop_loss': -5.0,  # 5% stop loss
                'take_profit': 10.0,  # 10% take profit
                'enabled': True,
                'performance': {'wins': 0, 'losses': 0, 'accuracy': 0}
            },
            'moderate': {
                'buy_threshold': 0.5,
                'sell_threshold': -0.5,
                'confidence_min': 70,
                'position_size': 0.2,
                'stop_loss': -3.0,
                'take_profit': 6.0,
                'enabled': True,
                'performance': {'wins': 0, 'losses': 0, 'accuracy': 0}
            },
            'conservative': {
                'buy_threshold': 2.0,
                'sell_threshold': -2.0,
                'confidence_min': 80,
                'position_size': 0.1,
                'stop_loss': -2.0,
                'take_profit': 4.0,
                'enabled': True,
                'performance': {'wins': 0, 'losses': 0, 'accuracy': 0}
            }
        }
    
    def record_prediction(self, stock_name, prediction_data):
        """Record a new prediction"""
        date_key = datetime.now().strftime('%Y-%m-%d')
        
        if stock_name not in self.predictions_history:
            self.predictions_history[stock_name] = {}
        
        self.predictions_history[stock_name][date_key] = {
            'predicted_price': prediction_data['prediction'],
            'current_price': prediction_data['current_price'],
            'predicted_change': prediction_data['change_pct'],
            'confidence': prediction_data['confidence'],
            'model': prediction_data['model'],
            'timestamp': datetime.now().isoformat(),
            'validated': False
        }
        
        self.save_predictions()
    
    def validate_predictions(self, stock_name, current_data):
        """Validate all unvalidated predictions for a stock"""
        if stock_name not in self.predictions_history:
            return []
        
        validations = []
        
        for date_key, prediction in self.predictions_history[stock_name].items():
            if prediction['validated']:
                continue
            
            # Check if we have actual data for that date + 1 day
            pred_date = datetime.strptime(date_key, '%Y-%m-%d')
            next_day = pred_date + timedelta(days=1)
            
            # Find actual price
            actual_row = current_data[current_data['Date'] == next_day.strftime('%Y-%m-%d')]
            
            if not actual_row.empty:
                actual_price = actual_row['Close'].iloc[0]
                predicted_price = prediction['predicted_price']
                
                # Calculate validation metrics
                error = abs(predicted_price - actual_price)
                error_pct = (error / actual_price) * 100
                accuracy = 100 - error_pct
                
                validation = {
                    'date': date_key,
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'error': error,
                    'error_pct': error_pct,
                    'accuracy': accuracy,
                    'status': 'accurate' if error_pct < 5 else 'moderate' if error_pct < 10 else 'poor',
                    'validated_at': datetime.now().isoformat()
                }
                
                # Mark as validated
                prediction['validated'] = True
                prediction['validation'] = validation
                
                validations.append(validation)
        
        self.save_predictions()
        return validations
    
    def adjust_strategies(self, validation_results):
        """Auto-adjust strategies based on validation results"""
        if not validation_results:
            return
        
        # Calculate overall accuracy
        accuracies = [v['accuracy'] for v in validation_results]
        avg_accuracy = np.mean(accuracies)
        
        adjustments = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy['enabled']:
                continue
            
            # Adjust thresholds based on accuracy
            if avg_accuracy < 60:  # Poor performance
                # Make strategies more conservative
                strategy['buy_threshold'] *= 1.2
                strategy['sell_threshold'] *= 1.2
                strategy['confidence_min'] = min(90, strategy['confidence_min'] + 5)
                strategy['position_size'] *= 0.8
                adjustments.append(f"{strategy_name}: Made more conservative (low accuracy)")
                
            elif avg_accuracy > 80:  # Good performance
                # Make strategies more aggressive
                strategy['buy_threshold'] *= 0.9
                strategy['sell_threshold'] *= 0.9
                strategy['confidence_min'] = max(50, strategy['confidence_min'] - 5)
                strategy['position_size'] = min(0.5, strategy['position_size'] * 1.1)
                adjustments.append(f"{strategy_name}: Made more aggressive (high accuracy)")
        
        if adjustments:
            self.save_strategies()
            print("\n🔧 Auto-Adjusted Strategies:")
            for adj in adjustments:
                print(f"   • {adj}")
        
        return adjustments
    
    def get_recommendation(self, stock_name, prediction_data):
        """Get trading recommendation based on current strategies"""
        recommendations = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy['enabled']:
                continue
            
            predicted_change = prediction_data['change_pct']
            confidence = prediction_data['confidence']
            
            # Check if meets strategy criteria
            if confidence < strategy['confidence_min']:
                continue
            
            if predicted_change > strategy['buy_threshold']:
                action = 'BUY'
                reason = f"Predicted +{predicted_change:.2f}% > threshold {strategy['buy_threshold']:.2f}%"
            elif predicted_change < strategy['sell_threshold']:
                action = 'SELL'
                reason = f"Predicted {predicted_change:.2f}% < threshold {strategy['sell_threshold']:.2f}%"
            else:
                action = 'HOLD'
                reason = f"Change {predicted_change:.2f}% within hold range"
            
            recommendations.append({
                'strategy': strategy_name,
                'action': action,
                'reason': reason,
                'position_size': strategy['position_size'],
                'stop_loss': strategy['stop_loss'],
                'take_profit': strategy['take_profit'],
                'confidence': confidence
            })
        
        return recommendations
    
    def generate_daily_report(self, stock_name):
        """Generate daily validation report"""
        if stock_name not in self.predictions_history:
            return None
        
        predictions = self.predictions_history[stock_name]
        validated = [p for p in predictions.values() if p.get('validated', False)]
        
        if not validated:
            return None
        
        # Calculate statistics
        validations = [p['validation'] for p in validated]
        accuracies = [v['accuracy'] for v in validations]
        
        report = {
            'stock': stock_name,
            'total_predictions': len(predictions),
            'validated_predictions': len(validated),
            'avg_accuracy': np.mean(accuracies),
            'best_accuracy': np.max(accuracies),
            'worst_accuracy': np.min(accuracies),
            'accurate_count': len([v for v in validations if v['status'] == 'accurate']),
            'moderate_count': len([v for v in validations if v['status'] == 'moderate']),
            'poor_count': len([v for v in validations if v['status'] == 'poor']),
            'generated_at': datetime.now().isoformat()
        }
        
        return report

def load_stock_csv(file_path):
    """Load stock CSV handling yfinance multi-index format"""
    import pandas as pd
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

def run_auto_validation(stock_file, stock_name):
    """Run automatic validation and adjustment"""
    print(f"\n{'='*70}")
    print(f"Auto-Validation System: {stock_name}")
    print(f"{'='*70}\n")
    
    # Load data with proper format handling
    try:
        data = load_stock_csv(stock_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize validator
    validator = AutoValidator()
    
    # Validate predictions
    print("Validating previous predictions...")
    validations = validator.validate_predictions(stock_name, data)
    
    if validations:
        print(f"✓ Validated {len(validations)} predictions\n")
        
        # Display results
        print("Validation Results:")
        for v in validations[-5:]:  # Show last 5
            status_icon = "✅" if v['status'] == 'accurate' else "⚠️" if v['status'] == 'moderate' else "❌"
            print(f"   {status_icon} {v['date']}: Accuracy {v['accuracy']:.2f}% (Error: {v['error_pct']:.2f}%)")
        
        # Auto-adjust strategies
        print("\n")
        adjustments = validator.adjust_strategies(validations)
        
        # Generate report
        report = validator.generate_daily_report(stock_name)
        if report:
            print(f"\n📊 Performance Report:")
            print(f"   Total Predictions: {report['total_predictions']}")
            print(f"   Validated: {report['validated_predictions']}")
            print(f"   Average Accuracy: {report['avg_accuracy']:.2f}%")
            print(f"   Accurate: {report['accurate_count']}, Moderate: {report['moderate_count']}, Poor: {report['poor_count']}")
    else:
        print("No predictions to validate yet.")
    
    print(f"\n{'='*70}\n")
    
    return validator

if __name__ == "__main__":
    import os
    
    if os.path.exists("EOD/RELIANCE.csv"):
        validator = run_auto_validation("EOD/RELIANCE.csv", "RELIANCE")
    else:
        print("Please run fetch_stocks.py first!")

