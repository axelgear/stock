"""
Flask Backend API for Stock Trading Dashboard
RESTful API with filtering, predictions, and recommendations
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import glob
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_predictor import AIStockPredictor
from auto_validator import AutoValidator
from database import (
    init_database, save_stock_to_db, get_stock_from_db,
    get_latest_prices_all_stocks, get_sync_metadata,
    get_database_stats, get_stocks_list,
    save_prediction, get_predictions_all_stocks, get_prediction_for_stock
)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})  # Enable CORS for frontend with explicit headers

# Cache for predictions
prediction_cache = {}
validator = AutoValidator()

# Initialize database on startup
init_database()

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Get list of all available stocks"""
    try:
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        stock_files = glob.glob(os.path.join(eod_dir, '*.csv'))
        stocks = [os.path.basename(f).replace('.csv', '') for f in stock_files]
        
        return jsonify({
            'success': True,
            'data': sorted(stocks),
            'count': len(stocks)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def load_stock_csv(file_path):
    """Load stock CSV handling yfinance multi-index format"""
    # Read first 3 rows to check format
    with open(file_path, 'r') as f:
        lines = [f.readline().strip() for _ in range(3)]
    
    # Check if it's the old yfinance multi-index format
    if lines[0].startswith('Price,') and lines[1].startswith('Ticker,') and lines[2].startswith('Date,'):
        # Old format: skip first 3 rows, use first column as Date
        df = pd.read_csv(file_path, skiprows=3, header=None)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    else:
        # New format: normal CSV
        df = pd.read_csv(file_path)
    
    # Ensure Date column exists and is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("Date column not found")
    
    return df

@app.route('/api/stock/<stock_name>', methods=['GET'])
def get_stock_data(stock_name):
    """Get stock data with optional filtering"""
    try:
        # Get query parameters
        period = request.args.get('period', '1y')  # 1m, 3m, 6m, 1y, 2y, 5y, all
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Load data
        file_path = os.path.join(os.path.dirname(__file__), '..', 'EOD', f'{stock_name}.csv')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        # Load with proper format handling
        df = load_stock_csv(file_path)
        
        # Filter by period
        if period != 'all':
            days_map = {'1m': 30, '3m': 90, '6m': 180, '1y': 365, '2y': 730, '5y': 1825}
            if period in days_map:
                cutoff_date = datetime.now() - timedelta(days=days_map[period])
                df = df[df['Date'] >= cutoff_date]
        
        # Filter by date range
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Calculate additional metrics
        df['Returns'] = df['Close'].pct_change() * 100
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        # Convert to JSON format and replace NaN with None (null in JSON)
        data = df.to_dict('records')
        
        # Replace NaN with None for valid JSON
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = None
        
        # Calculate summary statistics with safety checks
        try:
            current_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_close
            
            summary = {
                'current_price': current_close,
                'open_price': float(df['Open'].iloc[-1]),
                'high_price': float(df['High'].iloc[-1]),
                'low_price': float(df['Low'].iloc[-1]),
                'volume': int(df['Volume'].iloc[-1]),
                'change': current_close - prev_close,
                'change_pct': ((current_close - prev_close) / prev_close * 100) if prev_close != 0 else 0,
                'high_52w': float(df['High'].tail(min(252, len(df))).max()),
                'low_52w': float(df['Low'].tail(min(252, len(df))).min()),
                'avg_volume': float(df['Volume'].mean())
            }
        except Exception as e:
            # Fallback summary with safe defaults
            summary = {
                'current_price': float(df['Close'].iloc[-1]) if len(df) > 0 else 0,
                'open_price': 0,
                'high_price': 0,
                'low_price': 0,
                'volume': 0,
                'change': 0,
                'change_pct': 0,
                'high_52w': 0,
                'low_52w': 0,
                'avg_volume': 0
            }
        
        return jsonify({
            'success': True,
            'data': data,
            'summary': summary,
            'count': len(data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/<stock_name>', methods=['GET'])
def predict_stock(stock_name):
    """Get AI prediction for a stock"""
    try:
        # Check cache
        cache_key = f"{stock_name}_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in prediction_cache:
            return jsonify(prediction_cache[cache_key])
        
        # Load data
        file_path = os.path.join(os.path.dirname(__file__), '..', 'EOD', f'{stock_name}.csv')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        # Load with proper format handling
        df = load_stock_csv(file_path)
        
        # Initialize predictor
        predictor = AIStockPredictor(df)
        
        # Train ensemble model
        accuracy = predictor.train_ensemble()
        
        # Get prediction
        prediction = predictor.predict_next_day('ensemble')
        
        if not prediction:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
        
        # Generate signal
        signal = predictor.generate_trading_signal()
        
        # Get recommendations
        recommendations = validator.get_recommendation(stock_name, prediction)
        
        # Record prediction
        validator.record_prediction(stock_name, prediction)
        
        result = {
            'success': True,
            'stock': stock_name,
            'prediction': {
                'current_price': float(prediction['current_price']),
                'predicted_price': float(prediction['prediction']),
                'change': float(prediction['change']),
                'change_pct': float(prediction['change_pct']),
                'confidence': float(prediction['confidence']),
                'model': prediction['model']
            },
            'signal': signal,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        prediction_cache[cache_key] = result
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/validate/<stock_name>', methods=['GET'])
def validate_stock(stock_name):
    """Validate previous predictions"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'EOD', f'{stock_name}.csv')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        # Load with proper format handling
        df = load_stock_csv(file_path)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Validate predictions
        validations = validator.validate_predictions(stock_name, df)
        
        # Get report
        report = validator.generate_daily_report(stock_name)
        
        return jsonify({
            'success': True,
            'validations': validations,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/filter', methods=['POST'])
def filter_stocks():
    """Filter stocks by various criteria"""
    try:
        criteria = request.json
        
        # Get all stocks
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        stock_files = glob.glob(os.path.join(eod_dir, '*.csv'))
        
        results = []
        
        for file_path in stock_files:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                df = load_stock_csv(file_path)
                if df.empty:
                    continue
                
                latest = df.iloc[-1]
                
                # Calculate metrics
                price = latest['Close']
                volume = latest['Volume']
                
                # Calculate returns
                returns_1d = ((latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']) * 100 if len(df) > 1 else 0
                returns_5d = ((latest['Close'] - df.iloc[-6]['Close']) / df.iloc[-6]['Close']) * 100 if len(df) > 5 else 0
                returns_1m = ((latest['Close'] - df.iloc[-22]['Close']) / df.iloc[-22]['Close']) * 100 if len(df) > 21 else 0
                
                # 52-week high/low
                high_52w = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
                low_52w = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
                
                # Apply filters
                if 'price_min' in criteria and price < criteria['price_min']:
                    continue
                if 'price_max' in criteria and price > criteria['price_max']:
                    continue
                if 'volume_min' in criteria and volume < criteria['volume_min']:
                    continue
                if 'returns_1d_min' in criteria and returns_1d < criteria['returns_1d_min']:
                    continue
                if 'returns_1m_min' in criteria and returns_1m < criteria['returns_1m_min']:
                    continue
                
                results.append({
                    'stock': stock_name,
                    'price': float(price),
                    'volume': int(volume),
                    'returns_1d': float(returns_1d),
                    'returns_5d': float(returns_5d),
                    'returns_1m': float(returns_1m),
                    'high_52w': float(high_52w),
                    'low_52w': float(low_52w)
                })
                
            except Exception:
                continue
        
        # Sort results
        sort_by = criteria.get('sort_by', 'returns_1d')
        sort_order = criteria.get('sort_order', 'desc')
        
        results = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=(sort_order == 'desc'))
        
        # Limit results
        limit = criteria.get('limit', 50)
        results = results[:limit]
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get current trading strategies"""
    try:
        strategies = validator.strategies
        return jsonify({
            'success': True,
            'strategies': strategies
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/strategies', methods=['PUT'])
def update_strategies():
    """Update trading strategies"""
    try:
        new_strategies = request.json
        validator.strategies = new_strategies
        validator.save_strategies()
        
        return jsonify({
            'success': True,
            'message': 'Strategies updated successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# DATABASE CACHE ENDPOINTS - For fast loading and filtering
# ============================================================================

@app.route('/api/db/sync-all', methods=['POST'])
def sync_all_stocks_to_db():
    """Load all stocks from CSV files into database - ONE TIME OPERATION"""
    try:
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        stock_files = glob.glob(os.path.join(eod_dir, '*.csv'))
        
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for file_path in stock_files:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                df = load_stock_csv(file_path)
                success, result = save_stock_to_db(stock_name, df)
                
                if success:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'stock': stock_name,
                        'error': result
                    })
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'stock': stock_name,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'message': f"Synced {results['success']} stocks to database",
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/all-stocks-fast', methods=['GET'])
def get_all_stocks_fast():
    """Get all stocks data from database with 7-day performance - OPTIMIZED!"""
    try:
        df_stocks = get_latest_prices_all_stocks()
        df_predictions = get_predictions_all_stocks()
        
        # Merge predictions if available
        if not df_predictions.empty:
            df = df_stocks.merge(
                df_predictions[['stock_name', 'predicted_price', 'predicted_change_pct', 'confidence']],
                on='stock_name',
                how='left'
            )
        else:
            df = df_stocks
            df['predicted_price'] = None
            df['predicted_change_pct'] = None
            df['confidence'] = None
        
        # Convert to list of dicts
        stocks_data = []
        for _, row in df.iterrows():
            stocks_data.append({
                'stock_name': row['stock_name'],
                'today_date': str(row['today_date']) if pd.notna(row['today_date']) else '',
                'today_close': float(row['today_close']) if pd.notna(row['today_close']) else 0,
                'today_open': float(row['today_open']) if pd.notna(row['today_open']) else 0,
                'today_high': float(row['today_high']) if pd.notna(row['today_high']) else 0,
                'today_low': float(row['today_low']) if pd.notna(row['today_low']) else 0,
                'today_change': float(row['today_change']) if pd.notna(row['today_change']) else 0,
                'today_pct': float(row['today_pct']) if pd.notna(row['today_pct']) else 0,
                'yesterday_pct': float(row['yesterday_pct']) if pd.notna(row['yesterday_pct']) else 0,
                'day2_pct': float(row['day2_pct']) if pd.notna(row['day2_pct']) else 0,
                'day3_pct': float(row['day3_pct']) if pd.notna(row['day3_pct']) else 0,
                'day4_pct': float(row['day4_pct']) if pd.notna(row['day4_pct']) else 0,
                'day5_pct': float(row['day5_pct']) if pd.notna(row['day5_pct']) else 0,
                'day6_pct': float(row['day6_pct']) if pd.notna(row['day6_pct']) else 0,
                'week_7day_pct': float(row['week_7day_pct']) if pd.notna(row['week_7day_pct']) else 0,
                'week_7to14_pct': float(row['week_7to14_pct']) if pd.notna(row['week_7to14_pct']) else 0,
                'week_14to21_pct': float(row['week_14to21_pct']) if pd.notna(row['week_14to21_pct']) else 0,
                'week_21to31_pct': float(row['week_21to31_pct']) if pd.notna(row['week_21to31_pct']) else 0,
                'intraday_change': float(row['intraday_change']) if pd.notna(row['intraday_change']) else 0,
                'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                'volume_ratio': float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else 0,
                'week_52_high': float(row['week_52_high']) if pd.notna(row['week_52_high']) else 0,
                'week_52_low': float(row['week_52_low']) if pd.notna(row['week_52_low']) else 0,
                'week_52_position': float(row['week_52_position']) if pd.notna(row['week_52_position']) else 0,
                'predicted_price': float(row['predicted_price']) if pd.notna(row['predicted_price']) else None,
                'predicted_change_pct': float(row['predicted_change_pct']) if pd.notna(row['predicted_change_pct']) else None,
                'confidence': float(row['confidence']) if pd.notna(row['confidence']) else None
            })
        
        return jsonify({
            'success': True,
            'data': stocks_data,
            'count': len(stocks_data),
            'cached': True,
            'has_predictions': not df_predictions.empty
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/stats', methods=['GET'])
def get_db_stats():
    """Get database statistics"""
    try:
        stats = get_database_stats()
        metadata = get_sync_metadata()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_syncs': metadata.head(10).to_dict('records') if not metadata.empty else []
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/update-today', methods=['POST'])
def update_today_data():
    """Update only today's data for all stocks - FAST UPDATE"""
    try:
        import yfinance as yf
        
        # Get list of stocks from database
        stocks = get_stocks_list()
        
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        today = datetime.now().date()
        
        for stock_name in stocks[:50]:  # Limit to 50 for now
            try:
                # Download only last 2 days
                ticker = yf.Ticker(f"{stock_name}.NS")
                df = ticker.history(period='2d')
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Only save if we have today's data
                    latest_date = pd.to_datetime(df['Date'].iloc[-1]).date()
                    if latest_date == today:
                        save_stock_to_db(stock_name, df.tail(1))
                        results['success'] += 1
                    else:
                        results['skipped'] += 1
                else:
                    results['skipped'] += 1
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'stock': stock_name,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'message': f"Updated {results['success']} stocks",
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/predict-all', methods=['POST'])
def predict_all_stocks():
    """Generate predictions for all stocks - BULK PREDICTION"""
    try:
        # Get parameters
        data = request.get_json() or {}
        limit = data.get('limit', 50)  # Limit for performance
        force_refresh = data.get('force_refresh', False)
        
        # Get list of stocks
        stocks = get_stocks_list()[:limit]
        
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        for stock_name in stocks:
            try:
                # Check if prediction already exists (unless force refresh)
                if not force_refresh:
                    existing = get_prediction_for_stock(stock_name)
                    if existing and existing['prediction_date'] == prediction_date:
                        results['skipped'] += 1
                        continue
                
                # Load data and predict
                file_path = os.path.join(eod_dir, f'{stock_name}.csv')
                if not os.path.exists(file_path):
                    results['failed'] += 1
                    continue
                
                df = load_stock_csv(file_path)
                if len(df) < 60:  # Need enough data
                    results['skipped'] += 1
                    continue
                
                # Simple prediction using last price and moving average
                current_price = float(df['Close'].iloc[-1])
                ma_20 = df['Close'].tail(20).mean()
                ma_50 = df['Close'].tail(50).mean() if len(df) >= 50 else ma_20
                
                # Calculate trend
                recent_trend = ((current_price - ma_20) / ma_20) * 100
                long_trend = ((ma_20 - ma_50) / ma_50) * 100 if ma_50 > 0 else 0
                
                # Simple prediction: weighted average of trends
                predicted_change = (recent_trend * 0.6 + long_trend * 0.4) * 0.3  # Conservative
                predicted_price = current_price * (1 + predicted_change / 100)
                
                # Confidence based on volatility
                volatility = df['Close'].tail(20).std() / current_price
                confidence = min(max(70 - volatility * 100, 40), 90)
                
                # Save prediction
                save_prediction(
                    stock_name=stock_name,
                    prediction_date=prediction_date,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    predicted_change_pct=predicted_change,
                    confidence=confidence,
                    model_used='MA_Trend'
                )
                
                results['success'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'stock': stock_name,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'message': f"Generated predictions for {results['success']} stocks",
            'results': results,
            'prediction_date': prediction_date
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Serve frontend
@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    # If file not found, return index.html for client-side routing
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Stock Trading API Server")
    print("="*70)
    print("\n📡 API Endpoints:")
    print("   GET  /api/stocks              - List all stocks")
    print("   GET  /api/stock/<name>        - Get stock data")
    print("   GET  /api/predict/<name>      - AI prediction")
    print("   GET  /api/validate/<name>     - Validate predictions")
    print("   POST /api/filter              - Filter stocks")
    print("   GET  /api/strategies          - Get strategies")
    print("\n💾 Database Cache Endpoints (OPTIMIZED!):")
    print("   POST /api/db/sync-all         - Load all CSVs into DB")
    print("   GET  /api/db/all-stocks-fast  - Get all stocks + 7-day % (INSTANT!)")
    print("   POST /api/db/update-today     - Update today's data only")
    print("   POST /api/db/predict-all      - Generate predictions for all stocks")
    print("   GET  /api/db/stats            - Database statistics")
    print("   PUT  /api/strategies          - Update strategies")
    print("\n🌐 Server running on: http://localhost:5050")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5050, debug=True)

