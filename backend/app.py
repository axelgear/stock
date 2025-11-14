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

try:
    from advanced_ai_predictor import AdvancedStockPredictor
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    from ai_predictor import AIStockPredictor
    ADVANCED_AI_AVAILABLE = False
    print("⚠️ Advanced AI predictor not available, using basic predictor")

from auto_validator import AutoValidator
from database import (
    init_database, save_stock_to_db, get_stock_from_db,
    get_latest_prices_all_stocks, get_sync_metadata,
    get_database_stats, get_stocks_list,
    save_prediction, get_predictions_all_stocks, get_prediction_for_stock,
    save_validation_result, get_validation_results, get_model_performance_stats,
    get_top_performing_stocks, get_daily_accuracy_trend, optimize_database
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
    """Load stock CSV handling various yfinance formats"""
    # Read first few rows to check format
    with open(file_path, 'r') as f:
        lines = [f.readline().strip() for _ in range(5)]
    
    print(f"Loading {os.path.basename(file_path)}...")
    print(f"First line: {lines[0]}")
    print(f"Second line: {lines[1]}")
    
    # Check if it's the old yfinance multi-index format
    if lines[0].startswith('Price,') and lines[1].startswith('Ticker,') and lines[2].startswith('Date,'):
        # Old format: skip first 3 rows, use first column as Date
        df = pd.read_csv(file_path, skiprows=3, header=None)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    elif lines[1].count('.NS') >= 3:  # Second row contains ticker symbols
        # Skip the problematic second row
        df = pd.read_csv(file_path, skiprows=[1])
        print(f"Skipped ticker row, loaded {len(df)} rows")
    else:
        # Normal CSV format
        df = pd.read_csv(file_path)
    
    # Clean up any empty Date values (first column might be empty)
    if df.iloc[0, 0] == '' or pd.isna(df.iloc[0, 0]):
        df = df.dropna(subset=[df.columns[0]])
    
    # Ensure Date column exists and is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Remove any rows with invalid dates
        df = df.dropna(subset=['Date'])
    else:
        raise ValueError("Date column not found")
    
    # Ensure we have the required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert price columns to numeric, handling any non-numeric values
    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in price_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid price data
    df = df.dropna(subset=price_columns)
    
    print(f"Successfully loaded {len(df)} valid rows")
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
        
        # Initialize predictor (use advanced if available)
        if ADVANCED_AI_AVAILABLE:
            predictor = AdvancedStockPredictor(df)
        
            # Train multiple models for better accuracy
            accuracies = predictor.train_all_models()
            
            # Use best model or ensemble
            best_model = max(accuracies, key=accuracies.get) if accuracies else 'advanced_ensemble'
            prediction = predictor.generate_ensemble_prediction()
            
            if not prediction:
                # Fallback to best individual model
                prediction = predictor.predict_next_day(best_model)
            
            accuracy = accuracies.get(best_model, 0) if accuracies else 0
        else:
            # Use basic predictor
            predictor = AIStockPredictor(df)
            accuracy = predictor.train_ensemble()
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
        print("\n" + "="*70)
        print("🚀 STARTING DATABASE SYNC")
        print("="*70)
        
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        stock_files = glob.glob(os.path.join(eod_dir, '*.csv'))
        
        print(f"📁 EOD Directory: {eod_dir}")
        print(f"📊 Found {len(stock_files)} CSV files")
        
        if len(stock_files) == 0:
            return jsonify({
                'success': False,
                'error': f'No CSV files found in {eod_dir}',
                'results': {'success': 0, 'failed': 0, 'errors': []}
            })
        
        # Initialize database first
        print("🏗️ Initializing database...")
        init_database()
        
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        # Process first few files for testing
        test_files = stock_files[:5]  # Test with first 5 files
        print(f"🧪 Testing sync with first {len(test_files)} files...")
        
        for i, file_path in enumerate(test_files, 1):
            stock_name = os.path.basename(file_path).replace('.csv', '')
            print(f"\n[{i}/{len(test_files)}] Processing {stock_name}...")
            
            try:
                # Load CSV
                df = load_stock_csv(file_path)
                if df.empty:
                    raise ValueError("Empty dataframe after loading CSV")
                
                print(f"   ✓ Loaded {len(df)} rows from CSV")
                
                # Save to database
                success, result = save_stock_to_db(stock_name, df)
                
                if success:
                    results['success'] += 1
                    print(f"   ✅ Saved {result} records to database")
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'stock': stock_name,
                        'error': result
                    })
                    print(f"   ❌ Failed to save: {result}")
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = str(e)
                results['errors'].append({
                    'stock': stock_name,
                    'error': error_msg
                })
                print(f"   ❌ Error processing {stock_name}: {error_msg}")
        
        # If test was successful, process all files
        if results['success'] > 0:
            print(f"\n✅ Test successful! Processing remaining {len(stock_files) - len(test_files)} files...")
            
            for i, file_path in enumerate(stock_files[len(test_files):], len(test_files) + 1):
                stock_name = os.path.basename(file_path).replace('.csv', '')
                
                try:
                    df = load_stock_csv(file_path)
                    if df.empty:
                        continue
                    
                    success, result = save_stock_to_db(stock_name, df)
                    
                    if success:
                        results['success'] += 1
                        if i % 50 == 0:  # Print progress every 50 stocks
                            print(f"   📊 Processed {i}/{len(stock_files)} stocks...")
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
        
        print("\n" + "="*70)
        print("📊 SYNC COMPLETED")
        print("="*70)
        print(f"✅ Successful: {results['success']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📁 Total files: {len(stock_files)}")
        
        # Verify data was saved
        try:
            stats = get_database_stats()
            print(f"💾 Database now contains: {stats['total_records']} records for {stats['total_stocks']} stocks")
        except Exception as e:
            print(f"⚠️ Could not verify database stats: {e}")
        
        return jsonify({
            'success': True,
            'message': f"Synced {results['success']} stocks to database (Failed: {results['failed']})",
            'results': results,
            'total_files': len(stock_files)
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ SYNC FAILED: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/api/db/test-sync', methods=['POST'])
def test_sync():
    """Test the sync functionality with one stock"""
    try:
        print("\n🧪 TESTING SYNC FUNCTIONALITY")
        
        # Test database connection
        try:
            stats = get_database_stats()
            print(f"✅ Database accessible: {stats['total_records']} records")
        except Exception as e:
            print(f"❌ Database error: {e}")
            init_database()
            print("✅ Database initialized")
        
        # Test CSV loading
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        test_file = os.path.join(eod_dir, 'RELIANCE.csv')
        
        if not os.path.exists(test_file):
            test_files = glob.glob(os.path.join(eod_dir, '*.csv'))
            if test_files:
                test_file = test_files[0]
            else:
                return jsonify({
                    'success': False, 
                    'error': f'No CSV files found in {eod_dir}'
                })
        
        stock_name = os.path.basename(test_file).replace('.csv', '')
        print(f"🧪 Testing with: {stock_name}")
        
        # Load CSV
        df = load_stock_csv(test_file)
        print(f"✅ Loaded {len(df)} rows from CSV")
        
        # Save to database
        success, result = save_stock_to_db(stock_name, df)
        
        if success:
            print(f"✅ Saved {result} records to database")
            
            # Verify data was saved
            stats = get_database_stats()
            
            return jsonify({
                'success': True,
                'message': f'Test successful! Saved {result} records for {stock_name}',
                'test_stock': stock_name,
                'records_saved': result,
                'database_stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to save {stock_name}: {result}'
            })
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Test failed: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

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
        
        # Helper function to safely convert to float, handling NaN and Infinity
        def safe_float(value, default=0):
            """Convert value to float, handling NaN, Infinity, and None"""
            if pd.isna(value):
                return default
            try:
                f = float(value)
                if np.isinf(f) or np.isnan(f):
                    return default
                return f
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Convert value to int, handling NaN and invalid values"""
            if pd.isna(value):
                return default
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return default
        
        # Convert to list of dicts
        stocks_data = []
        for _, row in df.iterrows():
            stocks_data.append({
                'stock_name': row['stock_name'],
                'today_date': str(row['today_date']) if pd.notna(row['today_date']) else '',
                'today_close': safe_float(row['today_close']),
                'today_open': safe_float(row['today_open']),
                'today_high': safe_float(row['today_high']),
                'today_low': safe_float(row['today_low']),
                'today_change': safe_float(row['today_change']),
                'today_pct': safe_float(row['today_pct']),
                'yesterday_pct': safe_float(row['yesterday_pct']),
                'day2_pct': safe_float(row['day2_pct']),
                'day3_pct': safe_float(row['day3_pct']),
                'day4_pct': safe_float(row['day4_pct']),
                'day5_pct': safe_float(row['day5_pct']),
                'day6_pct': safe_float(row['day6_pct']),
                'week_7day_pct': safe_float(row['week_7day_pct']),
                'week_7to14_pct': safe_float(row['week_7to14_pct']),
                'week_14to21_pct': safe_float(row['week_14to21_pct']),
                'week_21to31_pct': safe_float(row['week_21to31_pct']),
                'intraday_change': safe_float(row['intraday_change']),
                'volume': safe_int(row['volume']),
                'volume_ratio': safe_float(row['volume_ratio']),
                'week_52_high': safe_float(row['week_52_high']),
                'week_52_low': safe_float(row['week_52_low']),
                'week_52_position': safe_float(row['week_52_position']),
                'predicted_price': safe_float(row['predicted_price'], None) if pd.notna(row['predicted_price']) else None,
                'predicted_change_pct': safe_float(row['predicted_change_pct'], None) if pd.notna(row['predicted_change_pct']) else None,
                'confidence': safe_float(row['confidence'], None) if pd.notna(row['confidence']) else None
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
    """Generate predictions for all stocks - BULK PREDICTION with Advanced AI"""
    try:
        # Get parameters
        data = request.get_json() or {}
        limit = data.get('limit', 5000)  # Limit for performance
        force_refresh = data.get('force_refresh', False)
        use_advanced_ai = data.get('use_advanced_ai', True)
        model_type = data.get('model_type', 'advanced_ensemble')  # lstm, polynomial, advanced_ensemble
        
        # Get list of stocks
        stocks = get_stocks_list()[:limit]
        
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'model_accuracies': {}
        }
        
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"🚀 Starting bulk prediction for {len(stocks)} stocks using {'Advanced AI' if use_advanced_ai and ADVANCED_AI_AVAILABLE else 'Basic AI'}")
        
        for i, stock_name in enumerate(stocks, 1):
            try:
                print(f"[{i}/{len(stocks)}] Processing {stock_name}...")
                
                # Check if prediction already exists (unless force refresh)
                if not force_refresh:
                    existing = get_prediction_for_stock(stock_name)
                    if existing and existing['prediction_date'] == prediction_date:
                        results['skipped'] += 1
                        print(f"   ⏭️ Skipped (prediction exists)")
                        continue
                
                # Load data and predict
                file_path = os.path.join(eod_dir, f'{stock_name}.csv')
                if not os.path.exists(file_path):
                    results['failed'] += 1
                    print(f"   ❌ File not found")
                    continue
                
                df = load_stock_csv(file_path)
                if len(df) < 100:  # Need enough data for advanced models
                    results['skipped'] += 1
                    print(f"   ⏭️ Insufficient data ({len(df)} rows)")
                    continue
                
                current_price = float(df['Close'].iloc[-1])
                
                if use_advanced_ai and ADVANCED_AI_AVAILABLE:
                    # Use Advanced AI Predictor
                    try:
                        predictor = AdvancedStockPredictor(df)
                        
                        # Train model if not cached (simplified for bulk prediction)
                        if model_type == 'advanced_ensemble':
                            accuracy = predictor.train_advanced_ensemble()
                        elif model_type == 'polynomial':
                            accuracy = predictor.train_polynomial_regression()
                        elif model_type == 'lstm':
                            accuracy = predictor.train_lstm_model()
                        else:
                            # Default to ensemble
                            accuracy = predictor.train_advanced_ensemble()
                            model_type = 'advanced_ensemble'
                        
                        # Generate prediction
                        prediction = predictor.predict_next_day(model_type)
                        
                        if prediction:
                            predicted_price = prediction['prediction']
                            predicted_change = prediction['change_pct']
                            confidence = prediction['confidence']
                            model_used = f"Advanced_{model_type}"
                            results['model_accuracies'][stock_name] = accuracy
                            print(f"   ✅ Advanced AI prediction: {predicted_change:+.2f}% (conf: {confidence:.1f}%)")
                        else:
                            raise Exception("Advanced prediction failed, falling back to simple method")
                        
                    except Exception as e:
                        print(f"   ⚠️ Advanced AI failed ({e}), using simple method")
                        # Fallback to simple method
                        predicted_price, predicted_change, confidence, model_used = simple_prediction(df, current_price)
                else:
                    # Use simple prediction method
                    predicted_price, predicted_change, confidence, model_used = simple_prediction(df, current_price)
                    print(f"   ✅ Simple prediction: {predicted_change:+.2f}% (conf: {confidence:.1f}%)")
                
                # Save prediction
                save_prediction(
                    stock_name=stock_name,
                    prediction_date=prediction_date,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    predicted_change_pct=predicted_change,
                    confidence=confidence,
                    model_used=model_used
                )
                
                results['success'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'stock': stock_name,
                    'error': str(e)
                })
                print(f"   ❌ Failed: {e}")
        
        # Calculate average accuracy if using advanced AI
        avg_accuracy = 0
        if results['model_accuracies']:
            avg_accuracy = sum(results['model_accuracies'].values()) / len(results['model_accuracies'])
        
        response_data = {
            'success': True,
            'message': f"Generated predictions for {results['success']} stocks",
            'results': results,
            'prediction_date': prediction_date,
            'advanced_ai_used': use_advanced_ai and ADVANCED_AI_AVAILABLE,
            'model_type': model_type,
            'average_accuracy': avg_accuracy
        }
        
        print(f"✅ Bulk prediction completed: {results['success']} success, {results['failed']} failed, {results['skipped']} skipped")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def simple_prediction(df, current_price):
    """Simple prediction fallback method"""
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
    
    return predicted_price, predicted_change, confidence, 'Simple_MA_Trend'

@app.route('/api/db/train-models/<stock_name>', methods=['POST'])
def train_stock_models(stock_name):
    """Train advanced AI models for a specific stock"""
    try:
        if not ADVANCED_AI_AVAILABLE:
            return jsonify({'success': False, 'error': 'Advanced AI not available'}), 400
        
        # Get parameters
        data = request.get_json() or {}
        models_to_train = data.get('models', ['advanced_ensemble', 'polynomial', 'lstm'])
        
        # Load data
        file_path = os.path.join(os.path.dirname(__file__), '..', 'EOD', f'{stock_name}.csv')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Stock not found'}), 404
        
        df = load_stock_csv(file_path)
        if len(df) < 100:
            return jsonify({'success': False, 'error': 'Insufficient data for training'}), 400
        
        # Initialize predictor
        predictor = AdvancedStockPredictor(df)
        
        # Train specified models
        results = {}
        for model_name in models_to_train:
            try:
                if model_name == 'advanced_ensemble':
                    accuracy = predictor.train_advanced_ensemble()
                elif model_name == 'polynomial':
                    accuracy = predictor.train_polynomial_regression()
                elif model_name == 'lstm':
                    accuracy = predictor.train_lstm_model()
                else:
                    continue
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'status': 'success'
                }
            except Exception as e:
                results[model_name] = {
                    'accuracy': 0,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Save models
        try:
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'trained_models', stock_name)
            predictor.save_models(model_dir)
        except Exception as e:
            print(f"Warning: Could not save models: {e}")
        
        return jsonify({
            'success': True,
            'stock': stock_name,
            'results': results,
            'best_model': max(results, key=lambda x: results[x]['accuracy']) if results else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/validate-predictions', methods=['POST'])
def validate_all_predictions():
    """Validate predictions against actual prices"""
    try:
        # Get parameters
        data = request.get_json() or {}
        validation_date = data.get('date', (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
        
        # Get predictions for the date
        predictions_df = get_predictions_all_stocks(validation_date)
        
        if predictions_df.empty:
            return jsonify({
                'success': True,
                'message': 'No predictions found for validation',
                'validations': []
            })
        
        validations = []
        eod_dir = os.path.join(os.path.dirname(__file__), '..', 'EOD')
        
        for _, pred_row in predictions_df.iterrows():
            stock_name = pred_row['stock_name']
            predicted_price = pred_row['predicted_price']
            predicted_change_pct = pred_row['predicted_change_pct']
            confidence = pred_row['confidence']
            model_used = pred_row['model_used']
            
            try:
                # Load actual data
                file_path = os.path.join(eod_dir, f'{stock_name}.csv')
                if not os.path.exists(file_path):
                    continue
                
                df = load_stock_csv(file_path)
                
                # Find actual price on validation date
                validation_datetime = datetime.strptime(validation_date, '%Y-%m-%d')
                actual_row = df[df['Date'].dt.date == validation_datetime.date()]
                
                if not actual_row.empty:
                    actual_price = float(actual_row['Close'].iloc[0])
                    
                    # Calculate validation metrics
                    price_error = abs(predicted_price - actual_price)
                    price_error_pct = (price_error / actual_price) * 100
                    
                    # Direction accuracy (did we predict the right direction?)
                    actual_change_pct = ((actual_price - pred_row['current_price']) / pred_row['current_price']) * 100
                    direction_correct = (predicted_change_pct > 0) == (actual_change_pct > 0)
                    
                    validation = {
                        'stock_name': stock_name,
                        'predicted_price': predicted_price,
                        'actual_price': actual_price,
                        'predicted_change_pct': predicted_change_pct,
                        'actual_change_pct': actual_change_pct,
                        'price_error': price_error,
                        'price_error_pct': price_error_pct,
                        'direction_correct': direction_correct,
                        'confidence': confidence,
                        'model_used': model_used,
                        'accuracy_score': max(0, 100 - price_error_pct)
                    }
                    
                    validations.append(validation)
                    
                    # Save validation result to database for tracking
                    save_validation_result(
                        stock_name=stock_name,
                        validation_date=validation_date,
                        predicted_price=predicted_price,
                        actual_price=actual_price,
                        predicted_change_pct=predicted_change_pct,
                        actual_change_pct=actual_change_pct,
                        confidence=confidence,
                        model_used=model_used
                    )
                    
            except Exception as e:
                print(f"Error validating {stock_name}: {e}")
                continue
        
        # Calculate summary statistics
        if validations:
            avg_accuracy = sum(v['accuracy_score'] for v in validations) / len(validations)
            direction_accuracy = sum(1 for v in validations if v['direction_correct']) / len(validations) * 100
            avg_price_error_pct = sum(v['price_error_pct'] for v in validations) / len(validations)
        else:
            avg_accuracy = direction_accuracy = avg_price_error_pct = 0
        
        return jsonify({
            'success': True,
            'validation_date': validation_date,
            'validations': validations,
            'summary': {
                'total_validated': len(validations),
                'average_accuracy': avg_accuracy,
                'direction_accuracy': direction_accuracy,
                'average_price_error_pct': avg_price_error_pct
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/model-comparison', methods=['GET'])
def model_comparison():
    """Compare different AI models performance"""
    try:
        # Get predictions with different models
        predictions_df = get_predictions_all_stocks()
        
        if predictions_df.empty:
            return jsonify({
                'success': True,
                'message': 'No predictions found',
                'model_stats': {}
            })
        
        # Group by model
        model_stats = {}
        for model in predictions_df['model_used'].unique():
            model_preds = predictions_df[predictions_df['model_used'] == model]
            
            model_stats[model] = {
                'count': len(model_preds),
                'avg_confidence': float(model_preds['confidence'].mean()),
                'avg_predicted_change': float(model_preds['predicted_change_pct'].mean()),
                'stocks': model_preds['stock_name'].tolist()
            }
        
        return jsonify({
            'success': True,
            'model_stats': model_stats,
            'total_predictions': len(predictions_df)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/performance-metrics', methods=['GET'])
def get_performance_metrics():
    """Get comprehensive performance metrics for all models"""
    try:
        # Get database stats
        db_stats = get_database_stats()
        
        # Get recent predictions
        predictions_df = get_predictions_all_stocks()
        
        # Get model performance from validation table (if available)
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock_cache.db'))
        
        try:
            # Check if validation results exist
            validation_query = '''
                SELECT 
                    model_used,
                    AVG(CASE WHEN ABS(predicted_change_pct - actual_change_pct) < 5 THEN 1 ELSE 0 END) * 100 as accuracy_rate,
                    AVG(ABS(predicted_change_pct - actual_change_pct)) as avg_error,
                    COUNT(*) as validation_count
                FROM predictions p
                JOIN validation_results v ON p.stock_name = v.stock_name AND p.prediction_date = v.validation_date
                GROUP BY model_used
            '''
            
            validation_df = pd.read_sql_query(validation_query, conn)
            validation_stats = validation_df.to_dict('records')
            
        except Exception:
            # Table doesn't exist or no validation data
            validation_stats = []
        finally:
            conn.close()
        
        metrics = {
            'database_stats': db_stats,
            'prediction_stats': {
                'total_predictions': len(predictions_df),
                'unique_stocks': len(predictions_df['stock_name'].unique()) if not predictions_df.empty else 0,
                'latest_prediction_date': predictions_df['prediction_date'].max() if not predictions_df.empty else None
            },
            'model_validation': validation_stats,
            'advanced_ai_available': ADVANCED_AI_AVAILABLE
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/analytics/model-performance', methods=['GET'])
def get_model_analytics():
    """Get detailed model performance analytics"""
    try:
        # Get model performance stats
        model_stats = get_model_performance_stats()
        
        # Get daily accuracy trend
        daily_trend = get_daily_accuracy_trend(30)
        
        # Get top performing stocks
        top_stocks = get_top_performing_stocks(10)
        
        return jsonify({
            'success': True,
            'model_performance': model_stats,
            'daily_trend': daily_trend,
            'top_stocks': top_stocks
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/analytics/accuracy-trends', methods=['GET'])
def get_accuracy_trends():
    """Get accuracy trends for different time periods"""
    try:
        days = request.args.get('days', 30, type=int)
        
        # Get overall trends
        trends = get_daily_accuracy_trend(days)
        
        # Get model-specific trends
        model_trends = {}
        all_models = get_model_performance_stats()
        
        for model in all_models:
            model_name = model['model_used']
            validation_results = get_validation_results(model_used=model_name, days=days)
            
            if not validation_results.empty:
                daily_stats = validation_results.groupby('validation_date').agg({
                    'accuracy_score': 'mean',
                    'direction_correct': 'mean',
                    'confidence': 'mean'
                }).reset_index()
                
                model_trends[model_name] = daily_stats.to_dict('records')
        
        return jsonify({
            'success': True,
            'overall_trends': trends,
            'model_trends': model_trends
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/analytics/stock-performance', methods=['GET'])
def get_stock_performance():
    """Get individual stock prediction performance"""
    try:
        stock_name = request.args.get('stock')
        days = request.args.get('days', 30, type=int)
        
        if stock_name:
            # Get specific stock performance
            validation_results = get_validation_results(stock_name=stock_name, days=days)
            
            if validation_results.empty:
                return jsonify({
                    'success': True,
                    'message': f'No validation data for {stock_name}',
                    'stock_performance': []
                })
            
            performance = validation_results.to_dict('records')
        else:
            # Get all stocks performance summary
            performance = get_top_performing_stocks(50)
        
        return jsonify({
            'success': True,
            'stock_performance': performance
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/optimize', methods=['POST'])
def optimize_database_endpoint():
    """Optimize database for better performance"""
    try:
        success = optimize_database()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Database optimized successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Database optimization failed'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db/analytics/predictions-vs-actuals', methods=['GET'])
def get_predictions_vs_actuals():
    """Get predictions vs actual values for scatter plot analysis"""
    try:
        days = request.args.get('days', 30, type=int)
        model_used = request.args.get('model')
        
        validation_results = get_validation_results(model_used=model_used, days=days)
        
        if validation_results.empty:
            return jsonify({
                'success': True,
                'message': 'No validation data available',
                'data': []
            })
        
        # Prepare data for scatter plot
        scatter_data = []
        for _, row in validation_results.iterrows():
            scatter_data.append({
                'stock_name': row['stock_name'],
                'predicted_price': float(row['predicted_price']),
                'actual_price': float(row['actual_price']),
                'predicted_change_pct': float(row['predicted_change_pct']),
                'actual_change_pct': float(row['actual_change_pct']),
                'accuracy_score': float(row['accuracy_score']),
                'confidence': float(row['confidence']),
                'model_used': row['model_used'],
                'validation_date': row['validation_date']
            })
        
        # Calculate correlation
        predicted_prices = validation_results['predicted_price'].values
        actual_prices = validation_results['actual_price'].values
        correlation = np.corrcoef(predicted_prices, actual_prices)[0, 1]
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(actual_prices, predicted_prices)
        
        return jsonify({
            'success': True,
            'data': scatter_data,
            'statistics': {
                'correlation': float(correlation) if not np.isnan(correlation) else 0,
                'r2_score': float(r2),
                'total_predictions': len(scatter_data),
                'mean_accuracy': float(validation_results['accuracy_score'].mean()),
                'std_accuracy': float(validation_results['accuracy_score'].std())
            }
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

