"""
Database module for caching stock data in SQLite
This dramatically speeds up data loading and filtering
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock_cache.db')

def init_database():
    """Initialize the SQLite database with proper schema"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create stocks table with all necessary columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_name TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_name, date)
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stock_date 
        ON stock_data(stock_name, date DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_date 
        ON stock_data(date DESC)
    ''')
    
    # Create metadata table to track last update
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sync_metadata (
            stock_name TEXT PRIMARY KEY,
            last_sync TIMESTAMP,
            record_count INTEGER,
            last_date DATE,
            last_close REAL
        )
    ''')
    
    # Create predictions table for caching AI predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_name TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            current_price REAL,
            predicted_price REAL,
            predicted_change_pct REAL,
            confidence REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_name, prediction_date)
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_predictions_stock 
        ON predictions(stock_name, prediction_date DESC)
    ''')
    
    # Create validation results table for tracking accuracy
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_name TEXT NOT NULL,
            validation_date DATE NOT NULL,
            predicted_price REAL,
            actual_price REAL,
            predicted_change_pct REAL,
            actual_change_pct REAL,
            price_error REAL,
            price_error_pct REAL,
            direction_correct BOOLEAN,
            confidence REAL,
            model_used TEXT,
            accuracy_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_name, validation_date)
        )
    ''')
    
    # Create additional indexes for performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_validation_stock_date 
        ON validation_results(stock_name, validation_date DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_validation_model 
        ON validation_results(model_used, validation_date DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stock_close_date 
        ON stock_data(stock_name, close DESC, date DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_predictions_confidence 
        ON predictions(confidence DESC, prediction_date DESC)
    ''')
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized with optimized indexes: {DB_FILE}")

def save_stock_to_db(stock_name, df):
    """Save stock dataframe to database with improved error handling and deduplication"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Prepare data
        df_copy = df.copy()
        df_copy['stock_name'] = stock_name
        
        # Handle different date formats
        if 'Date' in df_copy.columns:
            df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.strftime('%Y-%m-%d')
        else:
            # If Date is index, reset it
            df_copy = df_copy.reset_index()
        df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.strftime('%Y-%m-%d')
        
        # Ensure all required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df_copy.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Select only needed columns
        columns_map = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'stock_name': 'stock_name'
        }
        
        df_save = df_copy[list(columns_map.keys())].rename(columns=columns_map)
        
        # Remove duplicates based on stock_name and date
        df_save = df_save.drop_duplicates(subset=['stock_name', 'date'], keep='last')
        
        # Delete existing data for this stock to avoid duplicates
        cursor = conn.cursor()
        cursor.execute('DELETE FROM stock_data WHERE stock_name = ?', (stock_name,))
        
        # Insert new data
        df_save.to_sql('stock_data', conn, if_exists='append', index=False)
        
        # Update metadata
        last_row = df_copy.iloc[-1]
        cursor.execute('''
            INSERT OR REPLACE INTO sync_metadata 
            (stock_name, last_sync, record_count, last_date, last_close)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            stock_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            len(df_save),
            last_row['Date'],
            float(last_row['Close'])
        ))
        
        conn.commit()
        return True, len(df_save)
        
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def get_stock_from_db(stock_name, limit=None):
    """Get stock data from database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = '''
            SELECT date, open, high, low, close, volume
            FROM stock_data
            WHERE stock_name = ?
            ORDER BY date ASC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        df = pd.read_sql_query(query, conn, params=(stock_name,))
        df['date'] = pd.to_datetime(df['date'])
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    finally:
        conn.close()

def get_latest_prices_all_stocks(page=None, per_page=None, sort_by='stock_name', sort_order='asc', search=None):
    """Get latest price data for all stocks with 7-day performance - OPTIMIZED with PAGINATION!
    
    Args:
        page: Page number (1-indexed), None for all results
        per_page: Results per page, None for all results
        sort_by: Column to sort by (stock_name, today_close, today_pct, week_7day_pct, etc)
        sort_order: 'asc' or 'desc'
        search: Search term for stock name filter
    
    Returns:
        tuple: (dataframe, total_count)
    """
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Single optimized query for all data
        query = '''
            WITH latest_dates AS (
                SELECT stock_name, MAX(date) as max_date
                FROM stock_data
                {search_clause}
                GROUP BY stock_name
            ),
            ranked_data AS (
                SELECT 
                    sd.stock_name,
                    sd.date,
                    sd.open,
                    sd.close,
                    sd.high,
                    sd.low,
                    sd.volume,
                    ROW_NUMBER() OVER (PARTITION BY sd.stock_name ORDER BY sd.date DESC) as rn
                FROM stock_data sd
                INNER JOIN latest_dates ld ON sd.stock_name = ld.stock_name
                WHERE sd.date >= date(ld.max_date, '-60 days')
            ),
            aggregated AS (
                SELECT 
                    stock_name,
                    MAX(CASE WHEN rn = 1 THEN date END) as today_date,
                    MAX(CASE WHEN rn = 1 THEN open END) as today_open,
                    MAX(CASE WHEN rn = 1 THEN close END) as today_close,
                    MAX(CASE WHEN rn = 1 THEN high END) as today_high,
                    MAX(CASE WHEN rn = 1 THEN low END) as today_low,
                    MAX(CASE WHEN rn = 1 THEN volume END) as volume,
                    MAX(CASE WHEN rn = 2 THEN close END) as day1_close,
                    MAX(CASE WHEN rn = 3 THEN close END) as day2_close,
                    MAX(CASE WHEN rn = 4 THEN close END) as day3_close,
                    MAX(CASE WHEN rn = 5 THEN close END) as day4_close,
                    MAX(CASE WHEN rn = 6 THEN close END) as day5_close,
                    MAX(CASE WHEN rn = 7 THEN close END) as day6_close,
                    MAX(CASE WHEN rn = 8 THEN close END) as day7_close,
                    MAX(CASE WHEN rn = 15 THEN close END) as day14_close,
                    MAX(CASE WHEN rn = 22 THEN close END) as day21_close,
                    MAX(CASE WHEN rn = 32 THEN close END) as day31_close
                FROM ranked_data
                WHERE rn <= 32
                GROUP BY stock_name
            ),
            stats_52w AS (
                SELECT 
                    stock_name,
                    MAX(high) as week_52_high,
                    MIN(low) as week_52_low,
                    AVG(volume) as avg_volume
                FROM stock_data
                WHERE date >= date('now', '-365 days')
                GROUP BY stock_name
            )
            SELECT 
                a.*,
                s.week_52_high,
                s.week_52_low,
                s.avg_volume
            FROM aggregated a
            LEFT JOIN stats_52w s ON a.stock_name = s.stock_name
        '''
        
        # Add search clause
        search_clause = ""
        if search:
            search_clause = f"WHERE stock_name LIKE '%{search}%'"
        
        query = query.format(search_clause=search_clause)
        
        # Get total count before pagination
        df_all = pd.read_sql_query(query, conn)
        
        # Convert all numeric columns to float (handle None values)
        numeric_cols = ['today_open', 'today_close', 'today_high', 'today_low', 'volume',
                       'day1_close', 'day2_close', 'day3_close', 'day4_close', 'day5_close',
                       'day6_close', 'day7_close', 'day14_close', 'day21_close', 'day31_close',
                       'week_52_high', 'week_52_low', 'avg_volume']
        for col in numeric_cols:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        
        # Calculate all metrics with safe division
        df_all['today_change'] = df_all['today_close'] - df_all['day1_close']
        df_all['today_pct'] = ((df_all['today_close'] - df_all['day1_close']) / df_all['day1_close'].replace(0, np.nan) * 100).round(2)
        df_all['yesterday_pct'] = ((df_all['day1_close'] - df_all['day2_close']) / df_all['day2_close'].replace(0, np.nan) * 100).round(2)
        
        # Individual day percentage changes
        df_all['day2_pct'] = ((df_all['day2_close'] - df_all['day3_close']) / df_all['day3_close'].replace(0, np.nan) * 100).round(2)
        df_all['day3_pct'] = ((df_all['day3_close'] - df_all['day4_close']) / df_all['day4_close'].replace(0, np.nan) * 100).round(2)
        df_all['day4_pct'] = ((df_all['day4_close'] - df_all['day5_close']) / df_all['day5_close'].replace(0, np.nan) * 100).round(2)
        df_all['day5_pct'] = ((df_all['day5_close'] - df_all['day6_close']) / df_all['day6_close'].replace(0, np.nan) * 100).round(2)
        df_all['day6_pct'] = ((df_all['day6_close'] - df_all['day7_close']) / df_all['day7_close'].replace(0, np.nan) * 100).round(2)
        
        # Period percentage changes
        df_all['week_7day_pct'] = ((df_all['today_close'] - df_all['day7_close']) / df_all['day7_close'].replace(0, np.nan) * 100).round(2)
        df_all['week_7to14_pct'] = ((df_all['day7_close'] - df_all['day14_close']) / df_all['day14_close'].replace(0, np.nan) * 100).round(2)
        df_all['week_14to21_pct'] = ((df_all['day14_close'] - df_all['day21_close']) / df_all['day21_close'].replace(0, np.nan) * 100).round(2)
        df_all['week_21to31_pct'] = ((df_all['day21_close'] - df_all['day31_close']) / df_all['day31_close'].replace(0, np.nan) * 100).round(2)
        
        # Fill NaN values with 0 for percentage columns
        pct_cols = ['today_pct', 'yesterday_pct', 'day2_pct', 'day3_pct', 'day4_pct', 'day5_pct', 'day6_pct',
                    'week_7day_pct', 'week_7to14_pct', 'week_14to21_pct', 'week_21to31_pct']
        df_all[pct_cols] = df_all[pct_cols].fillna(0)
        
        # Intraday change
        df_all['intraday_change'] = ((df_all['today_close'] - df_all['today_open']) / df_all['today_open'] * 100).round(2)
        
        # Volume ratio
        df_all['volume_ratio'] = (df_all['volume'] / df_all['avg_volume']).round(2)
        
        # 52-week position
        df_all['week_52_position'] = ((df_all['today_close'] - df_all['week_52_low']) / 
                                  (df_all['week_52_high'] - df_all['week_52_low']) * 100).round(1)
        
        total_count = len(df_all)
        
        # Apply sorting if specified
        if sort_by in df_all.columns:
            ascending = (sort_order.lower() == 'asc')
            df_all = df_all.sort_values(by=sort_by, ascending=ascending, na_position='last')
        
        # Apply pagination if specified
        if page is not None and per_page is not None:
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df = df_all.iloc[start_idx:end_idx]
        else:
            df = df_all
        
        return df, total_count
        
    finally:
        conn.close()

def get_sync_metadata():
    """Get synchronization metadata for all stocks"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = '''
            SELECT 
                stock_name,
                last_sync,
                record_count,
                last_date,
                last_close
            FROM sync_metadata
            ORDER BY stock_name
        '''
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def get_stocks_list():
    """Get list of all stocks in database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = 'SELECT DISTINCT stock_name FROM stock_data ORDER BY stock_name'
        df = pd.read_sql_query(query, conn)
        return df['stock_name'].tolist()
    finally:
        conn.close()

def clear_old_data(days_to_keep=730):
    """Clear old data to keep database size manageable (default: 2 years)"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM stock_data WHERE date < ?', (cutoff_date,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()

def get_database_stats():
    """Get database statistics"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        cursor = conn.cursor()
        
        # Total records
        cursor.execute('SELECT COUNT(*) FROM stock_data')
        total_records = cursor.fetchone()[0]
        
        # Total stocks
        cursor.execute('SELECT COUNT(DISTINCT stock_name) FROM stock_data')
        total_stocks = cursor.fetchone()[0]
        
        # Date range
        cursor.execute('SELECT MIN(date), MAX(date) FROM stock_data')
        min_date, max_date = cursor.fetchone()
        
        # Database file size
        db_size = os.path.getsize(DB_FILE) if os.path.exists(DB_FILE) else 0
        
        return {
            'total_records': total_records,
            'total_stocks': total_stocks,
            'date_range': f"{min_date} to {max_date}",
            'db_size_mb': round(db_size / (1024 * 1024), 2),
            'db_file': DB_FILE
        }
    finally:
        conn.close()

def save_prediction(stock_name, prediction_date, current_price, predicted_price, 
                   predicted_change_pct, confidence, model_used='Ensemble'):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO predictions 
            (stock_name, prediction_date, current_price, predicted_price, 
             predicted_change_pct, confidence, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (stock_name, prediction_date, current_price, predicted_price,
              predicted_change_pct, confidence, model_used))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        conn.close()

def get_predictions_all_stocks(prediction_date=None):
    """Get predictions for all stocks"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        if prediction_date is None:
            # Get latest predictions for each stock
            query = '''
                SELECT 
                    stock_name,
                    prediction_date,
                    current_price,
                    predicted_price,
                    predicted_change_pct,
                    confidence,
                    model_used,
                    created_at
                FROM predictions
                WHERE (stock_name, prediction_date) IN (
                    SELECT stock_name, MAX(prediction_date)
                    FROM predictions
                    GROUP BY stock_name
                )
                ORDER BY stock_name
            '''
            df = pd.read_sql_query(query, conn)
        else:
            query = '''
                SELECT 
                    stock_name,
                    prediction_date,
                    current_price,
                    predicted_price,
                    predicted_change_pct,
                    confidence,
                    model_used,
                    created_at
                FROM predictions
                WHERE prediction_date = ?
                ORDER BY stock_name
            '''
            df = pd.read_sql_query(query, conn, params=(prediction_date,))
        
        return df
    finally:
        conn.close()

def get_prediction_for_stock(stock_name):
    """Get latest prediction for a specific stock"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = '''
            SELECT 
                prediction_date,
                current_price,
                predicted_price,
                predicted_change_pct,
                confidence,
                model_used,
                created_at
            FROM predictions
            WHERE stock_name = ?
            ORDER BY prediction_date DESC
            LIMIT 1
        '''
        
        df = pd.read_sql_query(query, conn, params=(stock_name,))
        return df.to_dict('records')[0] if not df.empty else None
    finally:
        conn.close()

def clear_old_predictions(days_to_keep=7):
    """Clear old predictions"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions WHERE prediction_date < ?', (cutoff_date,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()

def save_validation_result(stock_name, validation_date, predicted_price, actual_price, 
                          predicted_change_pct, actual_change_pct, confidence, model_used):
    """Save validation result to database for accuracy tracking"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        cursor = conn.cursor()
        
        # Calculate metrics
        price_error = abs(predicted_price - actual_price)
        price_error_pct = (price_error / actual_price) * 100
        direction_correct = (predicted_change_pct > 0) == (actual_change_pct > 0)
        accuracy_score = max(0, 100 - price_error_pct)
        
        cursor.execute('''
            INSERT OR REPLACE INTO validation_results 
            (stock_name, validation_date, predicted_price, actual_price, 
             predicted_change_pct, actual_change_pct, price_error, price_error_pct,
             direction_correct, confidence, model_used, accuracy_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (stock_name, validation_date, predicted_price, actual_price,
              predicted_change_pct, actual_change_pct, price_error, price_error_pct,
              direction_correct, confidence, model_used, accuracy_score))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        conn.close()

def get_validation_results(stock_name=None, model_used=None, days=30):
    """Get validation results with optional filtering"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = '''
            SELECT * FROM validation_results 
            WHERE validation_date >= date('now', '-{} days')
        '''.format(days)
        
        params = []
        
        if stock_name:
            query += ' AND stock_name = ?'
            params.append(stock_name)
        
        if model_used:
            query += ' AND model_used = ?'
            params.append(model_used)
        
        query += ' ORDER BY validation_date DESC'
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

def get_model_performance_stats():
    """Get comprehensive model performance statistics"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = '''
            SELECT 
                model_used,
                COUNT(*) as total_predictions,
                AVG(accuracy_score) as avg_accuracy,
                AVG(CASE WHEN direction_correct = 1 THEN 100 ELSE 0 END) as direction_accuracy,
                AVG(price_error_pct) as avg_error_pct,
                AVG(confidence) as avg_confidence,
                MIN(validation_date) as first_prediction,
                MAX(validation_date) as last_prediction
            FROM validation_results
            WHERE validation_date >= date('now', '-30 days')
            GROUP BY model_used
            ORDER BY avg_accuracy DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records')
    finally:
        conn.close()

def get_top_performing_stocks(limit=10, metric='accuracy_score'):
    """Get top performing stocks by prediction accuracy"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = f'''
            SELECT 
                stock_name,
                COUNT(*) as prediction_count,
                AVG({metric}) as avg_metric,
                AVG(CASE WHEN direction_correct = 1 THEN 100 ELSE 0 END) as direction_accuracy,
                AVG(confidence) as avg_confidence,
                MAX(validation_date) as last_validation
            FROM validation_results
            WHERE validation_date >= date('now', '-30 days')
            GROUP BY stock_name
            HAVING prediction_count >= 3
            ORDER BY avg_metric DESC
            LIMIT {limit}
        '''
        
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records')
    finally:
        conn.close()

def get_daily_accuracy_trend(days=30):
    """Get daily accuracy trend for performance tracking"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = '''
            SELECT 
                validation_date,
                COUNT(*) as prediction_count,
                AVG(accuracy_score) as avg_accuracy,
                AVG(CASE WHEN direction_correct = 1 THEN 100 ELSE 0 END) as direction_accuracy,
                AVG(confidence) as avg_confidence
            FROM validation_results
            WHERE validation_date >= date('now', '-{} days')
            GROUP BY validation_date
            ORDER BY validation_date ASC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records')
    finally:
        conn.close()

def sync_pending_data_to_db(stock_name, df):
    """Sync only NEW/PENDING data for a stock (incremental update)"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Get the latest date in database for this stock
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(date) FROM stock_data WHERE stock_name = ?
        ''', (stock_name,))
        
        result = cursor.fetchone()
        last_date = result[0] if result[0] else None
        
        if last_date:
            # Convert last_date to datetime for comparison
            last_date = pd.to_datetime(last_date)
            
            # Filter dataframe to only include dates AFTER last_date
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            df_pending = df_copy[df_copy['Date'] > last_date]
            
            if df_pending.empty:
                return True, 0  # No new data to sync
            
            # Prepare pending data for insertion
            df_pending['stock_name'] = stock_name
            df_pending['Date'] = df_pending['Date'].dt.strftime('%Y-%m-%d')
            
            # Ensure all required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            columns_map = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'stock_name': 'stock_name'
            }
            
            df_save = df_pending[list(columns_map.keys())].rename(columns=columns_map)
            
            # Insert new data
            df_save.to_sql('stock_data', conn, if_exists='append', index=False)
            
            # Update metadata
            last_row = df_pending.iloc[-1]
            cursor.execute('''
                UPDATE sync_metadata 
                SET last_sync = ?,
                    record_count = record_count + ?,
                    last_date = ?,
                    last_close = ?
                WHERE stock_name = ?
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(df_save),
                last_row['Date'],
                float(last_row['Close']),
                stock_name
            ))
            
            conn.commit()
            return True, len(df_save)
        else:
            # No existing data, do full sync
            return save_stock_to_db(stock_name, df)
        
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def optimize_database():
    """Run database optimization commands"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        cursor = conn.cursor()
        
        # Analyze tables for better query planning
        cursor.execute('ANALYZE')
        
        # Vacuum to defragment and reclaim space
        cursor.execute('VACUUM')
        
        # Update statistics
        cursor.execute('PRAGMA optimize')
        
        conn.commit()
        print("✓ Database optimized successfully")
        return True
    except Exception as e:
        print(f"❌ Database optimization failed: {e}")
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    # Initialize database
    init_database()
    print("Database ready!")

