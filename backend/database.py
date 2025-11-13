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
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized: {DB_FILE}")

def save_stock_to_db(stock_name, df):
    """Save stock dataframe to database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Prepare data
        df_copy = df.copy()
        df_copy['stock_name'] = stock_name
        df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.strftime('%Y-%m-%d')
        
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
        
        # Insert or replace data
        df_save.to_sql('stock_data', conn, if_exists='append', index=False)
        
        # Update metadata
        last_row = df.iloc[-1]
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sync_metadata 
            (stock_name, last_sync, record_count, last_date, last_close)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            stock_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            len(df),
            last_row['Date'].strftime('%Y-%m-%d') if isinstance(last_row['Date'], pd.Timestamp) else str(last_row['Date']),
            float(last_row['Close'])
        ))
        
        conn.commit()
        return True, len(df)
        
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

def get_latest_prices_all_stocks():
    """Get latest price data for all stocks with 7-day performance - OPTIMIZED!"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Single optimized query for all data
        query = '''
            WITH latest_dates AS (
                SELECT stock_name, MAX(date) as max_date
                FROM stock_data
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
            ORDER BY a.stock_name
        '''
        
        df = pd.read_sql_query(query, conn)
        
        # Convert all numeric columns to float (handle None values)
        numeric_cols = ['today_open', 'today_close', 'today_high', 'today_low', 'volume',
                       'day1_close', 'day2_close', 'day3_close', 'day4_close', 'day5_close',
                       'day6_close', 'day7_close', 'day14_close', 'day21_close', 'day31_close',
                       'week_52_high', 'week_52_low', 'avg_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate all metrics with safe division
        df['today_change'] = df['today_close'] - df['day1_close']
        df['today_pct'] = ((df['today_close'] - df['day1_close']) / df['day1_close'].replace(0, np.nan) * 100).round(2)
        df['yesterday_pct'] = ((df['day1_close'] - df['day2_close']) / df['day2_close'].replace(0, np.nan) * 100).round(2)
        
        # Individual day percentage changes
        df['day2_pct'] = ((df['day2_close'] - df['day3_close']) / df['day3_close'].replace(0, np.nan) * 100).round(2)
        df['day3_pct'] = ((df['day3_close'] - df['day4_close']) / df['day4_close'].replace(0, np.nan) * 100).round(2)
        df['day4_pct'] = ((df['day4_close'] - df['day5_close']) / df['day5_close'].replace(0, np.nan) * 100).round(2)
        df['day5_pct'] = ((df['day5_close'] - df['day6_close']) / df['day6_close'].replace(0, np.nan) * 100).round(2)
        df['day6_pct'] = ((df['day6_close'] - df['day7_close']) / df['day7_close'].replace(0, np.nan) * 100).round(2)
        
        # Period percentage changes
        df['week_7day_pct'] = ((df['today_close'] - df['day7_close']) / df['day7_close'].replace(0, np.nan) * 100).round(2)
        df['week_7to14_pct'] = ((df['day7_close'] - df['day14_close']) / df['day14_close'].replace(0, np.nan) * 100).round(2)
        df['week_14to21_pct'] = ((df['day14_close'] - df['day21_close']) / df['day21_close'].replace(0, np.nan) * 100).round(2)
        df['week_21to31_pct'] = ((df['day21_close'] - df['day31_close']) / df['day31_close'].replace(0, np.nan) * 100).round(2)
        
        # Fill NaN values with 0 for percentage columns
        pct_cols = ['today_pct', 'yesterday_pct', 'day2_pct', 'day3_pct', 'day4_pct', 'day5_pct', 'day6_pct',
                    'week_7day_pct', 'week_7to14_pct', 'week_14to21_pct', 'week_21to31_pct']
        df[pct_cols] = df[pct_cols].fillna(0)
        
        # Intraday change
        df['intraday_change'] = ((df['today_close'] - df['today_open']) / df['today_open'] * 100).round(2)
        
        # Volume ratio
        df['volume_ratio'] = (df['volume'] / df['avg_volume']).round(2)
        
        # 52-week position
        df['week_52_position'] = ((df['today_close'] - df['week_52_low']) / 
                                  (df['week_52_high'] - df['week_52_low']) * 100).round(1)
        
        return df
        
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

if __name__ == '__main__':
    # Initialize database
    init_database()
    print("Database ready!")

