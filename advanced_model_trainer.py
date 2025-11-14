"""
Advanced Model Training System with Hyperparameter Optimization
Improves prediction accuracy through systematic model selection and tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Optional advanced libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class AdvancedModelTrainer:
    """Advanced model training with hyperparameter optimization and validation"""
    
    def __init__(self, data, target_column='Target', test_size=0.2):
        """
        Initialize advanced trainer
        
        Args:
            data: DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion for testing
        """
        self.data = data.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.feature_importance = None
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and split data for training"""
        # Remove non-feature columns
        feature_cols = [col for col in self.data.columns 
                       if col not in [self.target_column, 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = self.data[feature_cols]
        y = self.data[self.target_column]
        
        # Remove samples with missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]
        
        # Time series split
        split_idx = int(len(X) * (1 - self.test_size))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = feature_cols
        
        print(f"✓ Data prepared: {len(self.X_train)} train, {len(self.X_test)} test samples")
        print(f"✓ Features: {len(feature_cols)}")
    
    def optimize_random_forest(self, n_trials=50):
        """Optimize Random Forest hyperparameters"""
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available, using default parameters")
            model = RandomForestRegressor(random_state=42)
            model.fit(self.X_train_scaled, self.y_train)
            return model, self._evaluate_model(model)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            
            model = RandomForestRegressor(**params)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                   cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='rf_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Train best model
        best_model = RandomForestRegressor(**study.best_params)
        best_model.fit(self.X_train_scaled, self.y_train)
        
        return best_model, self._evaluate_model(best_model)
    
    def optimize_gradient_boosting(self, n_trials=50):
        """Optimize Gradient Boosting hyperparameters"""
        if not OPTUNA_AVAILABLE:
            model = GradientBoostingRegressor(random_state=42)
            model.fit(self.X_train_scaled, self.y_train)
            return model, self._evaluate_model(model)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            
            model = GradientBoostingRegressor(**params)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                   cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='gbm_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_model = GradientBoostingRegressor(**study.best_params)
        best_model.fit(self.X_train_scaled, self.y_train)
        
        return best_model, self._evaluate_model(best_model)
    
    def optimize_xgboost(self, n_trials=50):
        """Optimize XGBoost hyperparameters"""
        if not XGBOOST_AVAILABLE or not OPTUNA_AVAILABLE:
            print("⚠️ XGBoost or Optuna not available, skipping")
            return None, 0
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                   cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='xgb_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_model = xgb.XGBRegressor(**study.best_params)
        best_model.fit(self.X_train_scaled, self.y_train)
        
        return best_model, self._evaluate_model(best_model)
    
    def optimize_lightgbm(self, n_trials=50):
        """Optimize LightGBM hyperparameters"""
        if not LIGHTGBM_AVAILABLE or not OPTUNA_AVAILABLE:
            print("⚠️ LightGBM or Optuna not available, skipping")
            return None, 0
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                   cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='lgb_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_model = lgb.LGBMRegressor(**study.best_params)
        best_model.fit(self.X_train_scaled, self.y_train)
        
        return best_model, self._evaluate_model(best_model)
    
    def train_ensemble_models(self, n_trials=30):
        """Train and optimize multiple models"""
        print(f"\n{'='*70}")
        print("🚀 ADVANCED MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*70}\n")
        
        results = {}
        
        # Random Forest
        print("🌲 Optimizing Random Forest...")
        model, score = self.optimize_random_forest(n_trials)
        if model:
            results['random_forest'] = {'model': model, 'score': score}
            print(f"   ✓ Random Forest Score: {score:.4f}")
        
        # Gradient Boosting
        print("\n📈 Optimizing Gradient Boosting...")
        model, score = self.optimize_gradient_boosting(n_trials)
        if model:
            results['gradient_boosting'] = {'model': model, 'score': score}
            print(f"   ✓ Gradient Boosting Score: {score:.4f}")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            print("\n🚀 Optimizing XGBoost...")
            model, score = self.optimize_xgboost(n_trials)
            if model and score > 0:
                results['xgboost'] = {'model': model, 'score': score}
                print(f"   ✓ XGBoost Score: {score:.4f}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n💡 Optimizing LightGBM...")
            model, score = self.optimize_lightgbm(n_trials)
            if model and score > 0:
                results['lightgbm'] = {'model': model, 'score': score}
                print(f"   ✓ LightGBM Score: {score:.4f}")
        
        # Linear models (quick training)
        print("\n📊 Training Linear Models...")
        for name, model_class, params in [
            ('ridge', Ridge, {'alpha': 1.0}),
            ('lasso', Lasso, {'alpha': 0.1}),
            ('elastic_net', ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5})
        ]:
            model = model_class(**params)
            model.fit(self.X_train_scaled, self.y_train)
            score = self._evaluate_model(model)
            results[name] = {'model': model, 'score': score}
            print(f"   ✓ {name.replace('_', ' ').title()} Score: {score:.4f}")
        
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['score'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['score']
        
        print(f"\n{'='*70}")
        print("📊 TRAINING RESULTS SUMMARY")
        print(f"{'='*70}")
        
        for name, result in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
            status = "🏆" if name == best_model_name else "✅" if result['score'] > 0.7 else "⚠️"
            print(f"   {status} {name.replace('_', ' ').title():<20} {result['score']:.4f}")
        
        print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()} (Score: {self.best_score:.4f})")
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        print(f"{'='*70}\n")
        
        return results
    
    def _evaluate_model(self, model):
        """Evaluate model performance"""
        try:
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate accuracy as 100 - MAPE
            mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            accuracy = max(0, 100 - mape)
            
            return accuracy / 100  # Return as decimal for optimization
        except Exception as e:
            print(f"   ❌ Evaluation error: {e}")
            return 0
    
    def _calculate_feature_importance(self):
        """Calculate and display feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            
            print("\n🎯 TOP 15 MOST IMPORTANT FEATURES:")
            print("-" * 50)
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        elif hasattr(self.best_model, 'coef_'):
            # For linear models
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.best_model.coef_)
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            
            print("\n🎯 TOP 15 MOST IMPORTANT FEATURES (Linear Model):")
            print("-" * 50)
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Run train_ensemble_models() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_model_report(self):
        """Generate comprehensive model performance report"""
        if not self.models:
            return None
        
        report = {
            'best_model': {
                'name': max(self.models.keys(), key=lambda x: self.models[x]['score']),
                'score': self.best_score,
                'accuracy_pct': self.best_score * 100
            },
            'all_models': {},
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'training_data': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features': len(self.feature_names),
                'test_size': self.test_size
            }
        }
        
        for name, result in self.models.items():
            report['all_models'][name] = {
                'score': result['score'],
                'accuracy_pct': result['score'] * 100
            }
        
        return report
    
    def save_best_model(self, filepath):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        
        import joblib
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'score': self.best_score
        }, filepath)
        
        print(f"✅ Best model saved to {filepath}")

def train_advanced_stock_model(stock_file, stock_name):
    """Train advanced model for a specific stock"""
    print(f"\n{'='*70}")
    print(f"ADVANCED MODEL TRAINING: {stock_name}")
    print(f"{'='*70}\n")
    
    try:
        # Load and prepare data
        from advanced_ai_predictor import AdvancedStockPredictor
        
        # Load stock data
        if stock_file.endswith('.csv'):
            data = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
        else:
            data = stock_file  # Assume it's already a DataFrame
        
        # Create advanced predictor to get engineered features
        predictor = AdvancedStockPredictor(data)
        engineered_data = predictor.engineer_comprehensive_features()
        
        # Initialize advanced trainer
        trainer = AdvancedModelTrainer(engineered_data)
        
        # Train ensemble models
        results = trainer.train_ensemble_models()
        
        # Generate report
        report = trainer.get_model_report()
        
        # Save model
        import os
        model_dir = f"advanced_models/{stock_name}"
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_best_model(f"{model_dir}/best_model.pkl")
        
        return trainer, report, results
        
    except Exception as e:
        print(f"❌ Advanced training failed for {stock_name}: {e}")
        return None, None, None

if __name__ == "__main__":
    # Example usage
    import os
    
    stock_file = "EOD/RELIANCE.csv"
    if os.path.exists(stock_file):
        trainer, report, results = train_advanced_stock_model(stock_file, "RELIANCE")
        if trainer:
            print("\n✅ Advanced model training completed successfully!")
            if report:
                print(f"🏆 Best Model: {report['best_model']['name']} ({report['best_model']['accuracy_pct']:.2f}% accuracy)")
    else:
        print("Please ensure stock data is available in EOD/ directory")
