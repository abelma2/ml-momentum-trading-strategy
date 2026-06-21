import os
# Pin BLAS/OpenMP thread pools to 1 BEFORE importing numpy/sklearn/xgboost.
# With ProcessPoolExecutor running N parallel windows and each one fitting models
# whose underlying BLAS calls would otherwise spawn cpu_count threads each, we'd get
# N * cpu_count threads fighting over cpu_count cores. On a many-core machine this would
# go from useful parallelism to thread thrash. Single-threaded BLAS per worker keeps
# scaling linear.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
# Make stdout/stderr UTF-8 so status glyphs (checkmarks, etc.) print on any console.
# Windows consoles default to cp1252, which can't encode them and crash on print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import numpy as np
import sys
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Placeholder for actual performance metric
import xgboost as xgb
import os
import concurrent.futures

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
np.seterr(divide='ignore', invalid='ignore')
# Note: tensorflow/keras is commented out as it requires a separate, complex installation.
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import LSTM, Dense 

# --- MODULE 1: INDICATORS/FEATURES (From Paper Section 2.2) ---

class FeatureEngineer:
    """
    Generates all predictive features as described in the paper's
    Section 2.2: Feature Engineering Framework.
    """
    def __init__(self, lookbacks=None, target_period=21):
        self.lookbacks = lookbacks if lookbacks is not None else [5, 10, 20, 60, 120]
        self.target_period = target_period
        self.scaler = StandardScaler()
        self.constant_features_mask = None

    def add_price_features(self, data):
        """
        Adds price-based features from paper Section 2.2 using a robust
        unstack/stack method to prevent pandas alignment errors.
        """
        
        # Unstack data to have assets as columns, dates as index
        close_unstacked = data['Close'].unstack()
        
        # Create a container for all new features
        features_df = pd.DataFrame(index=data.index)

        for lookback in self.lookbacks:
            # 1. Returns
            return_unstacked = close_unstacked.pct_change(lookback)
            features_df[f'return_{lookback}d'] = return_unstacked.stack()

            # 2. Volatility
            vol_unstacked = close_unstacked.pct_change().rolling(window=lookback, min_periods=lookback).std()
            features_df[f'vol_{lookback}d'] = vol_unstacked.stack()

            # 3. Moving Average
            ma_unstacked = close_unstacked.rolling(window=lookback, min_periods=lookback).mean()
            features_df[f'ma_{lookback}d'] = ma_unstacked.stack()

        # Join the new features back to the main dataframe
        data = data.join(features_df)

        # Now, calculate derived features using the newly joined columns
        for lookback in self.lookbacks:
            # 4. Risk-adjusted return
            data[f'risk_adj_return_{lookback}d'] = data[f'return_{lookback}d'] / data[f'vol_{lookback}d']
            
            # 5. Distance from moving average
            data[f'dist_from_ma_{lookback}d'] = (data['Close'] - data[f'ma_{lookback}d']) / data[f'ma_{lookback}d']

        # Clean up NaNs created by rolling windows
        longest_lookback = max(self.lookbacks) if self.lookbacks else 0
        
        data = data.dropna(subset=[
            f'return_{longest_lookback}d',
            f'risk_adj_return_{longest_lookback}d',
            f'dist_from_ma_{longest_lookback}d'
        ])
        
        return data

    def add_volume_features(self, data):
        """Placeholder for volume/liquidity features from paper Section 2.2"""
        return data

    def add_cross_sectional_features(self, data):
        """Placeholder for cross-sectional features from paper Section 2.2"""
        return data

    def create_target_variable(self, data):
        """
        Creates the target variable (y) for the ML models.
        """
        grouped = data.groupby(level='asset')
        future_return = grouped['Close'].pct_change(self.target_period).shift(-self.target_period)
        data['target'] = (future_return > 0).astype(int)
        data = data.dropna(subset=['target'])
        return data

    def get_feature_names(self, data):
        """Helper to get all feature column names."""
        return [col for col in data.columns if 'return_' in col or 'vol_' in col or 'dist_' in col]

    def build_features_and_target(self, data):
        """
        Main pipeline method to build all features and target.
        """
        data = data.copy()
        data = self.add_price_features(data)
        data = self.add_volume_features(data)
        data = self.add_cross_sectional_features(data)
        data = self.create_target_variable(data)

        feature_cols = self.get_feature_names(data)
        
        if not feature_cols:
            return pd.DataFrame(), pd.Series(dtype='float64'), data
            
        # Use .copy() to avoid SettingWithCopyWarning
        X = data[feature_cols].copy() 
        y = data['target'].copy()

        # Scale features
        if X.empty:
            return X, y, data
            
        # More aggressive cleaning to prevent NaN/inf propagation
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN with 0, but also check for columns that are all NaN
        col_nan_counts = X.isna().sum()
        mostly_nan_cols = col_nan_counts[col_nan_counts > len(X) * 0.9].index
        if len(mostly_nan_cols) > 0:
            X.drop(columns=mostly_nan_cols, inplace=True)
        
        X.fillna(0, inplace=True)
        
        # Ensure no NaN values remain
        assert not X.isna().any().any(), "NaN values still present after cleaning"

        return X, y, data
        
    def fit_scaler(self, X_train):
        """Fits the scaler on the training data."""
        try:
            # Ensure no NaN/inf before computing statistics
            X_temp = X_train.copy()
            X_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_temp.fillna(0, inplace=True)
            
            # Remove constant features (zero variance) before fitting
            with np.errstate(divide='ignore', invalid='ignore'):
                feature_std = X_temp.std()
            
            # Filter out NaN stds and near-zero variance
            valid_std = feature_std[~feature_std.isna() & (feature_std > 1e-10)]
            non_constant_features = valid_std.index
            
            if len(non_constant_features) == 0:
                print("Warning: All features are constant. Proceeding unscaled.")
                self.scaler = None
                self.constant_features_mask = None
                return
            
            self.constant_features_mask = X_train.columns.isin(non_constant_features)
            X_to_fit = X_temp.loc[:, self.constant_features_mask]
            
            # Final check for NaN
            if X_to_fit.isna().any().any():
                X_to_fit = X_to_fit.fillna(0)
            
            self.scaler.fit(X_to_fit)
        except (ValueError, RuntimeWarning) as e:
            print(f"Warning: Error fitting scaler: {e}. Proceeding unscaled.")
            self.scaler = None
            self.constant_features_mask = None
            
    def scale_features(self, X):
        """Scales features using the already-fitted scaler."""
        if X.empty:
            return X
        
        X = X.copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        
        # Ensure columns match what was fitted
        if self.scaler and hasattr(self.scaler, 'mean_') and hasattr(self, 'constant_features_mask'):
            if self.constant_features_mask is not None:
                # Keep only columns that were present during fitting
                available_cols = X.columns[self.constant_features_mask]
                X_to_scale = X[available_cols]
                
                # Verify no NaN before scaling
                if X_to_scale.isna().any().any():
                    X_to_scale = X_to_scale.fillna(0)
                
                X_scaled = X.copy()
                X_scaled[available_cols] = self.scaler.transform(X_to_scale)
                return X_scaled
            else:
                return X
        else:
            return X


# --- MODULE 2: MODEL COMPETITION (From Paper Section 3.1) ---

class ModelCompetitor:
    """
    Trains and manages the suite of predictive models from
    paper Section 3.1: Base Model Specifications.
    """
    def __init__(self, n_workers: int = None):
        """
        n_workers: number of worker threads to use for parallel training/prediction.
        If None, defaults to min(4, os.cpu_count() or 1).
        We use ThreadPoolExecutor since many ML libraries release the GIL during heavy
        computation; this avoids adding extra dependencies and pickling issues.
        """
        # Use L2-regularized LogisticRegression (the paper's "Ridge classifier") instead of
        # sklearn.linear_model.Ridge (which is regression and doesn't expose predict_proba).
        # C=2.0 corresponds to alpha=0.5 (since C = 1/alpha in sklearn).
        # Set inner-model n_jobs=1 to avoid oversubscription with the outer
        # ProcessPoolExecutor (which already runs 4 windows in parallel). Previously
        # n_jobs=-1 made each of the 4 subprocesses spawn N-core thread pools, leading
        # to thread storm and ~9% CPU efficiency.
        self.models = {
            'Ridge': LogisticRegression(C=2.0, max_iter=3000, solver='lbfgs', random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1),
            'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        }
        self.trained_models = {}
        if n_workers is None:
            try:
                cpu = os.cpu_count() or 1
            except Exception:
                cpu = 1
            self.n_workers = min(4, cpu)
        else:
            self.n_workers = max(1, int(n_workers))

    def _fit_single(self, name, model, X, y):
        """Helper to fit a single model; used with executors."""
        print(f"Training {name}...")
        try:
            # XGBoost accepts verbose arg; sklearn does not
            if name == 'XGBoost':
                model.fit(X, y, verbose=False)
            else:
                model.fit(X, y)
            return name, model, None
        except Exception as e:
            return name, None, e

    def train_models(self, X_train, y_train):
        """Trains all models on the provided training data in parallel."""
        self.trained_models = {}
        # If there are no models or empty training set, do nothing
        if not self.models or X_train is None or X_train.empty:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {
                ex.submit(self._fit_single, name, model, X_train, y_train): name
                for name, model in self.models.items()
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    model_name, fitted_model, err = fut.result()
                    if err is not None:
                        print(f"Failed to train {model_name}: {err}")
                    elif fitted_model is not None:
                        self.trained_models[model_name] = fitted_model
                except Exception as e:
                    print(f"Unexpected error training {name}: {e}")

    def _predict_single(self, name, model, X):
        """Helper to get predictions from a single model."""
        try:
            if model is None:
                return name, np.full(X.shape[0], np.nan)

            if hasattr(model, 'predict_proba'):
                # Handle single-class models
                try:
                    classes = getattr(model, 'classes_', None)
                    if classes is None or len(classes) < 2:
                        print(f"Warning: {name} was trained on a single class. Predicting 0.5 probability.")
                        return name, np.full(X.shape[0], 0.5)
                    probs = model.predict_proba(X)
                    return name, probs[:, 1]
                except Exception as e:
                    # Fallback to predict if predict_proba fails
                    pred_values = model.predict(X)
                    return name, np.clip(pred_values, 0, 1)
            else:
                pred_values = model.predict(X)
                return name, np.clip(pred_values, 0, 1)
        except Exception as e:
            print(f"Failed to predict with {name}: {e}")
            return name, np.full(X.shape[0], np.nan)

    def get_predictions(self, X_test):
        """
        Generates predictions (signals) from all trained models in parallel.
        Returns a DataFrame of signals.
        """
        predictions = {}
        if X_test is None or X_test.empty:
            return pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {
                ex.submit(self._predict_single, name, model, X_test): name
                for name, model in self.trained_models.items()
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    model_name, preds = fut.result()
                    predictions[model_name] = preds
                except Exception as e:
                    name = futures.get(fut, '<unknown>')
                    print(f"Prediction future failed for {name}: {e}")
                    predictions[name] = np.full(X_test.shape[0], np.nan)

        return pd.DataFrame(predictions, index=X_test.index)


# --- MODULE 3: PORTFOLIO CONSTRUCTION (From Paper Chapter 4) ---

class PortfolioConstructor:
    """
    Constructs the final portfolio based on model signals
    and confidence, as described in paper Chapter 4.
    """
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

    def calculate_model_weights(self, model_performance):
        """
        Stage 1: Model Confidence Weighting (Paper Section 4.1)
        """
        avg_performance = model_performance.mean()
        avg_performance = avg_performance.fillna(0).replace([np.inf, -np.inf], 0)
        exp_performance = np.exp(avg_performance)
        sum_exp_performance = np.sum(exp_performance)
        
        if sum_exp_performance == 0 or np.isnan(sum_exp_performance):
            num_models = len(avg_performance)
            if num_models == 0:
                return pd.Series(dtype='float64')
            return pd.Series(1.0 / num_models, index=avg_performance.index)
            
        model_weights = exp_performance / sum_exp_performance
        return model_weights

    def calculate_positions(self, model_signals, model_weights, asset_vols):
        """
        Stage 2: SIMPLIFIED Long-Only Momentum Strategy
        """
        # Convert ML probabilities to signals: 0-1 -> -1 to +1
        position_signals = (model_signals - 0.5) * 2
        ensemble_signal = position_signals.dot(model_weights)
        
        # For each date, rank assets and go long top performers
        target_positions = pd.Series(0.0, index=ensemble_signal.index)
        
        for date in ensemble_signal.index.get_level_values('date').unique():
            date_signals = ensemble_signal.xs(date, level='date')
            
            # Sort by signal strength
            date_signals_sorted = date_signals.sort_values(ascending=False)
            
            # Take top 10% performers (decile strategy) - more concentrated
            n_assets = len(date_signals)
            n_long = max(5, int(n_assets * 0.10))  # At least 5 stocks
            
            # Weight by signal strength (stronger signals get more allocation)
            top_signals = date_signals_sorted.head(n_long)
            
            # Normalize signals to positive weights that sum to 1
            signal_weights = top_signals - top_signals.min() + 0.01  # Shift to positive
            signal_weights = signal_weights / signal_weights.sum()  # Normalize to sum=1
            
            for asset, weight in signal_weights.items():
                target_positions.loc[(date, asset)] = weight
        
        return target_positions


# --- MODULE 4: WALK-FORWARD BACKTESTER (From Paper Section 3.2) ---

def process_single_window(args):
    """
    Process a single walk-forward window.
    This function is designed to be run in parallel.
    """
    (data, unique_dates, start_index, training_window, validation_window, 
     test_window, lookbacks, longest_lookback, end_index) = args
    
    train_start_idx = start_index
    train_end_idx = train_start_idx + training_window
    test_start_idx = train_end_idx
    test_end_idx = test_start_idx + test_window
    
    if test_end_idx > end_index:
        return None
        
    train_dates = unique_dates[train_start_idx:train_end_idx]
    test_dates = unique_dates[test_start_idx:test_end_idx]

    print(f"Processing: Train {train_dates[0].date()} to {train_dates[-1].date()}, Test {test_dates[0].date()} to {test_dates[-1].date()}")

    # Initialize objects for this window
    feature_engineer = FeatureEngineer(lookbacks=lookbacks, target_period=test_window)
    model_competitor = ModelCompetitor(n_workers=1)  # Use single thread per window
    portfolio_constructor = PortfolioConstructor()

    # Get data slices
    train_feat_start_idx = max(0, train_start_idx - longest_lookback - 2)
    train_target_end_idx = min(end_index - 1, train_end_idx + feature_engineer.target_period)
    test_feat_start_idx = max(0, test_start_idx - longest_lookback - 2)
    test_target_end_idx = min(end_index - 1, test_end_idx + feature_engineer.target_period)

    train_slice_dates = unique_dates[train_feat_start_idx : train_target_end_idx]
    idx = pd.IndexSlice
    train_data_slice = data.loc[idx[train_slice_dates, :], :]
    X_train_all, y_train_all, train_data_with_features = feature_engineer.build_features_and_target(train_data_slice)
    
    test_slice_dates = unique_dates[test_feat_start_idx : test_target_end_idx]
    test_data_slice = data.loc[idx[test_slice_dates, :], :]
    X_test_all, y_test_all, test_data_with_features = feature_engineer.build_features_and_target(test_data_slice)

    # Filter for exact date ranges
    X_train = X_train_all[X_train_all.index.get_level_values('date').isin(train_dates)]
    y_train = y_train_all[y_train_all.index.get_level_values('date').isin(train_dates)]
    X_test = X_test_all[X_test_all.index.get_level_values('date').isin(test_dates)]
    
    if X_train.empty or X_test.empty:
        return None

    if y_train.nunique() < 2:
        return None

    # Split the training window into actual-train (first 189 days) + held-out validation
    # (last 63 days). The previous code fitted models on ALL 252 train days then "validated"
    # on the last 63 of those — meaning val accuracy was inflated by overfitting on data
    # the models had just been trained on, which over-weighted tree models in the ensemble.
    val_start_date = train_dates[-validation_window]
    train_actual_mask = X_train.index.get_level_values('date') < val_start_date
    val_mask = X_train.index.get_level_values('date') >= val_start_date

    X_train_actual = X_train[train_actual_mask]
    y_train_actual = y_train[train_actual_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]

    if X_train_actual.empty or y_train_actual.nunique() < 2:
        return None

    # Fit scaler on actual-train ONLY (no leakage from val/test)
    feature_engineer.fit_scaler(X_train_actual)
    X_train_scaled = feature_engineer.scale_features(X_train_actual)
    X_val_scaled = feature_engineer.scale_features(X_val) if not X_val.empty else X_val
    X_test_scaled = feature_engineer.scale_features(X_test)

    # Train models on actual-train only
    model_competitor.train_models(X_train_scaled, y_train_actual)
    model_signals = model_competitor.get_predictions(X_test_scaled)

    if model_signals.empty or model_signals.isnull().all().all():
        return None

    # Genuine out-of-sample validation for ensemble weighting
    model_performance = {}
    if not X_val_scaled.empty and y_val.nunique() >= 2:
        val_preds = model_competitor.get_predictions(X_val_scaled)
        for model_name in val_preds.columns:
            preds = (val_preds[model_name] > 0.5).astype(int)
            model_performance[model_name] = [accuracy_score(y_val, preds)]
    
    if not model_performance:
         model_performance = {model: [1.0] for model in model_competitor.trained_models.keys()}

    model_weights = portfolio_constructor.calculate_model_weights(pd.DataFrame(model_performance))
    
    try:
        last_train_day_data = train_data_with_features[train_data_with_features.index.get_level_values('date') == train_dates[-1]]
        asset_vols = last_train_day_data['vol_20d'].droplevel('date') 
    except Exception as e:
        asset_vols = pd.Series(0.01, index=data.index.get_level_values('asset').unique())

    target_positions = portfolio_constructor.calculate_positions(model_signals, model_weights, asset_vols)
    
    return target_positions


def run_walk_forward_backtest(data, n_workers=None):
    """
    Implements the Walk-Forward Validation Protocol from
    paper Section 3.2 with parallel processing.
    """
    print("Starting walk-forward backtest with parallel processing...")

    training_window = 252  # 1 year
    validation_window = 63 # ~3 months (Placeholder)
    test_window = 21       # ~1 month (Rebalancing period)
    step_size = 5          # 1 week, per the paper
    
    if n_workers is None:
        # Use cpu_count - 1 workers (one per core, minus one for the OS / main process)
        # so speedup scales with available cores rather than a fixed cap.
        # With OMP_NUM_THREADS=1 (set at top of file), each worker is single-threaded so
        # there's no oversubscription and scaling is linear up to cpu_count.
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_workers = max(1, cpu - 1)
    print(f"Backtest using {n_workers} parallel workers (cpu_count={os.cpu_count()})", flush=True)
    
    unique_dates = data.index.get_level_values('date').unique().sort_values()
    end_index = len(unique_dates)
    
    lookbacks = [5, 10, 20, 60, 120]  # Focus on shorter-term momentum
    longest_lookback = max(lookbacks) if lookbacks else 0
    
    # Check if we have enough data
    min_data_needed = longest_lookback + training_window + test_window + test_window
    if end_index < min_data_needed:
        print(f"Error: Not enough data. Need at least {min_data_needed} unique days, but only have {end_index}.")
        return pd.DataFrame()
    
    # Set start_index to the first day we can build a full feature set
    start_index = longest_lookback
    
    # Generate all window arguments
    window_args = []
    current_idx = start_index
    while current_idx + training_window + test_window <= end_index:
        window_args.append((
            data, unique_dates, current_idx, training_window, 
            validation_window, test_window, lookbacks, longest_lookback, end_index
        ))
        current_idx += step_size
    
    print(f"Total windows to process: {len(window_args)}")
    print(f"Using {n_workers} workers for parallel processing\n")
    
    # Process windows in parallel
    all_target_positions = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_single_window, args) for args in window_args]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                if result is not None:
                    all_target_positions.append(result)
                print(f"Completed {i}/{len(window_args)} windows")
            except Exception as e:
                print(f"Window processing error: {e}")

    print("\nWalk-forward backtest complete.")
    if not all_target_positions:
        print("No positions were generated. The dataset might be too small.")
        return pd.DataFrame()
        
    # Combine all positions and remove duplicate index entries from overlapping windows
    final_positions_df = pd.concat(all_target_positions)
    final_positions_df = final_positions_df[~final_positions_df.index.duplicated(keep='last')]
    final_positions_df.sort_index(inplace=True)
    
    return final_positions_df


# --- MODULE 5: BACKTESTING & PERFORMANCE METRICS ---

class PerformanceAnalyzer:
    """
    Calculates comprehensive performance metrics for the strategy.
    """
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = None
        
    def calculate_benchmark_returns(self, price_data):
        """
        Calculate equal-weighted market benchmark returns.
        """
        print("\nCalculating market benchmark returns...")
        
        all_dates = price_data.index.get_level_values('date').unique().sort_values()
        benchmark_returns_list = []
        
        for i in range(len(all_dates) - 1):
            current_date = all_dates[i]
            next_date = all_dates[i + 1]
            
            idx = pd.IndexSlice
            try:
                current_prices = price_data.loc[idx[current_date, :], 'Close'].droplevel('date')
                next_prices = price_data.loc[idx[next_date, :], 'Close'].droplevel('date')
                
                # Equal-weighted market return
                asset_returns = (next_prices - current_prices) / current_prices
                market_return = asset_returns.mean()
                
                benchmark_returns_list.append({
                    'date': next_date,
                    'return': market_return
                })
            except (KeyError, AttributeError):
                continue
        
        benchmark_df = pd.DataFrame(benchmark_returns_list)
        benchmark_df.set_index('date', inplace=True)
        
        print(f"Calculated {len(benchmark_df)} benchmark returns")
        if len(benchmark_df) > 0:
            print(f"Benchmark: mean={benchmark_df['return'].mean():.4%}, std={benchmark_df['return'].std():.4%}")
        
        return benchmark_df
    
    def calculate_portfolio_returns(self, positions_df, price_data):
        """
        Calculate daily portfolio returns based on positions and price changes.
        """
        print("\nCalculating portfolio returns...")
        
        # Get all available dates from price data
        all_dates = price_data.index.get_level_values('date').unique().sort_values()
        
        returns_list = []
        
        # For each date, get the most recent position and calculate return to next day
        for i in range(len(all_dates) - 1):
            current_date = all_dates[i]
            next_date = all_dates[i + 1]
            
            # Get the most recent positions at or before current_date
            available_position_dates = positions_df.index.get_level_values('date').unique()
            available_position_dates = available_position_dates[available_position_dates <= current_date]
            
            if len(available_position_dates) == 0:
                continue
                
            position_date = available_position_dates[-1]
            
            try:
                # positions_df is a Series with MultiIndex (date, asset)
                current_positions = positions_df.xs(position_date, level='date')
                # current_positions now has asset as index and position values
            except (KeyError, TypeError):
                continue
            
            # Get prices for current and next date
            idx = pd.IndexSlice
            try:
                current_prices = price_data.loc[idx[current_date, :], 'Close'].droplevel('date')
                next_prices = price_data.loc[idx[next_date, :], 'Close'].droplevel('date')
            except (KeyError, AttributeError):
                continue
            
            # Calculate returns for each asset
            asset_returns = (next_prices - current_prices) / current_prices
            
            # Align positions with returns - only for assets that have both position and price
            common_assets = current_positions.index.intersection(asset_returns.index)
            
            if len(common_assets) == 0:
                continue
            
            # Portfolio return = sum(position * asset_return) for common assets
            portfolio_return = (current_positions[common_assets] * asset_returns[common_assets]).sum()
            
            returns_list.append({
                'date': next_date,
                'return': portfolio_return,
                'position_date': position_date
            })
        
        if not returns_list:
            print("Warning: No returns calculated")
            return pd.DataFrame(columns=['return'])
        
        returns_df = pd.DataFrame(returns_list)
        returns_df.set_index('date', inplace=True)
        
        print(f"Calculated {len(returns_df)} daily returns")
        if len(returns_df) > 0:
            print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
            print(f"Sample returns: mean={returns_df['return'].mean():.4%}, std={returns_df['return'].std():.4%}")
        
        return returns_df
    
    def calculate_metrics(self, returns_df, benchmark_df=None):
        """
        Calculate key performance metrics.
        """
        returns = returns_df['return'].values
        
        # Basic statistics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Excess returns over risk-free rate
        daily_rf = self.risk_free_rate / 252
        total_excess_return = total_return - (self.risk_free_rate * len(returns) / 252)
        annual_excess_return = annual_return - self.risk_free_rate
        
        # Market benchmark statistics
        if benchmark_df is not None and not benchmark_df.empty:
            # Align dates
            common_dates = returns_df.index.intersection(benchmark_df.index)
            strategy_aligned = returns_df.loc[common_dates, 'return'].values
            benchmark_aligned = benchmark_df.loc[common_dates, 'return'].values
            
            # Market returns
            market_total_return = (1 + benchmark_aligned).prod() - 1
            market_annual_return = (1 + market_total_return) ** (252 / len(benchmark_aligned)) - 1
            market_volatility = benchmark_aligned.std() * np.sqrt(252)
            
            # Alpha (Jensen's Alpha) - excess return over market
            alpha_total = total_return - market_total_return
            alpha_annual = annual_return - market_annual_return
            
            # Beta (systematic risk)
            covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
            market_variance = np.var(benchmark_aligned)
            beta = covariance / market_variance if market_variance > 0 else 0
            
            # Information Ratio (tracking error adjusted alpha)
            active_returns = strategy_aligned - benchmark_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = (annual_return - market_annual_return) / tracking_error if tracking_error > 0 else 0
        else:
            market_total_return = 0
            market_annual_return = 0
            market_volatility = 0
            alpha_total = 0
            alpha_annual = 0
            beta = 0
            information_ratio = 0
            tracking_error = 0
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Average win/loss
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        
        # Profit factor
        total_wins = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Market Total Return': f"{market_total_return:.2%}",
            'Market Annual Return': f"{market_annual_return:.2%}",
            'Alpha (Total)': f"{alpha_total:.2%}",
            'Alpha (Annual)': f"{alpha_annual:.2%}",
            'Beta': f"{beta:.3f}",
            'Information Ratio': f"{information_ratio:.3f}",
            'Tracking Error': f"{tracking_error:.2%}",
            'Risk-Free Rate': f"{self.risk_free_rate:.2%}",
            'Excess Return vs RF': f"{annual_excess_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Market Volatility': f"{market_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Sortino Ratio': f"{sortino_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar_ratio:.3f}",
            'Win Rate': f"{win_rate:.2%}",
            'Avg Win': f"{avg_win:.4%}",
            'Avg Loss': f"{avg_loss:.4%}",
            'Profit Factor': f"{profit_factor:.3f}",
            'Number of Trades': len(returns)
        }
        
        return metrics, cumulative_returns, drawdown
    
    def print_performance_report(self, metrics):
        """
        Print formatted performance report.
        """
        print("\n" + "="*60)
        print(" " * 15 + "STRATEGY PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n{'Metric':<25} {'Value':>15}")
        print("-"*60)
        
        for metric, value in metrics.items():
            print(f"{metric:<25} {value:>15}")
        
        print("="*60)
    
    def plot_performance(self, cumulative_returns, drawdown, returns_df, output_dir='.'):
        """
        Generate performance visualizations (requires matplotlib).
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Cumulative returns
            axes[0].plot(returns_df.index, cumulative_returns, linewidth=2)
            axes[0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].grid(True, alpha=0.3)

            # Drawdown
            axes[1].fill_between(returns_df.index, drawdown * 100, 0, alpha=0.3, color='red')
            axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].grid(True, alpha=0.3)

            # Daily returns
            axes[2].bar(returns_df.index, returns_df['return'] * 100, alpha=0.6)
            axes[2].set_title('Daily Returns', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Return (%)')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(output_dir, 'strategy_performance.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nPerformance chart saved as '{out_path}'")

        except ImportError:
            print("\nNote: Install matplotlib to generate performance charts:")
            print("  pip install matplotlib")


# --- MAIN EXECUTION BLOCK (To load and run your data) ---

def load_and_preprocess_data(filepath):
    """
    Loads the user-provided data, renames columns,
    and sets the required MultiIndex.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, header=0) 
        
    except FileNotFoundError:
        print(f"---! ERROR: File not found at '{filepath}' !---")
        print(f"Please ensure '{filepath}' is in the same directory as the Python script.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

    df.rename(columns={
        'PX_OPEN': 'Open',
        'PX_HIGH': 'High',
        'PX_LOW': 'Low',
        'PX_LAST': 'Close',
        'VOLUME': 'Volume',
        'ticker': 'asset'
    }, inplace=True)
    
    try:
        df['date'] = pd.to_datetime(df['date'])
    except KeyError:
        print("Error: 'date' column not found after loading. Check CSV header.")
        return None
    except Exception as e:
        print(f"Error converting date column: {e}")
        return None
    
    if 'asset' not in df.columns:
        print("Error: 'ticker' (renamed to 'asset') column not found. Check CSV header.")
        return None
        
    df.set_index(['date', 'asset'], inplace=True)
    df.sort_index(inplace=True)
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in required_cols if col in df.columns]
    if 'Close' not in available_cols:
        print("Error: 'PX_LAST' (renamed to 'Close') column not found.")
        return None
        
    df = df[available_cols]
    df.dropna(subset=['Close'], inplace=True)
    
    print("Data preprocessing complete. Sample:")
    print(df.head())
    return df


if __name__ == "__main__":

    # --- CONFIGURATION ---
    # Use script-relative path so the script works regardless of cwd.
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_FILEPATH = os.path.join(HERE, 'df_2010.csv')
    OUTPUT_DIR = HERE  # write portfolio_returns.csv etc. next to the script
    FIG_DIR = os.path.join(HERE, 'figures')  # charts go in figures/
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Load and Preprocess Data
    processed_data = load_and_preprocess_data(DATA_FILEPATH)

    # 2. Run the Backtest
    if processed_data is not None and not processed_data.empty:
        
        # You can uncomment this line to run a faster test on a smaller date range
        # idx = pd.IndexSlice
        # processed_data = processed_data.loc[idx['2018-01-01':'2024-12-31', :], :]
        
        final_positions = run_walk_forward_backtest(processed_data)

        # 3. Display Results
        if not final_positions.empty:
            print("\n--- Final Target Positions (Sample) ---")
            try:
                sample_assets = final_positions.index.get_level_values('asset').unique()[:3]
                for asset in sample_assets:
                    print(f"\n--- Sample positions for asset: {asset} ---")
                    print(final_positions.xs(asset, level='asset').head(10))
            except Exception as e:
                print(f"Error displaying positions: {e}")
            
            # 4. Run Performance Analysis
            print("\n" + "="*60)
            print("RUNNING PERFORMANCE ANALYSIS")
            print("="*60)
            
            analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
            
            # Calculate benchmark returns
            benchmark_df = analyzer.calculate_benchmark_returns(processed_data)
            
            # Calculate portfolio returns
            returns_df = analyzer.calculate_portfolio_returns(final_positions, processed_data)
            
            if not returns_df.empty:
                # Calculate metrics
                metrics, cumulative_returns, drawdown = analyzer.calculate_metrics(returns_df, benchmark_df)
                
                # Print report
                analyzer.print_performance_report(metrics)
                
                # Try to plot (optional)
                analyzer.plot_performance(cumulative_returns, drawdown, returns_df, output_dir=FIG_DIR)
                
                # Save results to CSV (script-relative paths)
                returns_df.to_csv(os.path.join(OUTPUT_DIR, 'portfolio_returns.csv'))
                print(f"\nPortfolio returns saved to '{os.path.join(OUTPUT_DIR, 'portfolio_returns.csv')}'")

                final_positions.to_csv(os.path.join(OUTPUT_DIR, 'portfolio_positions.csv'))
                print(f"Portfolio positions saved to '{os.path.join(OUTPUT_DIR, 'portfolio_positions.csv')}'")
            else:
                print("\nCould not calculate returns - insufficient data.")

        else:
            print("\nBacktest finished with no positions.")
            training_window = 252
            test_window = 21
            lookbacks = [5, 10, 20, 60, 120]
            print(f"This can happen if the dataset (rows: {len(processed_data)}) is too small for the")
            print(f"minimum data requirement of ~{max(lookbacks) + training_window + test_window} days.")
    else:
        print("\nCould not run backtest as data failed to load or was empty.")
        print(f"Please check that '{DATA_FILEPATH}' is in the correct folder.")