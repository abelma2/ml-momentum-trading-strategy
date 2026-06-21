# ============================================================
# Momentum ML Framework – Out-of-Sample Diagnostics
# Indicator tests + signal tests + feature importance (OOS)
#
# Standalone out-of-sample diagnostics for the momentum strategy:
#   - univariate walk-forward AUC for each of the 25 features
#   - signal-strength forward-return analysis by decile bucket
#   - permutation feature importance (out-of-sample)
#   - per-window training/test diagnostics
#
# The information coefficient uses Spearman rank correlation via pandas (no SciPy).
# Excel output is written when openpyxl is available, else it falls back to CSV.
# Parallelized across cores (one single-threaded worker per core).
# ============================================================

import os
# Pin BLAS/OpenMP thread pools to 1 BEFORE importing numpy/sklearn/xgboost.
# With ProcessPoolExecutor running N parallel windows, each model's underlying BLAS
# would otherwise spawn cpu_count threads, leading to N*cpu_count threads on cpu_count
# cores. Single-threaded BLAS per worker keeps scaling linear up to physical cores.
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

import warnings
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.inspection import permutation_importance

import xgboost as xgb

# ----------------------------
# Global settings
# ----------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
np.seterr(divide="ignore", invalid="ignore")


# ============================================================
# Global perf knobs
# ============================================================
def _default_n_jobs() -> int:
    c = os.cpu_count() or 1
    return max(1, c - 1)  # "available core - 1"

N_JOBS = _default_n_jobs()

# Set these to trade speed vs precision (safe: does NOT change structure)
UNIV_STEP_SIZE_DEFAULT = 21         # monthly steps
MULTI_STEP_SIZE_DEFAULT = 5         # weekly steps (heavier)
N_PERM_REPEATS_DEFAULT = 5          # permutation repeats (heaviest component)


# ============================================================
# Small helpers
# ============================================================
def safe_to_excel(df: pd.DataFrame, path: str, index: bool = False) -> bool:
    """
    Try to write Excel. If openpyxl isn't installed (common on minimal installs),
    fall back gracefully (CSV already saved elsewhere).
    """
    try:
        df.to_excel(path, index=index)
        return True
    except ModuleNotFoundError as e:
        if "openpyxl" in str(e).lower():
            print(f"⚠️  Skipping Excel output (missing openpyxl): {path}")
            print("   Fix: pip install openpyxl")
            return False
        raise


def spearman_ic(y_true: pd.Series, p: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted probability and outcome (IC).
    No SciPy required.
    """
    try:
        s1 = pd.Series(np.asarray(p, dtype=float))
        s2 = pd.Series(pd.Series(y_true).astype(float).values)
        ic = s1.corr(s2, method="spearman")
        return float(ic) if ic is not None else float("nan")
    except Exception:
        return float("nan")


# ============================================================
# MODULE 1: DATA LOADING (normalize columns to date/asset/Close/Volume)
# ============================================================
class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(self.filepath)

        df = pd.read_csv(self.filepath)

        # Date
        if "date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                raise ValueError("Missing date column (date or Date).")
        df["date"] = pd.to_datetime(df["date"])

        # Asset
        if "asset" not in df.columns:
            if "ticker" in df.columns:
                df = df.rename(columns={"ticker": "asset"})
            elif "Ticker" in df.columns:
                df = df.rename(columns={"Ticker": "asset"})
            else:
                raise ValueError("Missing asset identifier (asset/ticker).")

        # Price -> Close
        price_candidates = ["Close", "PX_LAST", "price", "close", "Adj Close", "adj_close"]
        found = None
        for c in price_candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise ValueError(f"No price column found. Tried: {price_candidates}. Found: {df.columns.tolist()}")
        if found != "Close":
            df = df.rename(columns={found: "Close"})

        # Volume (optional)
        vol_candidates = ["Volume", "VOLUME", "PX_VOLUME", "volume"]
        for v in vol_candidates:
            if v in df.columns:
                if v != "Volume":
                    df = df.rename(columns={v: "Volume"})
                break

        df = df.sort_values(["date", "asset"])
        df = df.set_index(["date", "asset"]).sort_index()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Close"])
        return df


# ============================================================
# MODULE 2: FEATURE ENGINEERING
# ============================================================
class FeatureEngineer:
    """
    Builds the same 25 core families:
    returns, vol, MA, risk-adjusted return, dist-from-MA across lookbacks
    + optional volume changes and a cross-sectional z-score feature.

    NOTE: Uses groupby(...).transform(...) to keep index aligned.
    """

    def __init__(
        self,
        lookbacks: Optional[List[int]] = None,
        target_period: int = 21,
        label_mode: str = "cross_section_median",  # or "binary_positive"
    ):
        self.lookbacks = lookbacks or [5, 10, 20, 60, 120]
        self.target_period = int(target_period)
        self.label_mode = label_mode
        self.scaler = StandardScaler()

    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        g = data.groupby(level="asset")

        for lb in self.lookbacks:
            data[f"return_{lb}d"] = g["Close"].pct_change(lb)
            data[f"vol_{lb}d"] = g["Close"].transform(lambda x: x.pct_change().rolling(lb).std())
            data[f"ma_{lb}d"] = g["Close"].transform(lambda x: x.rolling(lb).mean())
            data[f"risk_adj_return_{lb}d"] = data[f"return_{lb}d"] / (data[f"vol_{lb}d"] + 1e-12)
            data[f"dist_from_ma_{lb}d"] = data["Close"] / (data[f"ma_{lb}d"] + 1e-12) - 1.0

        return data

    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Volume" not in data.columns:
            return data
        data = data.copy()
        g = data.groupby(level="asset")
        for lb in self.lookbacks:
            data[f"vol_chg_{lb}d"] = g["Volume"].pct_change(lb)
        return data

    def add_cross_sectional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "return_20d" in data.columns:
            tmp = data[["return_20d"]].reset_index()
            stats = tmp.groupby("date")["return_20d"].agg(["mean", "std"])
            m = data.index.get_level_values("date").map(stats["mean"])
            s = data.index.get_level_values("date").map(stats["std"])
            data["cs_z_return_20d"] = (data["return_20d"] - m) / (s + 1e-12)
        return data

    def create_target(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        g = data.groupby(level="asset")
        fut_ret = g["Close"].pct_change(self.target_period).shift(-self.target_period)

        if self.label_mode == "binary_positive":
            data["target"] = (fut_ret > 0).astype(int)
        elif self.label_mode == "cross_section_median":
            tmp = fut_ret.rename("fut").to_frame().dropna().reset_index()
            med = tmp.groupby("date")["fut"].median()
            m = data.index.get_level_values("date").map(med)
            data["target"] = (fut_ret > m).astype(int)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        return data.dropna(subset=["target"])

    def get_feature_cols(self, data: pd.DataFrame) -> List[str]:
        prefixes = ("return_", "vol_", "ma_", "risk_adj_return_", "dist_from_ma_", "vol_chg_", "cs_z_")
        return [c for c in data.columns if c.startswith(prefixes)]

    def build(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        data = self.add_price_features(data)
        data = self.add_volume_features(data)
        data = self.add_cross_sectional_features(data)
        data = self.create_target(data)

        feats = self.get_feature_cols(data)
        X = data[feats].replace([np.inf, -np.inf], np.nan).dropna()
        y = data.loc[X.index, "target"].astype(int)
        return X, y, data.loc[X.index]

    def fit_scaler(self, X: pd.DataFrame) -> None:
        self.scaler.fit(X.values)

    def scale(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.scaler.transform(X.values)
        return pd.DataFrame(Z, index=X.index, columns=X.columns)


# ============================================================
# MODULE 3: MODELS + METRICS + ENSEMBLE
# ============================================================
def eval_prob_metrics(y_true: pd.Series, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, 0.0, 1.0)
    yhat = (p > 0.5).astype(int)

    out = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "acc": float(accuracy_score(y, yhat)),
        "bal_acc": float(balanced_accuracy_score(y, yhat)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ic_spearman": float(spearman_ic(pd.Series(y_true), p)),
    }
    return out


class ModelSuite:
    def __init__(self, random_state: int = 42, n_jobs: int = 1):
        # Default n_jobs=1. With permutation_importance
        # making 25 * n_repeats sequential predict() calls per window, each model launching
        # all-cores OpenMP/joblib workers caused thread thrash (CPU stuck near ~10% over
        # 6+ hours in the prior overnight run). Single-threaded models keep the loop
        # honest and allow outer parallelism to be added cleanly later if desired.
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.models = {
            "RidgeLogit": LogisticRegression(C=2.0, max_iter=3000, solver="lbfgs"),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=10, n_jobs=self.n_jobs, random_state=self.random_state
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, random_state=self.random_state
            ),
            "XGBoost": xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        }
        self.trained: Dict[str, Any] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.trained = {}
        for name, m in self.models.items():
            self.trained[name] = m.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {}
        for name, m in self.trained.items():
            preds[name] = m.predict_proba(X)[:, 1]
        return pd.DataFrame(preds, index=X.index)

    @staticmethod
    def exp_weights_from_metric(
        metrics: Dict[str, Dict[str, float]],
        metric: str = "auc",
        alpha: float = 3.0
    ) -> Dict[str, float]:
        scores = {k: v.get(metric, np.nan) for k, v in metrics.items()}
        scores = {k: float(s) for k, s in scores.items() if s is not None and not np.isnan(s)}
        if not scores:
            return {}
        arr = np.array(list(scores.values()), dtype=float)
        arr = arr - np.max(arr)
        w = np.exp(alpha * arr)
        w = w / (w.sum() + 1e-12)
        return dict(zip(scores.keys(), w))


# ============================================================
# MODULE 4: WALK-FORWARD UNIVARIATE FEATURE TESTS (OOS)
# ============================================================
def run_univariate_oos_walkforward(
    data: pd.DataFrame,
    out_dir: str,
    lookbacks: Optional[List[int]] = None,
    label_mode: str = "cross_section_median",
    target_period: int = 21,
    training_window: int = 252,
    test_window: int = 21,
    step_size: int = UNIV_STEP_SIZE_DEFAULT,
    precomputed_panel: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Proper OOS indicator testing:
    - For each window:
        fit scaler on TRAIN only
        fit 1-feature logistic regression on TRAIN only
        evaluate on TEST only
    - Aggregate across windows.
    """
    os.makedirs(out_dir, exist_ok=True)

    lookbacks = lookbacks or [5, 10, 20, 60, 120]
    fe = FeatureEngineer(lookbacks=lookbacks, target_period=target_period, label_mode=label_mode)

    unique_dates = data.index.get_level_values("date").unique().sort_values()
    end_idx = len(unique_dates)
    longest = max(lookbacks)
    start_idx = longest

    # Precompute full feature panel once (or accept one from caller for cache reuse)
    if precomputed_panel is not None:
        X_all, y_all = precomputed_panel
    else:
        X_all, y_all, _ = fe.build(data)
    feat_cols = list(X_all.columns)

    rows_by_feature: Dict[str, List[Dict[str, float]]] = {c: [] for c in feat_cols}

    # Estimate total window count for progress reporting
    total_windows_est = max(1, (end_idx - start_idx - training_window - test_window) // step_size + 1)
    window_idx = 0

    cur = start_idx
    while cur + training_window + test_window <= end_idx:
        window_idx += 1
        train_dates = unique_dates[cur: cur + training_window]
        test_dates = unique_dates[cur + training_window: cur + training_window + test_window]

        train_mask = X_all.index.get_level_values("date").isin(train_dates)
        test_mask = X_all.index.get_level_values("date").isin(test_dates)

        X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
        X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

        # Diagnostics for this window
        n_obs_train = int(len(X_train))
        n_obs_test = int(len(X_test))
        n_assets_train = int(X_train.index.get_level_values("asset").nunique()) if n_obs_train else 0
        n_assets_test = int(X_test.index.get_level_values("asset").nunique()) if n_obs_test else 0

        # Require both classes in train/test
        if n_obs_train < 1000 or y_train.nunique() < 2 or n_obs_test < 200 or y_test.nunique() < 2:
            cur += step_size
            continue

        fe.fit_scaler(X_train)
        X_train_s = fe.scale(X_train)
        X_test_s = fe.scale(X_test)

        for c in feat_cols:
            try:
                clf = LogisticRegression(C=2.0, max_iter=2000, solver="lbfgs")
                clf.fit(X_train_s[[c]], y_train)
                p = clf.predict_proba(X_test_s[[c]])[:, 1]
                m = eval_prob_metrics(y_test, p)
                # attach window diagnostics (same for all features in this window)
                m.update({
                    "n_obs_train": n_obs_train,
                    "n_obs_test": n_obs_test,
                    "n_assets_train": n_assets_train,
                    "n_assets_test": n_assets_test,
                })
                rows_by_feature[c].append(m)
            except Exception:
                pass

        # Print every 20 windows so the run is visibly alive
        if window_idx % 20 == 0 or window_idx == total_windows_est:
            print(f"  [univariate] window {window_idx}/{total_windows_est} done", flush=True)

        cur += step_size

    # Aggregate per feature
    out_rows = []
    for feat, mets in rows_by_feature.items():
        if not mets:
            continue
        dfm = pd.DataFrame(mets)
        out_rows.append({
            "feature": feat,
            "auc_mean": float(dfm["auc"].mean()),
            "auc_std": float(dfm["auc"].std(ddof=0)),
            "acc_mean": float(dfm["acc"].mean()),
            "bal_acc_mean": float(dfm["bal_acc"].mean()),
            "logloss_mean": float(dfm["logloss"].mean()),
            "brier_mean": float(dfm["brier"].mean()),
            "ic_spearman_mean": float(dfm["ic_spearman"].mean()),
            "ic_spearman_std": float(dfm["ic_spearman"].std(ddof=0)),
            "n_windows": int(len(dfm)),
            "avg_n_assets_train": float(dfm["n_assets_train"].mean()) if "n_assets_train" in dfm else float("nan"),
            "avg_n_assets_test": float(dfm["n_assets_test"].mean()) if "n_assets_test" in dfm else float("nan"),
            "avg_n_obs_train": float(dfm["n_obs_train"].mean()) if "n_obs_train" in dfm else float("nan"),
            "avg_n_obs_test": float(dfm["n_obs_test"].mean()) if "n_obs_test" in dfm else float("nan"),
        })

    out = pd.DataFrame(out_rows).sort_values("auc_mean", ascending=False)

    out_csv = os.path.join(out_dir, "univariate_feature_oos_walkforward.csv")
    out_xlsx = os.path.join(out_dir, "univariate_feature_oos_walkforward.xlsx")
    out.to_csv(out_csv, index=False)
    safe_to_excel(out, out_xlsx, index=False)

    print("✓ Univariate OOS walk-forward saved:")
    print("  -", out_csv)
    print("  -", out_xlsx)
    return out


# ============================================================
# MODULE 5: WALK-FORWARD MULTIVARIATE MODEL + ENSEMBLE + IMPORTANCE
# ============================================================
@dataclass
class WindowResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model_metrics: Dict[str, Dict[str, float]]
    ensemble_metrics: Dict[str, float]
    weights: Dict[str, float]
    perm_importance: Optional[pd.Series]


# ----------------------------------------------------------------------------
# ProcessPoolExecutor worker plumbing for parallel multivariate walk-forward.
# Module-level so the worker function is picklable.
# ----------------------------------------------------------------------------
_DIAG_WORKER: Dict[str, Any] = {}


def _init_multivar_worker(X_all_pkl: bytes, y_all_pkl: bytes,
                          lookbacks: List[int], target_period: int, label_mode: str) -> None:
    """Runs once per worker subprocess. Loads the precomputed feature panel into module
    state so each task in this worker can access it without re-pickling per task."""
    import pickle
    _DIAG_WORKER["X_all"] = pickle.loads(X_all_pkl)
    _DIAG_WORKER["y_all"] = pickle.loads(y_all_pkl)
    _DIAG_WORKER["lookbacks"] = lookbacks
    _DIAG_WORKER["target_period"] = target_period
    _DIAG_WORKER["label_mode"] = label_mode


def _process_multivar_window(args):
    """Worker: process one walk-forward window. Returns (metric_rows, perm_imp_series_or_None)."""
    train_dates_arr, test_dates_arr, validation_window, weight_metric, n_perm_repeats = args
    train_dates = pd.DatetimeIndex(train_dates_arr)
    test_dates = pd.DatetimeIndex(test_dates_arr)

    X_all = _DIAG_WORKER["X_all"]
    y_all = _DIAG_WORKER["y_all"]
    lookbacks = _DIAG_WORKER["lookbacks"]
    target_period = _DIAG_WORKER["target_period"]
    label_mode = _DIAG_WORKER["label_mode"]
    feat_cols = list(X_all.columns)

    fe = FeatureEngineer(lookbacks=lookbacks, target_period=target_period, label_mode=label_mode)

    train_mask = X_all.index.get_level_values("date").isin(train_dates)
    test_mask = X_all.index.get_level_values("date").isin(test_dates)
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

    n_obs_train = int(len(X_train))
    n_obs_test = int(len(X_test))
    n_assets_train = int(X_train.index.get_level_values("asset").nunique()) if n_obs_train else 0
    n_assets_test = int(X_test.index.get_level_values("asset").nunique()) if n_obs_test else 0

    if y_train.nunique() < 2 or y_test.nunique() < 2 or X_train.empty or X_test.empty:
        return None
    if validation_window >= len(train_dates):
        return None

    val_start = train_dates[-validation_window]
    val_mask = X_train.index.get_level_values("date") >= val_start
    X_val, y_val = X_train.loc[val_mask], y_train.loc[val_mask]

    fe.fit_scaler(X_train)
    X_train_s = fe.scale(X_train)
    X_val_s = fe.scale(X_val) if not X_val.empty else X_val
    X_test_s = fe.scale(X_test)

    suite = ModelSuite(n_jobs=1)
    suite.train(X_train_s, y_train)

    val_preds = (
        suite.predict_proba(X_val_s)
        if (not X_val_s.empty and y_val.nunique() >= 2)
        else pd.DataFrame(index=X_val_s.index)
    )
    test_preds = suite.predict_proba(X_test_s)

    model_metrics: Dict[str, Dict[str, float]] = {}
    val_metrics: Dict[str, Dict[str, float]] = {}
    for m in test_preds.columns:
        model_metrics[m] = eval_prob_metrics(y_test, test_preds[m].values)
        if (m in val_preds.columns) and (not val_preds.empty) and (y_val.nunique() >= 2):
            val_metrics[m] = eval_prob_metrics(y_val, val_preds[m].values)

    weights = ModelSuite.exp_weights_from_metric(val_metrics, metric=weight_metric, alpha=3.0)
    if not weights:
        weights = {m: 1.0 / len(test_preds.columns) for m in test_preds.columns}

    w = np.array([weights[m] for m in test_preds.columns], dtype=float)
    w = w / (w.sum() + 1e-12)
    p_ens = test_preds.values @ w
    ens_metrics = eval_prob_metrics(y_test, p_ens)

    train_start_str, train_end_str = str(train_dates[0].date()), str(train_dates[-1].date())
    test_start_str, test_end_str = str(test_dates[0].date()), str(test_dates[-1].date())

    rows = []
    for m, mm in model_metrics.items():
        row = {
            "train_start": train_start_str, "train_end": train_end_str,
            "test_start": test_start_str, "test_end": test_end_str,
            "model": m,
            "weight": float(weights.get(m, np.nan)),
            "n_obs_train": n_obs_train,
            "n_obs_test": n_obs_test,
            "n_assets_train": n_assets_train,
            "n_assets_test": n_assets_test,
            **{f"test_{k}": v for k, v in mm.items()},
            **{f"val_{k}": v for k, v in val_metrics.get(m, {}).items()},
        }
        rows.append(row)

    rows.append({
        "train_start": train_start_str, "train_end": train_end_str,
        "test_start": test_start_str, "test_end": test_end_str,
        "model": "Ensemble",
        "weight": 1.0,
        "n_obs_train": n_obs_train,
        "n_obs_test": n_obs_test,
        "n_assets_train": n_assets_train,
        "n_assets_test": n_assets_test,
        **{f"test_{k}": v for k, v in ens_metrics.items()},
    })

    perm_imp = None
    if (not X_val_s.empty) and (y_val.nunique() >= 2):
        chosen_name = "XGBoost" if "XGBoost" in suite.trained else "RidgeLogit"
        chosen = suite.trained[chosen_name]
        try:
            r = permutation_importance(
                chosen, X_val_s, y_val,
                scoring="roc_auc",
                n_repeats=n_perm_repeats,
                random_state=42,
            )
            perm_imp = pd.Series(r.importances_mean, index=feat_cols)
        except Exception:
            pass

    return (rows, perm_imp, test_start_str)


def run_multivariate_walkforward_with_importance(
    data: pd.DataFrame,
    out_dir: str,
    lookbacks: Optional[List[int]] = None,
    label_mode: str = "cross_section_median",
    target_period: int = 21,
    training_window: int = 252,
    validation_window: int = 63,
    test_window: int = 21,
    step_size: int = MULTI_STEP_SIZE_DEFAULT,
    weight_metric: str = "auc",
    n_perm_repeats: int = N_PERM_REPEATS_DEFAULT,
    n_workers: Optional[int] = None,
    precomputed_panel: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel walk-forward over windows. Each window is processed in a subprocess
    via ProcessPoolExecutor; the precomputed feature panel is passed once per
    worker via initargs (not per task) to avoid pickling 100MB on every dispatch.

    Produces:
      - per-window model metrics (each model + ensemble) -> window_model_metrics.csv/.xlsx
      - aggregated permutation importance on validation sets -> permutation_importance_summary.csv/.xlsx

    Args:
        precomputed_panel: optional (X_all, y_all) computed once by the caller and shared
                           across all three diagnostic stages to avoid recomputing features.
    """
    os.makedirs(out_dir, exist_ok=True)

    lookbacks = lookbacks or [5, 10, 20, 60, 120]
    fe = FeatureEngineer(lookbacks=lookbacks, target_period=target_period, label_mode=label_mode)

    unique_dates = data.index.get_level_values("date").unique().sort_values()
    end_idx = len(unique_dates)
    longest = max(lookbacks)
    start_idx = longest

    if precomputed_panel is not None:
        X_all, y_all = precomputed_panel
    else:
        X_all, y_all, _ = fe.build(data)
    feat_cols = list(X_all.columns)

    # Build window argument list (no heavy data; just date arrays + scalars).
    window_args = []
    cur = start_idx
    while cur + training_window + test_window <= end_idx:
        train_dates = unique_dates[cur: cur + training_window]
        test_dates = unique_dates[cur + training_window: cur + training_window + test_window]
        # Convert to numpy datetime64 arrays so pickling is small/fast.
        window_args.append((
            np.array(train_dates),
            np.array(test_dates),
            validation_window,
            weight_metric,
            n_perm_repeats,
        ))
        cur += step_size

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    n_workers = min(n_workers, len(window_args)) if window_args else 1

    print(
        f"[multivar] dispatching {len(window_args)} windows to {n_workers} workers "
        f"(cpu_count={os.cpu_count()}, n_perm_repeats={n_perm_repeats})",
        flush=True,
    )

    # Pickle the panel once in main; workers unpickle once on init.
    import pickle as _pickle
    import time as _time
    X_all_pkl = _pickle.dumps(X_all)
    y_all_pkl = _pickle.dumps(y_all)
    print(
        f"[multivar] panel size: X_all={len(X_all_pkl)/(1024*1024):.1f}MB "
        f"y_all={len(y_all_pkl)/(1024*1024):.1f}MB (pickled once per worker via initargs)",
        flush=True,
    )

    metric_rows: List[Dict[str, Any]] = []
    perm_list: List[pd.Series] = []
    completed = 0
    t0 = _time.time()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_multivar_worker,
        initargs=(X_all_pkl, y_all_pkl, lookbacks, target_period, label_mode),
    ) as executor:
        futures = [executor.submit(_process_multivar_window, args) for args in window_args]
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                result = fut.result()
            except Exception as e:
                print(f"  [multivar] worker raised: {type(e).__name__}: {e}", flush=True)
                result = None

            if result is not None:
                rows, perm_imp, _ = result
                metric_rows.extend(rows)
                if perm_imp is not None:
                    perm_list.append(perm_imp)

            if completed % 10 == 0 or completed == len(window_args):
                elapsed = _time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta_min = ((len(window_args) - completed) / rate / 60.0) if rate > 0 else 0.0
                print(
                    f"  [multivar] {completed}/{len(window_args)} done | "
                    f"elapsed {elapsed/60:.1f}m | "
                    f"{rate*60:.1f} windows/min | "
                    f"ETA {eta_min:.1f}m",
                    flush=True,
                )

    # Sort metric rows by test_start so the CSV is chronological even though workers completed out of order.
    metric_rows.sort(key=lambda r: r["test_start"])

    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = os.path.join(out_dir, "window_model_metrics.csv")
    metrics_xlsx = os.path.join(out_dir, "window_model_metrics.xlsx")
    metrics_df.to_csv(metrics_csv, index=False)
    safe_to_excel(metrics_df, metrics_xlsx, index=False)

    print("✓ Window model metrics saved:")
    print("  -", metrics_csv)
    print("  -", metrics_xlsx)

    perm_df = pd.DataFrame()
    if perm_list:
        perm_df = pd.concat(perm_list, axis=1).fillna(0.0)
        perm_df.columns = [f"window_{i}" for i in range(perm_df.shape[1])]
        perm_df["mean_importance"] = perm_df.mean(axis=1)
        perm_df["std_importance"] = perm_df.drop(columns=["mean_importance"]).std(axis=1, ddof=0)
        perm_df = perm_df.sort_values("mean_importance", ascending=False)

        perm_csv = os.path.join(out_dir, "permutation_importance_summary.csv")
        perm_xlsx = os.path.join(out_dir, "permutation_importance_summary.xlsx")
        perm_df.to_csv(perm_csv)
        safe_to_excel(perm_df, perm_xlsx, index=True)

        print("✓ Permutation importance saved:")
        print("  -", perm_csv)
        print("  -", perm_xlsx)

    return metrics_df, perm_df


# ============================================================
# MODULE 6: SIMPLE SIGNAL FORWARD-EXPECTATION TEST
# (bucket forward returns by predicted probability)
# ============================================================
def forward_expectation_by_signal_bucket(
    data: pd.DataFrame,
    out_dir: str,
    label_mode: str = "cross_section_median",
    target_period: int = 21,
    lookbacks: Optional[List[int]] = None,
    training_window: int = 252,
    test_window: int = 21,
    step_size: int = UNIV_STEP_SIZE_DEFAULT,
    n_buckets: int = 10,
    precomputed_panel: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
) -> pd.DataFrame:
    """
    For each window:
      - fit multivariate RidgeLogit on train
      - predict prob on test
      - compute realized forward return over target_period
      - bucket by predicted prob (deciles by default)
    Aggregate across windows: average forward return per bucket.
    """
    os.makedirs(out_dir, exist_ok=True)

    lookbacks = lookbacks or [5, 10, 20, 60, 120]
    fe = FeatureEngineer(lookbacks=lookbacks, target_period=target_period, label_mode=label_mode)

    unique_dates = data.index.get_level_values("date").unique().sort_values()
    end_idx = len(unique_dates)
    longest = max(lookbacks)
    start_idx = longest

    if precomputed_panel is not None:
        X_all, y_all = precomputed_panel
    else:
        X_all, y_all, _ = fe.build(data)

    # realized forward return (per asset) computed directly from raw data
    g = data.groupby(level="asset")
    fwd_ret_full = g["Close"].pct_change(target_period).shift(-target_period).rename("fwd_ret")
    fwd_ret = fwd_ret_full.reindex(X_all.index)

    rows = []
    total_windows_est = max(1, (end_idx - start_idx - training_window - test_window) // step_size + 1)
    window_idx = 0
    cur = start_idx
    while cur + training_window + test_window <= end_idx:
        window_idx += 1
        train_dates = unique_dates[cur: cur + training_window]
        test_dates = unique_dates[cur + training_window: cur + training_window + test_window]

        train_mask = X_all.index.get_level_values("date").isin(train_dates)
        test_mask = X_all.index.get_level_values("date").isin(test_dates)

        X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
        X_test = X_all.loc[test_mask]

        if y_train.nunique() < 2 or X_train.empty or X_test.empty:
            cur += step_size
            continue

        fe.fit_scaler(X_train)
        X_train_s = fe.scale(X_train)
        X_test_s = fe.scale(X_test)

        clf = LogisticRegression(C=2.0, max_iter=3000, solver="lbfgs")
        clf.fit(X_train_s, y_train)
        p = clf.predict_proba(X_test_s)[:, 1]

        tmp = pd.DataFrame({"p": p}, index=X_test_s.index).join(fwd_ret, how="left").dropna()
        if tmp.empty:
            cur += step_size
            continue

        # bucket within each DATE to respect cross-sectional nature
        tmp = tmp.reset_index()
        for d, day in tmp.groupby("date"):
            if len(day) < n_buckets:
                continue
            day = day.copy()
            day["bucket"] = pd.qcut(day["p"], q=n_buckets, labels=False, duplicates="drop")
            rows.append(day[["date", "bucket", "fwd_ret"]])

        if window_idx % 20 == 0 or window_idx == total_windows_est:
            print(f"  [bucket] window {window_idx}/{total_windows_est} done", flush=True)

        cur += step_size

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    agg = out.groupby("bucket")["fwd_ret"].agg(["mean", "std", "count"]).reset_index()
    agg["bucket"] = agg["bucket"].astype(int)

    csv_path = os.path.join(out_dir, "signal_forward_expectation_by_bucket.csv")
    xlsx_path = os.path.join(out_dir, "signal_forward_expectation_by_bucket.xlsx")
    agg.to_csv(csv_path, index=False)
    safe_to_excel(agg, xlsx_path, index=False)

    print("✓ Forward expectation by signal bucket saved:")
    print("  -", csv_path)
    print("  -", xlsx_path)
    return agg

# ============================================================
# RUN MODE TOGGLE
# ============================================================
RUN_SMALL_SAMPLE = False   # <-- SET TO True only for fast smoke tests (10 assets, 1000 dates)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Script-relative paths so the script works regardless of the current directory.
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(HERE, "df_2010.csv")
    OUT = os.path.join(HERE, "outputs")
    os.makedirs(OUT, exist_ok=True)

    print("Loading data...")
    data = DataLoader(DATA).load_data()

    # ============================================================
    # OPTIONAL SMALL-SAMPLE DIAGNOSTIC MODE
    # ============================================================
    if RUN_SMALL_SAMPLE:
        print("\n⚠️ RUNNING SMALL-SAMPLE DIAGNOSTIC MODE")

        assets_keep = (
            data.index.get_level_values("asset")
            .unique()
            .sort_values()
            [:10]
        )
        data = data.loc[data.index.get_level_values("asset").isin(assets_keep)]

        # limit to first M dates
        dates_keep = (
            data.index.get_level_values("date")
            .unique()
            .sort_values()[:1000]
        )
        data = data.loc[data.index.get_level_values("date").isin(dates_keep)]

        print(
            f"Diagnostic sample: "
            f"{len(assets_keep)} assets, "
            f"{len(dates_keep)} dates, "
            f"{len(data):,} rows"
        )
    else:
        print("\n▶️ RUNNING FULL DATASET MODE")

    # build the feature panel ONCE here and share it across all three diagnostic
    # functions. Previously each function called fe.build(data) independently, paying
    # the ~30-60s feature-engineering cost three times.
    print("\nBuilding feature panel (shared across all 3 diagnostic stages)...")
    _fe_shared = FeatureEngineer(
        lookbacks=[5, 10, 20, 60, 120],
        target_period=21,
        label_mode="cross_section_median",
    )
    _X_all_shared, _y_all_shared, _ = _fe_shared.build(data)
    print(f"  Panel: {len(_X_all_shared):,} rows x {len(_X_all_shared.columns)} features", flush=True)
    SHARED_PANEL = (_X_all_shared, _y_all_shared)

    # ============================================================
    # 1) Proper OOS univariate indicator tests
    # ============================================================
    print("\nRunning UNIVARIATE OOS walk-forward indicator tests...")
    run_univariate_oos_walkforward(
        data=data,
        out_dir=OUT,
        label_mode="cross_section_median",
        training_window=252,
        test_window=21,
        step_size=UNIV_STEP_SIZE_DEFAULT,
        precomputed_panel=SHARED_PANEL,
    )

    # ============================================================
    # 2) Multivariate model + ensemble + importance
    # ============================================================
    print("\nRunning MULTIVARIATE walk-forward (model metrics + ensemble + importance)...")
    run_multivariate_walkforward_with_importance(
        data=data,
        out_dir=OUT,
        label_mode="cross_section_median",
        training_window=252,
        validation_window=63,
        test_window=21,
        step_size=MULTI_STEP_SIZE_DEFAULT,
        weight_metric="auc",
        n_perm_repeats=N_PERM_REPEATS_DEFAULT,
        precomputed_panel=SHARED_PANEL,
    )

    # ============================================================
    # 3) Signal forward expectation
    # ============================================================
    print("\nRunning SIGNAL forward expectation (bucket analysis)...")
    forward_expectation_by_signal_bucket(
        data=data,
        out_dir=OUT,
        label_mode="cross_section_median",
        training_window=252,
        test_window=21,
        step_size=UNIV_STEP_SIZE_DEFAULT,
        n_buckets=10,
        precomputed_panel=SHARED_PANEL,
    )

    print("\n✓ All required diagnostics completed.")
