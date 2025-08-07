# 제출용 train.py

# load libraries (standard libraries)
import argparse
import logging
import os
import joblib
import warnings

# load libraries (third-party libraries)
import pandas as pd
import numpy as np

# load Model libraries
import catboost as cb

# load libraries (sklearn libraries)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

# load libraries (declared functions libraries)
from utils import calculate_leaderboard_score, TargetScaler
from features import get_all_features

# disable warnings and set logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# get and cache features
def get_and_cache_features(df, args, is_train=True):
    # GIN feature usage -> cache file name
    gin_suffix = "_with_gin" if args.use_gin_features else "_no_gin"
    file_prefix = "train" if is_train else "test"
    cache_path = f"cache/{file_prefix}_features{gin_suffix}.pkl"
    
    os.makedirs('cache', exist_ok=True) # create cache directory

    if os.path.exists(cache_path) and not args.force_feature_regen:
        logging.info(f"load features from cache: {cache_path}")
        features = joblib.load(cache_path)
    else:
        logging.info("new features generated and saved...")
        features = get_all_features(df[args.smiles_col], args.use_gin_features)
        joblib.dump(features, cache_path)
        logging.info(f"features saved: {cache_path}")
    return features

def run_feature_selection(X_train, args):
    # feature selection
    logging.info(f"feature selection before: {X_train.shape[1]}")
    selector = None
    
    # variance-based feature selection
    if args.feature_selection_method in ['variance', 'all']:
        selector = VarianceThreshold(threshold=args.variance_threshold)
        X_train = selector.fit_transform(X_train)
        logging.info(f"[variance-based] feature selection after: {X_train.shape[1]}")
        # Save the selector
        joblib.dump(selector, 'models/variance_selector.pkl')
    
    return X_train


def train_and_save_model(X, y, args):
    # Create directory to save models
    os.makedirs('models', exist_ok=True)

    # K-Fold cross-validation
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_binned = pd.cut(y, bins=num_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    oof_predictions = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        logging.info(f"--- Fold {fold+1}/{args.n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = TargetScaler(use_log=(not args.no_log_transform))
        y_train_scaled = scaler.fit(y_train).transform(y_train)
        y_val_scaled = scaler.transform(y_val)
        
        model = cb.CatBoostRegressor(**get_default_params(args))
        
        model.fit(
            X_train, y_train_scaled,
            eval_set=[(X_val, y_val_scaled)],
            early_stopping_rounds=100,
            verbose=500
        )

        val_preds_scaled = model.predict(X_val)
        oof_predictions[val_idx] = scaler.inverse_transform(val_preds_scaled)
        
        model_path = f"models/catboost_fold_{fold+1}.cbm"
        scaler_path = f"models/scaler_fold_{fold+1}.pkl"
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Scaler saved to {scaler_path}")

    final_oof_score = calculate_leaderboard_score(y, oof_predictions)
    logging.info(f"\n final OOF Leaderboard Score: {final_oof_score:.6f}")
    
def get_default_params(args):
    return {
        'iterations': 10000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': args.seed,
        'task_type': 'GPU' if args.use_gpu else 'CPU',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8
    }

def main(args):
    logging.info("===== 1. load data =====")
    train_df = pd.read_csv(args.train_path)
    
    logging.info("===== 2. generate/load features =====")
    X_train = get_and_cache_features(train_df, args, is_train=True)
    y_train = train_df[args.target_col].values

    logging.info("===== 3. (Optional) feature selection =====")
    if args.feature_selection_method != 'none':
        # Note: Feature selection should be handled carefully for prediction consistency.
        # For simplicity, this example saves only the variance selector.
        X_train = run_feature_selection(X_train, args)
    
    logging.info("===== 4. train and save models =====")
    train_and_save_model(X_train, y_train, args)

    logging.info("===== 5. Training complete =====")
    logging.info("Models and scalers for each fold are saved in the 'models/' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Molecule Prediction Training Pipeline")
    
    # path and basic settings
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--smiles_col', type=str, default='Canonical_Smiles')
    parser.add_argument('--target_col', type=str, default='Inhibition')
    parser.add_argument('--force_feature_regen', action='store_true', help='ignore cache and regenerate features')

    # model and training settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--use_gpu', action='store_true', help='use GPU')

    # preprocessing and feature selection control
    parser.add_argument('--no_log_transform', action='store_true', help='disable target log transformation')
    parser.add_argument('--use_gin_features', action='store_true', help='include GIN features')
    parser.add_argument('--feature_selection_method', type=str, default='none', 
                        choices=['none', 'variance'], help='feature selection method')
    parser.add_argument('--variance_threshold', type=float, default=0.01)

    args = parser.parse_args()
    main(args)