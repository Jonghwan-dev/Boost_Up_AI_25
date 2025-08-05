''' main.py '''

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

def run_feature_selection(X_train, X_test, args):
    # feature selection
    logging.info(f"feature selection before: {X_train.shape[1]}")
    
    # variance-based feature selection
    if args.feature_selection_method in ['variance', 'all']:
        selector = VarianceThreshold(threshold=args.variance_threshold)
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test) # apply same transformation to test set
        logging.info(f"[variance-based] feature selection after: {X_train.shape[1]}")

    # correlation-based feature selection
    if args.feature_selection_method in ['correlation', 'all']:
        X_train_df = pd.DataFrame(X_train)
        corr_matrix = X_train_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_cols = [column for column in upper.columns if any(upper[column] > args.correlation_threshold)]
        
        X_train = X_train_df.drop(columns=to_drop_cols).values
        X_test = pd.DataFrame(X_test).drop(columns=to_drop_cols).values
        logging.info(f"[correlation-based] feature selection after: {X_train.shape[1]}")
    
    return X_train, X_test

def train_model(X, y, X_test, args):
    # K-Fold cross-validation (consider target value distribution for fold splitting)
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_binned = pd.cut(y, bins=num_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    oof_predictions = np.zeros(len(X))
    test_predictions = []
    
    scaler = TargetScaler(use_log=(not args.no_log_transform))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        logging.info(f"--- Fold {fold+1}/{args.n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # scaler is fitted only on train data
        y_train_scaled = scaler.fit(y_train).transform(y_train)
        y_val_scaled = scaler.transform(y_val) # apply same transformation to validation data
        
        model = cb.CatBoostRegressor(**get_default_params(args))
        
        # use eval_set to monitor validation performance and early stopping
        model.fit(
            X_train, y_train_scaled,
            eval_set=[(X_val, y_val_scaled)],
            # early stopping: stop if validation score doesn't improve for 100 rounds
            early_stopping_rounds=100,
            # verbose: print progress every 500 rounds
            verbose=500
        )

        # inverse transform to original scale
        val_preds_scaled = model.predict(X_val)
        oof_predictions[val_idx] = scaler.inverse_transform(val_preds_scaled)
        
        test_preds_scaled = model.predict(X_test)
        test_predictions.append(scaler.inverse_transform(test_preds_scaled))

    # calculate final OOF score and log
    final_oof_score = calculate_leaderboard_score(y, oof_predictions)
    logging.info(f"\n final OOF Leaderboard Score: {final_oof_score:.6f}")
    
    # K-Fold predictions -> final test predictions
    final_test_predictions = np.mean(test_predictions, axis=0)
    return final_test_predictions

    # stable default CatBoost parameters
def get_default_params(args):
    
    return {
        'iterations': 10000, # set sufficiently large for early stopping
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        # evaluation metric for early stopping
        'eval_metric': 'RMSE',
        'random_seed': args.seed,
        'task_type': 'GPU' if args.use_gpu else 'CPU',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8
    }

# main execution function
def main(args):
    logging.info("===== 1. load data =====")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    logging.info("===== 2. generate/load features =====")
    X_train = get_and_cache_features(train_df, args, is_train=True)
    X_test = get_and_cache_features(test_df, args, is_train=False)
    y_train = train_df[args.target_col].values

    logging.info("===== 3. feature selection =====")
    if args.feature_selection_method != 'none':
        X_train, X_test = run_feature_selection(X_train, X_test, args)
    
    logging.info("===== 4. train and predict =====")
    final_predictions = train_model(X_train, y_train, X_test, args)

    logging.info("===== 5. generate submission file =====")
    # submission file name
    gin_suffix = "GIN" if args.use_gin_features else "noGIN"
    log_suffix = "no_log" if args.no_log_transform else "log"
    # submission file generation
    os.makedirs('submission', exist_ok=True)
    submission_df = pd.read_csv(args.submission_path)
    submission_df[args.target_col] = final_predictions
    submission_filename = f"Final_{log_suffix}_{gin_suffix}_FS_{args.feature_selection_method}_submission.csv"
    submission_df.to_csv(f"submission/{submission_filename}", index=False)
    logging.info(f"'{submission_filename}' file generated")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Molecule Prediction Pipeline")
    
    # path and basic settings
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--submission_path', type=str, default='data/sample_submission.csv')
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
                        choices=['none', 'variance', 'correlation', 'all'], help='feature selection method')
    parser.add_argument('--variance_threshold', type=float, default=0.01)
    parser.add_argument('--correlation_threshold', type=float, default=0.95)

    args = parser.parse_args()
    main(args)