# 제출용 predict.py

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

# load libraries (declared functions libraries)
from features import get_all_features

# disable warnings and set logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_and_cache_features(df, args, is_train=False): # Prediction is always on test set
    gin_suffix = "_with_gin" if args.use_gin_features else "_no_gin"
    # Use a different name for prediction features to avoid conflicts
    file_prefix = "predict"
    cache_path = f"cache/{file_prefix}_features{gin_suffix}.pkl"
    
    os.makedirs('cache', exist_ok=True)

    if os.path.exists(cache_path) and not args.force_feature_regen:
        logging.info(f"load features from cache: {cache_path}")
        features = joblib.load(cache_path)
    else:
        logging.info("new features generated and saved...")
        features = get_all_features(df[args.smiles_col], args.use_gin_features)
        joblib.dump(features, cache_path)
        logging.info(f"features saved: {cache_path}")
    return features

def predict(X_test, args):
    test_predictions = []
    
    for fold in range(1, args.n_splits + 1):
        logging.info(f"--- Predicting with Fold {fold}/{args.n_splits} model ---")
        
        # === 모델 및 스케일러 불러오기 ===
        model_path = f"models/catboost_fold_{fold}.cbm"
        scaler_path = f"models/scaler_fold_{fold}.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logging.error(f"Model or scaler for fold {fold} not found! Please run train.py first.")
            raise FileNotFoundError
            
        model = cb.CatBoostRegressor()
        model.load_model(model_path)
        scaler = joblib.load(scaler_path)
        # =================================

        # Predict and inverse transform
        test_preds_scaled = model.predict(X_test)
        test_preds = scaler.inverse_transform(test_preds_scaled)
        test_predictions.append(test_preds)

    # K-Fold 예측 결과의 평균으로 최종 예측값 생성 (앙상블)
    final_predictions = np.mean(test_predictions, axis=0)
    return final_predictions

def main(args):
    logging.info("===== 1. load test data =====")
    # The actual test data path will be provided by the platform
    test_df = pd.read_csv(args.test_path)
    
    logging.info("===== 2. generate/load features for test data =====")
    X_test = get_and_cache_features(test_df, args, is_train=False)

    # Apply saved feature selection if used during training
    if args.feature_selection_method == 'variance':
        selector_path = 'models/variance_selector.pkl'
        if os.path.exists(selector_path):
            logging.info("Applying saved variance threshold selector...")
            selector = joblib.load(selector_path)
            X_test = selector.transform(X_test)
        else:
            logging.error("Variance selector not found. Please run train.py with feature selection.")
            raise FileNotFoundError

    logging.info("===== 3. predict with saved models =====")
    final_predictions = predict(X_test, args)

    logging.info("===== 4. generate submission file =====")
    os.makedirs('submission', exist_ok=True)
    submission_df = pd.read_csv(args.submission_path)
    submission_df[args.target_col] = final_predictions
    submission_df.to_csv("submission/submission.csv", index=False)
    logging.info("'submission/submission.csv' file generated")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Molecule Prediction Inference Pipeline")
    
    # The platform will provide the paths to the test and sample submission files
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--submission_path', type=str, default='data/sample_submission.csv')
    
    # These arguments must match the settings used in train.py
    parser.add_argument('--smiles_col', type=str, default='Canonical_Smiles')
    parser.add_argument('--target_col', type=str, default='Inhibition')
    parser.add_argument('--force_feature_regen', action='store_true', help='ignore cache and regenerate features')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--use_gin_features', action='store_true', help='include GIN features (must match training)')
    parser.add_argument('--feature_selection_method', type=str, default='none', 
                        choices=['none', 'variance'], help='feature selection method (must match training)')

    args = parser.parse_args()
    main(args)