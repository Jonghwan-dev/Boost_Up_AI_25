# utils.py

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Leaderboard Score Calculator
def calculate_leaderboard_score(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = np.max(y_true) - np.min(y_true)
    nrmse_A = rmse / y_range if y_range != 0 else 0

    # Pearson Correlation (If std is 0, return 0)
    if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        correlation_B = 0.0
    else:
        correlation_B = pearsonr(y_true, y_pred)[0]
        # if correlation is nan, set to 0
        if np.isnan(correlation_B):
            correlation_B = 0.0

    # Leaderboard Score
    score = 0.5 * (1 - min(nrmse_A, 1)) + 0.5 * correlation_B
    return score

# Target Scaler for log transformation
class TargetScaler:

    def __init__(self, use_log=True):
        self.use_log = use_log
        self.offset = 0

    # fit step: calculate offset for log transformation
    def fit(self, y):
        
        if self.use_log:
            min_val = np.min(y)

            # if min_val is negative, add 1 to avoid log(0)
            self.offset = abs(min_val) + 1 if min_val <= 0 else 0
        return self

    # transform function: log transformation
    def transform(self, y):
        if not self.use_log:
            return y
        return np.log1p(y + self.offset)

    # inverse_transform function: inverse log transformation
    def inverse_transform(self, y_scaled):

        if not self.use_log:
            return y_scaled

        # if y_scaled is negative, return 0
        return np.maximum(0, np.expm1(y_scaled) - self.offset)
