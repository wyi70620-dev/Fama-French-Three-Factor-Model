# === * Library Imports ===
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from joblib import Parallel, delayed
from itertools import product


# === 1. Data Loading and preprocessing ===
def preprocess_stock_data(stock_filename, factors_df):
    stock_df = pd.read_csv(stock_filename)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y-%m')
    merged_df = pd.merge(stock_df, factors_df, on='Date', how='inner')
    merged_df['Excess Return'] = merged_df['Monthly Return'] - merged_df['RF']/100
    merged_df = merged_df.rename(columns={
        'Mkt-RF': 'x1',
        'SMB': 'x2',
        'HML': 'x3'
    })
    return merged_df[['Date', 'Excess Return', 'x1', 'x2', 'x3']]

factors_df = pd.read_csv("~/Desktop/Fama_French_3_Factors_Monthly.csv")
factors_df['Date'] = pd.to_datetime(factors_df['Date'], format='%Y-%m')
amzn_df = preprocess_stock_data("~/Desktop/AMZN.csv", factors_df)
aapl_df = preprocess_stock_data("~/Desktop/AAPL.csv", factors_df)
m_df    = preprocess_stock_data("~/Desktop/M.csv", factors_df)

def train_test_split_time(df, test_ratio=0.2, date_col='Date'):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = np.asarray(train_df[['x1', 'x2', 'x3']])
    y_train = np.asarray(train_df['Excess Return'])
    X_test = np.asarray(test_df[['x1', 'x2', 'x3']])
    y_test = np.asarray(test_df['Excess Return'])

    return X_train, y_train, X_test, y_test

amzn_X_train, amzn_y_train, amzn_X_test, amzn_y_test = train_test_split_time(amzn_df)
aapl_X_train, aapl_y_train, aapl_X_test, aapl_y_test = train_test_split_time(aapl_df)
m_X_train, m_y_train, m_X_test, m_y_test = train_test_split_time(m_df)

def standardize_train_test(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s

amzn_X_train_s, amzn_X_test_s = standardize_train_test(amzn_X_train, amzn_X_test)
aapl_X_train_s, aapl_X_test_s = standardize_train_test(aapl_X_train, aapl_X_test)
m_X_train_s, m_X_test_s = standardize_train_test(m_X_train, m_X_test)


# === 2. Hyperparameter Tuning ===
def get_mean_mse(model_class, X, Y, param_dict, n_splits=10, scale=True, timeseries=True):
    if timeseries:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    for train_idx, val_idx in cv.split(X):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        if scale:
            model = make_pipeline(StandardScaler(), model_class(**param_dict))
        else:
            model = model_class(**param_dict)
        model.fit(X_train, Y_train)
        preds = model.predict(X_val)
        mse_list.append(mean_squared_error(Y_val, preds))
    return float(np.mean(mse_list))


def run_model_grid_search(model_class, X, y, param_ranges, n_jobs=-1, n_splits=10, scale=True, timeseries=True):
    keys = list(param_ranges.keys())
    combinations = list(product(*[param_ranges[k] for k in keys]))
    param_grid = pd.DataFrame(combinations, columns=keys)

    def evaluate(row):
        params = row.to_dict()
        cv_mse = get_mean_mse(model_class, X, y, params, n_splits=n_splits, scale=scale, timeseries=timeseries)
        return cv_mse

    cv_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(row) for _, row in param_grid.iterrows()
    )
    param_grid['CV_MSE'] = cv_results
    best_row = param_grid.loc[param_grid['CV_MSE'].idxmin()]
    return best_row.to_dict()


# === 3. Model Training and Prediction ===
def train_and_predict(model_class, param_dict, X_train, y_train, X_test):
    model = model_class(**param_dict)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_predict_baseline(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# OLS
ols_pred_amzn = train_and_predict_baseline(amzn_X_train_s, amzn_y_train, amzn_X_test_s)
ols_pred_aapl = train_and_predict_baseline(aapl_X_train_s, aapl_y_train, aapl_X_test_s)
ols_pred_m = train_and_predict_baseline(m_X_train_s, m_y_train, m_X_test_s)

# Huber regression + L2
'''
huber_param_range = {
    'epsilon': list(np.arange(1, 10, 0.01)),
    'alpha': list(np.arange(10, 200, 0.1))
}

best_amzn_huber_params = run_model_grid_search(HuberRegressor, amzn_X_train, amzn_y_train, huber_param_range)
best_aapl_huber_params = run_model_grid_search(HuberRegressor, aapl_X_train, aapl_y_train, huber_param_range)
best_m_huber_params = run_model_grid_search(HuberRegressor, m_X_train, m_y_train, huber_param_range)
'''
best_amzn_huber_params = {'epsilon': 3.12, 'alpha': 87.8}
best_aapl_huber_params = {'epsilon': 3.56, 'alpha': 56.4}
best_m_huber_params = {'epsilon': 8.54, 'alpha': 152.3}

huber_pred_amzn = train_and_predict(HuberRegressor, best_amzn_huber_params, amzn_X_train_s, amzn_y_train, amzn_X_test_s)
huber_pred_aapl = train_and_predict(HuberRegressor, best_aapl_huber_params, aapl_X_train_s, aapl_y_train, aapl_X_test_s)
huber_pred_m = train_and_predict(HuberRegressor, best_m_huber_params, m_X_train_s, m_y_train, m_X_test_s)

# quantile regression + L2
'''
quantile_param_range = {
    'quantile': list(np.arange(0.1, 0.9, 0.01)),
    'alpha': list(np.arange(0.01, 0.09, 0.001))
}

best_amzn_quantile_params = run_model_grid_search(QuantileRegressor, amzn_X_train, amzn_y_train, quantile_param_range)
best_aapl_quantile_params = run_model_grid_search(QuantileRegressor, aapl_X_train, aapl_y_train, quantile_param_range)
best_m_quantile_params = run_model_grid_search(QuantileRegressor, m_X_train, m_y_train, quantile_param_range)
'''
best_amzn_quantile_params =  {'quantile': 0.4, 'alpha': 0.0239}
best_aapl_quantile_params =  {'quantile': 0.41, 'alpha': 0.0069}
best_m_quantile_params =  {'quantile': 0.43, 'alpha': 0.03}

quantile_pred_amzn = train_and_predict(QuantileRegressor, best_amzn_quantile_params, amzn_X_train_s, amzn_y_train, amzn_X_test_s)
quantile_pred_aapl = train_and_predict(QuantileRegressor, best_aapl_quantile_params, aapl_X_train_s, aapl_y_train, aapl_X_test_s)
quantile_pred_m = train_and_predict(QuantileRegressor, best_m_quantile_params, m_X_train_s, m_y_train, m_X_test_s)


# === 4. Save Results ===
def get_test_dates(df, test_ratio=0.2, date_col='Date'):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    return df.loc[split_index:, date_col].values


desktop_path = os.path.expanduser("~/Desktop")
predictions_df = pd.DataFrame({
    "Date_AMZN": get_test_dates(amzn_df),
    "y_true_AMZN": amzn_y_test,
    "OLS_AMZN": ols_pred_amzn,
    "Huber_AMZN": huber_pred_amzn,
    "Quantile_AMZN": quantile_pred_amzn,

    "Date_AAPL": get_test_dates(aapl_df),
    "y_true_AAPL": aapl_y_test,
    "OLS_AAPL": ols_pred_aapl,
    "Huber_AAPL": huber_pred_aapl,
    "Quantile_AAPL": quantile_pred_aapl,

    "Date_M": get_test_dates(m_df),
    "y_true_M": m_y_test,
    "OLS_M": ols_pred_m,
    "Huber_M": huber_pred_m,
    "Quantile_M": quantile_pred_m
})

save_path = os.path.join(desktop_path, "stock_predictions.csv")
predictions_df.to_csv(save_path, index=False)
print(f"预测结果已保存到: {save_path}")
