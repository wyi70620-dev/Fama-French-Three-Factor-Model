# === * Library Imports ===
import pandas as pd
import numpy as np
import os
from pygam import LinearGAM, ExpectileGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler


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
def get_mean_mse(X, y, model_type, params, n_splits=6, timeseries=True, random_state=42):
    if timeseries:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_splines    = int(params["n_splines"])
    spline_order = int(params["spline_order"])
    lam          = float(params["lam"])
    expectile    = params.get("expectile", None)

    mse_list = []
    for tr_idx, va_idx in cv.split(X):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        gam = make_gam(Xtr, ytr, model_type, n_splines, spline_order, lam, expectile)
        pred = gam.predict(Xva)
        mse = mean_squared_error(yva, pred)
        mse_list.append(mse)

    return float(np.mean(mse_list))

def run_optuna_tpe_search(X, y, model_type, param_ranges, n_jobs=-1, n_splits=6, timeseries=True, n_trials=None):
    total_combos = 1
    for k in param_ranges:
        total_combos *= max(1, len(param_ranges[k]))
    if n_trials is None:
        n_trials = int(min(total_combos, 250))

    sampler = TPESampler(seed=42, multivariate=False, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=None)

    def objective(trial):
        params = {}
        for key, values in param_ranges.items():
            values_list = list(values) if not np.isscalar(values) else [values]
            params[key] = trial.suggest_categorical(key, values_list)

        cv_score = get_mean_mse(
            X=X,
            y=y,
            model_type=model_type,
            params=params,
            n_splits=n_splits,
            timeseries=timeseries
        )
        return cv_score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)

    best_params = study.best_trial.params.copy()
    return best_params


# === 3. Model Training and Prediction ===
def make_gam(X_train, y_train, model_type, n_splines, spline_order, lam, expectile=None):
    n_splines    = int(n_splines)
    spline_order = int(spline_order)
    
    terms = s(0, n_splines=n_splines, spline_order=spline_order) + s(1, n_splines=n_splines, spline_order=spline_order) + s(2, n_splines=n_splines, spline_order=spline_order)
    
    if model_type == "Gaussian":
        gam = LinearGAM(terms, lam=float(lam))
    elif model_type == "Expectile":
        if expectile is None:
            raise ValueError("ExpectileGAM must provide expectile.")
        gam = ExpectileGAM(terms, lam=float(lam), expectile=float(expectile))
    else:
        raise ValueError("model_type must be either 'Gaussian' or 'Expectile'.")

    gam.fit(X_train, y_train)
    return gam

# Gaussian GAM
Gaussian_gam_amzn_model = make_gam(amzn_X_train_s, amzn_y_train, "Gaussian", 12, 5, 0.01)
Gaussian_gam_amzn_pred = Gaussian_gam_amzn_model.predict(amzn_X_test_s)

Gaussian_gam_aapl_model = make_gam(aapl_X_train_s, aapl_y_train, "Gaussian", 12, 5, 0.01)
Gaussian_gam_aapl_pred = Gaussian_gam_aapl_model.predict(aapl_X_test_s)

Gaussian_gam_m_model = make_gam(m_X_train_s, m_y_train, "Gaussian", 12, 5, 0.01)
Gaussian_gam_m_pred = Gaussian_gam_m_model.predict(m_X_test_s)

# Expectile GAM
Expectile_gam_amzn_model = make_gam(amzn_X_train_s, amzn_y_train, "Expectile", 12, 5, 0.01, 0.1)
Expectile_gam_amzn_pred = Expectile_gam_amzn_model.predict(amzn_X_test_s)

Expectile_gam_aapl_model = make_gam(aapl_X_train_s, aapl_y_train, "Expectile", 12, 5, 0.01, 0.1)
Expectile_gam_aapl_pred = Expectile_gam_aapl_model.predict(aapl_X_test_s)

Expectile_gam_m_model = make_gam(m_X_train_s, m_y_train, "Expectile", 12, 5, 0.01, 0.1)
Expectile_gam_m_pred = Expectile_gam_m_model.predict(m_X_test_s)


# === 6. Save Results ===
def get_test_dates(df, test_ratio=0.2, date_col='Date'):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    return df.loc[split_index:, date_col].values


desktop_path = os.path.expanduser("~/Desktop")
predictions_df = pd.DataFrame({
    "Date_AMZN": get_test_dates(amzn_df),
    "y_true_AMZN": amzn_y_test,
    "Gaussian_AMZN": Gaussian_gam_amzn_pred,
    "Expectile_AMZN": Expectile_gam_amzn_pred,

    "Date_AAPL": get_test_dates(aapl_df),
    "y_true_AAPL": aapl_y_test,
    "Gaussian_AAPL": Gaussian_gam_aapl_pred,
    "Expectile_AAPL": Expectile_gam_aapl_pred,

    "Date_M": get_test_dates(m_df),
    "y_true_M": m_y_test,
    "Gaussian_m": Gaussian_gam_m_pred,
    "Expectile_m": Expectile_gam_m_pred,
})

save_path = os.path.join(desktop_path, "stock_predictions3.csv")
predictions_df.to_csv(save_path, index=False)
print(f"预测结果已保存到: {save_path}")