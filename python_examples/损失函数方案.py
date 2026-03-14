# -------------------------------
# *.Library Imports 
# -------------------------------
import optuna
import numpy as np
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor


# -------------------------------
# 1.Hyperparameter Tuning
# -------------------------------
def get_mean_mse(model_class, X, Y, param_dict, n_splits=10, timeseries=True):
    if timeseries:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mse_list = []
    for train_idx, val_idx in cv.split(X):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        model = model_class(**param_dict)
        model.fit(X_train, Y_train)
        preds = model.predict(X_val)
        mse_list.append(mean_squared_error(Y_val, preds))

    return float(np.mean(mse_list))

def run_optuna_tpe_search(model_class, X, y, param_ranges, n_jobs=-1, n_splits=10, timeseries=True, n_trials=None):
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

        cv_mse = get_mean_mse(
            model_class=model_class,
            X=X,
            Y=y,
            param_dict=params,
            n_splits=n_splits,
            timeseries=timeseries
        )
        return cv_mse

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)

    best_params = study.best_trial.params.copy()
    return best_params


# -------------------------------
# 2.Model Training and Prediction
# -------------------------------
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

# ---------- OLS ----------
ols_pred_amzn = train_and_predict_baseline(amzn_X_train_s, amzn_y_train, amzn_X_test_s)
ols_pred_aapl = train_and_predict_baseline(aapl_X_train_s, aapl_y_train, aapl_X_test_s)
ols_pred_m = train_and_predict_baseline(m_X_train_s, m_y_train, m_X_test_s)

# ---------- Huber regression + L2 ----------
huber_param_range = {
    'epsilon': list(np.arange(1, 10, 0.01)),
    'alpha': np.logspace(-3, 3, 20)
}

best_amzn_huber_params = run_optuna_tpe_search(HuberRegressor, amzn_X_train_s, amzn_y_train, huber_param_range)
best_aapl_huber_params = run_optuna_tpe_search(HuberRegressor, aapl_X_train_s, aapl_y_train, huber_param_range)
best_m_huber_params = run_optuna_tpe_search(HuberRegressor, m_X_train_s, m_y_train, huber_param_range)

huber_pred_amzn = train_and_predict(HuberRegressor, best_amzn_huber_params, amzn_X_train_s, amzn_y_train, amzn_X_test_s)
huber_pred_aapl = train_and_predict(HuberRegressor, best_aapl_huber_params, aapl_X_train_s, aapl_y_train, aapl_X_test_s)
huber_pred_m = train_and_predict(HuberRegressor, best_m_huber_params, m_X_train_s, m_y_train, m_X_test_s)

# ---------- quantile regression + L2 ----------
quantile_param_range = {
    'quantile': list(np.arange(0.01, 1, 0.01)),
    'alpha': np.logspace(-3, 3, 20)
}

best_amzn_quantile_params = run_optuna_tpe_search(QuantileRegressor, amzn_X_train_s, amzn_y_train, quantile_param_range)
best_aapl_quantile_params = run_optuna_tpe_search(QuantileRegressor, aapl_X_train_s, aapl_y_train, quantile_param_range)
best_m_quantile_params = run_optuna_tpe_search(QuantileRegressor, m_X_train_s, m_y_train, quantile_param_range)

quantile_pred_amzn = train_and_predict(QuantileRegressor, best_amzn_quantile_params, amzn_X_train_s, amzn_y_train, amzn_X_test_s)
quantile_pred_aapl = train_and_predict(QuantileRegressor, best_aapl_quantile_params, aapl_X_train_s, aapl_y_train, aapl_X_test_s)
quantile_pred_m = train_and_predict(QuantileRegressor, best_m_quantile_params, m_X_train_s, m_y_train, m_X_test_s)