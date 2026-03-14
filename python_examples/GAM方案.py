# -------------------------------
# *.Library Imports 
# -------------------------------
import optuna
import numpy as np
from optuna.samplers import TPESampler
from pygam import LinearGAM, ExpectileGAM, s
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit


# -------------------------------
# 1.Hyperparameter Tuning
# -------------------------------
def get_mean_mse(X, y, model_type, params, n_splits=6, timeseries=True, random_state=42):
    if timeseries:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mse_list = []
    for tr_idx, va_idx in cv.split(X):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        gam = make_gam(Xtr, ytr, model_type, params)
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


# -------------------------------
# 2.GAM Model
# -------------------------------
def make_gam(X_train, y_train, model_type, params):
    # hyperparameters for each factor
    n_splines_1    = int(params["n_splines_1"])
    n_splines_2    = int(params["n_splines_2"])
    n_splines_3    = int(params["n_splines_3"])
    spline_order_1 = int(params["spline_order_1"])
    spline_order_2 = int(params["spline_order_2"])
    spline_order_3 = int(params["spline_order_3"])

    # construct smooth term
    terms = (
        s(0, n_splines=n_splines_1, spline_order=spline_order_1) +
        s(1, n_splines=n_splines_2, spline_order=spline_order_2) +
        s(2, n_splines=n_splines_3, spline_order=spline_order_3)
    )

    # Regularization for each term
    lam_1 = float(params["lam_1"])
    lam_2 = float(params["lam_2"])
    lam_3 = float(params["lam_3"])
    lam_vec = [lam_1, lam_2, lam_3]

    if model_type == "Gaussian":
        gam = LinearGAM(terms, lam=lam_vec)
    elif model_type == "Expectile":
        if "expectile" not in params:
            raise ValueError("ExpectileGAM must provide expectile.")
        expectile = float(params["expectile"])
        gam = ExpectileGAM(terms, lam=lam_vec, expectile=expectile)
    else:
        raise ValueError("model_type must be either 'Gaussian' or 'Expectile'.")

    gam.fit(X_train, y_train)
    return gam


# -------------------------------
# 3.Model Training and Prediction
# -------------------------------
# Gaussian GAM
Gaussian_param_ranges = {
    "n_splines_1":    [12, 16, 20, 24, 28, 32, 36],
    "n_splines_2":    [12, 16, 20, 24, 28, 32, 36],
    "n_splines_3":    [12, 16, 20, 24, 28, 32, 36],
    "spline_order_1": [2, 3, 4, 5],
    "spline_order_2": [2, 3, 4, 5],
    "spline_order_3": [2, 3, 4, 5],
    "lam_1":          np.logspace(-6, -2, 13),
    "lam_2":          np.logspace(-6, -2, 13),
    "lam_3":          np.logspace(-6, -2, 13),
}

best_Gaussian_amzn_params = run_optuna_tpe_search(amzn_X_train_s, amzn_y_train, "Gaussian", Gaussian_param_ranges)
Gaussian_gam_amzn_model = make_gam(amzn_X_train_s, amzn_y_train, "Gaussian", best_Gaussian_amzn_params)
Gaussian_gam_amzn_pred = Gaussian_gam_amzn_model.predict(amzn_X_test_s)

best_Gaussian_aapl_params = run_optuna_tpe_search(aapl_X_train_s, aapl_y_train, "Gaussian", Gaussian_param_ranges)
Gaussian_gam_aapl_model = make_gam(aapl_X_train_s, aapl_y_train, "Gaussian", best_Gaussian_aapl_params)
Gaussian_gam_aapl_pred = Gaussian_gam_aapl_model.predict(aapl_X_test_s)

best_Gaussian_m_params = run_optuna_tpe_search(m_X_train_s, m_y_train, "Gaussian", Gaussian_param_ranges)
Gaussian_gam_m_model = make_gam(m_X_train_s, m_y_train, "Gaussian", best_Gaussian_m_params)
Gaussian_gam_m_pred = Gaussian_gam_m_model.predict(m_X_test_s)

# Expectile GAM
Expectile_param_ranges = {
    "n_splines_1":    [12, 16, 20, 24, 28, 32, 36],
    "n_splines_2":    [12, 16, 20, 24, 28, 32, 36],
    "n_splines_3":    [12, 16, 20, 24, 28, 32, 36],
    "spline_order_1": [2, 3, 4, 5],
    "spline_order_2": [2, 3, 4, 5],
    "spline_order_3": [2, 3, 4, 5],
    "lam_1":          np.logspace(-6, -2, 13),
    "lam_2":          np.logspace(-6, -2, 13),
    "lam_3":          np.logspace(-6, -2, 13),
    "expectile":    [0.1, 0.3, 0.5, 0.7, 0.9]
}

best_Expectile_amzn_params = run_optuna_tpe_search(amzn_X_train_s, amzn_y_train, "Expectile", Expectile_param_ranges)
Expectile_gam_amzn_model = make_gam(amzn_X_train_s, amzn_y_train, "Expectile", best_Expectile_amzn_params)
Expectile_gam_amzn_pred = Expectile_gam_amzn_model.predict(amzn_X_test_s)

best_Expectile_aapl_params = run_optuna_tpe_search(aapl_X_train_s, aapl_y_train, "Expectile", Expectile_param_ranges)
Expectile_gam_aapl_model = make_gam(aapl_X_train_s, aapl_y_train, "Expectile", best_Expectile_aapl_params)
Expectile_gam_aapl_pred = Expectile_gam_aapl_model.predict(aapl_X_test_s)

best_Expectile_m_params = run_optuna_tpe_search(m_X_train_s, m_y_train, "Expectile", Expectile_param_ranges)
Expectile_gam_m_model = make_gam(m_X_train_s, m_y_train, "Expectile", best_Expectile_m_params)
Expectile_gam_m_pred = Expectile_gam_m_model.predict(m_X_test_s)