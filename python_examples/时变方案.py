# -------------------------------
# *.Library Imports 
# -------------------------------
import optuna
import numpy as np
from optuna.samplers import TPESampler
from filterpy.kalman import KalmanFilter


# -------------------------------
# 1.Define an auxiliary function
# -------------------------------
def build_H(X):
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def build_F(k, state_model="RW", phi=None):
    if state_model == "RW":
        F = np.eye(k)
    elif state_model == "AR1":
        if phi is None:
            raise ValueError("state_model='AR1' must provide phi")
        if abs(float(phi)) >= 1.0:
            raise ValueError("state_model='AR1' must have abs(phi) < 1.0")
        F = float(phi) * np.eye(k)
    else:
        raise ValueError("Undefined state_model")
    return F

def initialization(k, state_model, Q_mat, phi, x0, P0):
    if x0 is None:
        x0_vec = np.zeros((k, 1))
    else:
        x0_vec = np.asarray(x0, dtype=float).reshape(k, 1)

    if P0 is None:
        if state_model == "RW":
            P0_mat = np.eye(k) * 1000.0
        elif state_model == "AR1":
            if phi is None:
                raise ValueError("state_model='AR1' must provide phi")
            if abs(float(phi)) >= 1.0:
                raise ValueError("state_model='AR1' must have abs(phi) < 1.0")          
            P0_mat = np.asarray(Q_mat, dtype=float) / (1.0 - phi**2)
        else:
            raise ValueError("Undefined state_model")
    else:
        P0_mat = np.asarray(P0, dtype=float)
        assert P0_mat.shape == (k, k), "P0 must be a (k, k) matrix"

    return x0_vec, P0_mat


# -------------------------------
# 2.Hyperparameter Tuning
# -------------------------------
def get_kf_loglik(y, H_rows, q, r, state_model="RW", phi=None, x0=None, P0=None, eps=1e-9):
    y = np.asarray(y, dtype=float).ravel()
    H_rows = np.asarray(H_rows, dtype=float)
    T, k = H_rows.shape
    F = build_F(k, state_model=state_model, phi=phi)
    Q = q * np.eye(k)
    R = float(r)
    assert y.shape[0] == T, "y and H_rows must have the same number of periods T"
    assert np.isscalar(q) and np.isscalar(r), "q and r must have to be scalar"
    assert r > 0 and q >= 0, "r and q must have to be: r > 0 and q >= 0"

    x0_vec, P0_mat = initialization(k, state_model=state_model, Q_mat=Q, phi=phi, x0=x0, P0=P0)
        
    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.F = F
    kf.Q = Q
    kf.R = R
    kf.x = x0_vec
    kf.P = P0_mat
        
    loglik = 0.0
    log2pi = np.log(2.0 * np.pi)

    for t in range(T):
        kf.predict()
        H_t = H_rows[t].reshape(1, k)
        z_t = float(y[t])
        kf.update(z_t, H=H_t)
        v_t = float(kf.y.item())
        S_t = float(kf.S.item())

        if S_t <= 0:
            S_t = eps

        loglik += -0.5 * (log2pi + np.log(S_t) + (v_t * v_t) / S_t)

    return float(loglik)

def run_optuna_tpe_search(y, H_rows, param_ranges, state_model="RW", x0=None, P0=None, n_jobs=-1, n_trials=None):
    total_combos = 1
    for k in param_ranges:
        total_combos *= max(1, len(param_ranges[k]))
    if n_trials is None:
        n_trials = int(min(total_combos, 250))
        
    sampler = TPESampler(seed=42, multivariate=False, n_startup_trials=min(10, n_trials))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = {}
        for key, values in param_ranges.items():
            values_list = list(values) if not np.isscalar(values) else [values]
            params[key] = trial.suggest_categorical(key, values_list)

        q = float(params["q"])
        r = float(params["r"])
        phi = params.get("phi", None)

        ll = get_kf_loglik(
            y, H_rows,
            q=q, r=r,
            state_model=state_model,
            phi=phi,
            x0=x0, P0=P0
        )
        return ll

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
    
    best_params = study.best_trial.params.copy()
    return best_params


# -------------------------------
# 3.kalman_methods
# -------------------------------
def kalman_methods(y, H_rows, Q, R, state_model="RW", phi=None, x0=None, P0=None):
    # Normativity and Examination
    y = np.asarray(y, dtype=float).ravel()
    H_rows = np.asarray(H_rows, dtype=float)
    T, k = H_rows.shape
    F = build_F(k, state_model=state_model, phi=phi)
    assert y.shape[0] == T, "y and H_rows must have the same number of periods T"
    
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 0:
        Q_mat = float(Q) * np.eye(k)
    elif Q.ndim == 1:
        assert Q.shape[0] == k, "Length of Q’s diagonal vector must equal the state dimension k"
        Q_mat = np.diag(Q)
    else:
        assert Q.shape == (k, k), "Q must be a (k, k) matrix"
        Q_mat = Q

    if np.isscalar(R):
        R_mat = np.array([[float(R)]], dtype=float)
    else:
        R_mat = np.asarray(R, dtype=float)
        assert R_mat.shape == (1, 1), "R must be a scalar or a (1, 1) matrix"
    
    # Initialization
    x0_vec, P0_mat = initialization(k, state_model=state_model, Q_mat=Q_mat, phi=phi, x0=x0, P0=P0)

    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.F = F
    kf.Q = Q_mat
    kf.R = R_mat
    kf.x = x0_vec
    kf.P = P0_mat
    xs_filt = np.zeros((T, k))
    Ps_filt = np.zeros((T, k, k))
    xs_pred = np.zeros((T, k))
    Ps_pred = np.zeros((T, k, k))
    
    # Kalman Filter
    for t in range(T):
        # Prediction
        kf.predict()
        xs_pred[t] = kf.x.ravel()
        Ps_pred[t] = kf.P
        
        # Update
        H_t = H_rows[t][None, :]
        z_t = y[t]
        kf.update(z_t, H=H_t)
        xs_filt[t] = kf.x.ravel()
        Ps_filt[t] = kf.P
    
    # Kalman Smoother
    xs_smooth = np.zeros_like(xs_filt)
    Ps_smooth = np.zeros_like(Ps_filt)
    xs_smooth[-1] = xs_filt[-1]
    Ps_smooth[-1] = Ps_filt[-1]
    F = kf.F
    
    for t in range(T-2, -1, -1):
        Pt_f = Ps_filt[t]
        Pt1_p = Ps_pred[t+1]
        C_t = Pt_f @ F.T @ np.linalg.inv(Pt1_p)
        x_update = xs_smooth[t+1] - xs_pred[t+1]
        xs_smooth[t] = xs_filt[t] + (C_t @ x_update)
        P_update = Ps_smooth[t+1] - Pt1_p
        Ps_smooth[t] = Pt_f + C_t @ P_update @ C_t.T
    
    # The last time point (as the prediction result)
    x_last = xs_smooth[-1].copy()
    P_last = Ps_smooth[-1].copy()
    
    return {
        'x_last': x_last,
        'P_last': P_last,
        'F': F
    }


# -------------------------------
# 4.Model Training and Prediction
# -------------------------------
def kf_train(X_train, y_train, param_ranges, state_model="RW", x0=None, P0=None):
    H_train = build_H(X_train)
    best_param = run_optuna_tpe_search(y_train, H_train, param_ranges, state_model=state_model, x0=x0, P0=P0, n_jobs=-1)
    Q, R = float(best_param['q']), float(best_param['r'])
    phi = best_param.get('phi', None)

    result = kalman_methods(y_train, H_train, Q, R, state_model=state_model, phi=phi, x0=x0, P0=P0)
    model = {
        "k": H_train.shape[1],
        "Q": Q, "R": R,
        "F": result["F"].copy(),
        "coefficient": result["x_last"].copy(),
        "covariance": result["P_last"].copy()
    }
    return model

def kf_predict(X_test, y_test, model, update_with_observed=False):
    H_test = build_H(X_test)
    k = model["k"]
    q, r = model["Q"], model["R"]
    F = model["F"]
    x = model["coefficient"].reshape(-1, 1).copy()
    P = model["covariance"].copy()
    y_pred = np.zeros(H_test.shape[0])

    if not update_with_observed:
        y_pred = (H_test @ x).ravel()
        return y_pred
    else:
        kf = KalmanFilter(dim_x=k, dim_z=1)
        kf.F = F
        kf.Q = q * np.eye(k)
        kf.R = float(r)
        kf.x = x
        kf.P = P

        for t in range(H_test.shape[0]):
            H_t = H_test[t:t+1, :]
            if y_test is None:
                raise ValueError("update_with_observed=True must provide y_test")
            if t % 10 == 0:  
                kf.update(float(y_test[t]), H=H_t)
            y_pred[t] = float((H_t @ kf.x).item())
            if t < H_test.shape[0] - 1:
                kf.predict()
        return y_pred

# Random-Walk
RW_param_ranges = {
    "q": np.logspace(-6, -2, 13),
    "r": np.logspace(-6, -2, 13)
}

RW_amzn_model = kf_train(amzn_X_train_s, amzn_y_train, RW_param_ranges, state_model="RW")
RW_amzn_pred = kf_predict(amzn_X_test_s, amzn_y_test, RW_amzn_model)

RW_aapl_model = kf_train(aapl_X_train_s, aapl_y_train, RW_param_ranges, state_model="RW")
RW_aapl_pred = kf_predict(aapl_X_test_s, aapl_y_test, RW_aapl_model)

RW_m_model = kf_train(m_X_train_s, m_y_train, RW_param_ranges, state_model="RW")
RW_m_pred = kf_predict(m_X_test_s, m_y_test, RW_m_model)

# Auto-Regressive(order 1)
AR_param_ranges = {
    "q": np.logspace(-6, -2, 13),
    "r": np.logspace(-6, -2, 13),
    "phi": list(np.arange(0.01, 1, 0.01))
}

AR_amzn_model = kf_train(amzn_X_train_s, amzn_y_train, AR_param_ranges, state_model="AR1")
AR_amzn_pred = kf_predict(amzn_X_test_s, amzn_y_test, AR_amzn_model)

AR_aapl_model = kf_train(aapl_X_train_s, aapl_y_train, AR_param_ranges, state_model="AR1")
AR_aapl_pred = kf_predict(aapl_X_test_s, aapl_y_test, AR_aapl_model)

AR_m_model = kf_train(m_X_train_s, m_y_train, AR_param_ranges, state_model="AR1")
AR_m_pred = kf_predict(m_X_test_S, m_y_test, AR_m_model)