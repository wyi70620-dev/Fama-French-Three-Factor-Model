# -------------------------------
# *.Library Imports 
# -------------------------------
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.nonparametric.smoothers_lowess import lowess


# -------------------------------
# 1.Data distribution analysis
# -------------------------------
def analyze_distribution(data, value_col="Excess Return", series_name="Series", fig_dir=None, bins=30, sw_max_n=5000, hill_k=None):

    # ---------- data prepare ----------
    if isinstance(data, pd.DataFrame):
        ys = pd.Series(data[value_col]).dropna()
    elif isinstance(data, pd.Series):
        ys = data.dropna()
    else:
        ys = pd.Series(data).dropna()

    if len(ys) == 0:
        raise ValueError("No valid (non-NA) data points found.")

    # ---------- Four moments ----------
    moments = {
        "mean": float(ys.mean()),
        "std": float(ys.std(ddof=1)),
        "skewness": float(stats.skew(ys, bias=False)),
        "kurtosis": float(stats.kurtosis(ys, fisher=True, bias=False)),
    }

    # ---------- Normality test ----------
    # Shapiro–Wilk
    if len(ys) <= sw_max_n:
        sw_stat, sw_p = stats.shapiro(ys)
        sw = {"stat": float(sw_stat), "pvalue": float(sw_p)}
    else:
        sw = {"stat": float("nan"), "pvalue": float("nan")}

    # Anderson–Darling
    ad_res = stats.anderson(ys, dist='norm')
    ad = {
        "stat": float(ad_res.statistic),
        "critical_values": [float(c) for c in ad_res.critical_values],
        "significance_levels": [float(s) for s in ad_res.significance_level],
    }

    normality = {"shapiro_wilk": sw, "anderson_darling": ad}

    # ---------- Hill tail index ----------
    def _hill_tail_index(arr, side="right", k=None):
        x = pd.Series(arr).dropna()
        if side not in {"right", "left"}:
            raise ValueError("side must be 'right' or 'left'")
        if side == "left":
            x = -x

        x = np.asarray(x, dtype=float)
        n_all = len(x)
        if n_all < 10:
            raise ValueError("Not enough data points for Hill estimation.")

        x = np.sort(x)
        x = x[x > 0]
        n_pos = len(x)
        if n_pos < 10:
            raise ValueError("Not enough positive tail data points for Hill estimation.")

        if k is None:
            k = int(math.sqrt(n_pos))
        if not (1 < k < n_pos):
            raise ValueError(f"k must satisfy 1 < k < {n_pos}, got {k}.")

        x_tail = x[-k:]
        x_k = x[-k-1]
        logs = np.log(x_tail) - math.log(x_k)
        hill = 1.0 / (np.mean(logs))
        return {"alpha": float(hill), "k": float(k), "n": float(n_all), "side": side}

    tail_right = _hill_tail_index(ys, side="right", k=hill_k)
    tail_left  = _hill_tail_index(ys, side="left",  k=hill_k)
    
    # ---------- Visualization ----------
    if fig_dir is not None:
        import pathlib
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

        # Histogram with KDE
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(ys, bins=bins, density=True, alpha=0.5)
        kde = stats.gaussian_kde(ys)
        xs = np.linspace(ys.min(), ys.max(), 512)
        ax.plot(xs, kde(xs), linewidth=2)
        ax.set_title(f"{series_name}: Histogram + KDE")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        fig.savefig(f"{fig_dir}/{series_name}_hist_kde.png", bbox_inches="tight")
        plt.close(fig)

        # Q–Q Plot
        fig = plt.figure()
        sm.qqplot(ys, line="45", fit=True)
        ax = plt.gca()
        ax.set_title(f"{series_name}: Q–Q Plot")
        plt.savefig(f"{fig_dir}/{series_name}_qq.png", bbox_inches="tight")
        plt.close(fig)

    # ---------- result ----------
    return {
        "moments": moments,
        "normality": normality,
        "tail_index_right": tail_right,
        "tail_index_left": tail_left,
    }


# -------------------------------
# 2.OLS fit diagnosing
# -------------------------------
def run_ols_with_diagnostics(data, value_col="Excess Return", x_cols=("x1", "x2", "x3"), prefix="OLS_Residuals", fig_dir=None, reset_power=2, sw_max_n=5000):

    # ---------- OLS fit ----------
    y = data[value_col].astype(float)
    X = data.loc[:, list(x_cols)].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()
    resid = results.resid.dropna()
    exog = results.model.exog

    # ---------- Normality test ----------
    # Shapiro–Wilk
    if len(resid) <= sw_max_n:
        sw_stat, sw_p = stats.shapiro(resid)
        sw = {"stat": float(sw_stat), "pvalue": float(sw_p)}
    else:
        sw = {"stat": float("nan"), "pvalue": float("nan")}

    # Anderson–Darling
    ad_res = stats.anderson(resid, dist='norm')
    ad = {
        "stat": float(ad_res.statistic),
        "critical_values": [float(c) for c in ad_res.critical_values],
        "significance_levels": [float(s) for s in ad_res.significance_level],
    }

    normality = {"shapiro_wilk": sw, "anderson_darling": ad}
    
    # ---------- Durbin–Watson ----------
    dw = float(durbin_watson(resid))
    
    # ---------- Breusch–Pagan ----------
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, exog)
    
    # ---------- Ramsey RESET ----------
    reset = linear_reset(results, power=reset_power, use_f=True)
    reset_result = {
        "F": float(getattr(reset, "fvalue", np.nan)),
        "pvalue": float(getattr(reset, "pvalue", np.nan)),
        "power": int(reset_power),
        "use_f": True
    }

    # ---------- ccpr_plots with LOWESS----------
    if fig_dir is not None:
        import pathlib
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig = sm.graphics.plot_ccpr_grid(results)
        fig.suptitle(f"{prefix}: Partial Residual (CCPR) Plots", y=1.02)
        for ax in fig.axes:
            x_data, y_data = None, None
            if ax.collections:
                offsets = ax.collections[0].get_offsets()
                if len(offsets) > 0:
                    xy = np.asarray(offsets)
                    x_data = xy[:, 0]
                    y_data = xy[:, 1]
            if (x_data is None or y_data is None) and ax.lines:
                x_data = ax.lines[0].get_xdata()
                y_data = ax.lines[0].get_ydata()
            if x_data is not None and y_data is not None and len(x_data) > 10:
                order = np.argsort(x_data)
                xy_sorted = np.column_stack([x_data[order], y_data[order]])
                lo = lowess(xy_sorted[:, 1], xy_sorted[:, 0], frac=0.3, it=0, return_sorted=True)
                ax.plot(lo[:, 0], lo[:, 1], linewidth=2)
        fig.tight_layout()
        outpath = f"{fig_dir}/{prefix}_ccpr_grid.png"
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        return outpath

    # ---------- result ----------
    return {
        "normality": normality,
        "durbin_watson": dw,
        "breusch_pagan": {"stat": float(bp_stat), "pvalue": float(bp_pvalue)},
        "ramsey_reset": reset_result
    }