# -------------------------------
# *.Library Imports 
# -------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -------------------------------
# 1.Performance Metrics Calculation Function
# -------------------------------
def prediction_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    # Error Class
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + eps)))

    # R² (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)

    # Directional Accuracy
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    da = np.mean(sign_true == sign_pred)

    # Pearson correlation coefficient
    if np.std(y_true) < eps or np.std(y_pred) < eps:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_pred, y_true)[0, 1])

    # Spearman’s rank correlation coefficient
    y_true_rank = pd.Series(y_true).rank(method='average').to_numpy()
    y_pred_rank = pd.Series(y_pred).rank(method='average').to_numpy()
    if np.std(y_true_rank) < eps or np.std(y_pred_rank) < eps:
        spearman = np.nan
    else:
        spearman = float(np.corrcoef(y_pred_rank, y_true_rank)[0, 1])

    # Theil’s U
    rmse_num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2))
    theil_u = float(rmse_num / (denom + eps))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'DA': da,
        'Pearson': pearson,
        'Spearman': spearman,
        'TheilU': theil_u
    }
    

# -------------------------------
# 2.Visualization
# -------------------------------
# Heat Map
def plot_metrics_heatmap(
    y_true, model_preds, stock,
    metrics_order=("RMSE", "MAE", "MAPE", "R2", "DA", "Pearson", "Spearman", "TheilU"),
    lower_better=("RMSE", "MAE", "MAPE", "TheilU"),
    cmap = "coolwarm_r", annotate=True, normalize_by_col=True, eps=1e-12
):
    rows = []
    for name, y_pred in model_preds.items():
        m = prediction_metrics(y_true, y_pred)
        m["Model"] = name
        rows.append(m)
    df_raw = pd.DataFrame(rows).set_index("Model").astype(float)
    missing = [m for m in metrics_order if m not in df_raw.columns]
    if missing:
        raise ValueError(f"metrics_order contains unknown metrics: {missing}")
    df_raw = df_raw[list(metrics_order)]

    df_color = df_raw.copy()
    lower_better = set(lower_better)
    for col in df_color.columns:
        if col in lower_better:
            df_color[col] = -df_color[col]
    for col in df_color.columns:
        vals = df_color[col].values
        if np.any(np.isnan(vals)):
            mean_val = np.nanmean(vals)
            df_color[col] = np.where(np.isnan(vals), mean_val, vals)

    if normalize_by_col:
        df_norm = df_color.copy()
        for col in df_norm.columns:
            v = df_norm[col].values.astype(float)
            vmin, vmax = np.min(v), np.max(v)
            if np.isclose(vmax, vmin):
                df_norm[col] = 0.5
            else:
                df_norm[col] = (v - vmin) / (vmax - vmin + eps)
        heat_values = df_norm.T.values
        heat_values = heat_values - 0.5
    else:
        heat_values = df_color.T.values

    metrics = list(df_raw.columns)
    models = list(df_raw.index)
    fig_w = max(8, 1.6 * len(models))
    fig_h = max(5, 0.7 * len(metrics) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat_values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticklabels(metrics, fontsize=12)
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        def fmt(metric, v):
            if np.isnan(v):
                return "nan"
            if metric in ("R2", "DA", "Pearson", "Spearman"):
                return f"{v:.4f}"
            if metric == "TheilU":
                return f"{v:.6f}"
            if metric in ("RMSE", "MAE"):
                return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
            if metric == "MAPE":
                return f"{v:.6f}"
            return f"{v:.4f}"
        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                v = df_raw.loc[model, metric]
                ax.text(j, i, fmt(metric, v),
                        ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)
    cbar.ax.text(0.5, 1.02, "Good",
              ha="center", va="bottom",
              transform=cbar.ax.transAxes,
              fontsize=11, fontweight="bold")

    cbar.ax.text(0.5, -0.02, "Bad",
              ha="center", va="top",
              transform=cbar.ax.transAxes,
              fontsize=11, fontweight="bold")

    ax.set_title(f"{stock}: Model Performance Heatmap", pad=12)
    plt.tight_layout()
    plt.show()
    return df_raw

# Time series
def plot_time_series(stock, dates, y_true, preds_dict):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, linewidth=2, label='Actual', alpha=0.9)
    for name, y_pred in preds_dict.items():
        plt.plot(dates, y_pred, linewidth=1.5, alpha=0.9, label=name)
    plt.xlabel('Date')
    plt.ylabel('Excess Return')
    plt.title(f'{stock}: Time Series')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Radar Plot
def plot_radar(y_true, model_preds, stock, lower_better=("RMSE", "MAE", "MAPE", "TheilU")):
    rows = []
    for name, y_pred in model_preds.items():
        m = prediction_metrics(y_true, y_pred)
        m["Model"] = name
        rows.append(m)

    df_metrics = pd.DataFrame(rows).set_index("Model").astype(float)
    df_norm = df_metrics.copy().astype(float)
    lower_better = set(lower_better)

    for col in df_norm.columns:
        col_vals = df_norm[col].values

        if np.any(np.isnan(col_vals)):
            mean_val = np.nanmean(col_vals)
            col_vals = np.where(np.isnan(col_vals), mean_val, col_vals)

        if col in lower_better:
            col_vals = -col_vals

        cmin, cmax = np.min(col_vals), np.max(col_vals)
        if np.isclose(cmax, cmin):
            norm_vals = np.ones_like(col_vals) * 0.5  
        else:
            norm_vals = (col_vals - cmin) / (cmax - cmin)
        df_norm[col] = norm_vals

    labels = list(df_norm.columns)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for model_name, row in df_norm.iterrows():
        values = row.values
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
    ax.set_title(f'{stock}: Radar Plot', fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()
