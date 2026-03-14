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
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + eps)))  # 注意：这里是比例，不乘100

    # R² (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)

    # Directional Accuracy (看符号是否一致)
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


# ===============================
# 2.Visualization (REPLACE THESE 3 FUNCTIONS)
# ===============================

def plot_time_series(stock, dates, y_true, preds_dict, ax=None, color_map=None):
    """Time series subplot. No title, no legend inside."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Actual
    c_actual = None if color_map is None else color_map.get("Actual", None)
    ax.plot(dates, y_true, linewidth=2, label="Actual", alpha=0.9, color=c_actual)

    # Models
    for name, y_pred in preds_dict.items():
        c = None if color_map is None else color_map.get(name, None)
        ax.plot(dates, y_pred, linewidth=1.5, alpha=0.9, label=name, color=c)

    ax.set_xlabel("Date")
    ax.set_ylabel("Excess Return")
    # 不要标题
    ax.set_title("")
    return ax


def plot_radar(y_true, model_preds, stock,
               lower_better=("RMSE", "MAE", "MAPE", "TheilU"),
               ax=None, color_map=None):
    """Radar subplot. No title, no legend inside. Uses same colors as time series."""
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

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
    else:
        if ax.name != "polar":
            raise ValueError("plot_radar requires a polar axis (projection='polar').")

    # 画每个模型（颜色与时间序列一致）
    for model_name, row in df_norm.iterrows():
        values = row.values
        values = np.concatenate([values, [values[0]]])
        c = None if color_map is None else color_map.get(model_name, None)
        ax.plot(angles, values, linewidth=2, label=model_name, color=c)
        ax.fill(angles, values, alpha=0.1, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)

    # 不要标题
    ax.set_title("")
    # 不要子图内部 legend
    return ax


def plot_time_series_and_radar(
    stock, dates, y_true, model_preds,
    figsize=(14, 5),
    width_ratios=(1.65, 1.0),
    wspace=0.25,
    add_panel_labels=True,
    legend_outside=True
):
    """
    One big figure:
      - Left: time series
      - Right: radar
    Changes:
      1) remove subplot titles
      2) one shared legend placed to the far right (outside)
      3) consistent colors across both subplots
    """
    # ---- 统一颜色表（保证两张子图同色）----
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(base_colors) == 0:
        base_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    # 顺序：Actual + 模型（保持 dict 顺序）
    names_order = ["Actual"] + list(model_preds.keys())
    color_map = {name: base_colors[i % len(base_colors)] for i, name in enumerate(names_order)}

    # ---- 画图布局 ----
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace)

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_rd = fig.add_subplot(gs[0, 1], projection="polar")

    plot_time_series(stock, dates, y_true, model_preds, ax=ax_ts, color_map=color_map)
    plot_radar(y_true, model_preds, stock, ax=ax_rd, color_map=color_map)

    # ---- 公用 legend：放在最右侧（整张图外部）----
    if legend_outside:
        # 用时间序列子图的 handles/labels（包含 Actual + 全部模型）
        handles, labels = ax_ts.get_legend_handles_labels()

        # 去重（以防万一）
        uniq = []
        seen = set()
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        handles, labels = zip(*uniq)

        # 在右侧外部放 legend
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(0.95, 0.5),
            frameon=True
        )

        # 给右侧 legend 留空间
        fig.tight_layout(rect=[0.0, 0.06, 0.86, 1.0])
    else:
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])

    # ---- (a)/(b) 标注（可选保留）----
    if add_panel_labels:
        fig.text(0.33, -0.03, "(a) Time Series Comparison", ha="center", va="bottom", fontsize=13)
        fig.text(0.77, -0.03, "(b) Radar Plot", ha="center", va="bottom", fontsize=13)

    plt.show()
    return fig, (ax_ts, ax_rd)



# -------------------------------
# NEW: Heatmap Plot (like your screenshot)
# -------------------------------
def plot_metrics_heatmap(
    y_true,
    model_preds,
    stock,
    metrics_order=("RMSE", "MAE", "MAPE", "R2", "DA", "Pearson", "Spearman", "TheilU"),
    lower_better=("RMSE", "MAE", "MAPE", "TheilU"),
    cmap="coolwarm_r",
    annotate=True,
    normalize_by_col=True,
    eps=1e-12
):
    """
    输出：行=指标，列=模型 的热图。
    - 颜色：蓝=好，红=差（通过对 lower_better 取负统一方向）
    - annotate：显示原始数值（不是归一化值）
    - normalize_by_col：对每个指标在不同模型间做 min-max 归一化，避免量纲差异影响上色
    """
    # 1) build raw metrics DF: index=Model, columns=metrics
    rows = []
    for name, y_pred in model_preds.items():
        m = prediction_metrics(y_true, y_pred)
        m["Model"] = name
        rows.append(m)
    df_raw = pd.DataFrame(rows).set_index("Model").astype(float)

    # enforce metrics order
    missing = [m for m in metrics_order if m not in df_raw.columns]
    if missing:
        raise ValueError(f"metrics_order contains unknown metrics: {missing}")
    df_raw = df_raw[list(metrics_order)]

    # 2) build color DF with unified direction
    df_color = df_raw.copy()
    lower_better = set(lower_better)
    for col in df_color.columns:
        if col in lower_better:
            df_color[col] = -df_color[col]

    # handle NaNs for coloring
    for col in df_color.columns:
        vals = df_color[col].values
        if np.any(np.isnan(vals)):
            mean_val = np.nanmean(vals)
            df_color[col] = np.where(np.isnan(vals), mean_val, vals)

    # 3) optional per-column normalization (min-max)
    if normalize_by_col:
        df_norm = df_color.copy()
        for col in df_norm.columns:
            v = df_norm[col].values.astype(float)
            vmin, vmax = np.min(v), np.max(v)
            if np.isclose(vmax, vmin):
                df_norm[col] = 0.5
            else:
                df_norm[col] = (v - vmin) / (vmax - vmin + eps)
        heat_values = df_norm.T.values  # transpose: rows=metrics, cols=models
        heat_values = heat_values - 0.5  # center
    else:
        heat_values = df_color.T.values

    # 4) plot
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

    # grid like table
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotate with RAW numbers
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
                return f"{v:.6f}"  # MAPE is ratio
            return f"{v:.4f}"

        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                v = df_raw.loc[model, metric]
                ax.text(
                    j, i, fmt(metric, v),
                    ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold"
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)
    cbar.ax.text(
        0.5, 1.02, "Good",
        ha="center", va="bottom",
        transform=cbar.ax.transAxes,
        fontsize=11, fontweight="bold"
    )
    cbar.ax.text(
        0.5, -0.02, "Bad",
        ha="center", va="top",
        transform=cbar.ax.transAxes,
        fontsize=11, fontweight="bold"
    )

    ax.set_title(f"{stock}: Model Performance Heatmap", pad=12)
    plt.tight_layout()
    plt.show()

    return df_raw


# -------------------------------
# 3.Helper: 针对单只股票做分析
# -------------------------------
def analyze_single_stock(df, stock, date_prefix="Date_", ytrue_prefix="True_"):
    date_col = f"{date_prefix}{stock}"
    ytrue_col = f"{ytrue_prefix}{stock}"

    dates = pd.to_datetime(df[date_col])
    y_true = df[ytrue_col].values

    model_preds = {}
    for col in df.columns:
        # 以 _STOCK 结尾的是该股票相关列，但要排除日期和真实值
        if col.endswith(f"_{stock}") and col not in [date_col, ytrue_col]:
            model_name = col.replace(f"_{stock}", "")
            model_preds[model_name] = df[col].values

    # 计算各模型的统计指标
    rows = []
    for name, y_pred in model_preds.items():
        metrics = prediction_metrics(y_true, y_pred)
        metrics["Model"] = name
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows).set_index("Model")
    print(f"\n================ {stock}: Metrics ================")
    print(metrics_df.round(6))

    # NEW: 合并图（左时间序列 + 右雷达）
    plot_time_series_and_radar(stock, dates, y_true, model_preds)

    # 热图：指标×模型
    plot_metrics_heatmap(y_true, model_preds, stock)

    return metrics_df


# -------------------------------
# 4.Run
# -------------------------------
csv_path = "~/Desktop/stock_predictions.csv"
df = pd.read_csv(csv_path)

date_prefix = "Date_"
ytrue_prefix = "True_"

stocks = sorted([
    col.replace(ytrue_prefix, "")
    for col in df.columns
    if col.startswith(ytrue_prefix)
])

print("Detected stocks:", stocks)

if not stocks:
    print("没有检测到任何股票，请检查列名是否形如 'True_AAPL', 'Date_AAPL' 这类格式。")
else:
    all_metrics = {}

    for stock in stocks:
        metrics_df = analyze_single_stock(
            df,
            stock,
            date_prefix=date_prefix,
            ytrue_prefix=ytrue_prefix
        )
        all_metrics[stock] = metrics_df

    combined_metrics_list = []
    for stock, mdf in all_metrics.items():
        tmp = mdf.copy()
        tmp["Stock"] = stock
        combined_metrics_list.append(tmp.reset_index())

    combined_metrics_df = pd.concat(combined_metrics_list, ignore_index=True)
    combined_metrics_df = combined_metrics_df.set_index(["Stock", "Model"])

    print("\n================ All Stocks: Combined Metrics ================")
    print(combined_metrics_df.round(6))
