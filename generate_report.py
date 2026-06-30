#!/usr/bin/env python3
"""
UW-Madison Reproducible Report Generator

Usage:
    python generate_report.py
    python generate_report.py --config config.yaml --output output/report.html
"""
import argparse
import base64
import io
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# UW-Madison brand palette
# ---------------------------------------------------------------------------
UW = {
    "red":        "#C5050C",
    "white":      "#FFFFFF",
    "dark":       "#282728",
    "gray":       "#646569",
    "light_gray": "#f7f7f7",
    "gold":       "#f2a900",
    "border":     "#dadada",
}

_UW_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "uw_div", [UW["red"], "#ffffff", UW["dark"]]
)

# ---------------------------------------------------------------------------
# Style & figure helpers
# ---------------------------------------------------------------------------

def _apply_uw_style():
    plt.rcParams.update({
        "figure.facecolor": UW["light_gray"],
        "axes.facecolor":   "#ffffff",
        "axes.edgecolor":   UW["dark"],
        "axes.labelcolor":  UW["dark"],
        "text.color":       UW["dark"],
        "xtick.color":      UW["gray"],
        "ytick.color":      UW["gray"],
        "grid.color":       "#e4e4e4",
        "grid.alpha":       0.8,
        "font.family":      "sans-serif",
        "axes.grid":        True,
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ---------------------------------------------------------------------------
# Config & data loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    dc = config["data"]
    kwargs = {}
    if "delimiter" in dc:
        kwargs["sep"] = dc["delimiter"]
    return pd.read_csv(dc["path"], **kwargs)


def _var_groups(config: dict):
    variables = config["variables"]
    dependent   = [v for v in variables if v["role"] == "dependent"]
    independent = [v for v in variables if v["role"] == "independent"]
    return dependent, independent


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(series: pd.Series, var_type: str) -> dict:
    base = {
        "n":           int(len(series)),
        "missing":     int(series.isna().sum()),
        "missing_pct": f"{series.isna().mean() * 100:.1f}%",
    }
    if var_type == "continuous":
        s = series.dropna()
        base.update({
            "mean":     f"{s.mean():.4g}",
            "std":      f"{s.std():.4g}",
            "min":      f"{s.min():.4g}",
            "q25":      f"{s.quantile(0.25):.4g}",
            "median":   f"{s.median():.4g}",
            "q75":      f"{s.quantile(0.75):.4g}",
            "max":      f"{s.max():.4g}",
            "skewness": f"{s.skew():.3f}",
            "kurtosis": f"{s.kurtosis():.3f}",
        })
    else:
        vc = series.value_counts()
        base.update({
            "n_unique":  int(series.nunique()),
            "mode":      str(vc.index[0]) if len(vc) else "N/A",
            "mode_freq": f"{vc.iloc[0]} ({vc.iloc[0]/len(series)*100:.1f}%)" if len(vc) else "N/A",
        })
    return base


# ---------------------------------------------------------------------------
# Univariate plots
# ---------------------------------------------------------------------------

def plot_continuous(series: pd.Series, label: str) -> str:
    _apply_uw_style()
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(11, 4))
    data = series.dropna()

    ax_hist.hist(data, bins=30, color=UW["red"], alpha=0.72, edgecolor="white", linewidth=0.4)
    ax_hist.set_title(f"Distribution — {label}", fontweight="bold", color=UW["dark"], fontsize=11)
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("Count")

    ax2 = ax_hist.twinx()
    ax2.set_ylabel("Density", color=UW["gray"], fontsize=9)
    ax2.tick_params(colors=UW["gray"], labelsize=8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(UW["border"])
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 300)
    ax2.plot(xs, kde(xs), color=UW["dark"], linewidth=2, zorder=5)
    ax2.set_ylim(bottom=0)

    ax_box.boxplot(
        data, patch_artist=True,
        medianprops={"color": UW["gold"], "linewidth": 2.5},
        boxprops={"facecolor": UW["red"], "alpha": 0.45, "linewidth": 1.2},
        whiskerprops={"color": UW["gray"], "linewidth": 1.2},
        capprops={"color": UW["dark"], "linewidth": 1.5},
        flierprops={"markerfacecolor": UW["red"], "marker": "o",
                    "markersize": 4, "alpha": 0.5, "markeredgewidth": 0},
    )
    ax_box.set_title(f"Boxplot — {label}", fontweight="bold", color=UW["dark"], fontsize=11)
    ax_box.set_ylabel(label)
    ax_box.set_xticks([])

    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_categorical(series: pd.Series, label: str) -> str:
    _apply_uw_style()
    counts = series.value_counts().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(counts) * 0.45)))
    colors = [UW["red"] if i == 0 else UW["gray"] for i in range(len(counts))]
    bars = ax.barh(counts.index.astype(str)[::-1], counts.values[::-1],
                   color=colors[::-1], alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=UW["gray"], fontsize=9)

    ax.set_title(f"Value Counts — {label}", fontweight="bold", color=UW["dark"], fontsize=11)
    ax.set_xlabel("Count")
    ax.set_xlim(right=counts.max() * 1.12)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Bivariate plots
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, x_var: dict, y_var: dict) -> str:
    _apply_uw_style()
    xc, yc = x_var["name"], y_var["name"]
    xl, yl = x_var.get("label", xc), y_var.get("label", yc)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mask = df[[xc, yc]].notna().all(axis=1)
    ax.scatter(df.loc[mask, xc], df.loc[mask, yc],
               color=UW["red"], alpha=0.35, s=22, edgecolors="none", zorder=3)

    if mask.sum() > 2:
        xs = df.loc[mask, xc].values
        ys = df.loc[mask, yc].values
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(x_line, m * x_line + b, color=UW["dark"],
                linewidth=2, linestyle="--", label=f"Trend (slope={m:.2g})", zorder=4)
        ax.legend(fontsize=9)

    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{xl}  →  {yl}", fontweight="bold", color=UW["dark"], fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_boxplot_by_category(df: pd.DataFrame, cat_var: dict, cont_var: dict,
                              cat_is_dep: bool = False) -> str:
    _apply_uw_style()
    cc, nc = cat_var["name"], cont_var["name"]
    cl, nl = cat_var.get("label", cc), cont_var.get("label", nc)

    top_cats = df[cc].value_counts().head(10).index.tolist()
    sub = df[df[cc].isin(top_cats)].dropna(subset=[cc, nc])

    fig, ax = plt.subplots(figsize=(max(6, len(top_cats) * 0.9), 4.5))
    palette = [UW["red"], UW["dark"], UW["gray"], UW["gold"]] * 3
    groups = [sub.loc[sub[cc] == cat, nc].values for cat in top_cats]
    bp = ax.boxplot(
        groups, patch_artist=True,
        medianprops={"color": UW["gold"], "linewidth": 2},
        whiskerprops={"color": UW["gray"]},
        capprops={"color": UW["dark"]},
        flierprops={"markerfacecolor": UW["red"], "marker": "o",
                    "markersize": 4, "alpha": 0.5, "markeredgewidth": 0},
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.5)

    ax.set_xticks(range(1, len(top_cats) + 1))
    ax.set_xticklabels(top_cats, rotation=25, ha="right", fontsize=9)
    ax.set_xlabel(cl); ax.set_ylabel(nl)
    title = f"{nl}  by  {cl}" if not cat_is_dep else f"{cl}  distribution by  {nl}"
    ax.set_title(title, fontweight="bold", color=UW["dark"], fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, cols: list) -> str | None:
    if len(cols) < 2:
        return None
    _apply_uw_style()

    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 0.5)))

    im = ax.imshow(corr.values, cmap=_UW_CMAP, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    labels = [c.replace("_", " ").title() for c in cols]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.55 else UW["dark"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if abs(val) > 0.5 else "normal")

    ax.set_title("Correlation Matrix (Continuous Variables)",
                 fontweight="bold", color=UW["dark"], fontsize=12, pad=14)
    ax.grid(False)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Shared diagnostic plot (residuals vs fitted + Q-Q)
# ---------------------------------------------------------------------------

def _residual_plots(fitted) -> str:
    from scipy.stats import probplot
    _apply_uw_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # GLMResults uses resid_deviance; OLS/Logit/MixedLM use resid
    resids = getattr(fitted, "resid", None)
    if resids is None:
        resids = getattr(fitted, "resid_deviance", fitted.resid_response)

    ax1.scatter(fitted.fittedvalues, resids,
                color=UW["red"], alpha=0.35, s=20, edgecolors="none", zorder=3)
    ax1.axhline(0, color=UW["dark"], linestyle="--", linewidth=1.5, zorder=4)
    ax1.set_xlabel("Fitted Values"); ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted", fontweight="bold", color=UW["dark"], fontsize=11)

    (osm, osr), (slope, intercept, _) = probplot(resids)
    ax2.scatter(osm, osr, color=UW["red"], alpha=0.35, s=20, edgecolors="none", zorder=3)
    x_line = np.array([osm.min(), osm.max()])
    ax2.plot(x_line, slope * x_line + intercept,
             color=UW["dark"], linewidth=2, linestyle="--", zorder=4)
    ax2.set_xlabel("Theoretical Quantiles"); ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Q-Q Plot of Residuals", fontweight="bold", color=UW["dark"], fontsize=11)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Model fitting — shared data prep
# ---------------------------------------------------------------------------

def _prepare_xy(df: pd.DataFrame, model_cfg: dict,
                dependent: list, independent: list):
    """Encode categoricals, drop NAs, return (dep_var, dep_col, df_m, X_const, y)."""
    import statsmodels.api as sm

    dep_var  = dependent[0]
    dep_col  = dep_var["name"]
    ind_cols = [v["name"] for v in independent if v["name"] in df.columns]

    df_m     = df[[dep_col] + ind_cols].dropna()
    cat_cols = [c for c in ind_cols if not pd.api.types.is_numeric_dtype(df_m[c])]
    df_enc   = pd.get_dummies(df_m, columns=cat_cols, drop_first=True)

    y   = df_enc[dep_col].astype(float)
    X   = df_enc.drop(columns=[dep_col]).astype(float)
    X_c = sm.add_constant(X)

    return dep_var, dep_col, df_m, X_c, y


# ---------------------------------------------------------------------------
# Model fitting — individual model functions
# ---------------------------------------------------------------------------

def _fit_ols(df, model_cfg, dependent, independent):
    import statsmodels.api as sm
    dep_var, dep_col, df_m, X_c, y = _prepare_xy(df, model_cfg, dependent, independent)
    fitted = sm.OLS(y, X_c).fit()
    return {
        "type":          "linear_regression",
        "description":   model_cfg.get("description", ""),
        "n_obs":         len(df_m),
        "dep_label":     dep_var.get("label", dep_col),
        "summary_html":  fitted.summary().as_html(),
        "r_squared":     f"{fitted.rsquared:.4f}",
        "adj_r_squared": f"{fitted.rsquared_adj:.4f}",
        "f_pvalue":      f"{fitted.f_pvalue:.4g}",
        "residual_plot": _residual_plots(fitted),
    }


def _fit_logit(df, model_cfg, dependent, independent):
    import statsmodels.api as sm
    dep_var, dep_col, df_m, X_c, y = _prepare_xy(df, model_cfg, dependent, independent)
    fitted = sm.Logit(y, X_c).fit(disp=0, maxiter=200)
    return {
        "type":             "logistic_regression",
        "description":      model_cfg.get("description", ""),
        "n_obs":            len(df_m),
        "dep_label":        dep_var.get("label", dep_col),
        "summary_html":     fitted.summary().as_html(),
        "pseudo_r_squared": f"{fitted.prsquared:.4f}",
        "llr_pvalue":       f"{fitted.llr_pvalue:.4g}",
        "residual_plot":    _residual_plots(fitted),
    }


# Supported GLM families
_GLM_FAMILIES = None

def _glm_families():
    global _GLM_FAMILIES
    if _GLM_FAMILIES is None:
        import statsmodels.api as sm
        _GLM_FAMILIES = {
            "gaussian":          sm.families.Gaussian(),
            "poisson":           sm.families.Poisson(),
            "binomial":          sm.families.Binomial(),
            "gamma":             sm.families.Gamma(),
            "negative_binomial": sm.families.NegativeBinomial(),
            "inverse_gaussian":  sm.families.InverseGaussian(),
            "tweedie":           sm.families.Tweedie(var_power=1.5),
        }
    return _GLM_FAMILIES


def _fit_glm(df, model_cfg, dependent, independent):
    import statsmodels.api as sm
    family_name = model_cfg.get("family", "gaussian").lower()
    families    = _glm_families()
    if family_name not in families:
        raise ValueError(
            f"Unknown GLM family '{family_name}'. "
            f"Supported: {', '.join(families.keys())}"
        )

    dep_var, dep_col, df_m, X_c, y = _prepare_xy(df, model_cfg, dependent, independent)
    fitted = sm.GLM(y, X_c, family=families[family_name]).fit()

    return {
        "type":         "glm",
        "family":       family_name,
        "description":  model_cfg.get("description", ""),
        "n_obs":        len(df_m),
        "dep_label":    dep_var.get("label", dep_col),
        "summary_html": fitted.summary().as_html(),
        "aic":          f"{fitted.aic:.4g}",
        "bic":          f"{fitted.bic:.4g}",
        "deviance":     f"{fitted.deviance:.4g}",
        "pearson_chi2": f"{fitted.pearson_chi2:.4g}",
        "residual_plot": _residual_plots(fitted),
    }


def _fit_mixed_lm(df, model_cfg, dependent, independent):
    import statsmodels.api as sm
    group_col = model_cfg.get("group_col")
    if not group_col:
        raise ValueError("model.group_col is required for mixed_linear")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataset")

    dep_var  = dependent[0]
    dep_col  = dep_var["name"]
    ind_cols = [v["name"] for v in independent if v["name"] in df.columns]

    needed  = list(dict.fromkeys([dep_col, group_col] + ind_cols))
    df_m    = df[needed].dropna()
    cat_cols = [c for c in ind_cols
                if c != group_col and not pd.api.types.is_numeric_dtype(df_m[c])]
    df_enc  = pd.get_dummies(df_m, columns=cat_cols, drop_first=True)

    groups  = df_enc[group_col].astype(str)
    y       = df_enc[dep_col].astype(float)
    X_cols  = [c for c in df_enc.columns if c not in (dep_col, group_col)]
    X_c     = sm.add_constant(df_enc[X_cols].astype(float))

    fitted  = sm.MixedLM(y, X_c, groups=groups).fit()

    return {
        "type":           "mixed_linear",
        "group_col":      group_col,
        "description":    model_cfg.get("description", ""),
        "n_obs":          len(df_m),
        "n_groups":       int(groups.nunique()),
        "dep_label":      dep_var.get("label", dep_col),
        "summary_html":   fitted.summary().as_html(),
        "log_likelihood": f"{fitted.llf:.4g}",
        "aic":            f"{fitted.aic:.4g}",
        "bic":            f"{fitted.bic:.4g}",
        "residual_plot":  _residual_plots(fitted),
    }


# ---------------------------------------------------------------------------
# A/B test
# ---------------------------------------------------------------------------

def _cohens_d_label(d: float) -> str:
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def _plot_ab_continuous(ctrl, trt, ctrl_label, trt_label, metric_label) -> str:
    _apply_uw_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.histogram_bin_edges(
        np.concatenate([ctrl.dropna(), trt.dropna()]), bins=30
    )
    ax1.hist(ctrl.dropna(), bins=bins, alpha=0.6, color=UW["gray"],
             label=f"{ctrl_label} (n={len(ctrl)})", edgecolor="white", linewidth=0.3)
    ax1.hist(trt.dropna(), bins=bins, alpha=0.6, color=UW["red"],
             label=f"{trt_label} (n={len(trt)})", edgecolor="white", linewidth=0.3)
    ax1.axvline(float(ctrl.mean()), color=UW["gray"], linestyle="--", linewidth=2,
                label=f"Mean {ctrl_label}")
    ax1.axvline(float(trt.mean()),  color=UW["red"],  linestyle="--", linewidth=2,
                label=f"Mean {trt_label}")
    ax1.set_xlabel(metric_label); ax1.set_ylabel("Count")
    ax1.set_title("Distribution Comparison", fontweight="bold", color=UW["dark"], fontsize=11)
    ax1.legend(fontsize=8)

    bp = ax2.boxplot(
        [ctrl.dropna().values, trt.dropna().values],
        patch_artist=True, tick_labels=[ctrl_label, trt_label],
        medianprops={"color": UW["gold"], "linewidth": 2.5},
        whiskerprops={"color": UW["gray"]},
        capprops={"color": UW["dark"]},
        flierprops={"markerfacecolor": UW["red"], "marker": "o",
                    "markersize": 4, "alpha": 0.5, "markeredgewidth": 0},
    )
    for patch, color in zip(bp["boxes"], [UW["gray"], UW["red"]]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax2.set_ylabel(metric_label)
    ax2.set_title("Group Comparison (IQR)", fontweight="bold", color=UW["dark"], fontsize=11)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_ab_categorical(ctrl, trt, ctrl_label, trt_label, metric_label) -> str:
    _apply_uw_style()
    cats = sorted(set(ctrl.dropna().unique()) | set(trt.dropna().unique()))
    ctrl_pct = ctrl.value_counts(normalize=True) * 100
    trt_pct  = trt.value_counts(normalize=True) * 100

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 1.2), 4.5))
    ax.bar(x - w/2, [ctrl_pct.get(c, 0) for c in cats], w,
           color=UW["gray"], alpha=0.8, label=f"{ctrl_label} (n={len(ctrl)})", edgecolor="white")
    ax.bar(x + w/2, [trt_pct.get(c, 0) for c in cats], w,
           color=UW["red"],  alpha=0.8, label=f"{trt_label} (n={len(trt)})",  edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Proportion (%)"); ax.legend(fontsize=9)
    ax.set_title(f"Distribution of {metric_label} by Group",
                 fontweight="bold", color=UW["dark"], fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fit_ab_test(df, model_cfg, dependent, independent):
    from scipy import stats as sp_stats

    dep_var   = dependent[0]
    dep_col   = dep_var["name"]
    dep_label = dep_var.get("label", dep_col)
    dep_type  = dep_var.get("type", "continuous")

    treatment_col = model_cfg.get("treatment_col")
    if not treatment_col:
        raise ValueError("model.treatment_col is required for ab_test")
    if treatment_col not in df.columns:
        raise ValueError(f"Column '{treatment_col}' not found in dataset")

    ctrl_label = str(model_cfg.get("control_label",   "control"))
    trt_label  = str(model_cfg.get("treatment_label", "treatment"))
    alpha      = float(model_cfg.get("alpha", 0.05))

    test_type = model_cfg.get("test", "auto")
    if test_type == "auto":
        test_type = (
            "t_test"     if dep_type == "continuous" else
            "proportion" if dep_type == "binary"     else
            "chi2"
        )

    df_ab = df[[dep_col, treatment_col]].dropna()
    ctrl  = df_ab[df_ab[treatment_col].astype(str) == ctrl_label][dep_col]
    trt   = df_ab[df_ab[treatment_col].astype(str) == trt_label][dep_col]

    if len(ctrl) == 0 or len(trt) == 0:
        raise ValueError(
            f"No rows found for '{ctrl_label}' or '{trt_label}' in column '{treatment_col}'. "
            f"Unique values: {df[treatment_col].unique().tolist()}"
        )

    def _num_stats(s, label):
        return {"label": label, "n": len(s),
                "mean":   f"{s.mean():.4g}", "std":    f"{s.std():.4g}",
                "median": f"{s.median():.4g}",
                "min":    f"{s.min():.4g}",  "max":    f"{s.max():.4g}"}

    def _cat_stats(s, label):
        vc = s.value_counts()
        return {"label": label, "n": len(s),
                "mode":     str(vc.index[0]) if len(vc) else "N/A",
                "mode_pct": f"{vc.iloc[0]/len(s)*100:.1f}%" if len(vc) else "N/A"}

    is_numeric = pd.api.types.is_numeric_dtype(ctrl)
    stat_fn    = _num_stats if is_numeric else _cat_stats

    out = {
        "type":             "ab_test",
        "description":      model_cfg.get("description", ""),
        "dep_label":        dep_label,
        "dep_type":         dep_type,
        "is_numeric":       is_numeric,
        "treatment_col":    treatment_col,
        "ctrl_label":       ctrl_label,
        "trt_label":        trt_label,
        "alpha":            alpha,
        "test_type":        test_type,
        "n_obs":            len(df_ab),
        "ctrl_stats":       stat_fn(ctrl, ctrl_label),
        "trt_stats":        stat_fn(trt, trt_label),
    }

    # ── Statistical test ──────────────────────────────────────────────────
    if test_type == "t_test":
        stat, pvalue = sp_stats.ttest_ind(ctrl, trt, equal_var=False)
        mean_diff = float(trt.mean() - ctrl.mean())
        var_ctrl  = float(ctrl.var(ddof=1))
        var_trt   = float(trt.var(ddof=1))
        se        = float(np.sqrt(var_ctrl/len(ctrl) + var_trt/len(trt)))
        df_w      = (var_ctrl/len(ctrl) + var_trt/len(trt))**2 / (
                        (var_ctrl/len(ctrl))**2/(len(ctrl)-1) +
                        (var_trt/len(trt))**2/(len(trt)-1))
        t_crit    = sp_stats.t.ppf(1 - alpha/2, df=df_w)
        pooled    = float(np.sqrt((ctrl.std()**2 + trt.std()**2) / 2))
        cohens_d  = mean_diff / pooled if pooled > 0 else 0.0

        out.update({
            "test_name":   "Welch's Two-Sample t-Test",
            "stat_label":  "t",
            "statistic":   f"{stat:.4f}",
            "pvalue":      f"{pvalue:.4g}",
            "significant": bool(pvalue < alpha),
            "effect":      f"Δ mean = {mean_diff:+.4g}",
            "ci":          f"{int((1-alpha)*100)}% CI: [{mean_diff - t_crit*se:.4g}, {mean_diff + t_crit*se:.4g}]",
            "effect_size": f"Cohen's d = {cohens_d:.3f} ({_cohens_d_label(abs(cohens_d))})",
        })

    elif test_type == "mannwhitney":
        stat, pvalue = sp_stats.mannwhitneyu(ctrl, trt, alternative="two-sided")
        r_rb = float(stat / (len(ctrl) * len(trt)))
        out.update({
            "test_name":   "Mann-Whitney U Test (non-parametric)",
            "stat_label":  "U",
            "statistic":   f"{stat:.4f}",
            "pvalue":      f"{pvalue:.4g}",
            "significant": bool(pvalue < alpha),
            "effect":      f"Δ median = {float(trt.median() - ctrl.median()):+.4g}",
            "ci":          "",
            "effect_size": f"Rank-biserial r = {r_rb:.3f}",
        })

    elif test_type == "proportion":
        from statsmodels.stats.proportion import proportions_ztest
        cb = ctrl.astype(float); tb = trt.astype(float)
        stat, pvalue = proportions_ztest([tb.sum(), cb.sum()], [len(tb), len(cb)])
        p_c, p_t = float(cb.mean()), float(tb.mean())
        diff = p_t - p_c
        se   = float(np.sqrt(p_c*(1-p_c)/len(cb) + p_t*(1-p_t)/len(tb)))
        z_c  = sp_stats.norm.ppf(1 - alpha/2)
        h    = 2 * (np.arcsin(np.sqrt(max(0, p_t))) - np.arcsin(np.sqrt(max(0, p_c))))
        out.update({
            "test_name":   "Two-Proportion z-Test",
            "stat_label":  "z",
            "statistic":   f"{stat:.4f}",
            "pvalue":      f"{pvalue:.4g}",
            "significant": bool(pvalue < alpha),
            "effect":      f"Δ proportion = {diff:+.4f} ({p_c:.3f} → {p_t:.3f})",
            "ci":          f"{int((1-alpha)*100)}% CI: [{diff - z_c*se:.4f}, {diff + z_c*se:.4f}]",
            "effect_size": f"Cohen's h = {h:.3f}",
        })

    elif test_type == "chi2":
        ct = pd.crosstab(df_ab[treatment_col].astype(str), df_ab[dep_col].astype(str))
        chi2, pvalue, dof, _ = sp_stats.chi2_contingency(ct)
        v = float(np.sqrt(chi2 / (len(df_ab) * (min(ct.shape) - 1))))
        out.update({
            "test_name":   "Chi-Squared Test of Independence",
            "stat_label":  "χ²",
            "statistic":   f"{chi2:.4f} (df={dof})",
            "pvalue":      f"{pvalue:.4g}",
            "significant": bool(pvalue < alpha),
            "effect":      "",
            "ci":          "",
            "effect_size": f"Cramér's V = {v:.3f}",
        })
    else:
        raise ValueError(
            f"Unknown test '{test_type}'. Use: t_test, mannwhitney, proportion, chi2"
        )

    if is_numeric:
        out["dist_plot"] = _plot_ab_continuous(ctrl, trt, ctrl_label, trt_label, dep_label)
    else:
        out["dist_plot"] = _plot_ab_categorical(ctrl, trt, ctrl_label, trt_label, dep_label)

    return out


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------

_DISPATCHERS = {
    "linear_regression":  _fit_ols,
    "logistic_regression": _fit_logit,
    "glm":                _fit_glm,
    "mixed_linear":       _fit_mixed_lm,
    "ab_test":            _fit_ab_test,
}


def fit_model(df: pd.DataFrame, config: dict,
              dependent: list, independent: list) -> dict | None:
    model_cfg = config.get("model")
    if not model_cfg or not dependent:
        return None

    model_type = model_cfg.get("type", "linear_regression")
    fn = _DISPATCHERS.get(model_type)

    if fn is None:
        return {
            "type": model_type,
            "error": (
                f"Unknown model type '{model_type}'. "
                f"Supported: {', '.join(_DISPATCHERS.keys())}"
            ),
        }

    try:
        return fn(df, model_cfg, dependent, independent)
    except Exception as exc:
        return {"type": model_type, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_report(config_path: str, output_path: str):
    config = load_config(config_path)
    df     = load_data(config)
    dependent, independent = _var_groups(config)
    all_vars = dependent + independent

    print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Univariate sections ────────────────────────────────────────────────
    var_sections = []
    for var in all_vars:
        col   = var["name"]
        label = var.get("label", col)
        vtype = var.get("type", "continuous")
        role  = var["role"]

        if col not in df.columns:
            print(f"  [WARN] '{col}' not in dataset — skipping")
            continue

        print(f"  Plotting {label} ({vtype}, {role})")
        stats = summary_stats(df[col], vtype)
        plot  = (plot_continuous(df[col], label)
                 if vtype == "continuous"
                 else plot_categorical(df[col], label))

        var_sections.append({
            "name": col, "label": label, "type": vtype, "role": role,
            "stats": stats, "plot": plot,
        })

    # ── Correlation matrix ─────────────────────────────────────────────────
    cont_cols = [
        v["name"] for v in all_vars
        if v.get("type") == "continuous" and v["name"] in df.columns
    ]
    print("  Building correlation matrix")
    corr_plot = plot_correlation_matrix(df, cont_cols)

    # ── Bivariate: each independent vs dependent ───────────────────────────
    bivariate_plots = []
    dep_var = dependent[0] if dependent else None
    if dep_var and dep_var["name"] in df.columns:
        dep_type = dep_var.get("type", "continuous")
        for ind_var in independent:
            ind_col  = ind_var["name"]
            ind_type = ind_var.get("type", "continuous")
            if ind_col not in df.columns:
                continue

            ind_label = ind_var.get("label", ind_col)
            dep_label = dep_var.get("label", dep_var["name"])
            print(f"  Bivariate: {ind_label} → {dep_label}")

            if dep_type == "continuous" and ind_type == "continuous":
                plot  = plot_scatter(df, ind_var, dep_var)
                title = f"{ind_label}  →  {dep_label}"
            elif dep_type == "continuous" and ind_type in ("categorical", "binary", "ordinal"):
                plot  = plot_boxplot_by_category(df, ind_var, dep_var, cat_is_dep=False)
                title = f"{dep_label}  by  {ind_label}"
            elif dep_type in ("categorical", "binary") and ind_type == "continuous":
                plot  = plot_boxplot_by_category(df, dep_var, ind_var, cat_is_dep=True)
                title = f"{ind_label}  by  {dep_label}"
            else:
                continue

            bivariate_plots.append({"title": title, "plot": plot})

    # ── Model ──────────────────────────────────────────────────────────────
    print("  Fitting model")
    model_results = fit_model(df, config, dependent, independent)

    # ── Render HTML ────────────────────────────────────────────────────────
    overview = {
        "n_rows":      len(df),
        "n_cols":      len(df.columns),
        "n_missing":   int(df.isna().sum().sum()),
        "missing_pct": f"{df.isna().mean().mean() * 100:.1f}%",
    }

    tpl_dir = Path(__file__).parent / "templates"
    env     = Environment(loader=FileSystemLoader(str(tpl_dir)), autoescape=False)
    tpl     = env.get_template("report.html.j2")

    html = tpl.render(
        project         = config.get("project", {}),
        overview        = overview,
        var_sections    = var_sections,
        corr_plot       = corr_plot,
        bivariate_plots = bivariate_plots,
        model_results   = model_results,
        generated_at    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config_path     = config_path,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nReport saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a UW-Madison styled reproducible HTML report."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--output", default="output/report.html",
                        help="Output HTML path (default: output/report.html)")
    args = parser.parse_args()
    generate_report(args.config, args.output)
