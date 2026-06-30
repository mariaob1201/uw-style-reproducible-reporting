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
# Helpers
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

    # Histogram + KDE overlay
    ax_hist.hist(data, bins=30, color=UW["red"], alpha=0.72, edgecolor="white", linewidth=0.4)
    ax_hist.set_title(f"Distribution — {label}", fontweight="bold", color=UW["dark"], fontsize=11)
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("Count")

    ax2 = ax_hist.twinx()
    ax2.set_ylabel("Density", color=UW["gray"], fontsize=9)
    ax2.tick_params(colors=UW["gray"], labelsize=8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(UW["border"])
    # KDE via numpy
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 300)
    ax2.plot(xs, kde(xs), color=UW["dark"], linewidth=2, zorder=5)
    ax2.set_ylim(bottom=0)

    # Boxplot
    bp = ax_box.boxplot(
        data,
        patch_artist=True,
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
    xl = x_var.get("label", xc)
    yl = y_var.get("label", yc)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mask = df[[xc, yc]].notna().all(axis=1)
    ax.scatter(df.loc[mask, xc], df.loc[mask, yc],
               color=UW["red"], alpha=0.35, s=22, edgecolors="none", zorder=3)

    # OLS trend line
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
    cl = cat_var.get("label", cc)
    nl = cont_var.get("label", nc)

    top_cats = df[cc].value_counts().head(10).index.tolist()
    sub = df[df[cc].isin(top_cats)].dropna(subset=[cc, nc])

    fig, ax = plt.subplots(figsize=(max(6, len(top_cats) * 0.9), 4.5))

    groups = [sub.loc[sub[cc] == cat, nc].values for cat in top_cats]
    palette = [UW["red"], UW["dark"], UW["gray"], UW["gold"]] * 3
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        medianprops={"color": UW["gold"], "linewidth": 2},
        whiskerprops={"color": UW["gray"]},
        capprops={"color": UW["dark"]},
        flierprops={"markerfacecolor": UW["red"], "marker": "o",
                    "markersize": 4, "alpha": 0.5, "markeredgewidth": 0},
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticks(range(1, len(top_cats) + 1))
    ax.set_xticklabels(top_cats, rotation=25, ha="right", fontsize=9)
    ax.set_xlabel(cl); ax.set_ylabel(nl)
    title = (f"{nl}  by  {cl}" if not cat_is_dep
             else f"{cl}  distribution by  {nl}")
    ax.set_title(title, fontweight="bold", color=UW["dark"], fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Correlation heatmap (pure matplotlib — no seaborn)
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

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [c.replace("_", " ").title() for c in cols]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

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
# Model fitting
# ---------------------------------------------------------------------------

def fit_model(df: pd.DataFrame, config: dict,
              dependent: list, independent: list) -> dict | None:
    model_cfg = config.get("model")
    if not model_cfg or not dependent or not independent:
        return None

    model_type = model_cfg.get("type", "linear_regression")
    dep_var    = dependent[0]
    dep_col    = dep_var["name"]
    ind_cols   = [v["name"] for v in independent]
    all_cols   = [dep_col] + ind_cols

    try:
        import statsmodels.api as sm

        df_m = df[[c for c in all_cols if c in df.columns]].dropna()
        cat_cols = [c for c in ind_cols
                    if not pd.api.types.is_numeric_dtype(df_m[c])]
        df_enc = pd.get_dummies(df_m, columns=cat_cols, drop_first=True)
        # Ensure bool columns are int
        bool_cols = df_enc.select_dtypes(include="bool").columns
        df_enc[bool_cols] = df_enc[bool_cols].astype(int)

        y = df_enc[dep_col].astype(float)
        X = df_enc.drop(columns=[dep_col]).astype(float)
        X_c = sm.add_constant(X)

        out = {
            "type":        model_type,
            "description": model_cfg.get("description", ""),
            "n_obs":       len(df_m),
            "dep_label":   dep_var.get("label", dep_col),
        }

        if model_type == "linear_regression":
            fitted = sm.OLS(y, X_c).fit()
            out["summary_html"]    = fitted.summary().as_html()
            out["r_squared"]       = f"{fitted.rsquared:.4f}"
            out["adj_r_squared"]   = f"{fitted.rsquared_adj:.4f}"
            out["f_pvalue"]        = f"{fitted.f_pvalue:.4g}"
            out["residual_plot"]   = _residual_plots(fitted)

        elif model_type == "logistic_regression":
            fitted = sm.Logit(y, X_c).fit(disp=0, maxiter=200)
            out["summary_html"]      = fitted.summary().as_html()
            out["pseudo_r_squared"]  = f"{fitted.prsquared:.4f}"
            out["llr_pvalue"]        = f"{fitted.llr_pvalue:.4g}"

        else:
            out["error"] = f"Unknown model type '{model_type}'. Use: linear_regression, logistic_regression"

        return out

    except Exception as exc:
        return {"type": model_type, "error": str(exc)}


def _residual_plots(fitted) -> str:
    from scipy.stats import probplot
    _apply_uw_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.scatter(fitted.fittedvalues, fitted.resid,
                color=UW["red"], alpha=0.35, s=20, edgecolors="none", zorder=3)
    ax1.axhline(0, color=UW["dark"], linestyle="--", linewidth=1.5, zorder=4)
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted", fontweight="bold", color=UW["dark"], fontsize=11)

    (osm, osr), (slope, intercept, _) = probplot(fitted.resid)
    ax2.scatter(osm, osr, color=UW["red"], alpha=0.35, s=20, edgecolors="none", zorder=3)
    x_line = np.array([osm.min(), osm.max()])
    ax2.plot(x_line, slope * x_line + intercept,
             color=UW["dark"], linewidth=2, linestyle="--", zorder=4)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Q-Q Plot of Residuals", fontweight="bold", color=UW["dark"], fontsize=11)

    fig.tight_layout()
    return _fig_to_b64(fig)


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

    # ── Correlation matrix ────────────────────────────────────────────────
    cont_cols = [
        v["name"] for v in all_vars
        if v.get("type") == "continuous" and v["name"] in df.columns
    ]
    print("  Building correlation matrix")
    corr_plot = plot_correlation_matrix(df, cont_cols)

    # ── Bivariate: each independent vs dependent ──────────────────────────
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
                plot = plot_scatter(df, ind_var, dep_var)
                title = f"{ind_label}  →  {dep_label}"
            elif dep_type == "continuous" and ind_type in ("categorical", "binary", "ordinal"):
                plot = plot_boxplot_by_category(df, ind_var, dep_var, cat_is_dep=False)
                title = f"{dep_label}  by  {ind_label}"
            elif dep_type in ("categorical", "binary") and ind_type == "continuous":
                plot = plot_boxplot_by_category(df, dep_var, ind_var, cat_is_dep=True)
                title = f"{ind_label}  by  {dep_label}"
            else:
                continue

            bivariate_plots.append({"title": title, "plot": plot})

    # ── Model ─────────────────────────────────────────────────────────────
    print("  Fitting model")
    model_results = fit_model(df, config, dependent, independent)

    # ── Render HTML ───────────────────────────────────────────────────────
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
        project       = config.get("project", {}),
        overview      = overview,
        var_sections  = var_sections,
        corr_plot     = corr_plot,
        bivariate_plots = bivariate_plots,
        model_results = model_results,
        generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config_path   = config_path,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nReport saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a UW-Madison styled reproducible HTML report from a config file."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--output", default="output/report.html",
                        help="Output HTML path (default: output/report.html)")
    args = parser.parse_args()
    generate_report(args.config, args.output)
