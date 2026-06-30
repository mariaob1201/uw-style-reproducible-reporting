#!/usr/bin/env python3
"""
UW-Madison Reproducible Report Generator — pipeline manager.

Usage:
    python generate_report.py
    python generate_report.py --config config.yaml --output output/report.html
"""
import argparse
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from report_logic import (
    load_config,
    load_data,
    get_variable_groups,
    summary_stats,
    plot_continuous,
    plot_categorical,
    plot_scatter,
    plot_boxplot_by_category,
    plot_correlation_matrix,
    fit_model,
)


def generate_report(config_path: str, output_path: str):
    config = load_config(config_path)
    df     = load_data(config)
    dependent, independent = get_variable_groups(config)
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
