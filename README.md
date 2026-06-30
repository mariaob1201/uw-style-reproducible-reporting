# UW-Madison Reproducible Report Framework

A config-driven Python framework that generates a self-contained HTML data exploration report with [UW-Madison](https://www.wisc.edu) branding. Point it at a CSV, describe your variables in a YAML file, and get a styled report with EDA plots, summary statistics, and a fitted model — no notebooks required.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the sample dataset (optional — only needed for the demo config)
python data/create_sample.py

# 3. Run the report
python generate_report.py

# Output: output/report.html
```

Open `output/report.html` in any browser. The file is fully self-contained — all plots are embedded as base64 images, no internet connection required.

---

## How it works

Edit `config.yaml` to describe your dataset. The script reads the config, loads the CSV, and produces a report with:

| Section | What it shows |
|---|---|
| **Overview** | Row count, column count, missing value summary |
| **Per-variable EDA** | Histogram + KDE + boxplot (continuous); bar chart (categorical); descriptive stats for both |
| **Correlation matrix** | Pearson heatmap of all continuous variables |
| **Bivariate analysis** | Scatter + trend line (continuous × continuous); boxplots (categorical × continuous) |
| **Model results** | Statsmodels regression table, R², F-test p-value, residuals vs fitted, Q-Q plot |

---

## Config reference

```yaml
project:
  title: "My Analysis"
  description: "Short description shown in the report header"
  author: "Your Name"

data:
  path: "data/my_dataset.csv"
  delimiter: ","          # optional, defaults to ","

variables:
  - name: "salary"        # must match column name in CSV exactly
    type: "continuous"    # continuous | categorical | binary | ordinal
    role: "dependent"     # dependent | independent | id | excluded
    label: "Annual Salary ($)"   # optional human-readable label

  - name: "education_level"
    type: "categorical"
    role: "independent"
    label: "Education Level"

  # id and excluded roles are loaded but skipped in analysis
  - name: "employee_id"
    type: "continuous"
    role: "id"

model:
  type: "linear_regression"    # linear_regression | logistic_regression
  description: "Narrative description shown in the report"
```

### Variable types

| Type | Description | EDA plot |
|---|---|---|
| `continuous` | Numeric, unbounded | Histogram + KDE, boxplot |
| `categorical` | Unordered discrete | Horizontal bar chart |
| `binary` | Two-valued (0/1, True/False, Yes/No) | Horizontal bar chart |
| `ordinal` | Ordered discrete | Horizontal bar chart |

### Variable roles

| Role | Description |
|---|---|
| `dependent` | Outcome / response variable (one per config) |
| `independent` | Predictor / feature variable |
| `id` | Row identifier — loaded but excluded from all analysis |
| `excluded` | Excluded from analysis entirely |

### Supported models

| Model type | When to use | Dependent variable |
|---|---|---|
| `linear_regression` | Continuous outcome (OLS via statsmodels) | `continuous` |
| `logistic_regression` | Binary outcome (Logit via statsmodels) | `binary` or `categorical` with 2 levels |

Categorical independent variables are automatically one-hot encoded. Missing rows are dropped before model fitting.

---

## Project structure

```
uw-style-reproducible-reporting/
├── config.yaml              # ← edit this for your data
├── generate_report.py       # ← run this
├── requirements.txt
├── data/
│   ├── create_sample.py     # generates demo CSV
│   └── sample_employees.csv # 350-row synthetic demo dataset
└── templates/
    └── report.html.j2       # Jinja2 template (UW-Madison CSS)
```

The `output/` directory is created automatically and is excluded from version control.

---

## Using your own data

1. Copy `config.yaml` or edit it in place.
2. Set `data.path` to your CSV.
3. List each column you want analyzed under `variables` with its `type` and `role`.
4. Set `model.type` to match your dependent variable.
5. Run `python generate_report.py --config config.yaml --output output/report.html`.

You can maintain multiple configs for different analyses:

```bash
python generate_report.py --config configs/model_a.yaml --output output/model_a.html
python generate_report.py --config configs/model_b.yaml --output output/model_b.html
```

---

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, scipy, statsmodels, scikit-learn, jinja2, pyyaml

Install everything with:

```bash
pip install -r requirements.txt
```

> **Note:** Tested with pandas 3.x. String columns with `StringDtype` are handled correctly for categorical encoding.

---

## Design

Report styling follows the [UW-Madison visual identity](https://brand.wisc.edu):

- **Cardinal Red** `#C5050C` — primary headings, KPI cards, chart fills
- **Gold** `#f2a900` — accent (chart medians, footer bar)
- **Dark** `#282728` — body text, sidebar background
- **Gray** `#646569` — secondary text, axis labels
