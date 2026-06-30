"""Run this once to generate sample_employees.csv for all example configs."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 350

education_levels = ["High School", "Bachelor", "Master", "PhD"]
departments      = ["Engineering", "Marketing", "Finance", "Operations", "Research"]
regions          = ["North", "South", "East", "West", "Central"]

education  = np.random.choice(education_levels, n, p=[0.18, 0.44, 0.26, 0.12])
department = np.random.choice(departments, n, p=[0.28, 0.18, 0.20, 0.16, 0.18])
region     = np.random.choice(regions, n)
age        = np.random.normal(38, 10, n).clip(22, 65).astype(int)

edu_bonus  = {"High School": 0, "Bachelor": 14_000, "Master": 30_000, "PhD": 52_000}
dept_bonus = {"Engineering": 22_000, "Marketing": 4_000, "Finance": 16_000,
              "Operations": 0, "Research": 26_000}

years_experience = (age - 22 + np.random.normal(0, 2, n)).clip(0, 40).round(1)

salary = (
    42_000
    + years_experience * 2_400
    + np.array([edu_bonus[e] for e in education])
    + np.array([dept_bonus[d] for d in department])
    + np.random.normal(0, 9_000, n)
).clip(25_000, 260_000).round(-2).astype(int)

# A/B variant column: treatment group gets a ~5% salary bump on average
variant      = np.where(np.arange(n) % 2 == 0, "control", "treatment")
salary_ab    = salary.copy().astype(float)
trt_idx      = variant == "treatment"
salary_ab[trt_idx] += np.random.normal(3_500, 8_000, trt_idx.sum())
salary_ab    = salary_ab.clip(25_000, 260_000).round(-2).astype(int)

# Count variable: training sessions attended (Poisson-distributed)
# More experienced & higher-ed employees attend more sessions
lam = np.exp(
    0.5
    + 0.02 * years_experience
    + np.array([0, 0.1, 0.3, 0.6])[
        pd.Categorical(education, categories=education_levels).codes
    ]
)
training_sessions = np.random.poisson(lam).clip(0, 15)

# Inject ~5% missing values in years_experience
yrs = years_experience.astype(float)
yrs[np.random.choice(n, int(0.05 * n), replace=False)] = np.nan

df = pd.DataFrame({
    "salary":             salary,
    "salary_ab":          salary_ab,     # for A/B test example
    "years_experience":   yrs,
    "age":                age,
    "education_level":    education,
    "department":         department,
    "region":             region,        # grouping variable for mixed LM
    "training_sessions":  training_sessions,  # count variable for GLM Poisson
    "variant":            variant,       # A/B group column
})

out = Path(__file__).parent / "sample_employees.csv"
df.to_csv(out, index=False)
print(f"Created {out} — {len(df)} rows, {df.isna().sum().sum()} missing values")
print(f"Columns: {list(df.columns)}")
