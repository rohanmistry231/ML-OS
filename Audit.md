# Audit.md

### The Complete ML Pipeline Audit Guide — Every Stage, Every Failure Point, Every Check That Stands Between You and Silent Model Collapse

> **Why this file exists:** A model hit 11.5% weighted MAPE. The team celebrated. Then a routine feature audit revealed three features were computed using the target variable itself. After removing them, error jumped to 71.3%. The model had never been forecasting — it had been reading the answer sheet. `Audit.md` exists so that never happens to you.
>
> **Drop this file into every ML project repo. Run it at every stage. No exceptions.**

---

## Table of Contents

1. [Philosophy of ML Auditing](#1-philosophy-of-ml-auditing)
2. [Stage 1 — Problem Definition Audit](#2-stage-1--problem-definition-audit)
3. [Stage 2 — Data Collection & Ingestion Audit](#3-stage-2--data-collection--ingestion-audit)
4. [Stage 3 — Exploratory Data Analysis (EDA) Audit](#4-stage-3--exploratory-data-analysis-eda-audit)
5. [Stage 4 — Train / Validation / Test Split Audit](#5-stage-4--train--validation--test-split-audit)
6. [Stage 5 — Feature Engineering Audit ⚠️ CRITICAL](#6-stage-5--feature-engineering-audit-️-critical)
7. [Stage 6 — Preprocessing & Transformation Audit](#7-stage-6--preprocessing--transformation-audit)
8. [Stage 7 — Model Selection Audit](#8-stage-7--model-selection-audit)
9. [Stage 8 — Model Training Audit](#9-stage-8--model-training-audit)
10. [Stage 9 — Model Evaluation Audit](#10-stage-9--model-evaluation-audit)
11. [Stage 10 — Hyperparameter Tuning Audit](#11-stage-10--hyperparameter-tuning-audit)
12. [Stage 11 — Interpretability & Explainability Audit](#12-stage-11--interpretability--explainability-audit)
13. [Stage 12 — Deployment Readiness Audit](#13-stage-12--deployment-readiness-audit)
14. [Stage 13 — Post-Deployment Monitoring Audit](#14-stage-13--post-deployment-monitoring-audit)
15. [Stage 14 — Time-Series & Forecasting Audit](#15-stage-14--time-series--forecasting-audit)
16. [Data Leakage — Master Reference](#16-data-leakage--master-reference)
17. [Bias & Fairness Audit](#17-bias--fairness-audit)
18. [Class Imbalance & Sparse Target Audit](#18-class-imbalance--sparse-target-audit)
19. [Feature Store & Reproducibility Audit](#19-feature-store--reproducibility-audit)
20. [Statistical Testing Audit](#20-statistical-testing-audit)
21. [Pipeline Automation & CI/CD Audit](#21-pipeline-automation--cicd-audit)
22. [Model Documentation & Governance Audit](#22-model-documentation--governance-audit)
23. [Audit Templates & Checklists](#23-audit-templates--checklists)
24. [Appendix — Quick Reference Card](#24-appendix--quick-reference-card)

---

## 1. Philosophy of ML Auditing

### What Is an ML Audit?

An ML audit is a systematic, checkpoint-driven process of verifying the correctness, fairness, reliability, and generalizability of every decision made during a machine learning pipeline.

It is not a one-time review. It is not a post-mortem after something breaks. It is a continuous discipline embedded into every stage — the same way software engineering has code review, testing, and CI/CD. ML needs audit gates.

### The Three Laws of ML Auditing

```
LAW 1 — The Law of Temporal Integrity
    No feature used at training time should contain information
    that would not be available at the exact moment of prediction.
    This is the single most violated law in applied ML.

LAW 2 — The Law of Split Sanctity
    The test set is sacred. Once created, it must never influence
    any decision — not preprocessing, not feature selection,
    not tuning, not even a "quick look."

LAW 3 — The Law of Skeptical Performance
    If your model performs suspiciously well, assume it is wrong
    before assuming you are brilliant. Investigate until proven otherwise.
```

### Why Most Teams Skip Audits (And Pay For It)

| Excuse | What Actually Happens |
|---|---|
| "We don't have time" | 3× more time debugging production failures that a 20-minute audit would have caught |
| "Our data is clean" | Data is never clean. Ever. The question is where it's dirty. |
| "We trust our pipeline" | Even senior engineers introduce leakage — it's a systematic problem, not a competence problem |
| "Metrics look good" | Metrics look good **because** of leakage, not despite it |
| "We'll audit at the end" | Bugs compound. Early leakage corrupts every downstream decision. |

### Audit Severity Tiers

Every check in this guide is tagged with a severity level:

```
🔴 CRITICAL    — Skipping this can silently invalidate your entire model.
                 Data leakage, temporal violations, target contamination.
                 These checks are non-negotiable.

🟠 HIGH        — Skipping this will degrade model quality or reliability.
                 Overfitting signals, metric misselection, preprocessing bugs.
                 Skip only with documented justification.

🟡 MODERATE    — Skipping this introduces risk that compounds over time.
                 Distribution drift, subgroup performance, calibration.
                 Address before production deployment.

🔵 ADVISORY    — Best practice. Improves robustness and maintainability.
                 Documentation, versioning, reproducibility.
                 Prioritize based on team maturity.
```

### Audit Mindset Hierarchy

```
Level 1 — Reactive:    "Something broke, let's investigate"          ❌ Too late
Level 2 — Periodic:    "Let's review the model every quarter"        ⚠️ Better
Level 3 — Systematic:  "Every pipeline stage has a gate"             ✅ Good
Level 4 — Automated:   "Audit checks run in CI/CD on every commit"   ✅✅ Best
```

This guide is written to get you from Level 1 to Level 3 immediately, and provides automation patterns for Level 4.

---

## 2. Stage 1 — Problem Definition Audit

> Most ML projects fail before a single line of code is written. A misspecified problem means you optimize for the wrong thing — and no amount of engineering fixes a wrong objective.

### Audit Checklist

#### 1.1 — Target Variable Definition 🔴 CRITICAL

- [ ] Target variable is precisely and unambiguously defined in writing
- [ ] There is exactly ONE definition of the target — not multiple interpretations across teams
- [ ] Target is measured at the correct unit of analysis (user-level? transaction-level? daily? weekly?)
- [ ] Target variable will be available at the same granularity in production
- [ ] Label generation logic is documented, reviewed, and tested with edge cases
- [ ] Target variable does not encode future information (e.g., `days_until_churn` computed using the churn date itself)

**Red Flags:**
- Target means different things to different stakeholders ("churn" to product ≠ "churn" to data science)
- Target leaks into feature space (e.g., `total_refund_amount` when predicting whether an order will be returned)
- Target computed using data from after the prediction point

#### 1.2 — Business Objective Alignment 🟠 HIGH

- [ ] ML metric (RMSE, AUC, F1, wMAPE) maps to an actual business outcome (revenue, cost, safety)
- [ ] There is a clear chain: model score → business decision → measurable impact
- [ ] Stakeholders have explicitly agreed on the primary metric (not assumed)
- [ ] Decision threshold is defined (e.g., "flag if predicted probability > 0.6")
- [ ] Cost asymmetry is explicitly discussed — is a false positive or false negative more expensive?

**Cost Asymmetry Framework:**
```
For every ML task, fill this in before writing code:

Business Question:     [What decision does this model support?]
ML Task:               [Classification / Regression / Ranking / Forecasting]
Primary Metric:        [Chosen metric and why]
Cost of FP:            [What happens when the model says YES incorrectly?]
Cost of FN:            [What happens when the model says NO incorrectly?]
Which is worse?        [FP / FN / Equal]
Threshold implication: [Higher threshold = fewer FP / Lower threshold = fewer FN]
```

#### 1.3 — Data Availability & Latency Audit 🔴 CRITICAL

- [ ] Every data source needed at training time is also available at inference time
- [ ] Latency of each data source is documented (some features arrive hours or days late)
- [ ] There is a minimum data freshness requirement — and it's enforced
- [ ] Fallback behavior is defined for when a data source is unavailable

**This check alone would have prevented countless production failures.** A model trained on features that don't exist at inference time is useless — no matter how good the offline metrics look.

#### 1.4 — Prediction Horizon & Granularity Audit 🔴 CRITICAL

- [ ] Prediction horizon is explicitly defined (predict what, how far ahead?)
- [ ] Granularity is locked (daily, weekly, per-user, per-item)
- [ ] The gap between "when the model runs" and "when the prediction is needed" is documented
- [ ] Staleness tolerance is defined (how old can a prediction be before it's useless?)

---

## 3. Stage 2 — Data Collection & Ingestion Audit

### Audit Checklist

#### 2.1 — Source Integrity 🔴 CRITICAL

- [ ] Data lineage is documented for every column (where did it come from? what system? what query?)
- [ ] If multiple sources are joined, the join key is validated (no silent many-to-many explosions)
- [ ] Data pull is reproducible — same query on same date range produces identical output
- [ ] Timestamps on records are trustworthy and consistent (timezone, format, granularity)
- [ ] No records have timestamps in the future relative to your training cutoff date

**Silent Join Failures:**
```python
# Before any join, always check:
assert left_df[join_key].is_unique, f"Duplicate keys in left table on '{join_key}'"
assert right_df[join_key].is_unique, f"Duplicate keys in right table on '{join_key}'"

# After the join:
assert len(merged) == len(left_df), \
    f"Row count changed after join: {len(left_df)} → {len(merged)}. Check for duplicates."
```

#### 2.2 — Schema Validation 🟠 HIGH

- [ ] Column names, data types, and value ranges match expectations
- [ ] All expected columns are present
- [ ] Unexpected extra columns are flagged (could be leakage vectors)
- [ ] Data types are correct (no integers stored as strings, no dates stored as floats)

```python
def schema_audit(df, expected_schema: dict) -> list:
    """
    Validate DataFrame schema against expectations.
    expected_schema = {'column_name': 'expected_dtype', ...}
    Returns list of issues found.
    """
    issues = []

    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            issues.append(f"🔴 MISSING COLUMN: '{col}'")
        elif str(df[col].dtype) != expected_dtype:
            issues.append(
                f"🟠 TYPE MISMATCH: '{col}' — expected {expected_dtype}, got {df[col].dtype}"
            )

    extra_cols = set(df.columns) - set(expected_schema.keys())
    if extra_cols:
        issues.append(f"⚠️  UNEXPECTED COLUMNS (potential leakage vector): {extra_cols}")

    return issues if issues else ["✅ Schema validation passed."]
```

#### 2.3 — Volume & Completeness Audit 🟠 HIGH

- [ ] Row count is within expected range (flag if >20% deviation from historical baseline)
- [ ] Exact duplicate rows identified and counted
- [ ] Near-duplicate rows with different labels identified (label noise)
- [ ] Missing value percentage per column documented
- [ ] Missingness mechanism investigated — MCAR, MAR, or MNAR?

```python
def completeness_audit(df):
    """Profile missing values and flag potential issues."""
    total = len(df)
    report = []

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct = round((n_missing / total) * 100, 2)
        severity = (
            "🔴" if pct > 50 else
            "🟠" if pct > 20 else
            "🟡" if pct > 5 else
            "✅"
        )
        if n_missing > 0:
            report.append({
                "column": col,
                "missing_count": n_missing,
                "missing_pct": pct,
                "severity": severity,
            })

    return sorted(report, key=lambda x: x["missing_pct"], reverse=True)
```

#### 2.4 — Temporal Integrity Audit 🔴 CRITICAL (for any time-dependent data)

- [ ] Earliest and latest record timestamps are correct and expected
- [ ] No gaps in time coverage (missing days, missing weeks)
- [ ] Data collection frequency is consistent
- [ ] No periods of abnormal data density (could indicate a logging bug or system event)

```python
def temporal_integrity_audit(df, date_col, expected_freq="D"):
    """Check for gaps and anomalies in time coverage."""
    dates = pd.to_datetime(df[date_col]).sort_values()
    full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=expected_freq)
    missing_dates = full_range.difference(dates.dt.normalize().unique())

    print(f"Date range:    {dates.min()} → {dates.max()}")
    print(f"Expected days: {len(full_range)}")
    print(f"Missing days:  {len(missing_dates)}")

    if len(missing_dates) > 0:
        print(f"🟠 First 10 missing: {list(missing_dates[:10])}")

    return missing_dates
```

#### 2.5 — Data Drift at Ingestion 🟡 MODERATE

- [ ] Compare current data pull distributions against historical baseline
- [ ] Flag columns where distribution has shifted significantly (KS test, PSI)
- [ ] Investigate whether drift is real-world change or a pipeline bug

---

## 4. Stage 3 — Exploratory Data Analysis (EDA) Audit

> **Critical Rule:** EDA must be performed on the TRAINING SET ONLY. Looking at the full dataset during EDA leaks test set information into your mental model and feature decisions.

### Audit Checklist

#### 3.1 — Univariate Distribution Audit 🟠 HIGH

- [ ] Every feature's distribution reviewed against domain expectations
- [ ] Outliers classified as errors vs. genuine extreme values
- [ ] Near-zero variance features identified (low information — consider dropping)
- [ ] Constant features identified and removed (zero predictive value)
- [ ] Categorical feature cardinalities are manageable (not 50,000 unique categories)

#### 3.2 — Target Variable Audit 🔴 CRITICAL

- [ ] Target distribution documented (class balance for classification, distribution shape for regression)
- [ ] Class imbalance quantified (if applicable)
- [ ] Zero-inflation quantified (for count/demand data: what % of target values are zero?)
- [ ] Target stability over time verified (distribution shift in the target = concept drift)
- [ ] Target values are within physically plausible range

**Zero-Inflation Check (common in demand, fraud, medical events):**
```python
def target_sparsity_audit(y, zero_threshold=0.5):
    """Flag if target is dominated by zeros — affects metric and model choice."""
    zero_pct = (y == 0).mean()
    print(f"Target zero-rate: {zero_pct:.1%}")

    if zero_pct > zero_threshold:
        print(f"🔴 Target is {zero_pct:.1%} zeros. Implications:")
        print(f"   → Accuracy/RMSE will be misleading (predicting 0 always 'works')")
        print(f"   → Consider: wMAPE, MAE, or two-stage model (classify then regress)")
        print(f"   → Evaluate on non-zero subset separately")
    return zero_pct
```

#### 3.3 — Correlation & Leakage Suspicion Audit ⚠️ 🔴 CRITICAL

- [ ] Correlation of every feature with target computed and reviewed
- [ ] Any feature with |correlation| > 0.85 with target flagged for manual investigation
- [ ] Pairwise feature correlations computed — highly collinear pairs (>0.95) flagged
- [ ] Single-feature model trained for top correlated features — suspiciously high performance = leakage signal

```python
# Single-Feature Leakage Detector — Classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def leakage_audit_classification(df, target_col, threshold=0.90):
    """Train one-feature models. AUC above threshold = suspected leakage."""
    suspects = []
    feature_cols = [c for c in df.columns if c != target_col]

    for col in feature_cols:
        X = df[[col]].fillna(-9999)
        y = df[target_col]
        model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

        if auc > threshold:
            suspects.append({"feature": col, "single_feature_AUC": round(auc, 4)})

    return pd.DataFrame(suspects).sort_values("single_feature_AUC", ascending=False)
```

```python
# Single-Feature Leakage Detector — Regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def leakage_audit_regression(df, target_col, baseline_mae=None):
    """
    Train one-feature models. If a single feature achieves MAE
    dramatically below baseline, investigate for leakage.
    """
    suspects = []
    feature_cols = [c for c in df.columns if c != target_col]
    y = df[target_col]

    if baseline_mae is None:
        baseline_mae = mean_absolute_error(y, np.full(len(y), y.mean()))

    for col in feature_cols:
        X = df[[col]].fillna(-9999)
        model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)
        mae = mean_absolute_error(y, model.predict(X))
        reduction = 1 - (mae / baseline_mae)

        if reduction > 0.70:  # Single feature explains >70% of baseline error
            suspects.append({
                "feature": col,
                "single_feature_MAE": round(mae, 4),
                "error_reduction_vs_baseline": f"{reduction:.1%}",
            })

    return pd.DataFrame(suspects).sort_values("single_feature_MAE")
```

#### 3.4 — Temporal Pattern Audit 🟠 HIGH (for time-dependent data)

- [ ] Target variable plotted over time — sudden jumps or drops investigated
- [ ] Feature distributions plotted across time — drift patterns identified
- [ ] Certain time periods are not over/under-represented
- [ ] Gap between event and observation is consistent

---

## 5. Stage 4 — Train / Validation / Test Split Audit

> **This is the most structurally important stage. Errors here silently corrupt everything downstream. There is no recovery.**

### Audit Checklist

#### 4.1 — Split Timing Audit 🔴 CRITICAL

- [ ] Split happens **BEFORE** any preprocessing, scaling, encoding, imputation, or feature selection
- [ ] Split indices are saved and frozen — identical split used across all experiments
- [ ] No feature engineering decisions were informed by test set data

```python
# ❌ WRONG — Scaling before split leaks test statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)             # fit() sees ALL data including test
X_train, X_test = train_test_split(X_scaled)   # Test data is already contaminated

# ✅ CORRECT — Split first, then fit on train only
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit() sees train only
X_test_scaled = scaler.transform(X_test)         # transform() only — no fitting
```

#### 4.2 — Temporal Split Audit 🔴 CRITICAL (for any time-dependent data)

- [ ] Time-based split used — NOT random split
- [ ] Training data is strictly from the past; test data is strictly from the future
- [ ] No data from the test period leaks into training features (lag features, rolling means)
- [ ] A buffer gap exists between training end and test start (prevents near-future leakage)

```python
def temporal_split_with_buffer(df, date_col, train_end, buffer_days, test_start=None):
    """
    Split with explicit buffer gap to prevent near-future leakage.

    Example:
        train_end   = '2024-06-30'
        buffer_days = 7
        test_start  = '2024-07-08'  (auto-calculated if not provided)
    """
    df[date_col] = pd.to_datetime(df[date_col])

    if test_start is None:
        test_start = pd.to_datetime(train_end) + pd.Timedelta(days=buffer_days)

    train = df[df[date_col] <= train_end].copy()
    test = df[df[date_col] >= test_start].copy()
    buffer = df[(df[date_col] > train_end) & (df[date_col] < test_start)]

    print(f"Train:  {len(train):,} rows  |  {df[date_col].min()} → {train_end}")
    print(f"Buffer: {len(buffer):,} rows  |  EXCLUDED")
    print(f"Test:   {len(test):,} rows   |  {test_start} → {df[date_col].max()}")

    assert len(train) > 0, "Empty training set"
    assert len(test) > 0, "Empty test set"
    assert len(train) + len(buffer) + len(test) == len(df), "Row count mismatch"

    return train, test
```

#### 4.3 — Group Leakage Audit 🔴 CRITICAL

- [ ] If data has group structure (multiple rows per entity — per user, per product, per sensor), verify:
  - Same entity does NOT appear in both train and test
  - Use `GroupKFold` or `GroupShuffleSplit` if group-level features exist

```python
def group_overlap_audit(train_df, test_df, group_col):
    """Detect entities appearing in both train and test — a common leakage source."""
    train_groups = set(train_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())
    overlap = train_groups & test_groups

    if overlap:
        print(f"🔴 {len(overlap)} groups appear in BOTH train and test!")
        print(f"   If features are group-level aggregations, this is leakage.")
        print(f"   Sample overlapping groups: {list(overlap)[:5]}")
    else:
        print(f"✅ No group overlap between train and test.")

    return overlap
```

#### 4.4 — Stratification Audit 🟡 MODERATE

- [ ] For imbalanced classification — stratified split preserves class ratios
- [ ] Class ratios verified in both train and test after splitting
- [ ] For regression with skewed targets — consider stratified binning for the split

#### 4.5 — Split Proportion Audit 🟡 MODERATE

- [ ] Test set is large enough for statistically meaningful evaluation
- [ ] Validation set is representative (not a tiny sliver)
- [ ] For time-series: test period covers at least one full seasonal cycle

---

## 6. Stage 5 — Feature Engineering Audit ⚠️ CRITICAL

> **This is where most leakage enters the pipeline.** Every new feature you create is a potential vulnerability. Treat feature creation like you'd treat writing security-sensitive code — with gates, reviews, and paranoia.

### The Golden Rule of Feature Engineering

```
For EVERY feature you create, answer this question:

    "At the exact moment of prediction in production,
     using ONLY information available up to that point,
     could I compute this feature's exact value?"

If the answer is NO, MAYBE, or "I think so" — investigate immediately.
```

### Audit Checklist

#### 5.1 — Feature Availability Gate (Per Feature) 🔴 CRITICAL

**Every single feature** must pass this gate before entering the pipeline:

| Question | Required Answer |
|---|---|
| What is the data source of this feature? | Documented |
| When was this data generated relative to the prediction point? | Before |
| Is this available at inference time in production? | Yes |
| Does this use any future information, even indirectly? | No |
| Does this use information from the test set? | No |
| Does this feature use the target variable in its computation? | No |
| Single-feature model performance | Below suspicion threshold |

#### 5.2 — Aggregation Feature Audit 🔴 CRITICAL

Aggregation features (means, counts, sums, rolling windows) are the #1 leakage source.

- [ ] Rolling statistics use a strictly backward-looking window
- [ ] "Current period" value is NOT included in the aggregation when predicting the current period
- [ ] Group-level aggregations computed on training data only, then merged onto test

```python
# ❌ WRONG — Aggregation computed on all data including test period
df['category_avg_sales'] = df.groupby('category')['sales'].transform('mean')

# ✅ CORRECT — Compute on train, apply to both
train_agg = train_df.groupby('category')['sales'].mean().reset_index()
train_agg.columns = ['category', 'category_avg_sales']

train_df = train_df.merge(train_agg, on='category', how='left')
test_df = test_df.merge(train_agg, on='category', how='left')
```

#### 5.3 — Target Encoding Audit 🔴 CRITICAL

Target encoding is one of the most common leakage sources in practice.

- [ ] Target encoding computed on training data only
- [ ] Smoothing applied to reduce overfitting to rare categories
- [ ] If using cross-validation, target encoding recomputed within each fold

```python
# ❌ WRONG — Target encoding on full dataset
df['region_encoded'] = df.groupby('region')['target'].transform('mean')

# ✅ CORRECT — Fit on train, apply to test
train_enc = train_df.groupby('region')['target'].mean().reset_index()
train_enc.columns = ['region', 'region_encoded']

train_df = train_df.merge(train_enc, on='region', how='left')
test_df = test_df.merge(train_enc, on='region', how='left')
```

#### 5.4 — Lag Feature Audit 🔴 CRITICAL

- [ ] Lag direction verified — shifting backward (into the past), not forward
- [ ] `shift(-1)` = future value (LEAKAGE); `shift(1)` = past value (correct)
- [ ] Rows where lag introduces NaNs at series boundaries are handled explicitly
- [ ] Lag features respect entity boundaries (lag of entity A doesn't bleed into entity B)

```python
# ❌ WRONG — Accidental future leak
df['sales_next_week'] = df.groupby('product_id')['sales'].shift(-1)

# ✅ CORRECT — Past value only
df['sales_last_week'] = df.groupby('product_id')['sales'].shift(1)

# ✅ VERIFY — After creating lags, check for NaN pattern
assert df.groupby('product_id')['sales_last_week'].apply(
    lambda x: x.iloc[0]
).isna().all(), "First row per group should be NaN for lag features"
```

#### 5.5 — Derived Ratio & Interaction Feature Audit 🟠 HIGH

- [ ] Ratio features have denominator-zero protection
- [ ] Interaction features don't accidentally encode the target
- [ ] Features derived from other features inherit the strictest temporal constraint of their inputs

#### 5.6 — Feature Audit Log (Mandatory) 🔴 CRITICAL

**Maintain this log. Every feature must have an entry before it enters the pipeline.**

```
| Feature Name         | Source        | Stage Created | Uses Future? | Avail at Inference? | Single-Model Perf | Domain OK? | Status  |
|----------------------|---------------|---------------|--------------|---------------------|--------------------| -----------|---------|
| item_weight_kg       | Product DB    | Ingestion     | No           | Yes                 | MAE: 48.2         | Yes        | ✅ PASS |
| avg_sales_last_4w    | Sales DB      | Feature Eng   | No           | Yes                 | MAE: 31.7         | Yes        | ✅ PASS |
| total_annual_demand  | Sales DB      | Feature Eng   | YES ⚠️       | No                  | MAE: 2.1          | N/A        | ❌ FAIL |
```

A feature that can't fill out this row doesn't belong in your pipeline.

---

## 7. Stage 6 — Preprocessing & Transformation Audit

### Audit Checklist

#### 6.1 — Fit vs. Transform Audit 🔴 CRITICAL

The single most common preprocessing mistake in applied ML:

- [ ] All transformers (scalers, encoders, imputers) are fit ONLY on training data
- [ ] Test data is ONLY transformed — never used to fit
- [ ] This applies to: `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OrdinalEncoder`, `LabelEncoder`, `PCA`, `TF-IDF`, and any custom transformer

```python
# ✅ Use sklearn Pipeline to enforce correct behavior automatically
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

X_train_processed = preprocessing.fit_transform(X_train)  # Fit + transform on train
X_test_processed = preprocessing.transform(X_test)          # Transform only — no fitting
```

#### 6.2 — Imputation Audit 🟠 HIGH

- [ ] Imputation strategy is appropriate for the data type and missingness pattern
- [ ] Imputation value does not accidentally encode target information
- [ ] Missing values in the test set are handled using training-set statistics only
- [ ] If missingness is informative, a binary `_was_missing` indicator column is added

**Example of informative missingness:**
```
Feature: 'customer_credit_score'
Missing for 30% of records.
Investigation: Missing = customer never applied for credit = different risk profile.
→ Missingness itself is a signal. Add 'credit_score_was_missing' as a feature.
→ But don't impute with 0 — that looks like a real score to the model.
```

#### 6.3 — Encoding Audit 🟠 HIGH

- [ ] All categorical values in the test set were seen in training (unseen categories cause crashes or silent errors)
- [ ] Ordinal encoding order is correct and meaningful
- [ ] One-hot encoding uses `drop='first'` or equivalent to avoid perfect multicollinearity
- [ ] High-cardinality categoricals are handled appropriately (frequency encoding, hashing, embedding — not 10,000 one-hot columns)

```python
def unseen_category_audit(train_df, test_df, cat_cols):
    """Flag categories in test that were never seen in training."""
    for col in cat_cols:
        train_cats = set(train_df[col].dropna().unique())
        test_cats = set(test_df[col].dropna().unique())
        unseen = test_cats - train_cats

        if unseen:
            print(f"🟠 '{col}': {len(unseen)} unseen categories in test: {list(unseen)[:5]}")
        else:
            print(f"✅ '{col}': all test categories seen in training.")
```

#### 6.4 — Distribution Shift Post-Processing Audit 🟡 MODERATE

- [ ] After preprocessing, train and test distributions are compared (KS test)
- [ ] Large shifts may indicate temporal bias, selection bias, or a preprocessing bug

```python
from scipy.stats import ks_2samp

def distribution_drift_audit(train_df, test_df, numeric_cols, p_threshold=0.01):
    """Flag features where train and test distributions diverge significantly."""
    drifted = []
    for col in numeric_cols:
        stat, p_value = ks_2samp(
            train_df[col].dropna(), test_df[col].dropna()
        )
        if p_value < p_threshold:
            drifted.append({
                "feature": col,
                "KS_statistic": round(stat, 4),
                "p_value": f"{p_value:.2e}",
            })

    if drifted:
        print(f"🟠 {len(drifted)} features show significant distribution drift:")
    return pd.DataFrame(drifted).sort_values("KS_statistic", ascending=False)
```

---

## 8. Stage 7 — Model Selection Audit

### Audit Checklist

#### 7.1 — Baseline Audit 🔴 CRITICAL (Non-Negotiable First Step)

- [ ] A naive baseline is established BEFORE any ML model is trained
- [ ] Your final model must beat this baseline by a meaningful margin

**Baseline Options by Task Type:**

| Task | Baseline |
|---|---|
| Classification (balanced) | Majority class prediction |
| Classification (imbalanced) | Always predict majority class → measure F1, not accuracy |
| Regression | Predict global mean |
| Time-series forecasting | Predict last known value (naive persistence) |
| Demand forecasting | Predict same-period-last-year value |
| Ranking | Random ranking |
| Anomaly detection | Flag nothing (or flag everything) |

**If your model doesn't beat the baseline — something is wrong with the model, the features, or the problem definition. Do not proceed.**

#### 7.2 — Model Complexity vs. Data Size Audit 🟠 HIGH

- [ ] Model complexity is appropriate for dataset size

```
< 1,000 rows     → Linear/Logistic Regression, shallow trees
1K – 100K rows   → Gradient Boosting (XGBoost, LightGBM), Random Forest, SVMs
100K – 10M rows  → Gradient Boosting, Neural Networks viable
> 10M rows       → Neural Networks, distributed training, aggressive feature selection
```

- [ ] Number of features is reasonable relative to number of samples
- [ ] Simpler models tried first — complexity added only when justified by validation performance

#### 7.3 — Assumption Audit Per Model Type 🟡 MODERATE

| Model | Key Assumptions to Verify |
|---|---|
| Linear / Logistic Regression | Linearity, no perfect multicollinearity, no extreme outliers |
| Decision Tree | Check for overfitting (unrestricted depth) |
| Random Forest | Features are at least weakly predictive, sufficient trees |
| XGBoost / LightGBM | Learning rate vs. n_estimators tradeoff, early stopping configured |
| Neural Network | Sufficient data, proper normalization, learning rate tuned |
| KNN | Features are on same scale, distance metric is meaningful |
| SVM | Features scaled, kernel choice appropriate for data geometry |
| Naive Bayes | Feature independence assumption — verify if it even approximately holds |

---

## 9. Stage 8 — Model Training Audit

### Audit Checklist

#### 8.1 — Training Process Audit 🟠 HIGH

- [ ] Random seed set and documented for reproducibility
- [ ] All library versions logged (`requirements.txt`, `conda.yml`, or `poetry.lock`)
- [ ] Early stopping configured to prevent overfitting (for gradient boosting, neural nets)
- [ ] Training loss monitored for divergence, NaN, or sudden jumps
- [ ] Training data is shuffled appropriately (per epoch for NNs; NOT for time-series)

#### 8.2 — Cross-Validation Audit 🔴 CRITICAL

- [ ] Cross-validation performed on training data only — test set never touched
- [ ] All preprocessing steps are INSIDE the CV loop (not fitted before CV)
- [ ] For time-series data: `TimeSeriesSplit` or walk-forward validation used — NOT random `KFold`
- [ ] CV folds are consistent across experiments (same fold indices)

```python
# ❌ WRONG — Preprocessing outside CV leaks information across folds
X_scaled = scaler.fit_transform(X_train)    # fit() sees all training folds
scores = cross_val_score(model, X_scaled, y_train, cv=5)

# ✅ CORRECT — Preprocessing inside CV via Pipeline
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier()),
])
scores = cross_val_score(pipe, X_train, y_train, cv=5)
```

#### 8.3 — Overfitting Audit 🟠 HIGH

- [ ] Train vs. validation performance gap is monitored

```
Overfitting Severity Guide (Classification):
Train AUC − Val AUC < 0.02    → ✅ Healthy
Train AUC − Val AUC 0.02–0.05 → 🟡 Mild overfit — monitor
Train AUC − Val AUC 0.05–0.10 → 🟠 Moderate overfit — add regularization
Train AUC − Val AUC > 0.10    → 🔴 Severe overfit or data leakage — investigate

Overfitting Severity Guide (Regression):
Train MAPE − Val MAPE < 2%    → ✅ Healthy
Train MAPE − Val MAPE 2–5%    → 🟡 Mild overfit
Train MAPE − Val MAPE 5–15%   → 🟠 Moderate overfit
Train MAPE − Val MAPE > 15%   → 🔴 Severe overfit or leakage
```

- [ ] Learning curves plotted — training and validation should converge, not diverge
- [ ] Regularization applied appropriately (L1/L2 for linear, max_depth/min_samples for trees, dropout for NNs)

#### 8.4 — Underfitting Audit 🟡 MODERATE

- [ ] Both train and validation metrics are above baseline (if not → underfitting)
- [ ] Model is not predicting a constant or near-constant value
- [ ] Check: model actually uses the features (not ignoring them due to scaling issues, NaN handling, etc.)

---

## 10. Stage 9 — Model Evaluation Audit

### Audit Checklist

#### 9.1 — Metric Selection Audit 🔴 CRITICAL

- [ ] Chosen metric is appropriate for the task AND the data distribution

| Scenario | Recommended Metrics | Avoid |
|---|---|---|
| Balanced binary classification | AUC-ROC, F1, Accuracy | — |
| Imbalanced binary classification | AUC-PR, F1, Recall@K | Accuracy |
| Multi-class classification | Macro F1, per-class metrics | Overall accuracy alone |
| Regression (normal errors) | RMSE, R² | — |
| Regression (skewed target) | MAE, wMAPE, MedAE | RMSE (inflated by outliers) |
| Demand forecasting (sparse) | wMAPE, MAE, bias | MAPE (÷0), RMSE |
| Ranking | NDCG, MAP, MRR | — |
| Anomaly detection | AUC-ROC, Precision@K | Accuracy |

- [ ] Multiple metrics reported — no single number tells the full story
- [ ] Accuracy is NOT used as primary metric on imbalanced data

#### 9.2 — Test Set Evaluation Audit 🔴 CRITICAL

- [ ] Final evaluation done on the held-out test set exactly ONCE
- [ ] If you evaluated multiple times and chose the best run → test set is contaminated
- [ ] No feature engineering or model design decisions were made after looking at test performance
- [ ] Test set was never used for hyperparameter tuning

#### 9.3 — Suspiciously Good Performance Audit 🔴 CRITICAL

**If your results look too good — they probably are. Run these checks:**

```
Suspicion Triggers:
  • AUC > 0.99
  • Accuracy > 99% on non-trivial data
  • MAPE < 5% on a noisy real-world dataset
  • Near-zero RMSE
  • Model performance barely drops when you remove 80% of features

Investigation Checklist:
  Check 1: Run single-feature leakage audit           → Any feature too predictive?
  Check 2: Run group overlap audit                     → Same entities in train + test?
  Check 3: Manual column review                        → Is the target hiding in features?
  Check 4: Verify split timing                         → Split happened before feature eng?
  Check 5: Verify evaluation data                      → Correct test set indices?
  Check 6: Remove top 3 features by importance         → Does performance collapse?
           (If yes — those features ARE the model. Investigate them.)
```

#### 9.4 — Confusion Matrix / Error Analysis Audit 🟠 HIGH

**For classification:**
- [ ] Confusion matrix reviewed with a domain expert
- [ ] Most misclassified examples inspected manually (at least 20–50)
- [ ] Errors are not clustered in specific subgroups

**For regression:**
- [ ] Residual distribution is approximately centered at zero
- [ ] Residuals plotted against predicted values — no systematic patterns
- [ ] Largest errors inspected manually — are they data quality issues or model limitations?

```python
def residual_audit(y_true, y_pred, top_n=20):
    """Inspect largest prediction errors for patterns."""
    residuals = y_true - y_pred
    abs_errors = abs(residuals)

    print(f"Mean residual:   {residuals.mean():.4f}  (should be ≈ 0)")
    print(f"Residual std:    {residuals.std():.4f}")
    print(f"Median |error|:  {abs_errors.median():.4f}")
    print(f"95th pct |error|: {abs_errors.quantile(0.95):.4f}")
    print(f"Max |error|:     {abs_errors.max():.4f}")

    # Return worst predictions for manual inspection
    worst = pd.DataFrame({
        "actual": y_true, "predicted": y_pred,
        "error": residuals, "abs_error": abs_errors,
    }).nlargest(top_n, "abs_error")

    return worst
```

#### 9.5 — Threshold Audit 🟠 HIGH (Classification)

- [ ] Default 0.5 threshold is almost never optimal — audit this explicitly
- [ ] Threshold selected based on Precision-Recall tradeoff and business cost asymmetry
- [ ] Threshold choice is documented with business justification

---

## 11. Stage 10 — Hyperparameter Tuning Audit

### Audit Checklist

#### 10.1 — Tuning Data Isolation Audit 🔴 CRITICAL

- [ ] Tuning uses cross-validation on training data only
- [ ] Test set is never used during tuning — not even to "peek" at comparative results
- [ ] Validation set used during tuning is separate from the final test set

#### 10.2 — Search Space Audit 🟡 MODERATE

- [ ] Search space is reasonable — not too narrow (misses optimum) or too wide (wastes compute)
- [ ] Random search or Bayesian optimization used over grid search for large parameter spaces
- [ ] Number of trials is sufficient for the dimensionality of the search space

#### 10.3 — Overfitting to Validation Audit 🟠 HIGH

- [ ] Excessive tuning iterations can overfit to the validation set
- [ ] Final reported performance comes from the held-out test set, not the best validation score
- [ ] If hundreds/thousands of configurations were tried, apply multiple comparison correction

---

## 12. Stage 11 — Interpretability & Explainability Audit

### Audit Checklist

#### 11.1 — Feature Importance Audit 🔴 CRITICAL

- [ ] Feature importance computed and reviewed with a domain expert
- [ ] Top features make intuitive, logical sense for the problem
- [ ] Features that rank high but SHOULDN'T → investigate for leakage
- [ ] Permutation importance used instead of impurity-based importance (which is biased toward high-cardinality features)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std,
}).sort_values('importance_mean', ascending=False)

# 🔴 FLAG: Any feature with high importance that you can't explain → investigate
```

#### 11.2 — SHAP Analysis Audit 🟠 HIGH

- [ ] SHAP values computed for a representative sample
- [ ] SHAP summary plot reviewed with domain expert
- [ ] SHAP dependency plots checked for unexpected feature interactions

**What to look for in SHAP:**
- Feature consistently pushes predictions toward the correct label with unreasonable magnitude → leakage
- Missing-value indicator features have massive SHAP values → the missingness IS the signal (investigate why)
- Monotonic relationships that violate domain knowledge → data quality problem

#### 11.3 — Prediction Sanity Audit 🟠 HIGH

- [ ] Manually inspect 20–50 predictions (correct and incorrect)
- [ ] Look at the model's most confident predictions — do they make physical/business sense?
- [ ] Look at the model's most confident WRONG predictions — understand the failure mode

---

## 13. Stage 12 — Deployment Readiness Audit

### Audit Checklist

#### 12.1 — Feature Availability at Inference 🔴 CRITICAL

- [ ] Every feature used in training is available at inference time in production
- [ ] Data pipelines to generate each feature are built, tested, and monitored
- [ ] Latency of each feature source is within acceptable limits
- [ ] Fallback logic exists for when a feature source is unavailable or delayed

**Feature Availability Matrix (required before deployment):**
```
| Feature            | Source       | Latency    | Fallback if Unavailable   |
|--------------------|-------------|------------|--------------------------|
| user_age_days      | User DB      | Real-time  | Use cached value          |
| avg_orders_30d     | Orders DB    | 1 hour     | Use last available value  |
| weather_forecast   | Weather API  | Daily      | Use historical average    |
| inventory_level    | Warehouse DB | Real-time  | Raise alert, use last    |
```

#### 12.2 — Pipeline Consistency Audit 🔴 CRITICAL

- [ ] Exact same preprocessing pipeline (scaler, imputer, encoder) used in training and production
- [ ] Artifacts serialized from training and loaded in production — NOT refit
- [ ] Serialized artifact versions are tied to model version

#### 12.3 — Prediction Output Audit 🟠 HIGH

- [ ] Output format is correct (probability vs. class label vs. score vs. point forecast)
- [ ] Score range is as expected (0–1 for probabilities, non-negative for demand, etc.)
- [ ] Prediction distribution on recent data matches training-period distribution

#### 12.4 — Performance Gate 🟠 HIGH

- [ ] Minimum acceptable metric thresholds defined BEFORE deployment
- [ ] Model fails the deployment gate if it falls below thresholds
- [ ] Thresholds documented with business justification

#### 12.5 — Latency & Throughput Audit 🟡 MODERATE

- [ ] Model inference time is within SLA requirements
- [ ] Batch prediction pipeline can handle expected data volume
- [ ] Memory footprint is within infrastructure limits

---

## 14. Stage 13 — Post-Deployment Monitoring Audit

### Audit Checklist

#### 14.1 — Data Drift Monitoring 🟠 HIGH

- [ ] Input feature distributions monitored continuously
- [ ] Alerts fire when drift exceeds defined thresholds
- [ ] PSI (Population Stability Index) computed for key features on a regular schedule

```
PSI Interpretation:
PSI < 0.10    → ✅ No significant drift
PSI 0.10–0.25 → 🟡 Moderate drift — investigate
PSI > 0.25    → 🔴 Major drift — model retraining likely required
```

#### 14.2 — Prediction Drift Monitoring 🟠 HIGH

- [ ] Distribution of model output scores tracked over time
- [ ] Significant shifts in score distribution trigger investigation
- [ ] Alert thresholds set on prediction distribution statistics (mean, std, quantiles)

#### 14.3 — Performance Degradation Monitoring 🔴 CRITICAL

- [ ] Ground truth labels collected post-deployment (even with delay)
- [ ] Actual vs. predicted performance tracked on rolling basis
- [ ] Retraining trigger defined: performance drops below X for Y consecutive periods

#### 14.4 — Shadow / A-B Testing Audit 🟡 MODERATE

- [ ] New models run in shadow mode before full rollout
- [ ] Shadow predictions logged and compared to production model
- [ ] Statistical significance of performance difference validated before swapping

---

## 15. Stage 14 — Time-Series & Forecasting Audit

> **This stage is mandatory for any model where time is a dimension of the data.** Forecasting, demand prediction, anomaly detection on sequential data, sensor-based models — all fall here. Time introduces an entire class of failure modes that don't exist in cross-sectional ML.

### Audit Checklist

#### 14.1 — Temporal Ordering Audit 🔴 CRITICAL

- [ ] Data is sorted by time before any time-dependent operations
- [ ] No operation (lag, rolling window, diff) accidentally uses future values
- [ ] Entity boundaries respected — rolling window of entity A doesn't include entity B's data

#### 14.2 — Forecast Horizon Audit 🔴 CRITICAL

- [ ] Forecast horizon explicitly matches the business requirement
- [ ] Model is evaluated at the ACTUAL horizon it will serve — not a convenient shorter one
- [ ] Multi-step-ahead forecasts account for error accumulation (step 5 is less accurate than step 1)

```
Common Mistake:
Train model to predict 1 step ahead.
Evaluate on 1-step-ahead predictions.
Deploy to forecast 5 steps ahead by recursively applying 1-step model.
→ Errors compound at each step. Real-world performance is much worse.
→ Evaluate at the actual deployment horizon.
```

#### 14.3 — Walk-Forward Validation Audit 🔴 CRITICAL

- [ ] Walk-forward (expanding window or sliding window) validation used — NOT random K-Fold
- [ ] Each validation fold trains on past data only and predicts future data only
- [ ] Sufficient number of walk-forward folds to cover different time regimes

```python
def walk_forward_split(df, date_col, n_splits=5, test_size_days=30, gap_days=7):
    """
    Generate walk-forward validation splits.
    Each fold: train on past, skip gap, test on future.
    """
    df = df.sort_values(date_col)
    max_date = df[date_col].max()
    splits = []

    for i in range(n_splits, 0, -1):
        test_end = max_date - pd.Timedelta(days=(i - 1) * test_size_days)
        test_start = test_end - pd.Timedelta(days=test_size_days)
        train_end = test_start - pd.Timedelta(days=gap_days)

        train_idx = df[df[date_col] <= train_end].index
        test_idx = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)].index

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
            print(f"Fold {len(splits)}: train → {train_end.date()} | "
                  f"gap {gap_days}d | test {test_start.date()} → {test_end.date()} | "
                  f"train={len(train_idx):,} test={len(test_idx):,}")

    return splits
```

#### 14.4 — Seasonality & Calendar Audit 🟠 HIGH

- [ ] Known seasonal patterns are accounted for (weekly, monthly, annual, holiday)
- [ ] Training data covers at least one full seasonal cycle (ideally two)
- [ ] Calendar features (day of week, month, holiday flags) are included where appropriate
- [ ] Model performance evaluated across different seasonal periods separately

#### 14.5 — Stationarity Audit 🟡 MODERATE

- [ ] Target variable tested for stationarity (ADF test, KPSS test)
- [ ] If non-stationary, differencing or detrending applied appropriately
- [ ] Trend and seasonality decomposition inspected visually

#### 14.6 — Feature-Target Temporal Alignment Audit 🔴 CRITICAL

- [ ] Every feature's timestamp is strictly BEFORE the target's timestamp
- [ ] No "same-period" features that wouldn't be available at prediction time
- [ ] Lag between feature availability and prediction time is explicitly documented

```
Example of same-period leakage in forecasting:

Predicting:    weekly_sales for week 12
Feature:       website_traffic for week 12     ← 🔴 LEAKAGE (same period)
Correct:       website_traffic for week 11     ← ✅ Available at prediction time
```

---

## 16. Data Leakage — Master Reference

> **Leakage is the #1 silent killer of ML models in production.** This section is a comprehensive reference for every type of leakage and how to detect it.

### Type 1 — Target Leakage 🔴

A feature directly encodes the target or is derived from it.

```
Examples:
❌ Feature: 'refund_amount' when predicting 'will_return_order' (refund happens AFTER return)
❌ Feature: 'treatment_outcome' when predicting 'diagnosis' (treatment follows diagnosis)
❌ Feature: 'velocity' computed as (actual_demand / time) when predicting demand
❌ Feature: 'claim_status' when predicting 'fraud' (claim reviewed after prediction needed)

Detection: Single-feature model achieves near-perfect performance.
```

### Type 2 — Temporal Leakage 🔴

A feature uses information from after the prediction point.

```
Examples:
❌ Feature: 'avg_sales_next_30_days' (future window)
❌ Feature: 'year_end_total' when predicting mid-year values
❌ Rolling average computed without respecting time ordering
❌ Lag feature with wrong sign: shift(-1) instead of shift(1)

Detection: Verify every feature's timestamp is strictly before the target's timestamp.
```

### Type 3 — Train-Test Contamination 🔴

Information from the test set influences training.

```
Examples:
❌ Scaler fit on full dataset before split
❌ Target encoding computed on full dataset
❌ Feature selection based on full-dataset correlations
❌ Hyperparameter tuning on test set
❌ SMOTE applied before split → synthetic test samples
❌ PCA fit on full dataset → test variance shapes components

Detection: Code review of pipeline ordering. Split must precede everything.
```

### Type 4 — Group Leakage 🔴

Entities appear in both train and test with group-level features.

```
Examples:
❌ User A in both train and test + feature 'user_avg_spend' computed across all periods
❌ Product B in train and test + feature 'product_return_rate' from all data
→ Train data about entity X in the feature reveals test data about entity X

Detection: Group overlap audit (see Section 5, Stage 4.3).
```

### Type 5 — Preprocessing Leakage 🟠

Statistical information from test data is used in preprocessing.

```
Examples:
❌ PCA fit on full dataset — test variance shapes the components
❌ Imputing with full-dataset median (includes test values)
❌ Feature normalization using full-dataset min/max
❌ TF-IDF vocabulary built on full corpus including test documents

Detection: Verify every .fit() call operates on training data only.
```

### The Leakage Detection Flowchart

```
NEW FEATURE CREATED
       │
       ▼
Q1: Does this feature use ANY data from AFTER the prediction point?
       │ YES → ❌ DROP IMMEDIATELY
       │ NO  ↓
Q2: Would this feature's exact value be available in production at inference time?
       │ NO  → ❌ DROP (or find a proxy that IS available)
       │ YES ↓
Q3: Was this feature computed using test set data at any point in the pipeline?
       │ YES → ❌ RECOMPUTE correctly on training data only
       │ NO  ↓
Q4: Does a single-feature model using ONLY this feature achieve suspiciously high performance?
       │ YES → ⚠️ INVESTIGATE DEEPLY before proceeding
       │ NO  ↓
Q5: Can a domain expert explain WHY this feature should be predictive?
       │ NO  → ⚠️ INVESTIGATE — unexplainable predictors are often leakage
       │ YES ↓
       ✅ FEATURE APPROVED — LOG IT IN THE FEATURE AUDIT LOG
```

---

## 17. Bias & Fairness Audit

### Audit Checklist

#### 17.1 — Data Representation Audit 🟠 HIGH

- [ ] All relevant subgroups are adequately represented in training data
- [ ] Subgroups with very few samples identified (model will underperform on them)
- [ ] Data collection bias investigated (e.g., urban-heavy, English-only, weekday-only)

#### 17.2 — Performance Disparity Audit 🟠 HIGH

- [ ] Performance metrics computed per subgroup
- [ ] Subgroups with significantly worse performance flagged
- [ ] Performance gap between best and worst subgroup documented

```python
def subgroup_performance_audit(y_true, y_pred, subgroups, metric_fn, metric_name="metric"):
    """Compute metric per subgroup and flag disparities."""
    results = []
    for name, mask in subgroups.items():
        score = metric_fn(y_true[mask], y_pred[mask])
        results.append({"subgroup": name, "n": mask.sum(), metric_name: round(score, 4)})

    results_df = pd.DataFrame(results).sort_values(metric_name)
    best = results_df[metric_name].max()
    worst = results_df[metric_name].min()
    gap = best - worst

    print(f"Best subgroup:  {best:.4f}")
    print(f"Worst subgroup: {worst:.4f}")
    print(f"Gap:            {gap:.4f}")

    if gap > 0.10:
        print(f"🟠 Significant performance disparity across subgroups.")

    return results_df
```

#### 17.3 — Prediction Fairness Audit 🟡 MODERATE

- [ ] Certain subgroups are not systematically over/under-predicted
- [ ] Model confidence is reasonably calibrated across subgroups
- [ ] Disparate impact ratio is within acceptable bounds (if applicable)

---

## 18. Class Imbalance & Sparse Target Audit

### Audit Checklist

#### 18.1 — Imbalance Detection 🔴 CRITICAL

- [ ] Class ratio computed and documented
- [ ] If the majority class exceeds 90%, majority-class prediction already achieves 90% "accuracy" → accuracy is useless
- [ ] Primary metric switched to AUC-PR, F1, or Recall — NOT accuracy

#### 18.2 — Sparse / Zero-Inflated Target Audit 🔴 CRITICAL (Regression / Forecasting)

- [ ] Percentage of zero-valued targets documented
- [ ] If >50% of target is zero, consider two-stage modeling (classify zero/non-zero, then regress on non-zero subset)
- [ ] Evaluation metric accounts for sparsity (wMAPE over MAPE; MAE over RMSE)
- [ ] Performance evaluated separately on zero vs. non-zero subsets

```
Common Mistake with sparse targets:
Model predicts 0 for everything → RMSE looks okay because most actual values ARE 0.
But the model is useless — it never predicts the non-zero events that matter.

Fix: Evaluate on the non-zero subset separately. Report both overall and non-zero metrics.
```

#### 18.3 — Imbalance Handling Strategy Audit 🟠 HIGH

| Strategy | When to Use | Audit Check |
|---|---|---|
| Class weights | Mild imbalance, tree-based models | Weights set correctly in model config |
| SMOTE / Oversampling | Moderate imbalance, tabular data | Applied ONLY to training data, AFTER split |
| Undersampling | Very large majority class | Not losing critical minority patterns |
| Threshold adjustment | Any imbalance | Threshold tuned on validation set |
| Two-stage model | Extreme sparsity (>80% zeros) | Both stages evaluated independently |

- [ ] If SMOTE used: applied AFTER split, ONLY on training data

```
⚠️ Common SMOTE Mistake:
Apply SMOTE to full dataset → split → test set contains synthetic samples
→ Model evaluated on fake data = inflated metrics
```

---

## 19. Feature Store & Reproducibility Audit

### Audit Checklist

#### 19.1 — Experiment Reproducibility 🟠 HIGH

- [ ] All random seeds set and documented
- [ ] All library versions logged (exact versions, not ranges)
- [ ] Data version documented (snapshot date, query, hash)
- [ ] Feature engineering code is version-controlled
- [ ] Model artifacts saved with unique version identifiers

**Reproducibility Test:** Can a teammate, on a different machine, run your pipeline end-to-end and get the same results (within floating-point tolerance)?

#### 19.2 — Feature Store Audit 🟡 MODERATE (If applicable)

- [ ] Feature definitions are centrally stored and versioned
- [ ] Same feature definition used in training and serving (no training-serving skew)
- [ ] Feature freshness requirements are defined and enforced
- [ ] Feature computation is auditable — can trace any feature value back to its raw source

#### 19.3 — Training-Serving Skew Audit 🔴 CRITICAL

One of the most insidious production problems — the model receives different data than it was trained on:

```
Training-Serving Skew = Feature computed differently during training vs. production

Example:
Training:    avg_orders_30d = mean of past 30 calendar days (correctly computed offline)
Production:  avg_orders_30d = mean of past 30 available data points (different if days are missing)
→ Model receives subtly different feature values than it learned from
→ Performance degrades silently — no error, no crash, just worse predictions

Detection:
1. Feature computation code is shared (not duplicated) between training and serving
2. Log a sample of production feature values; compare distributions to training
3. Run serving pipeline on historical data and compare outputs to training pipeline
```

- [ ] Feature computation code shared between training and serving
- [ ] Production feature value distributions compared to training distributions regularly
- [ ] Historical backtest confirms serving pipeline reproduces training pipeline outputs

---

## 20. Statistical Testing Audit

### Audit Checklist

#### 20.1 — Model Comparison Audit 🟡 MODERATE

- [ ] Model A vs. Model B difference tested for statistical significance
- [ ] Use paired t-test on cross-validation fold scores, or McNemar's test for classifiers
- [ ] Never declare a winner based on a single train-test split

#### 20.2 — Performance Stability Audit 🟡 MODERATE

- [ ] Multiple experiments with different random seeds — performance is stable
- [ ] High variance across seeds → model is unstable → need more data or simpler model
- [ ] Report: mean ± std of metrics across seeds

#### 20.3 — Calibration Audit 🟠 HIGH (Classification)

- [ ] Predicted probabilities are calibrated (0.7 prediction should be correct ~70% of the time)
- [ ] Calibration curve (reliability diagram) plotted and reviewed
- [ ] If miscalibrated: Platt scaling or isotonic regression applied
- [ ] Miscalibrated models are dangerous when thresholds drive business decisions

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def calibration_audit(y_true, y_pred_proba, n_bins=10):
    """Plot and evaluate model calibration."""
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted, fraction_positive, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Quantify miscalibration
    ece = np.mean(np.abs(fraction_positive - mean_predicted))
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    if ece > 0.05:
        print("🟠 Model is miscalibrated. Consider Platt scaling or isotonic regression.")
```

---

## 21. Pipeline Automation & CI/CD Audit

> **Level 4 of the Audit Mindset Hierarchy: automated audit checks that run on every pipeline execution.** This section turns the manual checklists above into code that fails loudly when something is wrong.

### Audit Checklist

#### 21.1 — Automated Gate Functions 🔵 ADVISORY

Convert critical audit checks into assertion functions that run automatically:

```python
class PipelineAuditGates:
    """
    Drop these into your training pipeline.
    Each gate either passes silently or raises an AssertionError with details.
    """

    @staticmethod
    def gate_schema(df, expected_schema):
        """Run at data ingestion."""
        for col, dtype in expected_schema.items():
            assert col in df.columns, f"AUDIT FAIL: Missing column '{col}'"
            assert str(df[col].dtype) == dtype, \
                f"AUDIT FAIL: '{col}' type mismatch: expected {dtype}, got {df[col].dtype}"
        extra = set(df.columns) - set(expected_schema.keys())
        assert not extra, f"AUDIT FAIL: Unexpected columns (leakage risk): {extra}"

    @staticmethod
    def gate_no_future_dates(df, date_col, cutoff_date):
        """Run at data ingestion. No records from the future."""
        future = df[pd.to_datetime(df[date_col]) > pd.to_datetime(cutoff_date)]
        assert len(future) == 0, \
            f"AUDIT FAIL: {len(future)} records have dates after cutoff {cutoff_date}"

    @staticmethod
    def gate_no_group_overlap(train_df, test_df, group_col):
        """Run after split."""
        overlap = set(train_df[group_col]) & set(test_df[group_col])
        assert len(overlap) == 0, \
            f"AUDIT FAIL: {len(overlap)} groups in both train and test: {list(overlap)[:5]}"

    @staticmethod
    def gate_no_target_in_features(feature_cols, target_col):
        """Run before training."""
        assert target_col not in feature_cols, \
            f"AUDIT FAIL: Target '{target_col}' found in feature list"

    @staticmethod
    def gate_performance_above_baseline(model_metric, baseline_metric, metric_name="metric"):
        """Run after evaluation."""
        assert model_metric > baseline_metric, \
            f"AUDIT FAIL: Model {metric_name} ({model_metric:.4f}) " \
            f"does not beat baseline ({baseline_metric:.4f})"

    @staticmethod
    def gate_train_test_gap(train_metric, test_metric, max_gap=0.10, metric_name="metric"):
        """Run after evaluation. Catches severe overfitting."""
        gap = abs(train_metric - test_metric)
        assert gap < max_gap, \
            f"AUDIT FAIL: Train-test {metric_name} gap is {gap:.4f} (max allowed: {max_gap})"

    @staticmethod
    def gate_no_high_single_feature_auc(audit_results_df, threshold=0.90):
        """Run after single-feature leakage audit."""
        flagged = audit_results_df[
            audit_results_df['single_feature_AUC'] > threshold
        ]
        assert len(flagged) == 0, \
            f"AUDIT FAIL: {len(flagged)} features exceed single-feature AUC threshold:\n{flagged}"
```

#### 21.2 — Pipeline Integration Pattern 🔵 ADVISORY

```python
# Example: Audit gates integrated into a training pipeline

def train_pipeline(config):
    # 1. INGEST
    df = load_data(config)
    PipelineAuditGates.gate_schema(df, config.expected_schema)
    PipelineAuditGates.gate_no_future_dates(df, 'date', config.cutoff)

    # 2. SPLIT (before any preprocessing)
    train_df, test_df = temporal_split_with_buffer(df, 'date', config.train_end, config.buffer)
    PipelineAuditGates.gate_no_group_overlap(train_df, test_df, 'entity_id')

    # 3. FEATURE ENGINEERING (on train only, then merge to test)
    train_df, test_df = build_features(train_df, test_df)
    PipelineAuditGates.gate_no_target_in_features(feature_cols, config.target)

    # 4. PREPROCESS (fit on train only)
    pipeline = build_preprocessing_pipeline()
    X_train = pipeline.fit_transform(train_df[feature_cols])
    X_test = pipeline.transform(test_df[feature_cols])

    # 5. TRAIN
    model = train_model(X_train, train_df[config.target])

    # 6. EVALUATE
    train_score = evaluate(model, X_train, train_df[config.target])
    test_score = evaluate(model, X_test, test_df[config.target])

    PipelineAuditGates.gate_performance_above_baseline(test_score, config.baseline)
    PipelineAuditGates.gate_train_test_gap(train_score, test_score)

    return model
```

#### 21.3 — Versioning & Experiment Tracking 🔵 ADVISORY

- [ ] Use MLflow, Weights & Biases, DVC, or equivalent for experiment tracking
- [ ] Every training run logs: data version, code commit, hyperparameters, metrics, artifacts
- [ ] Model registry tracks which model version is in production
- [ ] Rollback procedure exists and is tested

---

## 22. Model Documentation & Governance Audit

### Audit Checklist

#### 22.1 — Model Card 🟡 MODERATE

Every model going to production should have a model card documenting:

```
MODEL CARD TEMPLATE:
─────────────────────────────────────────
Model Name:          [Name and version]
Task:                [What it predicts]
Owner:               [Team / individual]
Training Date:       [When last trained]
Training Data:       [Source, date range, size, version hash]
Features:            [Count, top features by importance]
Target Variable:     [Definition, distribution]
Primary Metric:      [Name, value on test set]
Baseline:            [Name, value]
Known Limitations:   [Subgroups with poor performance, edge cases]
Bias Assessment:     [Subgroup performance disparities]
Retraining Schedule: [Trigger criteria]
Deployment Target:   [System, endpoint, frequency]
─────────────────────────────────────────
```

#### 22.2 — Decision Audit Trail 🟡 MODERATE

- [ ] Every major pipeline decision is documented (why this model? why this metric? why this threshold?)
- [ ] Feature exclusion decisions are documented with reasoning
- [ ] Audit findings are logged and tracked to resolution

---

## 23. Audit Templates & Checklists

### Master Pre-Training Audit Checklist

```
□ Problem definition and target variable are unambiguous and documented
□ Business metric ↔ ML metric mapping is explicit
□ All data sources documented with lineage
□ Schema validation passed
□ Duplicate rows identified and handled
□ Missing value analysis complete
□ Temporal integrity verified (no future dates in historical records)
□ Train/test split done BEFORE any preprocessing
□ Split is time-based (for sequential data)
□ No group overlap between train and test
□ Split indices saved and frozen
□ EDA completed on TRAINING SET ONLY
□ Target sparsity / imbalance documented
□ All features pass the Temporal Availability Gate
□ All features pass the Single-Feature Leakage Audit
□ Feature Audit Log is complete for every feature
□ All preprocessing fit on training data only
□ Unseen category audit complete for categorical features
□ Distribution drift audit complete (KS test)
□ Baseline model established
□ Class imbalance / target sparsity addressed
```

### Master Post-Training Audit Checklist

```
□ Training vs. validation performance gap is acceptable
□ Test set evaluation done exactly once
□ Suspiciously good performance investigated
□ Error analysis completed (confusion matrix / residuals) with domain expert
□ Optimal threshold selected with business cost justification
□ Feature importance reviewed for leakage signals
□ SHAP analysis reviewed for unexpected patterns
□ Subgroup performance audit complete
□ Prediction calibration checked (classification)
□ Model reproduces with same seed and data
□ All artifacts versioned and saved
□ Training-serving skew analysis complete
□ Feature availability at inference verified
□ Deployment performance thresholds defined
□ Monitoring setup for data drift and performance degradation
□ Retraining trigger criteria defined
□ Model card completed
```

### Feature Audit Log Template

```
| # | Feature Name | Source | Computation Logic | Stage Created | Uses Future? | Available at Inference? | Single-Feature Performance | Domain Expert Approved? | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |
```

### Post-Mortem Template (When Leakage Is Discovered)

```
INCIDENT: Data Leakage in [Model Name]
Date Discovered:
Model Version:
Discovered By:
Detection Method:

For EACH leaked feature:
  ─ Feature Name:
  ─ Leakage Type:        [Target / Temporal / Train-Test / Group / Preprocessing]
  ─ Root Cause:           [Why did it enter the pipeline?]
  ─ Impact:               [Metric before vs. after removal]
  ─ Business Decisions Affected: [What was decided based on inflated metrics?]
  ─ Corrective Action:    [How was it fixed?]
  ─ Prevention Gate:      [Which audit check would have caught this?]

Process Changes:
  ─ New audit steps added:
  ─ At which pipeline stage:
  ─ Responsible owner:
  ─ Automation status:    [Manual / Scripted / CI/CD integrated]
```

---

## 24. Appendix — Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ML PIPELINE AUDIT — QUICK REFERENCE                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  GOLDEN RULE:  Split FIRST → then everything else                        ║
║                                                                           ║
║  ─── LEAKAGE SIGNALS ───                                                 ║
║  🔴 Single-feature AUC > 0.90 (classif) or >70% error reduction (reg)   ║
║  🔴 Train-Test metric gap < 0.01 (suspiciously close)                    ║
║  🔴 Feature uses future information → DROP                               ║
║  🔴 Feature unavailable at inference → DROP                              ║
║  🔴 Removing top 3 features collapses performance → INVESTIGATE          ║
║                                                                           ║
║  ─── OVERFITTING SIGNALS ───                                             ║
║  🟠 Train AUC − Val AUC > 0.05 → Regularize                             ║
║  🟡 Both metrics near baseline → Underfitting, not overfitting           ║
║                                                                           ║
║  ─── METRIC SELECTION ───                                                ║
║  Imbalanced classification → AUC-PR, F1  (NOT accuracy)                  ║
║  Sparse regression / demand → wMAPE, MAE (NOT RMSE, NOT MAPE)           ║
║  Any task → report multiple metrics, never just one                      ║
║                                                                           ║
║  ─── PREPROCESSING ───                                                   ║
║  fit_transform() → TRAINING DATA ONLY                                    ║
║  transform()     → test / production data                                ║
║                                                                           ║
║  ─── TIME-SERIES ───                                                     ║
║  Random K-Fold → ❌ NEVER for temporal data                              ║
║  Walk-forward validation → ✅ ALWAYS                                      ║
║  Buffer gap between train and test → ✅ REQUIRED                         ║
║                                                                           ║
║  ─── EVERY NEW FEATURE ───                                               ║
║  Must pass → Feature Availability Gate → Single-Feature Audit            ║
║           → Domain Expert Review → Feature Audit Log entry               ║
║                                                                           ║
║  ─── DEPLOYMENT ───                                                      ║
║  Training pipeline = Serving pipeline (same code, same artifacts)        ║
║  Monitor: data drift (PSI) + prediction drift + metric degradation       ║
║  Define retraining triggers BEFORE deployment                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

*`Audit.md` — Version 2.0*
*Scope: Domain-agnostic — applicable to classification, regression, forecasting, ranking, and anomaly detection pipelines*
*Covers: 14 pipeline stages, 5 leakage types, 100+ audit checks with severity tiers, automation-ready code*
*License: Use freely. Star the repo. Audit relentlessly.*
