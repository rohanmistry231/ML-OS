# Data.md

### The Complete Data Layer Reference — How to Acquire, Profile, Split, Sample, Label, and Sustain the Data That Powers Every ML System

> **Why this file exists:** Algorithms are interchangeable. Features are buildable. Metrics are computable. **Data is the foundation that everything else stands on — and the single thing most teams underinvest in.** This file is the systematic reference for data: how to get it, how to know it's good, how to split it without breaking the model, how to handle imbalance, how to detect drift, how to label, and how to keep all of this honest over time.
>
> **How to read this file:** §3 (Splits) and §6 (Leakage at the Data Layer) are non-negotiable — read them first. §7 (Imbalance) and §9 (Drift) are the operational sections most teams botch. The rest is reference material to consult by section.

---

## Table of Contents

1. [Philosophy — How to Think About Data](#1-philosophy--how-to-think-about-data)
2. [Data Profiling — Know Your Data Before You Model It](#2-data-profiling--know-your-data-before-you-model-it)
3. [Splits — The Single Most Important Decision](#3-splits--the-single-most-important-decision)
4. [Sampling Strategies](#4-sampling-strategies)
5. [Data Quality Audit](#5-data-quality-audit)
6. [Data Leakage at the Data Layer](#6-data-leakage-at-the-data-layer)
7. [Class Imbalance & Rare Events](#7-class-imbalance--rare-events)
8. [Labeling — Where Ground Truth Comes From](#8-labeling--where-ground-truth-comes-from)
9. [Drift — When Data Changes Under You](#9-drift--when-data-changes-under-you)
10. [Synthetic Data & Augmentation](#10-synthetic-data--augmentation)
11. [Data Versioning & Reproducibility](#11-data-versioning--reproducibility)
12. [Data Pipelines & ETL](#12-data-pipelines--etl)
13. [Privacy, PII & Compliance](#13-privacy-pii--compliance)
14. [Cross-Validation Strategies](#14-cross-validation-strategies)
15. [Holdout Strategy](#15-holdout-strategy)
16. [The Data Workflow](#16-the-data-workflow)
17. [Implementation Quick Reference](#17-implementation-quick-reference)

---

## 1. Philosophy — How to Think About Data

### The Three Laws of the Data Layer

```
LAW 1 — The Law of Garbage In, Gospel Out
    Models do not detect bad data; they propagate it.
    A model trained on biased, leaky, mislabeled, or stale data
    will produce confident predictions reflecting those exact
    flaws. The model is never the safety net — the data layer is.

LAW 2 — The Law of the Sacred Split
    The test set is created once and touched only at the end.
    Every decision — preprocessing, feature engineering, model
    selection, threshold tuning — must be made without ever
    looking at the test set. Looking is touching. Touching
    is contamination. There is no "just a quick check."

LAW 3 — The Law of Distribution Conservation
    Train, validation, test, and production must come from the
    same distribution — or your offline metrics are lying. If
    the production distribution differs (and it always will,
    eventually), you must detect it, quantify it, and account
    for it. Models do not adapt; data layers do.
```

### Why Data Work Is Undervalued

Data work is undervalued in proportion to how invisible it is when done well. A clean dataset looks effortless; a polished model looks impressive. The result: teams spend months tuning a 0.3% accuracy improvement on top of a 30% labeling error rate that no one audited.

The honest hierarchy of ML impact:

```
Better data           →  10x to 100x improvement
Better features       →  2x to 10x improvement
Better model          →  1.1x to 2x improvement
Better hyperparams    →  1.01x to 1.1x improvement
```

Most teams invert this allocation. Don't.

### What "Good Data" Actually Means

| Property | Test |
|---|---|
| **Accurate** | Labels reflect reality, not labeler bias |
| **Complete** | All necessary fields are populated for enough samples |
| **Consistent** | Same entity is represented the same way across sources |
| **Timely** | Reflects the state of the world relevant to the prediction |
| **Representative** | Sample matches the production distribution |
| **Sufficient** | Enough examples to learn the patterns (especially edge cases) |
| **Documented** | Lineage, semantics, and known issues are written down |

Almost no real-world dataset satisfies all seven. The job is knowing which ones you're failing, how badly, and how to compensate.

### The Data Layer Hierarchy

```
Tier 0 — Raw:         Whatever lands in the warehouse / lake
Tier 1 — Profiled:    Schemas validated, distributions known, issues documented
Tier 2 — Cleaned:     Duplicates handled, types corrected, sentinel values mapped
Tier 3 — Split:       Train/val/test created correctly, frozen, versioned
Tier 4 — Sampled:     Class balance addressed, weights computed if needed
Tier 5 — Labeled:     Ground truth verified, label quality audited
Tier 6 — Monitored:   Drift tracked, freshness verified, pipeline health watched
```

You cannot skip tiers. A model trained on un-profiled, un-split, un-audited data is not a model — it's a noise generator with a confidence score.

---

## 2. Data Profiling — Know Your Data Before You Model It

> If you cannot answer fundamental questions about your data — How many rows? How many missing values per column? What's the distribution of the target? — you have no business training a model on it.

### 2.1 The Profiling Checklist

For every dataset, before any modeling:

#### Schema
- [ ] Number of rows and columns
- [ ] Data type of each column (and is it the *right* type?)
- [ ] Memory footprint
- [ ] Primary key — does it actually identify rows uniquely?
- [ ] Foreign keys — do they join cleanly to expected sources?

#### Per-Column
- [ ] Missing value count and pattern (random vs. structural)
- [ ] Cardinality (unique value count)
- [ ] Min, max, mean, median, std for numerical
- [ ] Top-N values + their frequencies for categorical
- [ ] Distribution shape (histogram)
- [ ] Outlier presence

#### Target Variable
- [ ] Distribution (balanced? skewed? bimodal?)
- [ ] Range and units
- [ ] Missingness pattern
- [ ] Temporal pattern (does the target distribution change over time?)
- [ ] Subgroup distribution (does it vary across user segments?)

#### Cross-Column
- [ ] Correlation matrix for numerical features
- [ ] Mutual information for mixed types
- [ ] Multi-collinearity (any feature pair with |r| > 0.95?)
- [ ] Constant or near-constant columns (variance ≈ 0)

#### Temporal Properties (if applicable)
- [ ] Date range covered
- [ ] Date column completeness
- [ ] Periodicity (daily / weekly / monthly cycles)
- [ ] Time zone consistency

### 2.2 Automated Profiling Tools

Don't write these checks from scratch:

| Tool | Use |
|---|---|
| `pandas-profiling` / `ydata-profiling` | One-line full report; great for first pass |
| `sweetviz` | Visual comparison of train vs. test |
| `pandera` | Schema validation as code |
| `great_expectations` | Production data quality framework |
| `dataprep` | Fast EDA |
| `pyjanitor` | Cleaning utilities |

**Critical:** Profile train and test separately, then compare. Distribution differences between them indicate either a leaky split or genuine drift — both worth investigating.

### 2.3 The Sanity-Check Questions

Before training, you must be able to answer:

1. "What's the shape of the target distribution?" — base rate matters for everything downstream
2. "What's the highest-cardinality feature?" — informs encoding strategy
3. "Which columns have > 30% missing values?" — flags imputation or drop decisions
4. "Is there any column that's a near-perfect predictor of the target?" — leakage check (see §6)
5. "What's the time range and granularity?" — informs split strategy
6. "Are there duplicates?" — affects metric validity
7. "What's the unit of analysis?" — row = user? session? transaction?

If you cannot answer these, you cannot model responsibly.

### 2.4 Duplicate Detection

Duplicates corrupt metrics silently. They can come from:
- **Ingestion bugs** — same row written twice
- **Join explosion** — a 1-to-N join treated as 1-to-1
- **Update logs** — multiple versions of the same entity over time

Check three forms:
```python
# 1. Exact duplicates
df.duplicated().sum()

# 2. Duplicates by primary key
df.duplicated(subset=['user_id', 'timestamp']).sum()

# 3. Near-duplicates (for text)
# - Hash-based: MinHash + LSH
# - Embedding-based: cosine similarity above threshold
```

**Critical:** Resolve duplicates **before** splitting. A duplicate landing in both train and test inflates metrics by giving the model the answer.

---

## 3. Splits — The Single Most Important Decision

> The split decision is the most consequential decision in the entire pipeline. Get it wrong and your metrics are meaningless — every downstream choice is built on a lie. Get it right and even a mediocre model produces honest, deployable estimates.

### 3.1 The Cardinal Rule

> **Split first. Then do everything else.**

Every preprocessing, feature engineering, scaling, encoding, imputation, and selection step happens **after** the split, using **only the training set**. Any deviation is leakage. (See `Audit.md` §16 and `Features.md` §17.)

### 3.2 Random Split

Standard `train_test_split` with a random seed.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)
```

**Use when:**
- IID assumption holds (rows are exchangeable)
- No temporal structure
- No grouping structure (no entity appears in multiple rows)

**Do NOT use when:**
- Data has time component (use time-based split)
- Same entity has multiple rows (use group split)
- Distribution shifts over time

### 3.3 Stratified Split

Preserves class proportions across train/test.

```python
train_test_split(..., stratify=y)
```

**Use for:** Imbalanced classification. Without stratification, a small random fluctuation can put zero positives in the test set, breaking everything.

### 3.4 Time-Based Split

The only valid split for temporal data.

```python
cutoff = '2025-01-01'
train = df[df['date'] < cutoff]
test = df[df['date'] >= cutoff]
```

**Use when:** Data has timestamps and the model will be used to predict future values.

**Critical detail:** Add a **buffer gap** between train and test if features have delayed information availability. If `feature_x` is only known 7 days after an event, train ends 7 days before test starts. Otherwise the training data sees information that production won't have.

```
TRAIN  ─────────────────────────[ buffer 7d ][ TEST ]─────────
                                 cutoff
```

### 3.5 Group-Based Split

When rows are grouped by an entity (user, session, customer, patient), all rows for one entity must be in the same split.

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['user_id']))
```

**Use when:**
- Predicting at the entity level (will user X churn?) but have multi-row observations per entity
- Sessions, transactions, patients with multiple visits, sensors with multiple readings
- Any scenario where entity-level patterns could leak across train/test

**Detection of group leakage:**
```python
# Should be 0 if split is correct
overlap = set(train['user_id']) & set(test['user_id'])
assert len(overlap) == 0
```

### 3.6 Combined Strategies

Real-world splits often need multiple constraints at once:

**Time + Group:** All of user X's data is in train if their first event is before cutoff; in test otherwise. Common in churn prediction.

**Stratified + Group:** Each group entirely in one fold, but fold class balance is preserved. Use `StratifiedGroupKFold`.

**Time + Stratified:** Preserve class balance across time-based folds. Useful when class distribution shifts over time.

### 3.7 The Three-Way Split

Most production models need **three** sets, not two:

```
TRAIN        — Fit the model
VALIDATION   — Tune hyperparameters, early stopping, model selection
TEST         — Final, single-use evaluation
```

Common ratios: 70/15/15, 80/10/10, or with separate temporal holdouts in production: train(all-but-last-month) / val(second-to-last-month) / test(last-month).

**The test set must be evaluated exactly once.** If you evaluate, retune, and re-evaluate, the test set becomes the validation set — and you no longer have an honest performance estimate.

### 3.8 Cross-Validation Splits

For small data or robust evaluation, use cross-validation (see §14).

### 3.9 The Common Split Disasters

| Mistake | What Happens |
|---|---|
| Random split on time series | Future leaks into past via random shuffling |
| No buffer on time-based split | Features with lag are unavailable in production |
| Group leakage | Same entity in train and test inflates metric |
| Stratifying on a contaminated target | Splits look balanced but are corrupted |
| Splitting after preprocessing | Test set contaminates fit (scaling, encoding) |
| Re-splitting after seeing test results | Test becomes validation, no honest estimate |
| Using val set as test by accident | Optimistic bias |

### 3.10 Freezing the Split

Once created, the split should be **frozen** — saved as a list of row indices or stored explicitly:

```python
np.save('train_idx.npy', train_idx)
np.save('test_idx.npy', test_idx)
```

Anyone re-running the experiment uses these indices. No re-shuffling. No "let me re-split with a different seed and see if results change." That second action is implicit test-set probing — leakage.

---

## 4. Sampling Strategies

### 4.1 Why Sample

- **Full dataset is too large** to train on (computational constraint)
- **Class imbalance** requires balancing techniques
- **Subgroup underrepresentation** requires oversampling minorities
- **Streaming / online data** requires reservoir sampling
- **Active learning** — sample what's most informative to label

### 4.2 Random Sampling

```python
df.sample(n=10000, random_state=42)
df.sample(frac=0.1, random_state=42)
```

**Use when:** Data is IID, just need a smaller working set.

### 4.3 Stratified Sampling

Preserve the distribution of one or more variables.

```python
from sklearn.model_selection import train_test_split
sample, _ = train_test_split(df, train_size=10000, stratify=df['target'])
```

**Use when:** Need to preserve class balance or subgroup proportions in a sample.

### 4.4 Cluster / Group Sampling

Sample whole groups (e.g., all events for a sampled set of users).

```python
sampled_users = df['user_id'].drop_duplicates().sample(n=1000)
sample = df[df['user_id'].isin(sampled_users)]
```

**Use when:** Group structure matters and you need representative groups, not individual rows.

### 4.5 Reservoir Sampling

Sample uniformly from a stream of unknown length.

```python
def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

**Use when:** Streaming data, can't fit all data in memory.

### 4.6 Importance Sampling

Sample more heavily from informative regions, then reweight.

**Use when:** Active learning, rare event modeling, expensive labeling budgets.

### 4.7 Active Learning Sampling

Sample the points the model is **most uncertain about** for labeling:

| Strategy | Description |
|---|---|
| **Uncertainty sampling** | Lowest-confidence predictions |
| **Query by committee** | Largest disagreement across models |
| **Expected model change** | Points that would change the model the most |
| **Diversity sampling** | Maximize coverage of feature space |

**Use when:** Labels are expensive (medical, legal, manual review). Active learning can cut labeling costs 5-10x.

---

## 5. Data Quality Audit

> Most data quality bugs are silent. They don't raise errors. They just produce wrong models. The audit makes them visible.

### 5.1 The Eight Dimensions of Data Quality

| Dimension | Question |
|---|---|
| **Completeness** | What percentage of values are present? |
| **Uniqueness** | Are entities represented only once? |
| **Consistency** | Same entity, same representation across sources? |
| **Validity** | Do values conform to expected format / range? |
| **Accuracy** | Do values match reality? |
| **Timeliness** | Are values current enough for the use case? |
| **Lineage** | Can you trace every value to its source? |
| **Documentation** | Is the data described, or just stored? |

### 5.2 The Audit Checklist

#### Schema Validation
- [ ] Every column has a documented type and meaning
- [ ] Type-checking enforced at ingestion (pandera, great_expectations)
- [ ] Required columns are non-null where expected
- [ ] Categorical columns have a defined set of allowed values

#### Range Validation
- [ ] Numerical columns within physical / business limits (age between 0 and 120, price ≥ 0, etc.)
- [ ] Dates within plausible range (no future dates in historical records)
- [ ] String lengths within expected bounds

#### Referential Integrity
- [ ] Foreign keys resolve to existing records
- [ ] Many-to-many relationships behave as expected
- [ ] Cascade behavior on deletes is correct

#### Logical Consistency
- [ ] `end_date >= start_date` always
- [ ] `total = sum of components`
- [ ] Mutually exclusive flags don't co-occur
- [ ] Hierarchical relationships are valid (subcategory belongs to parent category)

#### Pipeline Health
- [ ] Daily row counts within expected range
- [ ] No sudden zero-record days
- [ ] Feature distributions stable day-over-day
- [ ] Latency within SLO

### 5.3 The Silent Killers

These don't error — they just corrupt:

| Issue | Symptom | Detection |
|---|---|---|
| Encoding mismatch | Garbled characters in strings | String validation, charset checks |
| Truncated text | Random cutoffs in long fields | Length distribution check |
| Sentinel values masquerading as data | -999, 9999, "NA", "Unknown" treated as real | Domain-aware sentinel scanning |
| Timezone drift | Off-by-hours errors | Cross-source timestamp comparison |
| Off-by-one row alignment | Features shifted vs labels | Hash-based reconciliation |
| Schema evolution unhandled | New columns silently dropped | Schema versioning |
| Floating-point comparison | `0.1 + 0.2 ≠ 0.3` | Use tolerance comparisons |
| Locale-specific number parsing | `1,000` vs `1.000` | Standardize at ingestion |
| Null vs. empty string vs. zero | Three different "missing" states | Normalize at ingestion |

### 5.4 Sentinel Value Patterns

Many real-world datasets encode missing values as sentinels:

| Sentinel | Common Source |
|---|---|
| -1, -999, 9999 | Legacy systems where NULL wasn't allowed |
| "N/A", "NA", "Unknown", "?" | Survey data |
| "1900-01-01", "9999-12-31" | Default date stamps |
| 0 (when 0 is impossible) | Default value for unset fields |
| Empty string | Distinct from NULL in some databases |

**Always scan for these.** A model trained with -999 as a real value for "age missing" will produce nonsense predictions for actual elderly users.

```python
# Quick sentinel scan
for col in df.select_dtypes(include=np.number).columns:
    vc = df[col].value_counts(normalize=True).head(5)
    print(f"\n{col}:\n{vc}")
    # Look for suspiciously high frequencies of specific numbers
```

### 5.5 The Data Quality SLI / SLO

For production data pipelines, treat quality as a measurable SLO:

| Indicator | Example SLO |
|---|---|
| Daily completeness | > 99% of expected rows arrive |
| Schema conformance | 0 schema violations per day |
| Critical-field non-null rate | > 99.9% on required fields |
| Distribution stability | PSI < 0.1 vs. last 30-day baseline |
| Latency | 95th percentile under threshold |

Alerts when SLOs are breached. Treat data outages like service outages.

---

## 6. Data Leakage at the Data Layer

> See `Audit.md` §16 and `Features.md` §17 for the master leakage references. This section covers leakage that originates in the **data layer itself** — before any feature engineering happens.

### 6.1 The Five Data-Layer Leakage Patterns

#### 6.1.1 Pre-Split Preprocessing
Any transformation applied before splitting contaminates the train-test boundary.

**Examples:**
- Standardizing the whole dataset, then splitting
- Imputing with global mean, then splitting
- Encoding categoricals on the combined data
- PCA on the whole dataset

**Fix:** Split first. Always.

#### 6.1.2 Time Travel in Joins
Joining the "current" version of a dimension table to historical fact data:

```sql
-- WRONG: pulls today's user_segment for every historical event
SELECT events.*, users.segment
FROM events JOIN users ON events.user_id = users.user_id

-- RIGHT: pulls the user_segment as it was at event time
SELECT events.*, user_history.segment
FROM events
JOIN user_history
  ON events.user_id = user_history.user_id
 AND events.timestamp BETWEEN user_history.valid_from AND user_history.valid_to
```

**This is one of the most common silent leakage sources in industry.** Dimensional data changes over time. Joining the latest version into historical training data leaks future information.

**Fix:** Slowly Changing Dimension (SCD) Type 2 tables with valid_from / valid_to. Or use a feature store with point-in-time correctness.

#### 6.1.3 Future-Aware Aggregates
Computing aggregates on the entire dataset, then using them as features:

```python
# WRONG
df['user_lifetime_total'] = df.groupby('user_id')['amount'].transform('sum')
# This includes future transactions for each row

# RIGHT (expanding sum, time-respecting)
df = df.sort_values(['user_id', 'timestamp'])
df['user_total_so_far'] = df.groupby('user_id')['amount'].cumsum().shift(1)
```

#### 6.1.4 Sampled Data Leakage
Sampling before split can cause leakage if sampling preserves correlated information:

**Example:** Downsampling the majority class before splitting can place correlated points (same user, same session) in different splits.

**Fix:** Split first, then sample within train if needed.

#### 6.1.5 ID-Based Leakage
Including identifiers that correlate with the target:

- `user_id` where user_ids are assigned chronologically and the target has a temporal trend
- `record_id` where higher IDs were created in a different regime
- `experiment_id` where experiments are correlated with outcomes

**Fix:** Drop or audit any ID column. If you need entity-level signal, use aggregates, not the raw ID.

### 6.2 The Train/Test Distribution Comparison

After splitting, before modeling:

```python
from scipy.stats import ks_2samp

# For numerical features
for col in numerical_features:
    stat, p = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
    if p < 0.01:
        print(f"⚠️ {col}: distributions differ significantly (p={p:.4f})")
```

**Significant differences are a red flag for either:**
- Leaky split (group leakage, temporal violation)
- Genuine drift the model needs to handle
- Bug in the splitting logic

Either way: investigate before training.

### 6.3 The Single-Feature Audit

For every feature, train a model with only that feature. If single-feature performance is impossibly high:

```python
for col in feature_cols:
    if df[col].dtype == 'object':
        continue
    score = cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42),
        df[[col]].fillna(-999), y,
        cv=3, scoring='roc_auc'
    ).mean()
    if score > 0.90:
        print(f"🚨 {col}: single-feature AUC = {score:.3f} — investigate")
```

**Threshold heuristic:**
- AUC > 0.95 with one feature → almost certainly leakage
- AUC > 0.90 → suspicious, investigate
- AUC > 0.80 → strong feature, verify it's available at inference

---

## 7. Class Imbalance & Rare Events

> Class imbalance is mostly a metric problem, not a data problem. Most "fixes" for imbalance are placebos that distort calibration in exchange for slightly better accuracy on a poorly chosen metric. Read this section carefully before reaching for SMOTE.

### 7.1 What Imbalance Actually Is

A binary classification problem is "imbalanced" when:
- 95/5 → moderate
- 99/1 → severe
- 99.9/0.1 → extreme (fraud, manufacturing defects, rare diseases)

The problem is not that the model can't learn — modern algorithms handle imbalance fine. The problem is:
1. **Accuracy becomes useless** as a metric (predict majority → 99% accuracy)
2. **The decision threshold** of 0.5 is wrong for imbalanced problems
3. **Training signal** for the minority class is sparse

### 7.2 The Hierarchy of Imbalance Fixes

```
LEVEL 1 (always do):
  - Use the right metric: PR-AUC, F1, MCC, balanced accuracy
  - Tune the decision threshold based on business cost
  - Stratify your splits
  - Report performance per class, not just aggregate

LEVEL 2 (often enough):
  - Class weights (sample_weight or class_weight='balanced')
  - Cost-sensitive loss (custom loss weighting FN vs FP)

LEVEL 3 (when L1+L2 isn't enough):
  - Threshold moving with proper calibration
  - Focal loss (for neural networks)

LEVEL 4 (last resort, with caution):
  - Resampling (over/undersampling)
  - Synthetic data (SMOTE and variants)

LEVEL 5 (different problem):
  - Anomaly detection framing instead of classification
  - Reframe as ranking or one-class problem
```

**Most teams jump to Level 4 first. This is wrong.** Class weights and threshold tuning solve most imbalance problems without the calibration damage that resampling causes.

### 7.3 Class Weights

The simplest, safest, often-sufficient fix:

```python
# sklearn
LogisticRegression(class_weight='balanced')
RandomForestClassifier(class_weight='balanced')

# XGBoost
xgb.XGBClassifier(scale_pos_weight=neg_count / pos_count)

# LightGBM
lgb.LGBMClassifier(class_weight='balanced')
# or
lgb.LGBMClassifier(is_unbalance=True)
```

This re-weights the loss so minority-class errors count more. **Does not distort the data.** Calibration is largely preserved.

### 7.4 Threshold Moving

The decision threshold of 0.5 is rarely optimal. Pick the threshold based on business cost.

```python
from sklearn.metrics import precision_recall_curve

probs = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

# Pick threshold that maximizes F1
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
optimal_threshold = thresholds[f1_scores.argmax()]
```

Or pick by **expected cost**:
```
cost = FP_count × cost_per_FP + FN_count × cost_per_FN
Choose threshold minimizing cost.
```

### 7.5 Random Oversampling

Duplicate minority class samples.

**Pros:** Simple, doesn't fabricate data.
**Cons:** No new information, increased overfitting risk.

### 7.6 Random Undersampling

Drop majority class samples.

**Pros:** Faster training.
**Cons:** Loses information.

**Use when:** Massive datasets where undersampling still leaves enough majority data.

### 7.7 SMOTE (Synthetic Minority Oversampling Technique)

Create synthetic minority points by interpolating between existing ones in feature space.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Variants:** Borderline-SMOTE, ADASYN, SVMSMOTE.

**Critical:** Apply SMOTE **only to training data, after splitting.** Never to test or validation data.

### 7.8 Why SMOTE Often Doesn't Help

1. **Synthetic points in high-dim space are often meaningless** — interpolating between two rare events doesn't create a "more typical" rare event
2. **Calibration is destroyed** — your model now thinks the base rate is 50%, but production is still 1%
3. **Bias toward boundaries** — SMOTE creates points near the decision boundary, which can be detrimental
4. **Modern algorithms don't need it** — class weights work as well or better in most benchmarks

**Use SMOTE when:** You've exhausted Levels 1-3, the dataset is small, and empirical CV shows clear improvement.

### 7.9 Focal Loss

For neural networks, focal loss down-weights easy examples and focuses learning on hard ones:

```
FL(p) = -α(1-p)^γ × log(p)
```

Where `γ` (typically 2) controls the focusing strength.

**Use when:** Neural network with severe imbalance, especially detection / segmentation.

### 7.10 The Calibration Trap

**Resampling distorts predicted probabilities.** If you used SMOTE or undersampling, your model's `predict_proba` output no longer reflects true probabilities — it reflects probabilities under the resampled distribution.

If you need calibrated probabilities for downstream decisions (expected value calculations, ranking by risk), you must either:
1. Not resample (use class weights instead)
2. Recalibrate after resampling using `CalibratedClassifierCV` or Platt scaling on the original distribution
3. Adjust predictions analytically using the prior shift formula

### 7.11 Reframing as Anomaly Detection

For extreme imbalance (< 1% positive rate), classification often isn't the right framing. Anomaly detection treats the rare class as outliers in the distribution of "normal" data.

**See:** `Algorithms.md` §14 for anomaly detection methods (Isolation Forest, One-Class SVM, autoencoders).

---

## 8. Labeling — Where Ground Truth Comes From

> Labels are not given. They are produced. The quality of your labels caps the quality of your model — and most teams spend 100x more time on the model than on the labels.

### 8.1 Sources of Labels

| Source | Pros | Cons |
|---|---|---|
| **Direct observation** | High quality | Often expensive or impossible |
| **User behavior (implicit)** | Cheap, abundant | Often noisy proxy for true target |
| **Human annotators** | Flexible, can capture nuance | Cost, consistency, bias |
| **Expert labelers** | High accuracy on complex tasks | Very expensive, slow |
| **Crowdsourcing** | Scalable | Quality control critical |
| **Programmatic / heuristic** | Fast, scalable | Encodes the heuristic's biases |
| **Weak supervision** | Use multiple noisy sources | Requires careful aggregation (Snorkel) |
| **Self-supervised** | No labels needed | Limited tasks |
| **Distilled from larger model** | Cheap, scales | Bounded by the teacher model |

### 8.2 Label Quality Audit

Before training, audit a sample of labels:

- [ ] Pick a random sample (200-1000 examples)
- [ ] Have a second labeler annotate them blindly
- [ ] Compute inter-annotator agreement (Cohen's kappa, Krippendorff's alpha)
- [ ] Investigate disagreements — they reveal ambiguous label criteria

**Heuristic:**
- Agreement > 0.8 → labels are likely reliable
- 0.6 - 0.8 → ambiguity exists; refine guidelines
- < 0.6 → labels are noise; either fix the task or reformulate

### 8.3 The Label Noise Problem

Most production datasets have 5-20% label error rates. Effects:
- Caps model performance
- Inflates apparent overfitting (model fits true patterns but is judged against noisy labels)
- Disproportionate impact on minority classes

**Detection methods:**
- **Confident learning (`cleanlab`)** — identify likely mislabeled examples
- **Loss-based detection** — examples with consistently high training loss may be mislabeled
- **Cross-validation residuals** — examples where the model strongly disagrees with the label

```python
import cleanlab
from cleanlab.classification import CleanLearning
cl = CleanLearning(clf=RandomForestClassifier())
cl.fit(X, y)
label_issues = cl.find_label_issues(X, y)
```

### 8.4 Labeling Guidelines

For any annotation task:
- [ ] Written, versioned guidelines
- [ ] Example annotations for edge cases
- [ ] Decision rules for ambiguous cases
- [ ] Definition of when to skip vs. force a decision
- [ ] Calibration round before production labeling

### 8.5 Weak Supervision

Combine multiple noisy labeling sources to produce probabilistic labels:

| Source | Example |
|---|---|
| Heuristic rules | "If text contains 'free!!!' → spam" |
| Keyword lookups | Dictionary-based sentiment |
| Distant supervision | Knowledge base alignment |
| Existing models | Use a less accurate model to bootstrap |

**Snorkel** is the canonical framework — labels are aggregated using a generative model that learns each source's accuracy without ground truth.

### 8.6 Active Learning Loops

When labeling is expensive, prioritize the most informative examples:

```
1. Label a small seed set
2. Train initial model
3. Score unlabeled pool
4. Select uncertain / disagreed / diverse examples
5. Label those
6. Retrain
7. Repeat
```

Typical savings: 5-10x reduction in labeling cost for the same model performance.

### 8.7 Programmatic Labeling

When the target can be derived from other available data:

**Examples:**
- "Did this customer churn?" → look at activity 30 days later
- "Did this ad get clicked?" → join with clickstream
- "Is this transaction fraudulent?" → join with chargeback table

**Critical:** The derivation logic IS the label definition. Any error in the logic = labels are wrong.

### 8.8 Synthetic Labels from Larger Models

In the LLM era, common pattern:
1. Use a large model (GPT-4, Claude) to label data
2. Train a smaller, cheaper model on those labels
3. Deploy the smaller model

**Caveats:**
- Smaller model's ceiling is the large model's quality on this task
- Large model biases are inherited
- Compliance: are you allowed to use the large model's output for training?

---

## 9. Drift — When Data Changes Under You

> Every model is trained on a snapshot of a moving world. Drift is the inevitable divergence between that snapshot and current reality. Detecting drift is the difference between catching a degrading model in week one and discovering it in month six.

### 9.1 Types of Drift

```
DATA DRIFT (Covariate Shift) — P(X) changes
    The distribution of features changes but the relationship
    between features and target stays the same.
    Example: Average user age in your platform shifts upward.

CONCEPT DRIFT — P(Y|X) changes
    The relationship between features and target changes.
    Example: A feature that predicted fraud last year no longer
    does because fraudsters adapted.

LABEL DRIFT — P(Y) changes
    The base rate of the target changes.
    Example: Fraud rate doubles after a new attack vector emerges.

PRIOR PROBABILITY SHIFT — P(Y) changes, P(X|Y) stays
    Class proportions change but conditional distributions don't.
    A specific form of label drift.
```

### 9.2 Why It Matters

A model that was 92% accurate at launch can degrade to 75% in three months without anyone noticing — performance metrics aren't usually measured continuously in production because **ground truth isn't immediately available** (you don't know if a flagged transaction was actually fraud until days later).

Drift detection catches degradation before performance does.

### 9.3 Detection Methods — Numerical Features

#### Kolmogorov-Smirnov Test
```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(reference, current)
if p < 0.01:
    print("Distribution drift detected")
```

#### Population Stability Index (PSI)
```python
def psi(reference, current, bins=10):
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    ref_counts, _ = np.histogram(reference, breakpoints)
    cur_counts, _ = np.histogram(current, breakpoints)
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
    return np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
```

**Interpretation:**
- PSI < 0.1 → no significant drift
- 0.1 ≤ PSI < 0.25 → moderate drift, monitor
- PSI ≥ 0.25 → significant drift, investigate

#### Wasserstein Distance
Earth mover's distance — quantifies how much "mass" must move to transform one distribution into another. Good for continuous variables with similar shapes but shifted means.

### 9.4 Detection Methods — Categorical Features

#### Chi-squared test
```python
from scipy.stats import chi2_contingency
ref_counts = reference[col].value_counts()
cur_counts = current[col].value_counts()
# Align categories
all_cats = ref_counts.index.union(cur_counts.index)
contingency = pd.DataFrame({
    'reference': ref_counts.reindex(all_cats, fill_value=0),
    'current': cur_counts.reindex(all_cats, fill_value=0),
})
chi2, p, _, _ = chi2_contingency(contingency)
```

#### Categorical PSI
Same formula as numerical PSI but using observed categories instead of bins.

### 9.5 Detection Methods — Multivariate

Single-feature drift detection misses joint distribution changes. For these:

| Method | Notes |
|---|---|
| **Domain classifier** | Train a model to distinguish reference vs current. If AUC > 0.7, distributions differ. |
| **MMD (Maximum Mean Discrepancy)** | Kernel-based two-sample test |
| **Autoencoder reconstruction error** | Drift = reconstruction error increases |

#### Domain Classifier (most practical)
```python
ref_df['is_current'] = 0
cur_df['is_current'] = 1
combined = pd.concat([ref_df, cur_df])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
auc = cross_val_score(
    RandomForestClassifier(n_estimators=100),
    combined.drop(columns='is_current'),
    combined['is_current'],
    scoring='roc_auc'
).mean()
print(f"Domain classifier AUC: {auc:.3f}")
# AUC > 0.7 → significant drift
```

### 9.6 Concept Drift Detection

Concept drift is harder — needs ground truth or proxies.

| Method | Notes |
|---|---|
| **Performance monitoring** | Track metric on labeled production data (with delay) |
| **Prediction drift** | If P(Y_hat) shifts but P(X) doesn't, suggests concept drift |
| **DDM, EDDM, ADWIN** | Sequential drift detectors based on error rates |
| **Page-Hinkley test** | Detects mean shifts in a sequence |

### 9.7 Production Monitoring Stack

| Layer | What to Monitor |
|---|---|
| **Input data** | Schema conformance, missing rates, distribution stats |
| **Features** | PSI vs training reference, feature freshness |
| **Predictions** | Prediction distribution, score distribution |
| **Outcomes** | Performance metrics (with lag), business KPIs |
| **System** | Latency, throughput, error rates |

**Tools:**
- `evidently` — open source, comprehensive drift detection
- `whylogs / WhyLabs` — data and model profiling
- `Arize`, `Fiddler`, `Truera` — commercial ML observability
- `nannyML` — performance estimation without ground truth

### 9.8 Drift Response Playbook

When drift is detected:

1. **Confirm** — is this real or measurement noise? Bootstrap CI on the drift score.
2. **Investigate** — which features drifted? What changed upstream?
3. **Quantify** — is performance actually degraded, or just inputs?
4. **Decide** — retrain, recalibrate, hot-fix, or accept?
5. **Document** — what was the cause? What's the remediation?

**Critical:** Define **retraining triggers** before deployment, not after. (See `Audit.md` §14.)

---

## 10. Synthetic Data & Augmentation

### 10.1 When Synthetic Data Helps

- Privacy-sensitive contexts where real data can't be shared
- Rare class augmentation
- Edge case generation (failure modes, adversarial examples)
- Pretraining or warm-starting models
- Testing pipelines without exposing real data

### 10.2 When It Doesn't

- Replacing real data entirely (synthetic data inherits and amplifies biases of the generator)
- Improving baseline accuracy on a well-represented task (rarely helps)
- Tasks where the underlying distribution is poorly understood

### 10.3 Synthetic Data Methods

| Method | Use |
|---|---|
| **Statistical sampling** (parametric) | When distributions are well-understood |
| **GANs / VAEs** | Image, tabular, sequence synthesis |
| **Diffusion models** | High-quality image / video generation |
| **LLM-generated** | Text augmentation, instruction data |
| **CTGAN, TVAE** | Tabular synthesis libraries |
| **Faker** | Synthetic structured data (names, addresses) |

### 10.4 Data Augmentation (Different from Synthetic)

Augmentation transforms **existing** data to create variations:

| Domain | Common Augmentations |
|---|---|
| Images | Rotate, crop, flip, color jitter, cutout, mixup |
| Text | Back-translation, synonym replacement, paraphrasing, EDA |
| Audio | Time shift, pitch shift, noise injection, SpecAugment |
| Tabular | Mixup, SMOTE variants, Gaussian noise on numerical |

**Critical:** Augmentation applies to **training data only.** Augmenting validation/test data invalidates metrics.

### 10.5 Synthetic Data Validation

When using synthetic data, validate:
- [ ] Statistical similarity to real data (univariate and bivariate)
- [ ] Privacy guarantees (no memorization of training records)
- [ ] Utility test: model trained on synthetic + tested on real performs near model trained on real
- [ ] Diversity (no mode collapse)

---

## 11. Data Versioning & Reproducibility

### 11.1 Why Version Data

Without data versioning, "I trained the model on dataset X" is meaningless — dataset X today is not dataset X last month.

Effects:
- Cannot reproduce past results
- Cannot diagnose performance regressions (was it the model or the data?)
- Cannot audit decisions made on prior model versions
- Cannot compare models trained on different data fairly

### 11.2 What to Version

| Artifact | Version |
|---|---|
| Raw data snapshots | Yes |
| Cleaned / transformed data | Yes |
| Train/val/test split indices | Yes |
| Feature definitions | Yes (as code) |
| Label sources | Yes |
| Configuration | Yes |
| Random seeds | Yes |

### 11.3 Tools

| Tool | Approach |
|---|---|
| **DVC (Data Version Control)** | Git-like, file-based, decoupled storage |
| **LakeFS** | Git-like for data lakes |
| **Delta Lake / Iceberg / Hudi** | Versioned table formats |
| **Pachyderm** | Data pipeline versioning |
| **MLflow** | Logs data references with experiments |
| **Weights & Biases Artifacts** | Versioned artifact tracking |

### 11.4 The Reproducibility Bundle

For each model version, save:

```
model_v1.2.3/
  ├── data_hash.txt        # Hash of training data
  ├── train_idx.npy        # Frozen split
  ├── test_idx.npy
  ├── config.yaml          # All hyperparameters
  ├── feature_def.py       # Feature engineering code
  ├── model.pkl            # Trained artifact
  ├── metrics.json         # Test set performance
  ├── env.txt              # Library versions (pip freeze)
  └── README.md            # Decisions, owner, date
```

If any one is missing, reproducibility is gone.

### 11.5 Deterministic Pipelines

For full reproducibility:
- Pin all random seeds (`numpy`, `random`, `torch`, `tensorflow`, library-specific)
- Pin library versions
- Pin data versions
- Pin hardware where order matters (GPU non-determinism in some ops)
- Use deterministic CUDA operations where applicable

```python
import numpy as np
import random
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 12. Data Pipelines & ETL

### 12.1 The Pipeline Stages

```
EXTRACT     → Pull from sources (databases, APIs, files, streams)
       ↓
VALIDATE    → Schema checks, range checks, null checks
       ↓
TRANSFORM   → Clean, normalize, derive
       ↓
LOAD        → Write to destination (warehouse, feature store, lake)
       ↓
MONITOR     → Quality checks, freshness, anomalies
```

### 12.2 Pipeline Patterns

| Pattern | Use |
|---|---|
| **Batch ETL** | Daily / hourly snapshots, classic warehouse |
| **Streaming** | Real-time features, sub-second latency |
| **Lambda** | Batch + streaming layers reconciled |
| **Kappa** | Streaming-only, replay from log for batch |
| **Medallion** | Bronze (raw) → Silver (clean) → Gold (curated) |

### 12.3 Orchestration Tools

| Tool | Notes |
|---|---|
| **Airflow** | Industry standard, DAG-based, mature |
| **Prefect** | Pythonic, dynamic workflows |
| **Dagster** | Asset-based, strong on data lineage |
| **Argo Workflows** | Kubernetes-native |
| **dbt** | SQL-based transformations within a warehouse |
| **Mage**, **Kestra** | Newer entrants |

### 12.4 Validation in Pipelines

Treat data validation as code:

```python
import pandera as pa
from pandera import Column, DataFrameSchema

schema = DataFrameSchema({
    'user_id': Column(int, checks=pa.Check.ge(0)),
    'age': Column(int, checks=pa.Check.in_range(0, 120)),
    'email': Column(str, checks=pa.Check.str_matches(r'^[\w\.-]+@[\w\.-]+\.\w+$')),
    'created_at': Column(pa.DateTime),
    'amount': Column(float, checks=pa.Check.ge(0), nullable=True),
})

validated_df = schema.validate(df)
```

**Failed validation = pipeline failure.** Bad data should never silently pass to downstream consumers.

### 12.5 Idempotency

A pipeline is idempotent if running it twice produces the same result as running it once. This is critical because:
- Pipelines fail and get retried
- Backfills need to overwrite cleanly
- Late-arriving data needs to be reprocessed

**Pattern:** Use deterministic partitioning (by date, ID) and overwrite-by-partition, not append.

---

## 13. Privacy, PII & Compliance

### 13.1 What Counts as PII

| Direct PII | Indirect PII |
|---|---|
| Name | IP address |
| Email | Device ID |
| Phone | Browser fingerprint |
| SSN / national ID | Precise location |
| Credit card | Voice print |
| Biometric data | Genetic data |

**Quasi-identifiers** (combinations that re-identify): zip code + birthdate + gender → 87% of US population identifiable.

### 13.2 Minimization Principles

- **Don't collect what you don't need**
- **Don't store longer than necessary**
- **Don't share more broadly than required**
- **Don't process more granularly than the use case demands**

### 13.3 Anonymization Techniques

| Technique | Use |
|---|---|
| **Removal** | Drop the field entirely |
| **Pseudonymization** | Replace with stable but non-identifying token |
| **Hashing** | One-way transform (still vulnerable to dictionary attacks) |
| **Bucketing / generalization** | Replace exact value with range (age → age group) |
| **Noise injection** | Add random noise |
| **Differential privacy** | Mathematically guaranteed privacy bounds |
| **k-anonymity** | Each record indistinguishable from k-1 others |

**Caveat:** "Anonymized" datasets have been famously re-identified (Netflix Prize, AOL search logs). True anonymization is hard; differential privacy is the strongest formal guarantee.

### 13.4 Differential Privacy

Add calibrated noise to outputs such that the result is statistically indistinguishable whether or not any single record is included.

| Concept | Notes |
|---|---|
| ε (epsilon) | Privacy budget. Lower ε = more privacy, less utility. |
| Composition | Multiple queries consume budget |
| Local DP | Noise added at the source |
| Central DP | Noise added by trusted aggregator |

**Libraries:** Opacus (PyTorch), TF Privacy, Google DP library.

### 13.5 Regulatory Landscape

| Regulation | Scope |
|---|---|
| **GDPR** (EU) | Personal data of EU residents |
| **CCPA / CPRA** (California) | California consumer data |
| **HIPAA** (US) | Protected health information |
| **PCI-DSS** | Payment card data |
| **LGPD** (Brazil) | Similar to GDPR |
| **DPDP Act** (India) | India's data protection law |

**Key rights commonly granted:**
- Right to access
- Right to deletion ("right to be forgotten")
- Right to rectification
- Right to data portability
- Right to object to automated decision-making

### 13.6 ML-Specific Privacy Issues

- **Memorization** — large models can memorize training records verbatim
- **Membership inference** — attackers can determine if a record was in the training set
- **Model inversion** — reconstruct training data from model outputs
- **Training data extraction** — particularly relevant for LLMs

**Mitigations:** Differential privacy in training, regularization, deduplication, careful data handling.

---

## 14. Cross-Validation Strategies

### 14.1 Why Cross-Validate

A single train/test split has variance — different splits give different metrics. CV averages over multiple splits for a more reliable estimate.

### 14.2 k-Fold Cross-Validation

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

Each fold serves once as validation; rest as training. Average the metric.

**Standard k:** 5 or 10. Higher k = more stable estimate but more compute.

### 14.3 Stratified k-Fold

Preserves class proportions in each fold.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Use for:** Classification, especially imbalanced.

### 14.4 Group k-Fold

Ensures entities don't span folds.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=user_ids):
    ...
```

**Use for:** Any data with entity-level structure.

### 14.5 Stratified Group k-Fold

```python
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5)
```

Preserves both class balance and group constraint.

### 14.6 Time Series CV

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=0)
```

Each fold's training data is strictly before its validation data. **Train set grows; test set is always future.**

For **walk-forward validation** with a sliding window:
```python
TimeSeriesSplit(n_splits=5, gap=0, max_train_size=365)
```

### 14.7 Nested CV

For unbiased model selection + hyperparameter tuning:

```
OUTER LOOP: estimate generalization
  for outer_fold in outer_kfold:
    INNER LOOP: tune hyperparameters
      for inner_fold in inner_kfold:
        train on inner_train, evaluate on inner_val
        select best hyperparameters
    train on outer_train with best hyperparams
    evaluate on outer_val
  report mean outer_val performance
```

**Expensive but unbiased.** Use when stakes are high (publications, business-critical decisions).

### 14.8 Leave-One-Out CV (LOOCV)

k = n. Each row gets a turn as validation.

**Use when:** Very small datasets (< 100 rows). Otherwise too expensive.

### 14.9 The CV Pitfalls

- **CV before split** — folding into pre-existing test set
- **CV with leakage in preprocessing** — fitting scalers/encoders on the entire dataset
- **CV on time series with shuffle** — destroys temporal order
- **CV on grouped data without GroupKFold** — group leakage across folds
- **Different CV folds at tuning vs. evaluation** — invalidates comparison

**The discipline:** CV is part of the modeling process. The test set is still untouched after all CV is done.

---

## 15. Holdout Strategy

### 15.1 The Final Holdout

Beyond train/val splits, a true production model needs a **final, untouched holdout set** — typically a temporal slice that simulates production conditions.

```
Historical data         → train + val
Most recent N weeks     → final holdout (production simulation)
```

Touch the holdout exactly once before deployment. If performance differs significantly from CV, investigate before launching.

### 15.2 Production Holdout

After deployment, set aside a fraction of production traffic with ground truth eventually labeled. This becomes:
- The drift-detection reference
- The retraining performance benchmark
- The fairness audit dataset

### 15.3 Adversarial Holdouts

For high-stakes models, create curated test sets:
- **Edge cases** — hand-crafted hard examples
- **Subgroup tests** — performance per protected attribute
- **Adversarial tests** — perturbed inputs to test robustness
- **Counterfactual tests** — examples that should produce specific predictions

These don't replace random/temporal holdouts but supplement them.

---

## 16. The Data Workflow

### 16.1 End-to-End Sequence

```
1. PROBLEM        — Define the prediction task and unit of analysis
2. ACQUIRE        — Identify and pull all relevant sources
3. PROFILE        — Understand the data before touching it
4. AUDIT          — Quality checks, lineage, sentinel scan
5. LABEL          — Verify ground truth source and quality
6. SPLIT          — Time-respecting, group-aware, frozen
7. SAMPLE         — Address imbalance if needed, within training set only
8. VALIDATE       — Confirm train/test similarity (with awareness)
9. VERSION        — Snapshot data + split + config
10. MONITOR       — Set up drift detection before deployment
11. DOCUMENT      — Data card, lineage, decisions made
```

### 16.2 The Data Card Template

Every dataset used for a production model should have:

```
DATA CARD
─────────────────────────────────────────────
Dataset Name:        [Name]
Version:             [Hash or version tag]
Owner:               [Team or individual]
Description:         [What this dataset contains]
Source(s):           [Where it comes from]
Time Range:          [Date range covered]
Unit of Analysis:    [User / session / transaction / etc.]
Size:                [Rows, columns, disk size]
Target Variable:     [Definition, source, distribution]
Known Limitations:   [Biases, gaps, label noise rate]
Refresh Cadence:     [How often updated]
PII Status:          [Yes / No / Pseudonymized]
Compliance:          [GDPR / HIPAA / etc.]
Last Audited:        [Date]
─────────────────────────────────────────────
```

(Modeled on Google's Data Cards / dataset documentation standards.)

### 16.3 The Pre-Training Data Checklist

```
□ Problem and target definition are unambiguous
□ All data sources have documented lineage
□ Schema validation passes
□ No duplicate rows (or duplicates intentionally handled)
□ Missing value patterns documented
□ Sentinel values mapped
□ PII handled per compliance requirements
□ Train/test split is time-respecting (if temporal)
□ Train/test split is group-respecting (if grouped)
□ Split is frozen and saved
□ Stratification applied where appropriate
□ Single-feature leakage audit complete
□ Pre-split preprocessing audit complete
□ Class imbalance strategy chosen
□ Label quality audited
□ Data versioned and snapshot saved
□ Drift detection reference set captured
□ Data card written
```

---

## 17. Implementation Quick Reference

### Profiling

```python
import pandas as pd
from ydata_profiling import ProfileReport

report = ProfileReport(df, title="Data Profile")
report.to_file("profile.html")

# Quick manual profile
print(df.info())
print(df.describe(include='all'))
print(df.isna().sum())
print(df.nunique())
```

### Splitting

```python
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
)

# Random + stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Time-based
df_sorted = df.sort_values('date')
cutoff = df_sorted['date'].quantile(0.8)
train = df_sorted[df_sorted['date'] < cutoff]
test = df_sorted[df_sorted['date'] >= cutoff]

# Save split indices (freeze!)
import numpy as np
np.save('train_idx.npy', train.index.values)
np.save('test_idx.npy', test.index.values)
```

### Imbalance Handling

```python
# Class weights (preferred default)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight='balanced')

# XGBoost
import xgboost as xgb
neg, pos = (y == 0).sum(), (y == 1).sum()
clf = xgb.XGBClassifier(scale_pos_weight=neg / pos)

# Only if class weights insufficient:
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# NEVER fit_resample on validation or test
```

### Drift Detection

```python
import pandas as pd
from scipy.stats import ks_2samp

def detect_numerical_drift(ref, cur, alpha=0.01):
    drifts = {}
    for col in ref.select_dtypes(include='number').columns:
        stat, p = ks_2samp(ref[col].dropna(), cur[col].dropna())
        drifts[col] = {'ks_stat': stat, 'p_value': p, 'drift': p < alpha}
    return pd.DataFrame(drifts).T

# Or use evidently for a comprehensive report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("drift_report.html")
```

### Label Quality

```python
import cleanlab
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression

cl = CleanLearning(LogisticRegression())
cl.fit(X, y)
issues = cl.find_label_issues(X, y)
print(f"Suspected label errors: {issues.sum()} ({issues.mean()*100:.1f}%)")
```

### Versioning

```bash
# DVC
dvc init
dvc add data/raw_dataset.csv
git add data/raw_dataset.csv.dvc .gitignore
git commit -m "Add raw dataset v1"
dvc push  # to remote storage
```

### Validation

```python
import pandera as pa
from pandera import Column, DataFrameSchema

schema = DataFrameSchema({
    'user_id': Column(int, checks=pa.Check.ge(0), nullable=False),
    'age': Column(int, checks=pa.Check.in_range(0, 120)),
    'signup_date': Column(pa.DateTime, nullable=False),
    'plan': Column(str, checks=pa.Check.isin(['free', 'basic', 'pro'])),
}, strict=True)

# Validate — fails loudly on schema violations
validated = schema.validate(df)
```

---

## Summary — The 10 Data Principles That Actually Matter

If you remember nothing else, internalize these 10:

1. **Profile before you model.** If you don't know the data, you don't know what you're predicting.
2. **Split first, then everything else.** Every preprocessing decision after the split, using training data only.
3. **The test set is sacred.** Touch it once, at the end. Looking is touching.
4. **Time series gets time-based splits, with a buffer.** Random splits on temporal data are silent killers.
5. **Group structure means group-based splits.** Same entity in train and test inflates everything.
6. **Class weights before SMOTE.** Most imbalance problems don't need resampling.
7. **Labels are produced, not given.** Audit label quality before audit model quality.
8. **Drift is inevitable.** Build detection before deployment, not after.
9. **Version the data, not just the model.** Reproducibility starts with knowing what was trained on.
10. **Minimize what you collect.** Privacy isn't a checkbox; it's a discipline that compounds.

The data layer is where 90% of ML quality is determined. The other 10% — algorithms, features, hyperparameters — only matters if the data layer is sound.

---

*`Data.md` — Version 1.0*
*Scope: Comprehensive reference for the data layer of ML systems — from acquisition through monitoring*
*Companion to `Metrics.md` (how to measure), `Audit.md` (how to verify), `Algorithms.md` (what to fit), `Features.md` (how to build signal)*
*Use this as the single source of truth for "is my data ready, and will it stay ready?"*
