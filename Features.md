# Features.md

### The Complete Feature Engineering Reference — How Raw Data Becomes Signal, By Data Type, With Every Encoding, Transformation, and Pitfall

> **Why this file exists:** Algorithms get the credit. Features do the work. The single biggest gap between a mediocre model and a winning one is almost never the algorithm — it's how the features were constructed. This file is the systematic reference for turning raw columns into useful signal, organized by data type, with the failure modes called out explicitly.
>
> **How to read this file:** Find your data type in §2, then read that section deeply. Come back to §3 (encoding) and §4 (scaling) for cross-cutting techniques. §15 (target encoding) and §17 (leakage) are the highest-leverage sections — most production bugs live there.

---

## Table of Contents

1. [Philosophy — How to Think About Feature Engineering](#1-philosophy--how-to-think-about-feature-engineering)
2. [Feature Engineering Decision Tree](#2-feature-engineering-decision-tree)
3. [Categorical Feature Encoding](#3-categorical-feature-encoding)
4. [Numerical Feature Transformations](#4-numerical-feature-transformations)
5. [Feature Scaling](#5-feature-scaling)
6. [Datetime Features](#6-datetime-features)
7. [Text Features](#7-text-features)
8. [Image Features](#8-image-features)
9. [Time Series Features](#9-time-series-features)
10. [Geospatial Features](#10-geospatial-features)
11. [Interaction & Polynomial Features](#11-interaction--polynomial-features)
12. [Aggregation Features](#12-aggregation-features)
13. [Missing Value Handling](#13-missing-value-handling)
14. [Outlier Handling](#14-outlier-handling)
15. [Target Encoding — The Dangerous One](#15-target-encoding--the-dangerous-one)
16. [Feature Selection](#16-feature-selection)
17. [Leakage in Feature Engineering](#17-leakage-in-feature-engineering)
18. [Feature Stores & Production Patterns](#18-feature-stores--production-patterns)
19. [The Feature Engineering Workflow](#19-the-feature-engineering-workflow)
20. [Implementation Quick Reference](#20-implementation-quick-reference)

---

## 1. Philosophy — How to Think About Feature Engineering

### The Three Laws of Feature Engineering

```
LAW 1 — The Law of Domain Signal
    Every useful feature encodes something a domain expert
    would have told you. If you can't explain why a feature
    might predict the target, it probably doesn't — and if it
    does, it's leakage. Domain knowledge is the only sustainable
    source of new signal.

LAW 2 — The Law of the Inference Boundary
    Every feature must be computable at the exact moment of
    prediction, using only information available at that moment.
    A feature that uses tomorrow's data, future aggregates, or
    join keys not yet known is not a feature — it's a bug
    waiting to corrupt your metrics.

LAW 3 — The Law of Diminishing Features
    The first 10 features capture 80% of the signal.
    The next 90 features capture 15%.
    The next 900 features capture 5%.
    More features ≠ better model. Past a point, every new feature
    is a new vector for leakage, drift, and overfitting.
```

### Why Feature Engineering Matters More Than Model Choice

In practice, the model accuracy gradient looks like this:

```
Random features + best model         →  ~50% performance
Good features + worst valid model    →  ~85% performance
Good features + best model           →  ~92% performance
Great features + best model          →  ~95% performance
```

The largest single jump comes from **features**, not algorithm. This is why Kaggle grandmasters spend 70% of their time on features and 30% on models — and why your work should too.

### What Makes a Feature Good

A feature is useful if and only if it has:

| Property | Test |
|---|---|
| **Predictive signal** | Correlates with target after controlling for existing features |
| **Stability** | Distribution doesn't drift catastrophically over time |
| **Availability** | Can be computed at inference time, on the timescale required |
| **Lineage** | You can trace exactly how it was computed and from what source |
| **Independence (kind of)** | Adds something other features don't already provide |

Features that fail the **availability** test are the #1 cause of "looked great offline, died in production" disasters. See §17.

### The Feature Engineering Hierarchy

```
Tier 0 — Raw:           The columns as they arrive
Tier 1 — Cleaned:       Outliers handled, missing values filled, types corrected
Tier 2 — Encoded:       Categoricals encoded, datetimes decomposed, text vectorized
Tier 3 — Scaled:        Numerical features brought to compatible ranges
Tier 4 — Engineered:    Interactions, aggregations, ratios, domain features
Tier 5 — Selected:      Pruned to the subset that actually helps
Tier 6 — Stored:        Versioned, computed offline, served online
```

Each tier must be complete before the next is worth the effort. Skipping tiers = building on sand.

---

## 2. Feature Engineering Decision Tree

### By Feature Type

```
START — for each column in your data:
  │
  ├── NUMERICAL (continuous)
  │     ├── Roughly normal?         → Standardize (z-score)
  │     ├── Bounded, non-normal?    → Min-Max scale
  │     ├── Heavy-tailed?           → Log transform, then standardize
  │     ├── Has outliers?           → Robust scale, or winsorize first
  │     └── Skewed distribution?    → Box-Cox / Yeo-Johnson
  │
  ├── NUMERICAL (discrete / count)
  │     ├── Low cardinality?        → Treat as categorical
  │     ├── High cardinality?       → Log(1 + x), bin, or leave as-is for trees
  │     └── Zero-inflated?          → Add binary "has_value" + log(1 + x)
  │
  ├── CATEGORICAL
  │     ├── Low cardinality (< 15)        → One-hot encoding
  │     ├── Medium cardinality (15-100)   → One-hot or target encoding (with CV)
  │     ├── High cardinality (100-1000)   → Target encoding or hashing
  │     ├── Very high (> 1000)            → Embedding (if NN) or hashing
  │     ├── Ordinal (has order)           → Ordinal encoding
  │     └── Tree-based model + native?    → LightGBM/CatBoost handle it directly
  │
  ├── DATETIME
  │     ├── Extract calendar parts (year, month, dow, hour, etc.)
  │     ├── Cyclical features (sin/cos for hour, day, month)
  │     ├── Time since reference event
  │     ├── Lag and rolling features (time series)
  │     └── Holiday / business day flags
  │
  ├── TEXT
  │     ├── Short, structured?     → Regex, length, keyword flags
  │     ├── Bag-of-words OK?       → TF-IDF + linear model
  │     ├── Semantic matters?      → Sentence embeddings (sentence-transformers)
  │     └── LLM-era?               → OpenAI/Cohere embeddings as features
  │
  ├── IMAGE
  │     ├── Pretrained model       → Use penultimate layer as features
  │     ├── Custom training        → Train CNN/ViT end-to-end
  │     └── Simple proxy           → Color histograms, basic stats
  │
  ├── GEOSPATIAL
  │     ├── Lat/lon                → H3 / Geohash, distance to landmarks
  │     ├── Addresses              → Geocode first, then as lat/lon
  │     └── Regions                → One-hot or target encode
  │
  └── ID / KEY
        ├── Direct use is leakage usually
        ├── Use as join key for aggregations
        └── Frequency encode if entity-level signal needed
```

### By Algorithm

```
USING TREE-BASED MODEL (RF, XGBoost, LightGBM, CatBoost)?
  ├── Scaling — NOT needed
  ├── One-hot encoding — usually unnecessary (use native categorical or label encoding)
  ├── Missing values — leave as NaN; trees handle natively
  ├── Outliers — much less sensitive
  └── Polynomial features — usually wasteful (trees find interactions)

USING LINEAR MODEL (Linear/Logistic, Ridge, Lasso)?
  ├── Scaling — REQUIRED, especially with regularization
  ├── One-hot encoding — REQUIRED for categoricals
  ├── Missing values — must impute
  ├── Outliers — sensitive, consider winsorizing or robust scaling
  ├── Polynomial features — helpful for non-linear relationships
  └── Interactions — must be explicit (model can't find them)

USING NEURAL NETWORK?
  ├── Scaling — REQUIRED (helps optimization)
  ├── Categoricals — embeddings, not one-hot
  ├── Missing values — must handle (imputation + indicator)
  ├── Text — tokenize + embed
  └── Images — minimal preprocessing if using pretrained features

USING kNN / SVM (RBF)?
  ├── Scaling — REQUIRED (distance-based)
  ├── Dimensionality reduction often helps
  └── Curse of dimensionality kicks in fast
```

---

## 3. Categorical Feature Encoding

### 3.1 One-Hot Encoding

Create a binary column for each category.

```
color: ['red', 'blue', 'green']
→ color_red, color_blue, color_green (one is 1, others are 0)
```

**Pros:**
- ✅ No false ordinal relationship
- ✅ Works with any model
- ✅ Interpretable

**Cons:**
- ❌ Cardinality explosion — 1000 categories = 1000 columns
- ❌ Sparse matrices
- ❌ Tree-based models inefficient with one-hot

**Use when:** Low cardinality (< 15 categories), linear models, neural networks (small categoricals).

**Drop one column?** For linear models with intercept, yes (avoid dummy variable trap). For tree models, doesn't matter. For regularized models, also doesn't matter.

### 3.2 Label / Ordinal Encoding

Map each category to an integer.

```
size: ['S', 'M', 'L'] → [0, 1, 2]
```

**Pros:** Compact, fast.

**Cons:** Imposes an ordinal relationship. `'red' = 0, 'blue' = 1, 'green' = 2` makes 'green' twice as far from 'red' as 'blue' — meaningless.

**Use when:**
- Feature is actually ordinal (sizes, education levels, ratings)
- Tree-based model where order doesn't impose linear assumption
- Quick prototyping

### 3.3 Target Encoding (Mean Encoding)

Replace each category with the **mean of the target** for that category.

```
city: ['NYC', 'LA', 'Chicago']
y:    [1, 0, 1, 0, 1, ...]
→ NYC = mean(y where city=NYC) = 0.65
→ LA = 0.40
→ Chicago = 0.55
```

**Pros:**
- ✅ Compact (one column regardless of cardinality)
- ✅ Captures category-target signal directly
- ✅ Powerful for high-cardinality features

**Cons:**
- ❌ **MASSIVE LEAKAGE RISK** if done naively (see §15)
- ❌ Sensitive to category frequency (rare categories overfit)
- ❌ Doesn't generalize to unseen categories

**Use when:** Very high cardinality (zip codes, user IDs, products), tree-based models, you understand the leakage risks.

**See §15 for the full safe implementation pattern.**

### 3.4 Frequency / Count Encoding

Replace each category with its frequency in the training data.

```
country: ['US', 'UK', 'IN'] with counts [10000, 2000, 5000]
→ US = 10000, UK = 2000, IN = 5000
```

**Pros:** Simple, no leakage if computed on training set, captures category prevalence.

**Cons:** Two categories with identical frequencies are indistinguishable. Doesn't capture target signal.

**Use when:** High cardinality, popularity is informative (popular products, popular cities).

### 3.5 Hashing (Feature Hashing / Hash Trick)

Hash each category to one of `n_buckets` columns.

```python
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=64, input_type='string')
```

**Pros:** Fixed dimensionality regardless of vocabulary, fast, no need to store mapping, handles new categories gracefully.

**Cons:** Hash collisions (different categories → same bucket), not interpretable.

**Use when:** Massive cardinality (millions), streaming data with growing vocabulary, online learning.

### 3.6 Binary Encoding

Convert category to integer, then to binary representation, then one column per bit.

```
8 categories → 3 columns (since 2³ = 8)
```

**Pros:** More compact than one-hot for medium cardinality.

**Cons:** Imposes arbitrary ordinal structure via bit patterns. Rarely worth the complexity.

**Use when:** Rarely. Specific compactness requirements with medium cardinality.

### 3.7 Embedding (Learned)

For neural networks: learn a dense vector representation for each category.

```python
# PyTorch
nn.Embedding(num_embeddings=10000, embedding_dim=32)
```

**Pros:** Captures complex similarities between categories, dense compact representation, can be transferred.

**Cons:** Requires neural network, needs sufficient training data, harder to interpret.

**Use when:** High-cardinality categoricals in deep learning (user IDs, product IDs, words).

**Rule of thumb for embedding dim:** `min(50, cardinality / 2)` or `cardinality ** 0.25 * 4`.

### 3.8 Native Categorical Handling

Some libraries handle categoricals directly without manual encoding:

| Library | Method |
|---|---|
| **CatBoost** | Ordered target statistics with permutations (leakage-resistant) |
| **LightGBM** | Pass `categorical_feature=['col1', 'col2']` — uses optimal split finding |
| **H2O** | Native categorical support |
| **XGBoost** | Recent versions support categorical natively (`enable_categorical=True`) |

**This is almost always the best option when using these libraries.** Don't manually encode if the library can do it better.

### 3.9 Decision Matrix

| Cardinality | Tree Model | Linear Model | Neural Net |
|---|---|---|---|
| Low (< 10) | One-hot or label | One-hot | One-hot or small embedding |
| Medium (10-100) | Native categorical | One-hot | Embedding |
| High (100-1000) | Target encoding (CV) or native | Target encoding (CV) | Embedding |
| Very high (> 1000) | Target/frequency/hashing | Hashing | Embedding (essential) |

---

## 4. Numerical Feature Transformations

### 4.1 When Numerical Features Need Transformation

Numerical features benefit from transformation when:
- Distribution is heavily skewed
- Range spans multiple orders of magnitude
- Linear/neural models are being used (trees less sensitive)
- Outliers dominate the feature

### 4.2 Log Transformation

```python
x_log = np.log1p(x)   # log(1 + x), handles x=0
```

**Use for:** Right-skewed distributions, monetary amounts, counts, ratios.

**Effect:** Compresses large values, expands small values. Multiplicative relationships become additive.

**Cannot use on:** Zero or negative values without offset. `log1p` fixes the zero case.

### 4.3 Square Root Transformation

```python
x_sqrt = np.sqrt(x)
```

**Use for:** Moderately right-skewed counts (e.g., Poisson-distributed). Less aggressive than log.

### 4.4 Box-Cox Transformation

Find the optimal power transformation to make data Gaussian.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')  # requires positive values
x_transformed = pt.fit_transform(x.reshape(-1, 1))
```

**Use for:** Making non-normal continuous data approximately normal. Useful for linear models that assume normality.

**Limitation:** Requires strictly positive values.

### 4.5 Yeo-Johnson Transformation

Like Box-Cox but **handles negative and zero values**.

```python
pt = PowerTransformer(method='yeo-johnson')
```

**Default choice over Box-Cox** unless you know your data is strictly positive.

### 4.6 Quantile Transformation (Rank-Based)

Map each value to its quantile (uniform or Gaussian output).

```python
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
```

**Use for:** Any distribution → normal/uniform. Robust to outliers. Destroys absolute magnitudes but preserves order.

**Caveat:** Won't generalize well to values outside training range.

### 4.7 Binning / Discretization

Convert continuous to categorical buckets.

**Strategies:**
- **Equal-width:** Same range per bin. Sensitive to outliers.
- **Equal-frequency (quantile):** Same count per bin. Robust.
- **Domain-driven:** Custom bin edges based on knowledge (e.g., age groups: 0-17, 18-34, 35-54, 55+).
- **Tree-based:** Use a decision tree to find optimal splits.

**When binning helps:**
- Non-linear relationship that's hard to model otherwise (e.g., U-shaped)
- Need to model interactions cleanly
- Want to handle outliers without removing them
- Working with linear models on inherently non-linear features

**When binning hurts:**
- Loses information unnecessarily
- Trees already do this implicitly — don't pre-bin for trees

### 4.8 Ratios and Differences

Often the most predictive features aren't raw values but **relationships between them**:

| Feature | Use |
|---|---|
| `purchases / sessions` | Conversion rate |
| `revenue / users` | ARPU |
| `current_value - 30day_avg` | Anomaly indicator |
| `clicks / impressions` | CTR |

**Domain-specific ratios are usually more predictive than the raw values they're derived from.**

---

## 5. Feature Scaling

### 5.1 Why Scale

Many algorithms assume features are on similar scales:
- **Distance-based** (kNN, K-Means, SVM with RBF): distance dominated by large-scale features
- **Gradient descent** (linear models, neural nets): converges faster with similar scales
- **Regularization** (L1, L2): penalty is unfair to large-scale features

**Trees are exempt:** Random Forest, XGBoost, LightGBM, CatBoost don't need scaling.

### 5.2 StandardScaler (Z-Score)

```
x' = (x - μ) / σ
```

Result: mean 0, std 1.

**Use when:** Data is roughly normal, no extreme outliers, default for linear models.

**Library:** `sklearn.preprocessing.StandardScaler`

### 5.3 MinMaxScaler

```
x' = (x - min) / (max - min)
```

Result: range [0, 1].

**Use when:** Need bounded output, neural network inputs (sometimes), images.

**Drawback:** Extremely sensitive to outliers — a single outlier can compress everything else.

### 5.4 RobustScaler

```
x' = (x - median) / IQR
```

**Use when:** Outliers present, you don't want to remove them but don't want them to dominate scaling.

### 5.5 MaxAbsScaler

```
x' = x / |max|
```

Result: range [-1, 1], preserves sign and sparsity.

**Use when:** Sparse data (don't want to shift zeros), already-centered data.

### 5.6 Normalization (L2 / L1)

Scale **rows** so each has unit norm. Different from per-column scaling above.

```
x' = x / ||x||
```

**Use when:** Text features (TF-IDF), comparing samples by direction not magnitude (cosine similarity).

### 5.7 Critical Rules

1. **Fit on training data only.** Transform train and test with the same fitted scaler.
2. **Pipeline it.** Use `sklearn.pipeline.Pipeline` to chain scaling with model. This prevents leakage and simplifies deployment.
3. **Save the scaler.** It's part of the model — must be applied identically in production.
4. **Watch for distribution shift.** Scaler fit on old data may not fit new data well.

### 5.8 Scaling Decision Matrix

| Algorithm | Scaling? | Recommended |
|---|---|---|
| Linear/Logistic Regression | Yes | StandardScaler |
| Ridge / Lasso / ElasticNet | **Required** | StandardScaler |
| SVM | **Required** | StandardScaler |
| kNN | **Required** | StandardScaler |
| K-Means | **Required** | StandardScaler |
| PCA | **Required** | StandardScaler |
| Neural Networks | Yes | StandardScaler or MinMaxScaler |
| Naive Bayes | No effect | — |
| Decision Tree | No | — |
| Random Forest | No | — |
| XGBoost / LightGBM / CatBoost | No | — |

---

## 6. Datetime Features

### 6.1 Calendar Decomposition

Extract obvious components from any datetime:

| Component | Why It Matters |
|---|---|
| Year | Long-term trends |
| Month | Annual seasonality |
| Day of month | Pay days, monthly patterns |
| Day of week | Weekly patterns (huge for B2C) |
| Hour | Daily patterns |
| Quarter | Business cycles |
| Week of year | Sub-annual seasonality |
| Is_weekend | Major behavior shift |

### 6.2 Cyclical Encoding

**The big trap:** December (12) and January (1) are adjacent, but the model sees them as 11 units apart. Same for hour 23 → hour 0.

**Fix:** Sin/cos encoding.

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

Now December and January are appropriately close in feature space.

**Critical for:** Linear models, neural networks, any model that takes Euclidean distance.

**Trees:** Less critical because trees can handle the discontinuity via splits, but it still often helps.

### 6.3 Time-Based Differences

| Feature | Example |
|---|---|
| `days_since_signup` | User tenure |
| `days_since_last_purchase` | Recency |
| `time_until_event` | Anticipation effect |
| `days_until_holiday` | Pre-holiday buildup |
| `time_in_session` | Engagement |

### 6.4 Business Calendar Features

| Feature | Why |
|---|---|
| `is_holiday` | Demand spikes/dips |
| `is_business_day` | B2B activity |
| `days_to_payday` | Spending patterns |
| `is_weekend_or_holiday` | Combined leisure flag |
| `is_school_term` | Family behavior |

**Use libraries:** `holidays` (Python) handles country-specific holidays automatically.

### 6.5 Common Datetime Pitfalls

- **Timezone bugs.** Always store and process in UTC; only convert at the edges. Mixing timezones is a silent killer.
- **Off-by-one in week numbers.** ISO vs US week numbering differ.
- **Daylight Saving Time.** Some hours don't exist; some hours exist twice. Hour features can be misleading.
- **Leap years / leap seconds.** Day-of-year features can shift.
- **Future dates in training data.** Always verify max date is before prediction time. (See `Audit.md` §5.)

---

## 7. Text Features

### 7.1 Basic Text Features

Before any heavy NLP, extract simple structural features:

| Feature | Notes |
|---|---|
| Length (chars, words, sentences) | Often surprisingly predictive |
| Number of uppercase letters | Shouty content signal |
| Number of digits, punctuation | Structure |
| Presence of URLs, emails, phone numbers | Spam / professional signals |
| Sentiment score (VADER, TextBlob) | Crude but cheap |
| Readability (Flesch-Kincaid) | Complexity |
| Language detection | Multilingual handling |

### 7.2 Bag-of-Words (BoW)

Count occurrences of each word.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X = cv.fit_transform(corpus)
```

**Pros:** Simple, fast, interpretable, sparse-friendly.

**Cons:** No semantics, word order ignored beyond n-grams, vocabulary blowup.

### 7.3 TF-IDF

Term Frequency × Inverse Document Frequency. Weight by how distinctive a word is across documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X = tfidf.fit_transform(corpus)
```

**The default for classical text ML.** Pair with logistic regression or LinearSVC — astonishingly strong baseline.

### 7.4 N-grams

Sequences of N consecutive tokens.

```
"not good" — unigrams capture "not" and "good" separately, missing the negation
n-grams (1,2) capture "not good" as a single feature
```

**Trade-off:** More expressive but vocabulary explodes. Use `min_df` to filter rare n-grams.

### 7.5 Word Embeddings (Pre-Deep)

Dense vectors per word, learned from large corpora:
- **Word2Vec** (Mikolov, 2013) — Skip-gram or CBOW
- **GloVe** (Stanford) — global co-occurrence statistics
- **FastText** (Facebook) — subword info, handles OOV

To use as features: average word vectors per document.

**Status:** Largely superseded by sentence/contextual embeddings, but still useful in low-resource settings.

### 7.6 Sentence / Document Embeddings

Modern default. One dense vector per document.

| Approach | Library | Notes |
|---|---|---|
| **Sentence-Transformers** | `sentence-transformers` | Free, ~384-768 dim, runs locally |
| **OpenAI embeddings** | API | `text-embedding-3-small`, `text-embedding-3-large` |
| **Cohere embeddings** | API | Multilingual options |
| **BGE / E5** | HuggingFace | Strong open-source models |

**Workflow:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)  # (n_docs, 384)

# Use embeddings as features in downstream model
clf = LogisticRegression().fit(embeddings, y)
```

**This is the modern default for any text classification problem.**

### 7.7 Topic Models

| Method | Notes |
|---|---|
| **LDA (Latent Dirichlet Allocation)** | Classical, interpretable topics |
| **NMF** | Non-negative factorization, often more coherent topics |
| **BERTopic** | Embedding-based, modern default |
| **Top2Vec** | Joint document and topic embeddings |

**Use when:** Need interpretable topic distributions as features.

### 7.8 Text Preprocessing Decisions

| Step | Pre-LLM era | Modern (Embeddings) |
|---|---|---|
| Lowercase | Yes | Often no (models handle case) |
| Remove punctuation | Yes | No |
| Remove stopwords | Yes | No |
| Stemming / Lemmatization | Yes | No |
| Remove rare words | Yes (min_df) | No |
| Normalize whitespace | Yes | Yes |
| Remove HTML / URLs | Depends | Depends |

**The trend is clear:** modern transformer-based embeddings need less preprocessing — they've seen messy text during pretraining.

---

## 8. Image Features

### 8.1 Classical Image Features

For when deep learning is overkill or compute-limited:

| Feature | Description |
|---|---|
| Color histograms | RGB / HSV distribution |
| Edge features | Sobel, Canny |
| Texture (Haralick, GLCM) | Co-occurrence matrices |
| HOG (Histogram of Oriented Gradients) | Shape and gradient distribution |
| SIFT / SURF / ORB | Local keypoints (largely legacy) |

### 8.2 Pretrained CNN Features

The modern default for "image as features": use a pretrained network, extract the penultimate layer.

```python
import torch
import torchvision.models as models
from torchvision import transforms

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # Remove final layer
model.eval()

# features = (batch_size, 2048) for ResNet-50
```

**Common backbones:**
- ResNet-50 / ResNet-101 (2048-dim)
- EfficientNet (varies)
- ViT (768-1024 dim)
- CLIP image encoder (512 dim) — aligned with text

### 8.3 CLIP Embeddings

Joint text-image embeddings — useful for cross-modal search and zero-shot classification.

```python
import clip
model, preprocess = clip.load("ViT-B/32")
image_features = model.encode_image(preprocess(image).unsqueeze(0))
text_features = model.encode_text(clip.tokenize(["a cat", "a dog"]))
```

**Use when:** Image classification with limited labels, image-text similarity, multimodal applications.

### 8.4 Image Metadata

Don't forget the EXIF data:
- Capture date / time
- GPS location
- Camera model
- Resolution
- Aspect ratio
- File size

Often as predictive as the image itself for problems like spam detection or content moderation.

### 8.5 Image Augmentation as Feature Engineering

For training (not inference):

| Augmentation | When |
|---|---|
| Random crop / resize | Always |
| Horizontal flip | Most natural images |
| Color jitter | Vary lighting conditions |
| Rotation | If orientation isn't critical |
| Cutout / Random Erasing | Regularization |
| Mixup / CutMix | Strong regularization |
| RandAugment | Automated augmentation policies |

**Critical:** Augmentation is for training only. Test-time augmentation (TTA) is a different technique.

---

## 9. Time Series Features

> **The single highest-leverage feature engineering domain.** Get this right and trees + lag features will beat most deep learning approaches.

### 9.1 Lag Features

Past values of the series itself.

```python
df['y_lag_1'] = df['y'].shift(1)    # Previous time step
df['y_lag_7'] = df['y'].shift(7)    # Same day last week
df['y_lag_28'] = df['y'].shift(28)  # Same day 4 weeks ago
df['y_lag_365'] = df['y'].shift(365) # Same day last year
```

**Critical:** Always group by entity if you have panel data.
```python
df['y_lag_7'] = df.groupby('store_id')['y'].shift(7)
```

### 9.2 Rolling Window Statistics

Statistics over a recent window — captures trends and volatility.

```python
df['y_rolling_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
df['y_rolling_std_7'] = df['y'].shift(1).rolling(window=7).std()
df['y_rolling_max_30'] = df['y'].shift(1).rolling(window=30).max()
df['y_rolling_min_30'] = df['y'].shift(1).rolling(window=30).min()
```

**The `.shift(1)` is non-negotiable** — without it, the rolling window includes the current point, which is target leakage. (See §17.)

**Window sizes to try:** Day, week, month, quarter, year — pick by domain.

### 9.3 Expanding Window Features

Statistics over all history up to current point.

```python
df['y_expanding_mean'] = df['y'].shift(1).expanding().mean()
df['y_expanding_max'] = df['y'].shift(1).expanding().max()
```

**Use when:** Long-term baseline is informative, no fixed-length window is natural.

### 9.4 Exponential Weighted Moving Average (EWMA)

Recent values weighted more than old.

```python
df['y_ewm_7'] = df['y'].shift(1).ewm(span=7).mean()
```

**Often better than simple rolling mean** for capturing recent regime.

### 9.5 Differencing

Capture change rather than level.

```python
df['y_diff_1'] = df['y'].diff(1)        # First difference
df['y_diff_7'] = df['y'].diff(7)        # Weekly change
df['y_pct_change'] = df['y'].pct_change(1)
```

**Use when:** Series has strong trend, modeling change is easier than modeling level.

### 9.6 Aggregation by Time Window

For higher-cardinality time series (many entities), aggregate at different windows:

```python
# For each store-product, daily features
agg = df.groupby(['store', 'product']).agg({
    'sales': ['mean', 'std', 'min', 'max', 'sum'],
    'price': ['mean', 'std'],
})
```

### 9.7 Fourier Features for Seasonality

For complex seasonality, encode periodic patterns directly:

```python
def fourier_features(t, period, K):
    """K pairs of (sin, cos) for given period."""
    features = {}
    for k in range(1, K + 1):
        features[f'sin_p{period}_k{k}'] = np.sin(2 * np.pi * k * t / period)
        features[f'cos_p{period}_k{k}'] = np.cos(2 * np.pi * k * t / period)
    return features
```

**Use when:** Multiple overlapping seasonalities (daily + weekly + yearly), Prophet-style modeling.

### 9.8 Special Time Series Patterns

| Feature | Use |
|---|---|
| `time_since_event` | Recency of last sale, click, error |
| `count_in_last_N_days` | Burstiness |
| `is_first_observation` | Cold-start indicator |
| `cumulative_sum` | Running totals |
| `rank_within_window` | Relative position |

### 9.9 The Critical Rule: No Future Data

Every time series feature must satisfy:

> **At time t, the feature uses only data from time ≤ t.**

Violations:
- ❌ Future aggregates leaking via `rolling(...).mean()` without shift
- ❌ Centered windows
- ❌ Forward-filling missing values
- ❌ Computing features on the entire dataset and then splitting

See `Audit.md` §16 for the master leakage reference.

---

## 10. Geospatial Features

### 10.1 From Raw Coordinates

| Feature | Description |
|---|---|
| Latitude, Longitude | Raw — sometimes useful as-is |
| Geohash (length 5-7) | Hierarchical spatial grid |
| H3 cell (resolution 7-9) | Uber's hexagonal grid (better neighbor properties) |
| S2 cell | Google's spherical grid |

```python
import h3
df['h3_cell'] = df.apply(lambda r: h3.geo_to_h3(r['lat'], r['lon'], resolution=8), axis=1)
```

### 10.2 Distance Features

| Feature | Use |
|---|---|
| Distance to nearest landmark | Distance to airport, downtown, etc. |
| Distance to nearest of category | Closest restaurant, closest competitor |
| Average distance to N nearest | Density proxy |
| Distance to centroid | Anomaly indicator |

### 10.3 Region-Based Features

Join coordinates to administrative regions:
- Country, state, city
- ZIP / postal code
- Census tract / block
- Custom polygons (delivery zones)

Then aggregate features at the region level (median income, population density, etc.).

### 10.4 Density Features

| Feature | Description |
|---|---|
| Points within radius R | Local density |
| Count of category within R | Competitor density, POI density |
| KDE (kernel density estimate) | Smoothed density at point |

### 10.5 Trajectory Features (for moving entities)

| Feature | Description |
|---|---|
| Speed | Distance / time |
| Acceleration | Δspeed / time |
| Heading | Direction of travel |
| Stop duration | Time stationary |
| Total distance traveled | Trip length |
| Detour ratio | Actual distance / straight-line distance |

---

## 11. Interaction & Polynomial Features

### 11.1 Interaction Features

Combine two or more features to capture joint effects.

```python
df['price_per_sqft'] = df['price'] / df['sqft']
df['age_x_income'] = df['age'] * df['income']
df['device_x_country'] = df['device'].astype(str) + '_' + df['country'].astype(str)
```

**When you need them:**
- **Linear models** can't find interactions automatically — must be explicit
- **Domain knowledge** suggests joint effects (e.g., income only matters in certain age groups)

**When you don't need them:**
- **Tree-based models** find interactions automatically via sequential splits
- **Neural networks** find them in hidden layers

### 11.2 Polynomial Features

Square, cube, and cross terms.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False)
X_poly = poly.fit_transform(X)
```

**Use sparingly:**
- Quickly explodes feature count: 10 features → 65 features with degree=2
- Often creates highly collinear features
- Trees find polynomial-like patterns natively

**Better than blind polynomial:** Targeted polynomial features based on EDA findings.

### 11.3 Pairwise Ratios

For features on the same scale:

```python
for f1, f2 in important_pairs:
    df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-8)
```

**Watch for:** Division by zero (add small epsilon), exploding values (clip).

### 11.4 Splines

Piecewise polynomial functions — capture non-linear shapes without overfitting like high-degree polynomials.

```python
from sklearn.preprocessing import SplineTransformer
spline = SplineTransformer(n_knots=5, degree=3)
X_spline = spline.fit_transform(X)
```

**Use when:** Need smooth non-linear features for linear models, classical statistics workflows.

---

## 12. Aggregation Features

### 12.1 Group-Level Statistics

For each entity, summarize its history or related records.

```python
# For each user: aggregates over their purchase history
user_stats = df.groupby('user_id').agg(
    total_purchases=('amount', 'count'),
    avg_purchase=('amount', 'mean'),
    max_purchase=('amount', 'max'),
    purchase_std=('amount', 'std'),
    unique_categories=('category', 'nunique'),
    days_since_first_purchase=('date', lambda x: (today - x.min()).days),
)
```

### 12.2 Common Aggregation Patterns

| Pattern | Example |
|---|---|
| Count of past events | `n_purchases_last_30d` |
| Sum | `total_revenue_last_year` |
| Mean / median | `avg_session_duration` |
| Min / max | `max_single_purchase` |
| Standard deviation | `purchase_volatility` |
| Unique count | `n_distinct_categories_browsed` |
| Most frequent value | `most_common_device` |
| First / last value | `first_product_purchased` |
| Time since first / last | `days_since_signup`, `days_since_last_active` |

### 12.3 Hierarchical Aggregations

Compute aggregates at multiple levels of a hierarchy:

```
user → user_stats
user × product → user_product_stats
user × category → user_category_stats
user × month → user_month_stats
```

**Then join back** to the main table. Trees love these features.

### 12.4 The Time-Windowed Aggregation Pattern

For every aggregation, compute it over multiple windows:

```
For user u at time t, compute:
  - mean_purchase_last_7_days
  - mean_purchase_last_30_days
  - mean_purchase_last_90_days
  - mean_purchase_lifetime
```

**Critical:** Use only data from before time t. This is the central feature pattern in fraud, churn, and demand forecasting — and the central leakage trap.

### 12.5 Cross-Entity Aggregation

Aggregate features from related entities:

| Pattern | Example |
|---|---|
| User × Product | "How many other users bought this product" |
| Item × Location | "How well does this item sell in this region" |
| User × Cohort | "How does user compare to similar users" |

---

## 13. Missing Value Handling

### 13.1 Why Values Are Missing

Diagnose before imputing. There are three patterns:

| Pattern | Meaning | Strategy |
|---|---|---|
| **MCAR** (missing completely at random) | Missingness unrelated to data | Simple imputation OK |
| **MAR** (missing at random) | Missingness depends on observed features | Model-based imputation |
| **MNAR** (missing not at random) | Missingness depends on the missing value itself | Add indicator; may be predictive |

**Critical insight:** Missingness itself is often predictive. Always add a "was_missing" indicator before imputing.

### 13.2 Strategies

#### 13.2.1 Drop
```python
df.dropna(subset=['important_col'])
```
**Use when:** Few rows affected, missingness is random, no production constraint.

#### 13.2.2 Constant Imputation
```python
df['col'].fillna(-999)  # Sentinel for trees
df['col'].fillna(0)
df['col'].fillna('UNKNOWN')
```
**Use when:** Tree models, sentinel values, missing means "none" semantically.

#### 13.2.3 Mean / Median / Mode
```python
df['col'].fillna(df['col'].median())  # Numerical
df['col'].fillna(df['col'].mode()[0])  # Categorical
```
**Use when:** Simple baseline, MCAR.

**Critical:** Compute fill values on **training set only**, apply to test/production.

#### 13.2.4 Forward / Backward Fill
```python
df['col'].fillna(method='ffill')
```
**Use when:** Time series where last known value is a good estimate. Never use `bfill` on time series — that's future data leakage.

#### 13.2.5 Model-Based Imputation
Train a model to predict missing values from other features.
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=0)
```
**Use when:** MAR pattern, many features, complex relationships.

#### 13.2.6 Native Handling
- **XGBoost, LightGBM, CatBoost** — handle NaN natively. Best option when available.

### 13.3 The Missing Indicator Pattern

```python
df['col_was_missing'] = df['col'].isna().astype(int)
df['col'] = df['col'].fillna(df['col'].median())
```

**Always do this when imputing.** The missingness itself is a feature.

### 13.4 Pitfalls

- **Imputing with the global mean before splitting** — leaks information across train/test. (See `Audit.md` §16.)
- **Forward-filling across entities** — fills user A's data with user B's. Always group first.
- **Using `bfill` on time series** — leaks future data into past.
- **Ignoring why values are missing** — sometimes missing is not random and carries information.

---

## 14. Outlier Handling

### 14.1 Detection

| Method | Use |
|---|---|
| **Z-score** (\|z\| > 3) | Univariate, assumes normality |
| **IQR** (outside Q1-1.5×IQR, Q3+1.5×IQR) | Univariate, robust |
| **Isolation Forest** | Multivariate, scales |
| **Local Outlier Factor** | Density-based |
| **Modified Z-score (MAD-based)** | Robust univariate |
| **Domain rules** | Physical limits, business rules |

### 14.2 Strategies

#### 14.2.1 Remove
**Use sparingly.** Only when you're confident the outlier is a data error.

#### 14.2.2 Cap / Winsorize
Clip to a percentile range.
```python
lower = df['col'].quantile(0.01)
upper = df['col'].quantile(0.99)
df['col'] = df['col'].clip(lower, upper)
```
**Use when:** Outliers are real but distort modeling. Linear models, neural networks benefit.

#### 14.2.3 Transform
Log, sqrt, or quantile transform compress the tail.

#### 14.2.4 Robust Models
Use models that don't care: trees, robust regression (Huber loss), median-based statistics.

#### 14.2.5 Treat as Separate Feature
Add an indicator: `is_outlier = (df['col'] > threshold).astype(int)`. Often the outlier itself is the signal.

### 14.3 When NOT to Remove Outliers

- They represent real, rare events that matter (fraud, equipment failures)
- They're the majority of value (top 1% of users drive 50% of revenue)
- You're not sure they're errors

**Removing outliers without understanding them is one of the most common data science mistakes.**

---

## 15. Target Encoding — The Dangerous One

> **The single highest-leverage feature engineering technique for high-cardinality categoricals — and the single most common source of subtle leakage. Most implementations you'll find online are wrong. Read this section carefully.**

### 15.1 What It Is

Replace each category with the mean of the target for that category.

```
city: 'NYC' → mean(y where city='NYC') = 0.62
city: 'LA'  → mean(y where city='LA')  = 0.41
```

For high-cardinality features (zip codes, user IDs, product IDs), this is far more efficient than one-hot encoding and often more powerful.

### 15.2 Why Naive Target Encoding Leaks

```python
# THIS IS WRONG — DO NOT DO THIS
encoding = df.groupby('city')['y'].mean()
df['city_encoded'] = df['city'].map(encoding)
```

The problem: the encoding for each row includes that row's own target. The model can essentially see the answer for each training point.

Train AUC goes through the roof. Test AUC barely moves. Classic leakage.

### 15.3 The Correct Pattern: Out-of-Fold Target Encoding

```python
from sklearn.model_selection import KFold

def target_encode_oof(X_train, y_train, X_test, col, n_splits=5, smoothing=10):
    """
    Target encode `col` using out-of-fold means.
    Test set uses the full training set's encoding.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_train = pd.Series(index=X_train.index, dtype=float)
    
    global_mean = y_train.mean()
    
    for train_idx, val_idx in kf.split(X_train):
        # For the fold's training portion, compute means
        fold_means = y_train.iloc[train_idx].groupby(X_train.iloc[train_idx][col]).mean()
        fold_counts = y_train.iloc[train_idx].groupby(X_train.iloc[train_idx][col]).count()
        
        # Apply smoothing toward global mean
        smoothed = (fold_means * fold_counts + global_mean * smoothing) / (fold_counts + smoothing)
        
        # Encode the validation fold
        encoded_train.iloc[val_idx] = X_train.iloc[val_idx][col].map(smoothed).fillna(global_mean)
    
    # For test, use full training set encoding (smoothed)
    full_means = y_train.groupby(X_train[col]).mean()
    full_counts = y_train.groupby(X_train[col]).count()
    full_smoothed = (full_means * full_counts + global_mean * smoothing) / (full_counts + smoothing)
    encoded_test = X_test[col].map(full_smoothed).fillna(global_mean)
    
    return encoded_train, encoded_test
```

### 15.4 Smoothing

Without smoothing, rare categories get extreme encodings.

```
Category 'rare_city' appears once with y=1 → encoded as 1.0 (overfit)
```

**Smoothing toward the global mean:**

```
smoothed_value = (count × category_mean + smoothing × global_mean) / (count + smoothing)
```

- High smoothing → rare categories pulled hard toward global mean
- Low smoothing → trust category means more

Typical `smoothing` values: 5 to 50.

### 15.5 CatBoost's Approach

CatBoost handles this elegantly with **ordered target statistics**:

1. Randomly permute the training data
2. For each row, encode using only **prior** rows in the permutation
3. Use multiple permutations to reduce variance

This is the safest target encoding implementation available. **Just use CatBoost** if you have many high-cardinality categoricals.

### 15.6 Variants

- **Mean encoding** — default, target mean per category
- **Frequency encoding** — count per category (no target, no leakage)
- **Weight of Evidence (WoE)** — used heavily in credit risk, log(P(y=1|cat) / P(y=0|cat))
- **Leave-one-out encoding** — each row encoded by mean of all other rows in its category

### 15.7 Critical Rules

1. **Never compute target encoding on the full dataset before splitting.**
2. **Always use out-of-fold computation for training data.**
3. **Always smooth toward the global mean.**
4. **Validate that training and test AUCs aren't suspiciously close** — if they are, your encoding is leaking.
5. **For time series, use only past data** (expanding mean), not future.

---

## 16. Feature Selection

### 16.1 Why Select Features

- Reduce overfitting
- Speed up training and inference
- Easier interpretation
- Cheaper to maintain in production
- Less drift surface area

### 16.2 Filter Methods (model-agnostic)

| Method | Description |
|---|---|
| **Variance threshold** | Drop features with near-zero variance |
| **Correlation** | Drop one of each pair of highly correlated features |
| **Mutual Information** | Captures non-linear relationships with target |
| **Chi-squared** | For categorical features vs. categorical target |
| **ANOVA F-test** | For numerical features vs. categorical target |

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
```

### 16.3 Wrapper Methods

Train models with different feature subsets.

- **Recursive Feature Elimination (RFE)** — start with all, remove worst iteratively
- **Sequential Forward Selection** — start with none, add best iteratively
- **Sequential Backward Selection** — start with all, remove worst iteratively

**Expensive but thorough.** Use for small-to-medium feature sets.

### 16.4 Embedded Methods

Feature selection happens during model training.

- **Lasso regression** — L1 penalty forces some coefficients to zero
- **Tree feature importance** — Gini importance, gain, permutation
- **SHAP values** — game-theoretic feature attribution

```python
import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
```

### 16.5 Permutation Importance

Shuffle each feature, measure performance drop.

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=10)
```

**Pros:** Model-agnostic, captures actual contribution.
**Cons:** Slow, can be misleading with correlated features.

### 16.6 Drop-Column Importance

Retrain the model dropping each feature, measure performance drop. **The most reliable importance measure, but expensive.**

### 16.7 SHAP

Game-theoretic feature attribution. Decomposes each prediction into per-feature contributions.

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

**Use for:** Understanding individual predictions, global feature importance, debugging.

### 16.8 Practical Workflow

1. **Drop obviously useless** (zero variance, near-constant)
2. **Drop one of each highly correlated pair** (|r| > 0.95)
3. **Train a model with all remaining features**
4. **Look at feature importance / SHAP**
5. **Drop bottom features iteratively, measuring CV performance**
6. **Stop when performance starts to degrade**

**Don't:** Use a single statistical test as your only filter. Use multiple methods and triangulate.

---

## 17. Leakage in Feature Engineering

> See `Audit.md` §16 for the master leakage reference. This section focuses on the **feature engineering–specific** leakage patterns.

### 17.1 The Five Feature Leakage Patterns

#### 17.1.1 Target Leakage
Feature directly encodes the target.

**Example:** Predicting whether an order will be returned, and a feature is `total_refund_amount`. The refund only exists if the order was returned.

**Detection:**
- Single-feature AUC > 0.95 → almost certain leakage
- Removing one feature drops performance from 0.95 to 0.65 → that feature was reading the answer

#### 17.1.2 Train-Test Contamination via Preprocessing

```python
# WRONG — fits scaler on combined data
scaler.fit(pd.concat([X_train, X_test]))

# RIGHT — fits only on train
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Same applies to:** imputers, encoders, feature selection, PCA, target encoding.

#### 17.1.3 Temporal Leakage

Using future information at training time.

**Examples:**
- Rolling window without `.shift(1)` — current row's value affects the window
- Joining latest version of a dimension table (which may differ from the version at the time of the row)
- Computing aggregates over the entire dataset before splitting time-wise

#### 17.1.4 Group Leakage

Same entity in train and test.

**Example:** Predicting churn at user level, but multiple sessions per user appear in both train and test. The model learns user-specific patterns and reports inflated accuracy.

**Fix:** Split by group (`GroupKFold`).

#### 17.1.5 Aggregation Leakage

Computing aggregations using the target itself.

**Example:** `avg_purchase_per_user` computed including the target row. If target is "did this transaction occur," you've encoded the answer.

**Fix:** Always exclude the current row when computing entity-level aggregates that are then joined back.

### 17.2 The Detection Toolkit

| Signal | Likely Cause |
|---|---|
| Single-feature AUC > 0.95 | Target leakage |
| Train AUC ≈ Test AUC, both very high | Train-test contamination |
| Validation score >> public benchmark | Leakage |
| Top feature is suspicious in name/source | Worth investigating |
| Removing top 3 features collapses performance | Investigate those features |
| Time-based split degrades performance vs random split | Temporal leakage |

### 17.3 The Prevention Checklist

For every feature, ask:
- [ ] At inference time, will I have this feature computable at the prediction moment?
- [ ] Was this feature computed using only data from before the prediction time?
- [ ] If the target were unknown, could this feature still be computed?
- [ ] Has this feature been computed using out-of-fold methods if it's target-derived?
- [ ] Has preprocessing (scaling, imputation, encoding) been fit on training data only?

If any answer is "no" or "I don't know" — investigate before training.

---

## 18. Feature Stores & Production Patterns

### 18.1 The Training-Serving Skew Problem

The #1 production failure for ML systems:

```
Training:  Features computed via SQL on data warehouse, joined offline
Serving:   Features computed via Python service on different data, different logic
Result:    Model sees different feature distributions in production than in training
```

This is fundamental and silent. Models can lose 30%+ of performance from training-serving skew alone, without any drift in actual data.

### 18.2 What a Feature Store Solves

A feature store provides:
- **Single source of truth** for feature definitions (one place, one logic)
- **Point-in-time correct training data** (no future leakage in historical joins)
- **Online and offline consistency** (same code paths)
- **Feature reuse** across teams and models
- **Monitoring and versioning** of features

### 18.3 Feature Store Architectures

| Component | Role |
|---|---|
| **Offline store** | Historical features for training (e.g., BigQuery, Snowflake, Delta Lake) |
| **Online store** | Low-latency feature serving (e.g., Redis, DynamoDB, BigTable) |
| **Feature definition layer** | Code/config defining how features are computed |
| **Materialization layer** | Computes features on schedule and writes to both stores |
| **SDK / API** | How models retrieve features |

### 18.4 Popular Tools

| Tool | Notes |
|---|---|
| **Feast** | Open source, lightweight, good for getting started |
| **Tecton** | Commercial, mature production features |
| **Hopsworks** | Open-core, end-to-end ML platform |
| **AWS SageMaker Feature Store** | Managed AWS option |
| **Vertex AI Feature Store** | GCP managed |
| **Databricks Feature Store** | If you're on Databricks |

### 18.5 Point-in-Time Joins (PIT)

The critical capability that distinguishes feature stores from databases.

```
For each (user_id, timestamp) in training data,
  retrieve features as they were known at that timestamp
```

Naive SQL joins get this wrong (typically pulling latest values). PIT joins get it right.

**This is what makes feature stores worth the complexity.**

### 18.6 Online-Offline Parity

Features must be computed identically in:
- Training pipelines (batch, historical)
- Serving infrastructure (real-time, online)

**Patterns:**
- Single feature definition compiled to both batch (SQL/Spark) and streaming (Python/Flink)
- Shared library used in both pipelines
- Shadow comparison: log online features and compare to recomputed offline features daily

### 18.7 Feature Monitoring

Once features are in production:

| Monitor | Why |
|---|---|
| Feature distribution (mean, std, quantiles) | Detect data drift |
| Missing rate per feature | Upstream pipeline failures |
| PSI vs training distribution | Quantified drift |
| Computation latency | SLO compliance |
| Feature freshness | Stale features = wrong predictions |

---

## 19. The Feature Engineering Workflow

### 19.1 The Order of Operations

```
1. Understand the problem and target
2. Profile the data (types, distributions, missingness, cardinality)
3. SPLIT FIRST — train/test/holdout, time-respecting if temporal
4. Establish baselines:
     - All-zeros / mean / mode
     - Simple model with raw features
5. Engineer features iteratively:
     - One feature family at a time
     - Measure CV improvement
     - Audit for leakage
6. Select features
7. Re-train, re-validate
8. Document every feature: source, logic, owner, expected behavior
9. Move to production: serve from same definitions
```

### 19.2 The Iteration Loop

```
HYPOTHESIS:   "Time-since-last-purchase should predict churn"
       ↓
IMPLEMENT:    Compute feature with proper time-respecting logic
       ↓
VALIDATE:     CV improvement?  Single-feature signal?  Leakage check?
       ↓
INSPECT:      Distribution sensible?  Available in production?
       ↓
DECIDE:       Keep / drop / refine
       ↓
DOCUMENT:     Add to feature audit log
```

### 19.3 The Feature Audit Log Template

```
| # | Feature | Source | Logic | Available at Inference | Single-Feature Signal | Status |
|---|---------|--------|-------|------------------------|----------------------|--------|
| 1 | days_since_last_purchase | events table | (now - max(event_date)) per user | Yes | AUC 0.72 | KEEP |
| 2 | total_refund_amount | orders table | sum(refund) per user | NO — leakage | AUC 0.99 | DROP |
```

Maintain this in your repo. (See `Audit.md` §6 for the full template.)

### 19.4 The 10-Feature Discipline

Before adding feature #11, ask: which of the existing 10 can I drop?

**Why:** Most production teams have 100+ features where 20 do all the work. The other 80 are maintenance debt — they break pipelines, drift silently, and add no signal. Prune aggressively.

---

## 20. Implementation Quick Reference

### sklearn-compatible Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numerical_features = ['age', 'income', 'tenure']
categorical_features = ['city', 'device', 'plan']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
])

# Full pipeline with model
from sklearn.linear_model import LogisticRegression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)  # All preprocessing fit on train only
y_pred = model.predict(X_test)  # Test data transformed using train-fit preprocessors
```

### Key Library Reference

| Task | Library | Notes |
|---|---|---|
| General ML preprocessing | `sklearn.preprocessing` | Standard toolkit |
| Categorical encoding | `category_encoders` | Many encoders beyond sklearn |
| Text features | `sklearn.feature_extraction.text` | TF-IDF, CountVectorizer |
| Sentence embeddings | `sentence-transformers` | Modern text features |
| Image features | `torchvision`, `timm` | Pretrained models |
| Time series features | `tsfresh`, `feature-engine` | Automated feature generation |
| Datetime | `pandas`, `holidays` | Calendar features |
| Geospatial | `h3-py`, `geopandas` | Spatial features |
| Imputation | `sklearn.impute`, `miceforest` | Including MICE |
| Outlier detection | `pyod` | Comprehensive outlier toolkit |
| Feature selection | `sklearn.feature_selection`, `shap`, `boruta` | Multiple paradigms |
| Feature stores | `feast`, `tecton`, `hopsworks` | Production patterns |
| Pipelines | `sklearn.pipeline`, `dagster`, `prefect` | Orchestration |

### Default Workflow Template

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. SPLIT FIRST
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. IDENTIFY TYPES
numerical = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(include='object').columns.tolist()

# 3. DEFINE PIPELINE (fit only on train)
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical),
    ('cat', categorical_pipeline, categorical),
])

# 4. ADD FEATURES INCREMENTALLY
# - Calendar features from datetime columns
# - Aggregations from grouping
# - Interactions where domain suggests
# - Each batch: CV evaluate, audit for leakage

# 5. SELECT
# - Drop near-constant
# - Drop highly correlated
# - Train model, inspect importance
# - Drop bottom features iteratively

# 6. DOCUMENT
# - Update feature audit log
# - Version the pipeline
# - Save with joblib / pickle for production
```

---

## Summary — The 10 Feature Engineering Principles That Actually Matter

If you remember nothing else, internalize these 10:

1. **Split before you do anything.** Every transformation fit on training data only.
2. **Domain knowledge beats algorithms.** A feature a domain expert would suggest is worth 10 from auto-FE.
3. **The inference boundary is sacred.** No feature uses data unavailable at prediction time.
4. **Trees and linear models want different features.** Don't blindly apply both treatments.
5. **Categorical cardinality dictates strategy.** Low → one-hot, high → target encoding (carefully), very high → embeddings or hashing.
6. **Cyclical features need sin/cos.** December and January are neighbors. Hour 23 and hour 0 are neighbors.
7. **Missing is information.** Always add the indicator before imputing.
8. **Target encoding is a loaded gun.** Out-of-fold, smoothed, or use CatBoost.
9. **Time series features must look backward only.** `.shift(1)` before every rolling operation.
10. **Production parity is non-negotiable.** Same logic, same source, same code, in training and serving.

Most production ML problems are feature engineering problems wearing model selection clothes. Master features and most of the rest takes care of itself.

---

*`Features.md` — Version 1.0*
*Scope: Comprehensive reference for feature engineering across all data types and production patterns*
*Companion to `Metrics.md` (how to measure), `Audit.md` (how to verify), `Algorithms.md` (what to fit)*
*Use this as the single source of truth for "how do I turn this column into signal?"*
