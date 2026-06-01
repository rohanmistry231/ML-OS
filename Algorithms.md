# Algorithms.md

### The Complete Machine Learning Algorithm Reference — Every Major Family, How It Works, When to Use It, How It Fails

> **Why this file exists:** Knowing metrics tells you *what* to measure. Knowing audits tells you *what to check*. This file tells you *what to actually build* — and crucially, *why* one algorithm beats another on a given problem. Most engineers can name 30 algorithms; very few can predict which one will win on a dataset before training. This file is built to close that gap.
>
> **How to read this file:** Don't read it linearly. Use the decision trees in §2 to find your family, then read that section. Come back when you hit a failure mode you don't recognize.

---

## Table of Contents

1. [Philosophy — How to Think About Algorithm Choice](#1-philosophy--how-to-think-about-algorithm-choice)
2. [The Algorithm Selection Decision Tree](#2-the-algorithm-selection-decision-tree)
3. [Linear Models](#3-linear-models)
4. [Tree-Based Models](#4-tree-based-models)
5. [Gradient Boosting](#5-gradient-boosting)
6. [Support Vector Machines](#6-support-vector-machines)
7. [k-Nearest Neighbors](#7-k-nearest-neighbors)
8. [Naive Bayes](#8-naive-bayes)
9. [Neural Networks — Foundations](#9-neural-networks--foundations)
10. [Deep Learning Architectures](#10-deep-learning-architectures)
11. [Transformers & Attention](#11-transformers--attention)
12. [Unsupervised — Clustering](#12-unsupervised--clustering)
13. [Unsupervised — Dimensionality Reduction](#13-unsupervised--dimensionality-reduction)
14. [Anomaly Detection](#14-anomaly-detection)
15. [Time Series Models](#15-time-series-models)
16. [Recommender Systems](#16-recommender-systems)
17. [Reinforcement Learning](#17-reinforcement-learning)
18. [Probabilistic & Bayesian Models](#18-probabilistic--bayesian-models)
19. [Ensemble Strategies](#19-ensemble-strategies)
20. [The Bias-Variance Framework](#20-the-bias-variance-framework)
21. [Hyperparameter Reference Card](#21-hyperparameter-reference-card)
22. [Algorithm Failure Modes Cheat Sheet](#22-algorithm-failure-modes-cheat-sheet)
23. [Implementation Quick Reference](#23-implementation-quick-reference)

---

## 1. Philosophy — How to Think About Algorithm Choice

### The Three Laws of Algorithm Selection

```
LAW 1 — The Law of the Baseline
    Every problem starts with the simplest possible model.
    If logistic regression wins, you don't need XGBoost.
    If linear regression wins, you don't need a neural network.
    Complexity must be earned, not assumed.

LAW 2 — The Law of Data Geometry
    Algorithm choice is dictated by data shape, not preference.
    Tabular small data → boosted trees, almost always.
    Sequential data → architecture that respects order.
    High-dimensional sparse → linear models with regularization.
    The algorithm should match the geometry of the problem.

LAW 3 — The Law of Inductive Bias
    Every algorithm encodes assumptions about the world.
    Linear models assume additivity.
    Trees assume axis-aligned splits.
    CNNs assume translation invariance.
    Transformers assume relational structure matters more than position.
    Picking the wrong inductive bias wastes data; picking the right one
    means a smaller model wins.
```

### What "Best Algorithm" Actually Means

There is no "best algorithm." There is only **best algorithm for this data, this constraint, this metric, this deployment target.** A model that wins on accuracy but takes 200ms to infer is a losing model in real-time bidding. A model that's 0.3% better but uninterpretable is a losing model in healthcare.

The five constraints that determine algorithm choice:

| Constraint | Question to Ask | Common Answer |
|---|---|---|
| **Accuracy ceiling** | What's the best possible performance? | Gradient boosting (tabular), Transformers (sequence) |
| **Data volume** | How much labeled data do you have? | < 10k rows → linear/trees; > 1M → neural |
| **Interpretability** | Does the business need to explain predictions? | Linear / shallow trees / GAMs |
| **Latency budget** | How fast must inference be? | Linear < trees < neural < transformer |
| **Update frequency** | How often does the model retrain? | Online learning vs. batch |

### The Hierarchy of Algorithm Complexity

```
Tier 0 — Baselines:          Mean, median, mode, last value, naive seasonal
Tier 1 — Linear:             Linear/logistic regression, Ridge, Lasso, GLMs
Tier 2 — Non-linear classic: kNN, Decision Tree, Naive Bayes, SVM
Tier 3 — Ensembles:          Random Forest, Gradient Boosting (XGBoost/LightGBM/CatBoost)
Tier 4 — Shallow neural:     MLPs, simple CNNs/RNNs
Tier 5 — Deep:               ResNet, LSTM/GRU, U-Net
Tier 6 — Transformer-era:    BERT, GPT, ViT, foundation models
Tier 7 — Frontier:           LLM agents, mixture of experts, multimodal
```

**Always start at Tier 0 and move up only when the previous tier hits a clear ceiling.** This is non-negotiable. Skipping tiers is the #1 cause of overengineered, underperforming systems.

---

## 2. The Algorithm Selection Decision Tree

### By Data Type

```
START
  │
  ├── TABULAR DATA (rows × columns of mixed types)
  │     ├── < 1,000 rows         → Linear models, Naive Bayes, kNN
  │     ├── 1k - 100k rows       → Random Forest, Gradient Boosting ★ default
  │     ├── 100k - 10M rows      → LightGBM/XGBoost, possibly TabNet/FT-Transformer
  │     └── > 10M rows           → Distributed boosting, factorization machines, deep tabular
  │
  ├── IMAGE DATA
  │     ├── Classification       → CNN (ResNet, EfficientNet) or ViT
  │     ├── Detection            → YOLO, Faster R-CNN, DETR
  │     ├── Segmentation         → U-Net, Mask R-CNN, SAM
  │     └── Generation           → Diffusion models, GANs
  │
  ├── TEXT DATA
  │     ├── Classification       → Fine-tuned BERT/RoBERTa, or LLM zero-shot
  │     ├── Generation           → GPT-family, T5, Llama
  │     ├── Embedding/Retrieval  → Sentence-Transformers, OpenAI embeddings
  │     └── Classic NLP          → TF-IDF + Logistic Regression (still a strong baseline)
  │
  ├── TIME SERIES
  │     ├── Univariate forecasting     → ARIMA, ETS, Prophet, N-BEATS
  │     ├── Multivariate w/ exog vars  → LightGBM with lag features, Temporal Fusion Transformer
  │     ├── Many related series        → DeepAR, hierarchical models
  │     └── Anomaly detection          → Isolation Forest, autoencoders, prophet residuals
  │
  ├── GRAPH DATA
  │     ├── Node classification  → GraphSAGE, GAT, GCN
  │     ├── Link prediction      → Node2Vec, GNN
  │     └── Graph classification → Graph Transformer, GIN
  │
  └── SEQUENTIAL / EVENT DATA
        ├── User behavior        → Transformer (self-attention over events)
        ├── Clickstream          → RNN, Transformer
        └── Survival             → Cox PH, Random Survival Forest, DeepSurv
```

### By Constraint

```
NEED INTERPRETABILITY?
  └── Linear models, GAMs, shallow trees, EBM (Explainable Boosting Machine)

NEED < 1ms INFERENCE?
  └── Linear models, small trees, quantized neural nets

NEED ONLINE / STREAMING UPDATES?
  └── SGD-based linear models, Hoeffding Trees, online learning variants

NEED PROBABILISTIC OUTPUTS WITH UNCERTAINTY?
  └── Bayesian models, Gaussian Processes, MC Dropout, conformal prediction

NEED TO HANDLE MISSING VALUES NATIVELY?
  └── XGBoost, LightGBM, CatBoost (others require imputation)

NEED TO HANDLE CATEGORICAL FEATURES NATIVELY?
  └── CatBoost, LightGBM (others require encoding)

VERY LIMITED LABELED DATA?
  └── Pretrained models + fine-tuning, semi-supervised learning, active learning
```

---

## 3. Linear Models

> **The most underrated algorithm family in modern ML.** Linear models are dismissed as "too simple" by people who have never properly tried them. On the right data, with the right features, a linear model can match or beat XGBoost — and run 1000x faster.

### 3.1 Linear Regression

**The model:**
```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ + ε
```

**How it learns:** Minimizes sum of squared residuals. Closed-form solution exists (Normal Equation), or solved via gradient descent for large data.

**Inductive bias:** The relationship between features and target is additive and linear. Effects don't interact unless you explicitly create interaction terms.

**Pros:**
- ✅ Fast to train and predict
- ✅ Fully interpretable — every coefficient has meaning
- ✅ Strong theoretical foundation (statistical inference, confidence intervals)
- ✅ Often the right answer for small, low-noise tabular data

**Cons:**
- ❌ Can't capture non-linear relationships without feature engineering
- ❌ Sensitive to outliers (squared loss)
- ❌ Assumes feature independence (multicollinearity is a problem)
- ❌ Assumes homoscedasticity (constant variance of errors)

**Key assumptions (the "LINE" mnemonic):**
- **L**inearity of relationship
- **I**ndependence of errors
- **N**ormality of residuals
- **E**qual variance (homoscedasticity)

**Use when:** Small tabular data, need interpretability, relationship is roughly linear, baseline.

**Avoid when:** Non-linear relationships, high-dimensional data without regularization, heavy outliers.

### 3.2 Ridge, Lasso, Elastic Net

These add **regularization** — a penalty for large coefficients — to linear regression.

| Method | Penalty | Effect |
|---|---|---|
| **Ridge (L2)** | λ × Σwᵢ² | Shrinks all coefficients toward zero, none exactly zero |
| **Lasso (L1)** | λ × Σ|wᵢ| | Forces some coefficients to exactly zero → feature selection |
| **Elastic Net** | α·L1 + (1-α)·L2 | Combines both; useful when features are correlated |

**When to use each:**
- **Ridge** — many features, all probably useful, multicollinearity is a problem
- **Lasso** — many features, suspect most are useless, want sparse model
- **Elastic Net** — Lasso but groups of correlated features should be selected together

**Critical detail:** Always **standardize features** before regularization (StandardScaler). Otherwise penalties unfairly target features with large scales.

**Hyperparameter:** `alpha` (λ in formulas) — the regularization strength. Tune via cross-validation. `LogisticRegressionCV` and `RidgeCV` do this automatically.

### 3.3 Logistic Regression

**The model:** Linear regression piped through a sigmoid to produce probabilities.
```
P(y=1|x) = 1 / (1 + e^(-(w₀ + w₁x₁ + ... + wₙxₙ)))
```

**How it learns:** Maximum likelihood estimation via gradient descent. No closed form.

**The decision boundary is linear in feature space** — this is the key limitation and also the key strength (regularizes against overfitting).

**Pros:**
- ✅ Outputs calibrated probabilities (better than most classifiers out of the box)
- ✅ Fast, scales to massive data
- ✅ Coefficients are interpretable as log-odds
- ✅ Strong baseline for any binary classification

**Cons:**
- ❌ Can't capture non-linear boundaries
- ❌ Struggles with class imbalance (use `class_weight='balanced'`)
- ❌ Sensitive to outliers in feature space

**Multiclass extensions:**
- **One-vs-Rest (OvR):** Train one binary classifier per class
- **Multinomial (Softmax):** Single model with softmax output — usually better

**Use when:** Binary classification baseline, need calibrated probabilities, high-dimensional sparse data (text via TF-IDF), interpretability matters.

### 3.4 Generalized Linear Models (GLMs)

A unified framework for regression problems with non-Gaussian errors:

| Distribution | Link Function | Use For |
|---|---|---|
| Gaussian | Identity | Standard regression |
| Binomial | Logit | Binary classification |
| Poisson | Log | Count data (number of events) |
| Negative Binomial | Log | Overdispersed counts |
| Gamma | Log/Inverse | Positive continuous (e.g., insurance claims) |
| Tweedie | Log | Zero-inflated continuous (e.g., demand) |

**Why this matters in practice:** Using OLS regression for count data, claim amounts, or time-to-event data is statistically wrong. GLMs give you the right tool.

**Library:** `statsmodels.GLM`, `sklearn` has limited GLM support, R's `glm()` is the gold standard.

### 3.5 When Linear Models Beat Everything Else

- **Very wide data** (more features than rows): high-dimensional genomics, text TF-IDF
- **Sparse data**: bag-of-words, one-hot encoded categoricals
- **Online learning**: SGD-based linear models update in milliseconds
- **Interpretability requirements**: medical, legal, financial decisions
- **Extremely tight latency** (sub-millisecond)

**Real example:** Most production ad-click predictors at scale use logistic regression or factorization machines, not deep learning. The data is so sparse and the latency budget so tight that complex models lose.

---

## 4. Tree-Based Models

### 4.1 Decision Trees

**The algorithm:** Recursively split the data on the feature/threshold that maximizes information gain (classification) or reduces variance (regression).

**Splitting criteria:**
- **Gini impurity** (classification): `1 - Σpᵢ²`
- **Entropy** (classification): `-Σpᵢ log(pᵢ)`
- **MSE / variance reduction** (regression)

**How a single tree fails:**
- Trees with no max_depth memorize training data → severe overfitting
- Small data perturbations create totally different trees → high variance
- Greedy splits → can miss the globally optimal tree

**Pros:**
- ✅ No feature scaling required
- ✅ Handles mixed data types
- ✅ Captures interactions automatically
- ✅ Highly interpretable (single tree)
- ✅ Robust to outliers

**Cons:**
- ❌ High variance — small data changes change the tree dramatically
- ❌ Can't extrapolate beyond training range
- ❌ Axis-aligned splits only (rotating the feature space changes everything)
- ❌ Biased toward features with many unique values

**Use a single tree only when:** You need a fully interpretable model with rules a human can read.

### 4.2 Random Forest

**The idea:** Train many decision trees, each on:
1. A bootstrap sample of the data (sampling with replacement)
2. A random subset of features at each split

Then **average their predictions** (regression) or **vote** (classification).

**Why this works:** Individual trees overfit in different directions; averaging reduces variance without much increase in bias. This is the **bagging principle**.

**Key hyperparameters:**

| Parameter | Effect |
|---|---|
| `n_estimators` | More trees = better, with diminishing returns. 100-500 typical. |
| `max_depth` | Controls per-tree complexity. None = grow until pure. |
| `max_features` | Features considered per split. √n for classification, n/3 for regression. |
| `min_samples_leaf` | Minimum samples per leaf. Higher = more regularization. |
| `bootstrap` | True = sample with replacement (default). |

**Pros:**
- ✅ Strong out-of-the-box performance on tabular data
- ✅ Resistant to overfitting (more trees doesn't hurt)
- ✅ Built-in feature importance (mean decrease in impurity)
- ✅ Out-of-bag error estimate (free cross-validation)
- ✅ Parallelizable

**Cons:**
- ❌ Memory-heavy (stores every tree)
- ❌ Slower inference than a single tree
- ❌ Less accurate than gradient boosting on most tabular benchmarks
- ❌ Feature importance is biased toward high-cardinality features

**Use when:** Tabular baseline beyond linear, need a robust default, want OOB estimates, parallelism available.

### 4.3 Extra Trees (Extremely Randomized Trees)

Like Random Forest, but:
- Uses the full dataset (no bootstrap)
- Random thresholds at each split (not optimal thresholds)

Result: even more variance reduction, slightly higher bias. Faster to train.

**Use when:** Random Forest is good but you want more regularization or faster training.

---

## 5. Gradient Boosting

> **The single most important algorithm family for tabular data in the last decade.** If you have rows × columns of features and a target, your first model after a linear baseline should be gradient boosting. Period.

### 5.1 The Core Idea

Train trees **sequentially**, where each new tree corrects the errors of the previous ensemble:

```
Prediction = Tree₁(x) + η × Tree₂(x) + η × Tree₃(x) + ... + η × Treeₙ(x)
```

Each tree is trained on the **gradient** (or residuals, in the simplest case) of the loss function with respect to the current predictions. The learning rate `η` controls how much each tree contributes.

**Key insight:** Where Random Forest reduces *variance* by averaging, boosting reduces *bias* by sequentially correcting errors. Different mechanism, often better result.

### 5.2 XGBoost

The original game-changer (2014). Brought gradient boosting to mainstream.

**Why it became dominant:**
- L1 + L2 regularization on leaf weights
- Sparsity-aware split finding (handles missing values natively)
- Cache-aware, parallel histogram-based splitting
- Built-in cross-validation, early stopping
- GPU support

**Critical hyperparameters:**

| Parameter | Range | Effect |
|---|---|---|
| `learning_rate` (eta) | 0.01 - 0.3 | Lower = better but slower; use early stopping |
| `n_estimators` | 100 - 10000 | Use early stopping to find optimal |
| `max_depth` | 3 - 10 | Deeper = more complex; 6 is a strong default |
| `min_child_weight` | 1 - 10 | Min samples per leaf; higher = more regularization |
| `subsample` | 0.5 - 1.0 | Row sampling per tree |
| `colsample_bytree` | 0.5 - 1.0 | Feature sampling per tree |
| `gamma` | 0 - 5 | Min loss reduction to split; higher = more conservative |
| `reg_alpha` | 0 - 10 | L1 regularization |
| `reg_lambda` | 0 - 10 | L2 regularization |

**Tuning strategy:**
1. Set `learning_rate = 0.1`, find optimal `n_estimators` via early stopping
2. Tune `max_depth` and `min_child_weight`
3. Tune `gamma`
4. Tune `subsample` and `colsample_bytree`
5. Tune `reg_alpha` and `reg_lambda`
6. Lower `learning_rate` to 0.01-0.05, increase `n_estimators` proportionally

### 5.3 LightGBM

Microsoft's gradient boosting. **The current default for production tabular ML.**

**Key innovations:**
- **Leaf-wise growth** (vs level-wise in XGBoost) — grows the leaf with maximum loss reduction. Faster convergence, but can overfit on small data.
- **GOSS (Gradient-based One-Side Sampling)** — keeps high-gradient instances, randomly samples low-gradient. Faster training.
- **EFB (Exclusive Feature Bundling)** — bundles mutually exclusive sparse features.
- **Native categorical handling** — no need to one-hot encode.

**When LightGBM beats XGBoost:**
- Large datasets (millions of rows)
- Many categorical features
- Memory-constrained environments
- Need fast training/iteration

**When XGBoost beats LightGBM:**
- Small datasets (LightGBM overfits more easily)
- Stable, well-tested production environments

**LightGBM-specific tip:** Use `max_depth` AND `num_leaves` together. Constraint: `num_leaves < 2^max_depth`. Common values: `num_leaves=31`, `max_depth=-1` (no limit).

### 5.4 CatBoost

Yandex's gradient boosting. Specialized for categorical features.

**Key innovations:**
- **Ordered boosting** — reduces a subtle form of target leakage that XGBoost and LightGBM are susceptible to
- **Native categorical encoding** — uses target statistics with permutations to prevent leakage
- **Symmetric trees** — same split condition at the same depth; faster inference, acts as regularization

**When CatBoost wins:**
- Many high-cardinality categorical features
- Small-to-medium datasets where overfitting risk is high
- Need minimal preprocessing
- Want best out-of-the-box performance without much tuning

**Real-world performance pattern:** On Kaggle and similar benchmarks, the three (XGBoost, LightGBM, CatBoost) often perform within 0.5% of each other. **Try all three, ensemble them.**

### 5.5 Loss Functions for Boosting

Boosting can optimize almost any differentiable loss:

| Loss | Use For |
|---|---|
| Squared Error | Regression (default) |
| Absolute Error | Robust regression |
| Huber | Regression with outliers |
| Quantile | Quantile regression / prediction intervals |
| Log Loss | Binary classification |
| Softmax | Multiclass classification |
| Poisson | Count regression |
| Tweedie | Zero-inflated regression (insurance, demand) |
| Cox | Survival analysis |
| LambdaRank / RankNet | Learning-to-rank |

### 5.6 Why Boosting Beats Deep Learning on Tabular Data

This is one of the most-debated topics in modern ML, but the empirical answer is clear: **on tabular data with < 10M rows, gradient boosting almost always beats deep learning.** Reasons:

1. **Tabular data lacks the smooth, hierarchical structure** that deep learning exploits (images have edges → textures → objects; tabular data has none of this)
2. **Trees handle mixed types, missing values, and outliers natively**
3. **Boosting is rotation-invariant** (or close to it) where neural nets aren't
4. **Less data needed** — boosting trains well on thousands of rows; neural nets need millions
5. **No architecture search** — boosting has well-understood hyperparameter ranges

Recent neural tabular models (TabNet, FT-Transformer, SAINT) close the gap on very large tabular datasets but rarely beat boosting unconditionally.

---

## 6. Support Vector Machines

### 6.1 The Core Idea

Find the hyperplane that **maximizes the margin** between classes. For non-linearly-separable data, use a **kernel** to project into a higher-dimensional space where separation is possible.

**The margin:** The distance from the hyperplane to the nearest training points (support vectors).

**The kernel trick:** Compute inner products in the high-dimensional space without explicitly transforming the data.

### 6.2 Common Kernels

| Kernel | Use For |
|---|---|
| **Linear** | High-dimensional data (text), when linear boundary is enough |
| **RBF (Gaussian)** | Default for non-linear problems; controlled by `gamma` |
| **Polynomial** | When you suspect polynomial relationships |
| **Sigmoid** | Rarely used; can behave like a neural network |

### 6.3 Hyperparameters

| Parameter | Effect |
|---|---|
| `C` | Inverse of regularization strength. Higher C = tighter fit, more overfitting risk |
| `gamma` (RBF) | Reach of a single point. High = local, low = global |
| `kernel` | Linear, RBF, poly, sigmoid |

**Tuning rule of thumb:** Grid search C ∈ [0.01, 0.1, 1, 10, 100] and gamma ∈ [0.001, 0.01, 0.1, 1] for RBF.

### 6.4 Pros and Cons

**Pros:**
- ✅ Effective in high-dimensional spaces
- ✅ Strong theoretical foundation
- ✅ Memory-efficient (only stores support vectors)
- ✅ Works well when number of features > number of samples

**Cons:**
- ❌ Scales poorly to large datasets (O(n²) to O(n³))
- ❌ Doesn't output probabilities natively (need Platt scaling)
- ❌ Sensitive to feature scaling — *must* standardize
- ❌ Choosing the right kernel and tuning is non-trivial
- ❌ Largely superseded by gradient boosting for tabular and neural nets for everything else

### 6.5 When SVMs Still Matter

- **Small to medium datasets** (< 50k samples) with clear margin structure
- **Text classification** with linear kernels (LinearSVC still competitive with logistic regression)
- **High-dimensional, low-sample data** (genomics, some imaging)
- **One-Class SVM** for anomaly detection

---

## 7. k-Nearest Neighbors

### 7.1 The Algorithm

**"No training"** — just store the data. To predict for a new point:
1. Find the k nearest training points (by some distance metric)
2. Classification: majority vote
3. Regression: mean (or weighted mean) of neighbors' values

### 7.2 Key Choices

| Choice | Options | Notes |
|---|---|---|
| **k** | 1, 3, 5, 7, ... | Small k = high variance, large k = high bias. Tune via CV. Use odd k for binary classification. |
| **Distance metric** | Euclidean, Manhattan, Cosine, Minkowski | Cosine for text; Euclidean default; Manhattan for high-dim or sparse |
| **Weighting** | Uniform or by inverse distance | Distance-weighted often better |

### 7.3 The Curse of Dimensionality

In high dimensions, **all points are roughly equidistant** from each other. kNN breaks down because the concept of "nearest" loses meaning. Rule of thumb: kNN starts struggling above ~20 features unless they're highly correlated.

### 7.4 Pros and Cons

**Pros:**
- ✅ No training time
- ✅ Trivially supports new classes (just add new points)
- ✅ Naturally handles multi-class
- ✅ Strong baseline for low-dimensional structured data

**Cons:**
- ❌ Slow inference (O(n) per query naively; O(log n) with KD-trees, only in low dim)
- ❌ Memory-heavy (stores all training data)
- ❌ Sensitive to feature scaling — *must* standardize
- ❌ Sensitive to irrelevant features
- ❌ Curse of dimensionality

**Use when:** Small dataset, low-dimensional, similarity-based reasoning makes intuitive sense, recommender baselines, embeddings-based retrieval (with ANN).

---

## 8. Naive Bayes

### 8.1 The Idea

Apply Bayes' theorem with a "naive" assumption: **all features are conditionally independent given the class.**

```
P(class | features) ∝ P(class) × ∏ P(featureᵢ | class)
```

### 8.2 Variants

| Variant | Use For |
|---|---|
| **Gaussian NB** | Continuous features, assumed normally distributed per class |
| **Multinomial NB** | Discrete counts (word counts in text classification) |
| **Bernoulli NB** | Binary features (word presence/absence) |
| **Complement NB** | Imbalanced text classification — robust to skewed distributions |

### 8.3 Pros and Cons

**Pros:**
- ✅ Extremely fast to train and predict
- ✅ Works well with high-dimensional sparse data (text)
- ✅ Strong baseline for text classification
- ✅ Surprisingly competitive despite the unrealistic independence assumption

**Cons:**
- ❌ The independence assumption is almost always wrong
- ❌ Probability estimates are poorly calibrated
- ❌ "Zero frequency problem" — needs Laplace smoothing
- ❌ Outperformed by virtually anything modern on non-text problems

**Use when:** Text classification baseline (especially spam/sentiment), need a fast model for real-time, very high-dimensional sparse features.

---

## 9. Neural Networks — Foundations

### 9.1 The Basic Architecture

A **neuron** computes:
```
y = activation(w · x + b)
```

A **layer** is many neurons in parallel. A **network** is layers stacked, with the output of each layer feeding the next.

**Universal approximation theorem:** A neural network with one hidden layer and enough neurons can approximate any continuous function. (In practice, depth helps more than width.)

### 9.2 Activation Functions

| Activation | Formula | Use |
|---|---|---|
| **ReLU** | max(0, x) | Default for hidden layers |
| **Leaky ReLU** | max(0.01x, x) | When dying ReLU is a problem |
| **GELU** | x · Φ(x) | Transformer hidden layers |
| **Swish/SiLU** | x · sigmoid(x) | Modern networks |
| **Tanh** | tanh(x) | Older RNNs |
| **Sigmoid** | 1/(1+e⁻ˣ) | Binary output layer only |
| **Softmax** | eˣⁱ/Σeˣʲ | Multi-class output layer |

**The dying ReLU problem:** Neurons stuck outputting 0 because their gradients are always 0. Fix: Leaky ReLU, lower learning rate, better initialization.

### 9.3 Training: Backpropagation

The chain rule applied to compute gradients of the loss with respect to every weight, propagating from output back to input. Combined with an optimizer (SGD, Adam) to update weights.

### 9.4 Optimizers

| Optimizer | Notes |
|---|---|
| **SGD** | Simple, well-understood, works with momentum |
| **SGD + Momentum** | Accelerates in consistent directions, dampens oscillations |
| **Adam** | Adaptive learning rates per parameter; default for most tasks |
| **AdamW** | Adam with decoupled weight decay; default for transformers |
| **RMSprop** | Older adaptive method; sometimes wins on RNNs |
| **Lion** | Newer, memory-efficient alternative to AdamW |

**Default for most problems:** AdamW with learning rate 1e-3 to 1e-4, scheduled to decay.

### 9.5 Regularization Techniques

| Technique | Effect |
|---|---|
| **L2 weight decay** | Penalizes large weights, smooths the function |
| **Dropout** | Randomly zeros activations during training |
| **Batch Normalization** | Normalizes layer activations; speeds training, adds slight regularization |
| **Layer Normalization** | Per-sample normalization; standard in transformers |
| **Early Stopping** | Stop training when validation loss stops improving |
| **Data Augmentation** | Modify inputs (rotate images, paraphrase text) to expand training distribution |
| **Mixup / CutMix** | Combine pairs of training examples |
| **Label Smoothing** | Soften one-hot targets to prevent overconfidence |

### 9.6 Initialization

How you initialize weights matters enormously:

| Init | Use For |
|---|---|
| **Xavier/Glorot** | tanh activations |
| **He / Kaiming** | ReLU and variants |
| **Orthogonal** | RNNs |

**Never initialize weights to zero** — all neurons compute the same thing and gradients are identical (symmetry breaking failure).

### 9.7 The Vanishing / Exploding Gradient Problem

In deep networks, gradients multiplied through many layers either shrink to zero (vanishing) or blow up (exploding). Solutions:
- Skip connections (ResNet)
- Proper initialization
- Batch/Layer Normalization
- Gradient clipping (for exploding)
- LSTM/GRU gates (for RNNs)

---

## 10. Deep Learning Architectures

### 10.1 Multi-Layer Perceptron (MLP)

The basic feedforward neural network. Fully connected layers.

**Use for:** Tabular data when gradient boosting isn't enough, simple function approximation. Rarely the first choice anymore — boosting wins on tabular, transformers on sequence/image.

### 10.2 Convolutional Neural Networks (CNNs)

**Inductive bias:** Translation invariance and local connectivity. The same feature detector slides across the image.

**Core components:**
- **Convolution layer** — applies learned filters
- **Pooling** (max/avg) — downsamples
- **Activation** (ReLU)
- **Fully connected** — final classification

**Landmark architectures:**

| Year | Architecture | Innovation |
|---|---|---|
| 1998 | LeNet | First successful CNN |
| 2012 | AlexNet | ReLU, dropout, GPU training; won ImageNet |
| 2014 | VGG | Demonstrated depth matters |
| 2014 | GoogLeNet/Inception | Multi-scale features via inception modules |
| 2015 | ResNet | Skip connections enable 100+ layer networks |
| 2017 | DenseNet | Dense connections between layers |
| 2019 | EfficientNet | Compound scaling of depth, width, resolution |
| 2020 | Vision Transformer (ViT) | Attention replaces convolution |
| 2022 | ConvNeXt | Modernized CNN matching ViT |

**Use when:** Images, audio spectrograms, anything with local spatial/temporal structure.

### 10.3 Recurrent Neural Networks (RNNs)

**Inductive bias:** Order matters. State propagates through time.

**Variants:**
- **Vanilla RNN** — suffers from vanishing gradients
- **LSTM** — gated cell solves vanishing gradients, handles long sequences
- **GRU** — simpler than LSTM, similar performance, fewer parameters
- **Bidirectional RNN** — processes sequence forward and backward

**The brutal truth:** For most sequence problems in 2024-26, **transformers have replaced RNNs**. RNNs remain useful for:
- Very long sequences where transformer memory is prohibitive
- Streaming / online settings
- Resource-constrained inference

### 10.4 Autoencoders

Networks trained to **reconstruct their input** through a bottleneck (compressed representation).

**Variants:**
- **Denoising AE** — input corrupted, target is clean
- **Variational AE (VAE)** — latent space is probabilistic; useful for generation
- **Sparse AE** — encourages few active neurons
- **Contractive AE** — penalizes sensitivity to small input changes

**Uses:**
- Dimensionality reduction (non-linear PCA)
- Anomaly detection (high reconstruction error = anomaly)
- Pre-training representations (largely superseded by self-supervised methods)
- Generative modeling (VAEs, less popular than diffusion now)

### 10.5 Generative Adversarial Networks (GANs)

Two networks in a game:
- **Generator** — produces fake samples
- **Discriminator** — tries to distinguish real from fake

Trained adversarially until the generator produces samples the discriminator can't distinguish from real data.

**Famous variants:** DCGAN, StyleGAN, CycleGAN, BigGAN

**Status:** Largely **superseded by diffusion models** for image generation, though still useful for certain tasks (image-to-image translation, super-resolution).

### 10.6 Diffusion Models

**The dominant approach to generative modeling in 2024-26.**

**The idea:** Train a model to gradually **denoise** an image. Generation = start from pure noise, iteratively denoise.

**Examples:** Stable Diffusion, DALL-E 3, Midjourney, Sora (video).

**Why they won over GANs:** More stable training, better mode coverage, higher-quality samples, easier to condition on text.

### 10.7 U-Net

Architecture with **encoder-decoder + skip connections** between symmetric layers.

**Originally for medical image segmentation. Now ubiquitous in:**
- Image segmentation (semantic and instance)
- Diffusion models (the denoising network is typically a U-Net)
- Any task with input-output spatial alignment

---

## 11. Transformers & Attention

> **The most important architecture of the 2020s.** Originally for translation, now everywhere — vision, speech, biology, code, robotics.

### 11.1 The Attention Mechanism

Given a query, attend to all keys, weighted by similarity, retrieving a weighted sum of values:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

Where Q (queries), K (keys), V (values) are linear projections of the input.

**Why this matters:** Unlike RNNs, attention sees the entire sequence at once. No information bottleneck, no vanishing gradients across sequence length.

### 11.2 Self-Attention

Q, K, V all come from the **same sequence**. Each token attends to every other token, learning relationships regardless of distance.

**Multi-head attention:** Multiple parallel attention operations with different learned projections, capturing different types of relationships.

### 11.3 The Transformer Block

```
x → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm → output
```

Repeated N times. Plus positional encodings (since attention is order-invariant by default).

### 11.4 Architectural Families

| Family | Architecture | Examples | Use |
|---|---|---|---|
| **Encoder-only** | Bidirectional attention | BERT, RoBERTa, DeBERTa | Classification, embeddings |
| **Decoder-only** | Causal (left-to-right) attention | GPT, Llama, Claude | Generation, chat |
| **Encoder-Decoder** | Separate encoder and decoder | T5, BART, original Transformer | Translation, summarization |

### 11.5 Vision Transformer (ViT)

Apply transformer to images by splitting them into **patches** treated as tokens. Surprisingly effective when pretrained on enough data.

**Modern variants:** Swin Transformer (hierarchical), DeiT (data-efficient), DINOv2 (self-supervised).

### 11.6 Why Transformers Won

1. **Parallelizable** — unlike RNNs, all positions process simultaneously
2. **Long-range dependencies** — no degradation with distance
3. **Scales with data and compute** — bigger models trained on more data keep improving
4. **Transfer learning** — pretrain once, fine-tune for many tasks
5. **Multi-modal** — same architecture handles text, images, audio, code

### 11.7 Current Limitations

- **Quadratic attention complexity** in sequence length (though many efficient variants exist: FlashAttention, Performer, Mamba)
- **Massive compute requirements** for training
- **Data hungry** — need huge pretraining corpora
- **Interpretability** is even worse than CNNs

---

## 12. Unsupervised — Clustering

### 12.1 K-Means

**Algorithm:**
1. Initialize k cluster centroids (random or k-means++)
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

**Pros:**
- ✅ Fast, scales to large data
- ✅ Simple and well-understood

**Cons:**
- ❌ Need to specify k in advance
- ❌ Assumes spherical, equal-sized clusters
- ❌ Sensitive to initialization (use k-means++)
- ❌ Sensitive to outliers
- ❌ Can't handle non-convex shapes

**How to choose k:** Elbow method, silhouette score, gap statistic, Bayesian Information Criterion.

### 12.2 DBSCAN

**Density-based clustering.** Defines clusters as dense regions separated by sparse regions.

**Parameters:**
- `eps` — neighborhood radius
- `min_samples` — minimum points to form a dense region

**Pros:**
- ✅ Doesn't require specifying k
- ✅ Finds arbitrarily shaped clusters
- ✅ Robust to outliers (labels them as noise)

**Cons:**
- ❌ Sensitive to `eps` parameter
- ❌ Struggles with varying density clusters
- ❌ Curse of dimensionality

### 12.3 Hierarchical Clustering

Build a tree (dendrogram) of nested clusters.

**Two approaches:**
- **Agglomerative** (bottom-up) — start with each point as its own cluster, merge
- **Divisive** (top-down) — start with one cluster, split

**Linkage methods:** Single, complete, average, Ward.

**Pros:**
- ✅ No need to specify k upfront (cut the dendrogram at chosen height)
- ✅ Produces interpretable hierarchy

**Cons:**
- ❌ O(n²) or O(n³) — doesn't scale
- ❌ Greedy — merges/splits can't be undone

### 12.4 Gaussian Mixture Models (GMM)

**Soft clustering** — each point has a probability of belonging to each cluster. Each cluster is modeled as a Gaussian.

Fit via Expectation-Maximization (EM).

**Pros:**
- ✅ Soft assignments
- ✅ Captures elliptical clusters
- ✅ Probabilistic foundation

**Cons:**
- ❌ Need to specify number of components
- ❌ Sensitive to initialization
- ❌ Assumes Gaussian clusters

### 12.5 HDBSCAN

Hierarchical DBSCAN. **The modern default for general-purpose clustering.**

**Improvements over DBSCAN:**
- No need to specify `eps`
- Handles varying density
- More robust to parameter choices

### 12.6 When to Use Which

| Situation | Use |
|---|---|
| You know k, clusters are roughly spherical | K-Means |
| Unknown k, varying density, outliers present | HDBSCAN |
| Need a hierarchy or dendrogram | Agglomerative |
| Soft assignments needed | GMM |
| Very large data | Mini-Batch K-Means |
| Text/embeddings | Spectral clustering or HDBSCAN on UMAP-reduced embeddings |

---

## 13. Unsupervised — Dimensionality Reduction

### 13.1 PCA (Principal Component Analysis)

Find orthogonal directions of maximum variance. Linear transformation.

**Pros:** Fast, deterministic, well-understood, preserves global structure.
**Cons:** Linear only, components hard to interpret, sensitive to scaling.

**Always standardize features before PCA.**

### 13.2 t-SNE

**Non-linear, preserves local structure.** Maps similar points to nearby positions in 2D/3D.

**Pros:** Beautiful visualizations, reveals cluster structure.
**Cons:** Slow (O(n²)), non-deterministic, distances between clusters are NOT meaningful, perplexity hyperparameter tricky.

**Critical:** t-SNE is for **visualization, not preprocessing for downstream models.**

### 13.3 UMAP

Modern alternative to t-SNE.

**Advantages over t-SNE:**
- Faster
- Preserves more global structure
- Can transform new data (t-SNE is non-parametric)
- Often used as preprocessing for clustering

**The current default for embedding visualization.**

### 13.4 Autoencoders

Non-linear dimensionality reduction via neural networks. Useful when you have lots of data and need a non-linear projection.

### 13.5 When to Use Which

| Goal | Use |
|---|---|
| Linear preprocessing for downstream model | PCA |
| Visualization (2D/3D) | UMAP > t-SNE |
| Clustering high-dim data | UMAP + HDBSCAN |
| Non-linear feature extraction | Autoencoder |
| Sparse data with topics | NMF, LDA |

---

## 14. Anomaly Detection

### 14.1 Statistical Methods

- **Z-score / Modified Z-score** — for univariate normal data
- **IQR-based** — robust to outliers in training
- **Mahalanobis distance** — multivariate, assumes Gaussian

### 14.2 Isolation Forest

**The default tabular anomaly detector.**

**Idea:** Anomalies are easier to isolate via random splits. Build random trees, anomalies have shorter average path length.

**Pros:** Scales well, no distance computation, handles high dimensions.

### 14.3 One-Class SVM

Learn a boundary around "normal" data. Anything outside is anomalous.

**Pros:** Theoretically grounded.
**Cons:** Slow, sensitive to hyperparameters.

### 14.4 Local Outlier Factor (LOF)

Compares local density of a point to its neighbors. Points in low-density regions are outliers.

**Use when:** Density-based reasoning makes sense, anomalies are in low-density regions.

### 14.5 Autoencoder-based

Train an autoencoder on normal data. High reconstruction error = anomaly.

**Use when:** Image, time series, or complex high-dimensional data.

### 14.6 When to Use Which

| Data | Method |
|---|---|
| Tabular | Isolation Forest |
| Time series | Prophet residuals, LSTM autoencoder, STL decomposition |
| Images | Autoencoder, GAN-based |
| Network / graph | Graph-based methods, embedding distance |
| Streaming | Online statistical methods, Half-Space Trees |

---

## 15. Time Series Models

### 15.1 Classical Statistical

**ARIMA (Autoregressive Integrated Moving Average)**
- AR: regress on past values
- I: differencing for stationarity
- MA: regress on past errors
- Needs stationarity, manual order selection (ACF/PACF plots) or auto_arima

**SARIMA** — ARIMA with seasonality

**ETS (Exponential Smoothing)** — Holt-Winters family. State-space form.

**When they win:** Univariate series, clear trend/seasonality, < 1000 observations, need interpretability.

### 15.2 Prophet

Facebook's library. Decomposes series into trend + seasonality + holidays.

**Pros:** Handles missing data, robust to outliers, easy to use, interpretable.
**Cons:** Often beaten by both classical methods and ML on accuracy benchmarks.

### 15.3 Tree-Based for Time Series

**The dominant approach for multivariate forecasting in industry.** Use LightGBM/XGBoost with:
- Lag features (`y[t-1]`, `y[t-7]`, `y[t-365]`)
- Rolling statistics (mean, std, min, max over windows)
- Date features (day of week, month, holiday flags)
- Exogenous variables

**Caveat:** Trees can't extrapolate. Detrend the series first if trend is strong.

### 15.4 Deep Learning for Time Series

| Model | Use |
|---|---|
| **DeepAR** | Probabilistic forecasting of many related series |
| **N-BEATS** | Pure deep learning, interpretable basis decomposition |
| **N-HiTS** | Successor to N-BEATS, hierarchical |
| **Temporal Fusion Transformer (TFT)** | Attention-based, mixes static + dynamic features |
| **PatchTST** | Transformer with patching, strong recent benchmark |
| **TimesNet** | Frequency-domain decomposition + 2D CNN |

**When DL wins:** Many related series (cross-learning), > 10k series, sufficient history.

### 15.5 Special Challenges

- **Intermittent demand** — many zeros (Croston's method, Tweedie regression)
- **Hierarchical forecasting** — totals must match subtotals (MinT reconciliation)
- **Cold start** — new items with no history (use meta-learning, item features)

---

## 16. Recommender Systems

### 16.1 Content-Based Filtering

Recommend items similar to those a user liked, based on **item features**.

**Pros:** Works for new users (if they've interacted), explainable.
**Cons:** Filter bubble (overspecialization), needs good item features.

### 16.2 Collaborative Filtering

Recommend based on **user-item interaction patterns**, ignoring item content.

**Memory-based (kNN):**
- User-based: find similar users, recommend what they liked
- Item-based: find items co-liked with the target item

**Model-based:**
- **Matrix Factorization (SVD, ALS)** — decompose user-item matrix into low-rank factors
- **Factorization Machines** — generalize MF to include side features

### 16.3 Deep Learning Recommenders

- **Neural Collaborative Filtering (NCF)** — replaces dot product with MLP
- **Two-Tower Models** — separate encoders for users and items, ANN at retrieval
- **Sequence Models (SASRec, BERT4Rec)** — model user history as a sequence
- **Graph-based (LightGCN, PinSAGE)** — bipartite user-item graph

### 16.4 Modern Production Stack

Most large-scale recommenders have **two stages:**
1. **Retrieval (candidate generation)** — narrow millions of items to hundreds. Two-tower model + ANN.
2. **Ranking** — score the hundreds with a heavier model. Often a deep cross-network or boosted trees.

### 16.5 Common Pitfalls

- **Cold start** for new users/items
- **Popularity bias** — popular items dominate
- **Filter bubble** — narrowing user exposure
- **Evaluation pitfalls** — offline metrics often don't match online A/B test outcomes
- **Position bias** in training data

---

## 17. Reinforcement Learning

### 17.1 The Core Setup

Agent in an environment, taking actions to maximize cumulative reward.

```
State → Agent (policy π) → Action → Environment → Reward + Next State
```

### 17.2 Categories

| Category | Algorithms | Use |
|---|---|---|
| **Value-based** | Q-learning, DQN | Discrete actions |
| **Policy-based** | REINFORCE, PPO | Continuous or discrete actions |
| **Actor-Critic** | A2C, A3C, PPO, SAC | Combines both, current default |
| **Model-based** | Dyna-Q, MuZero | When environment model is available or learnable |

### 17.3 Key Algorithms

- **Q-Learning / DQN** — learn action-value function. DQN brought RL to neural nets (Atari).
- **PPO (Proximal Policy Optimization)** — workhorse of modern RL. Used in RLHF for LLMs.
- **SAC (Soft Actor-Critic)** — strong for continuous control (robotics).
- **AlphaZero / MuZero** — Monte Carlo Tree Search + neural networks. Mastered Go, Chess, Shogi.

### 17.4 Reality Check

RL is **notoriously sample-inefficient and unstable in real-world deployment.** Most "RL" success stories are:
- Games / simulations (where samples are free)
- Bandits (contextual bandits, not full RL — common in industry)
- RLHF for language models (a stable use case)
- Specific robotics tasks with carefully shaped rewards

For most business problems, **bandits or supervised learning + decision rules** are more practical than full RL.

---

## 18. Probabilistic & Bayesian Models

### 18.1 Why Bayesian?

- **Uncertainty quantification** — full posterior distributions, not just point estimates
- **Incorporate prior knowledge** — formalize what you already know
- **Hierarchical structure** — natural for grouped/multilevel data
- **Small data** — Bayesian methods often win when data is scarce

### 18.2 Key Tools

- **Gaussian Processes** — non-parametric regression with uncertainty. Great for Bayesian optimization (e.g., hyperparameter tuning).
- **Bayesian Neural Networks** — neural nets with weight distributions. Computationally expensive but principled uncertainty.
- **MCMC (Markov Chain Monte Carlo)** — sample from posteriors. PyMC, Stan.
- **Variational Inference** — approximate posteriors as tractable distributions. Faster than MCMC.
- **Conformal Prediction** — distribution-free prediction intervals. Pragmatic alternative for production.

### 18.3 When to Reach for Bayesian

- Sample-efficient hyperparameter optimization (Bayesian optimization)
- Clinical / scientific contexts where uncertainty matters more than point accuracy
- Small datasets with informative priors
- A/B testing with sequential analysis

---

## 19. Ensemble Strategies

### 19.1 Why Ensembles Work

Different models make different errors. Combining them reduces variance (if errors are uncorrelated) or bias (if models complement each other).

### 19.2 Strategies

| Strategy | Description | Example |
|---|---|---|
| **Bagging** | Train models on bootstrap samples, average | Random Forest |
| **Boosting** | Sequential training, each correcting prior errors | XGBoost, AdaBoost |
| **Stacking** | Meta-model learns to combine base model predictions | Multi-level model |
| **Blending** | Like stacking but with holdout instead of CV | Simpler than stacking |
| **Voting** | Average predictions (soft) or majority vote (hard) | Easy baseline |

### 19.3 Stacking Best Practices

1. Use **diverse base models** (linear + tree + neural)
2. Generate out-of-fold predictions for the training set (avoid leakage)
3. Train meta-model on these predictions
4. Meta-model is usually a simple linear/logistic regression

### 19.4 When Ensembling Helps Most

- Models have **uncorrelated errors**
- You have budget for multiple inference passes
- Competition / accuracy-critical settings
- Last-mile improvement after individual models are tuned

**When it doesn't help:** Latency-critical applications, when one model dominates by a wide margin, when models are highly correlated.

---

## 20. The Bias-Variance Framework

### 20.1 The Decomposition

For any prediction model, the expected error decomposes as:

```
Total Error = Bias² + Variance + Irreducible Error
```

- **Bias** — systematic error from wrong assumptions. High = underfitting.
- **Variance** — error from sensitivity to training data fluctuations. High = overfitting.
- **Irreducible** — noise inherent in the data.

### 20.2 Where Each Algorithm Sits

```
                High Bias                    Low Bias
                ↑                            ↑
                Linear Reg                   Deep NN (small data)
                Logistic Reg                 Deep Tree
                Naive Bayes                  kNN (small k)
                                             RBF SVM (high C)
                ↓                            ↓
                Underfitting                 Overfitting
```

### 20.3 How to Diagnose

| Symptom | Diagnosis | Fix |
|---|---|---|
| Train error high, val error high | High bias (underfitting) | Bigger model, more features, less regularization |
| Train error low, val error high | High variance (overfitting) | Regularization, more data, simpler model |
| Train error low, val error low, but doesn't generalize | Train-test distribution shift | Better train/test split, address drift |
| Train error low, val error low, but suspiciously close | Data leakage | Audit features (see Audit.md §16) |

---

## 21. Hyperparameter Reference Card

### XGBoost / LightGBM Quick Defaults

```python
# Start here for any tabular problem
params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,           # LightGBM only
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 5000,        # use early stopping
    'early_stopping_rounds': 100,
}
```

### Random Forest Quick Defaults

```python
params = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'n_jobs': -1,
}
```

### Neural Network Quick Defaults

```python
# For most deep learning tasks
optimizer = AdamW(lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR
batch_size = 32 or 64 or 128
dropout = 0.1 to 0.3
warmup_steps = ~5% of total steps
gradient_clipping = 1.0
```

### Sensible Tuning Budgets

| Algorithm | Tuning Effort Needed |
|---|---|
| Logistic Regression | Low — mostly just `C` |
| Random Forest | Low — defaults are strong |
| XGBoost/LightGBM | Medium — 5-7 important params |
| Neural Network | High — architecture + training dynamics |
| Transformer | Very High — plus pretraining strategy |

---

## 22. Algorithm Failure Modes Cheat Sheet

| Algorithm | Most Common Failure Mode |
|---|---|
| Linear/Logistic Regression | Missing non-linear relationships; multicollinearity |
| Ridge/Lasso | Forgot to scale features before regularization |
| Decision Tree | Overfit to training data; no max_depth |
| Random Forest | Slow inference, biased feature importance on high-cardinality features |
| XGBoost/LightGBM | Overfitting on small data; categorical feature mishandling |
| SVM | Slow on large data, forgot to scale features |
| kNN | Curse of dimensionality, slow inference, forgot to scale |
| Naive Bayes | Poor probability calibration, independence assumption broken |
| Neural Networks | Underfitting (need more data/capacity), exploding/vanishing gradients |
| CNNs | Insufficient data augmentation, small receptive field |
| RNNs | Vanishing gradients, slow training |
| Transformers | Quadratic memory, data hunger, training instability |
| K-Means | Wrong k, non-spherical clusters, outlier sensitivity |
| DBSCAN | Wrong eps, varying density clusters |
| PCA | Used on non-linear data without kernel variant; forgot scaling |
| t-SNE / UMAP | Treating distance between clusters as meaningful |
| Isolation Forest | Wrong contamination parameter; high-dim data |
| ARIMA | Non-stationary input, wrong order, no exogenous variables |
| Collaborative Filtering | Cold start; popularity bias |
| RL | Reward hacking, sample inefficiency, distributional shift |

---

## 23. Implementation Quick Reference

### Tabular Default Stack

```python
# Always start here
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Cross-validation
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Tuning
import optuna  # The modern default for Bayesian hyperparameter optimization
```

### Deep Learning Default Stack

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer  # HuggingFace

# Training utilities
from accelerate import Accelerator    # Multi-GPU made easy
from torchmetrics import Accuracy, F1Score
import wandb  # Experiment tracking
```

### Time Series Default Stack

```python
# Classical
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# ML
import lightgbm as lgb  # With lag features

# Deep
from neuralforecast import NeuralForecast
from darts import TimeSeries  # Unified API for many models
```

### Model Selection Workflow

```python
# 1. ALWAYS start with a baseline
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_score = metric(y_test, baseline_pred)
print(f"Naive baseline: {baseline_score}")

# 2. Linear baseline
linear = Ridge(alpha=1.0).fit(X_train, y_train)
print(f"Linear: {metric(y_test, linear.predict(X_test))}")

# 3. Tree baseline
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1).fit(X_train, y_train)
print(f"Random Forest: {metric(y_test, rf.predict(X_test))}")

# 4. Gradient boosting
gbm = lgb.LGBMRegressor(n_estimators=5000).fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100)]
)
print(f"LightGBM: {metric(y_test, gbm.predict(X_test))}")

# 5. Only go deeper (neural, ensembling) if there's clear room above this
```

---

## Summary — The 12 Algorithms You Actually Need to Know Deeply

If you only master 12 algorithms in your career, make them these:

1. **Linear Regression / Ridge / Lasso** — foundation, baseline, interpretability
2. **Logistic Regression** — classification baseline, calibrated probabilities
3. **Random Forest** — strong robust default
4. **Gradient Boosting (XGBoost/LightGBM)** — the tabular winner
5. **k-Means** — clustering baseline
6. **PCA** — dimensionality reduction baseline
7. **MLP (Multi-Layer Perceptron)** — gateway to deep learning
8. **CNN** — vision
9. **Transformer** — sequence, vision, everything modern
10. **LSTM/GRU** — when you really need recurrence
11. **ARIMA / ETS** — classical forecasting
12. **PPO** — RL workhorse

Everything else is a variant, extension, or specialization of these.

---

*`Algorithms.md` — Version 1.0*
*Scope: Comprehensive reference for ML algorithm selection, characteristics, and failure modes*
*Companion to `Metrics.md` (how to measure) and `Audit.md` (how to verify)*
*Use this as the single source of truth for "which algorithm should I use here?"*
