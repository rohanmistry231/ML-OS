# Machine Learning Metrics — Complete Reference Guide

**Owner:** Rohan Mistry
**Purpose:** Authoritative reference for ML evaluation metrics across all problem types
**Scope:** Classification, Regression, Clustering, Ranking, Computer Vision, NLP, Time Series, Anomaly Detection, RL, Fairness, and Production Monitoring
**Last updated:** May 2026

---

## 📋 Table of Contents

1. [How to Choose a Metric — Decision Tree](#1-how-to-choose-a-metric--decision-tree)
2. [Core Terminology](#2-core-terminology)
3. [Classification Metrics](#3-classification-metrics)
   - 3.1 Confusion Matrix
   - 3.2 Accuracy
   - 3.3 Precision, Recall, F1
   - 3.4 ROC-AUC
   - 3.5 PR-AUC
   - 3.6 Log Loss / Cross-Entropy
   - 3.7 Brier Score
   - 3.8 MCC — Matthews Correlation Coefficient
   - 3.9 Cohen's Kappa
   - 3.10 Balanced Accuracy
   - 3.11 Specificity, Sensitivity, NPV, PPV
   - 3.12 Multi-Class Metrics
   - 3.13 Multi-Label Metrics
4. [Regression Metrics](#4-regression-metrics)
5. [Clustering Metrics](#5-clustering-metrics)
6. [Ranking & Recommendation Metrics](#6-ranking--recommendation-metrics)
7. [Computer Vision Metrics](#7-computer-vision-metrics)
8. [NLP Metrics](#8-nlp-metrics)
9. [Time Series Forecasting Metrics](#9-time-series-forecasting-metrics)
10. [Anomaly Detection Metrics](#10-anomaly-detection-metrics)
11. [Probabilistic & Calibration Metrics](#11-probabilistic--calibration-metrics)
12. [Generative Model Metrics](#12-generative-model-metrics)
13. [Reinforcement Learning Metrics](#13-reinforcement-learning-metrics)
14. [Survival Analysis Metrics](#14-survival-analysis-metrics)
15. [Fairness & Bias Metrics](#15-fairness--bias-metrics)
16. [Production Monitoring Metrics](#16-production-monitoring-metrics)
17. [Business / ROI Metrics](#17-business--roi-metrics)
18. [Common Pitfalls](#18-common-pitfalls)
19. [Quick Reference Cheat Sheet](#19-quick-reference-cheat-sheet)

---

## 1. How to Choose a Metric — Decision Tree

```
START
  │
  ├── What is the task type?
  │
  ├── CLASSIFICATION
  │     ├── Binary balanced → Accuracy, F1, ROC-AUC
  │     ├── Binary imbalanced → PR-AUC, F1, MCC, Recall@Precision
  │     ├── Multi-class → Macro/Weighted F1, Top-K Accuracy
  │     ├── Multi-label → Hamming Loss, Subset Accuracy, Jaccard
  │     └── Probabilistic output → Log Loss, Brier, Calibration
  │
  ├── REGRESSION
  │     ├── Normal scale → RMSE, MAE, R²
  │     ├── Different scales / log-distributed → MSLE, MAPE
  │     ├── Outlier-heavy → MAE, MedAE, Huber Loss
  │     └── Asymmetric cost → Quantile Loss
  │
  ├── RANKING / RECOMMENDATION
  │     ├── Top-K relevance → Precision@K, Recall@K
  │     ├── Ordered relevance → NDCG, MAP
  │     └── First-hit → MRR
  │
  ├── CLUSTERING
  │     ├── Labels known → ARI, NMI, V-measure
  │     └── Labels unknown → Silhouette, Davies-Bouldin, Calinski-Harabasz
  │
  ├── OBJECT DETECTION → mAP, IoU
  ├── SEGMENTATION → Dice, IoU, Pixel Accuracy
  ├── NLP GENERATION → BLEU, ROUGE, BERTScore
  ├── ASR → WER, CER
  ├── FORECASTING → WMAPE, MASE, Bias
  ├── ANOMALY → AUROC, AUPRC, Precision@K
  ├── RL → Cumulative Reward, Regret
  └── FAIRNESS → Demographic Parity, Equal Opportunity, Disparate Impact
```

---

## 2. Core Terminology

| Term | Meaning |
|---|---|
| **Ground Truth (y_true)** | Actual labels / values |
| **Prediction (y_pred)** | Model output (class label or value) |
| **Probability (y_proba)** | Model's confidence score in [0, 1] |
| **Threshold** | Cutoff converting probability → class |
| **TP** | True Positive — predicted positive, actually positive |
| **TN** | True Negative — predicted negative, actually negative |
| **FP** | False Positive — predicted positive, actually negative (Type I error) |
| **FN** | False Negative — predicted negative, actually positive (Type II error) |
| **Holdout / Test set** | Data unseen during training, used for evaluation |
| **K-fold CV** | Split data into K folds, train K times rotating the holdout fold |
| **Stratified split** | Preserve class proportions across train/test |
| **Class imbalance** | One class much rarer than others (e.g., 99% / 1%) |
| **Macro avg** | Unweighted mean across classes |
| **Micro avg** | Pool predictions across classes, then compute |
| **Weighted avg** | Class-weighted by support |
| **Support** | Number of true instances of each class |

---

## 3. Classification Metrics

### 3.1 Confusion Matrix

The foundation of all classification metrics:

```
                  Predicted
                  Positive    Negative
Actual Positive   TP          FN
Actual Negative   FP          TN
```

All other metrics derive from these 4 cells.

**sklearn:** `sklearn.metrics.confusion_matrix(y_true, y_pred)`

---

### 3.2 Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Plain English:** Fraction of predictions that are correct.

**Range:** 0 to 1 (higher is better)

**Pros:**
- ✅ Intuitive
- ✅ Works well for balanced classes

**Cons:**
- ❌ **Misleading on imbalanced data.** 99% accuracy on a 99/1 split = predicting majority class = useless model
- ❌ Hides class-specific errors

**Use when:** Balanced classes, error costs equal.

**Avoid when:** Class imbalance, asymmetric error costs (e.g., fraud, disease detection).

**sklearn:** `sklearn.metrics.accuracy_score`

---

### 3.3 Precision, Recall, F1

These three are the core of binary classification.

#### Precision
```
Precision = TP / (TP + FP)
```
**Plain English:** Of the items predicted positive, how many are actually positive?
**Use when:** False positives are costly (e.g., spam filter — don't mark legitimate email as spam).

#### Recall (Sensitivity / True Positive Rate)
```
Recall = TP / (TP + FN)
```
**Plain English:** Of the actual positives, how many did we catch?
**Use when:** False negatives are costly (e.g., cancer screening — don't miss any positives).

#### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Plain English:** Harmonic mean of precision and recall. Punishes extreme imbalance between them.
**Range:** 0 to 1 (higher is better)

#### F-Beta (generalized F-score)
```
F_β = (1 + β²) × (P × R) / (β² × P + R)
```
- β = 1 → F1 (balanced)
- β = 2 → F2 (recall weighted 2x more than precision)
- β = 0.5 → F0.5 (precision weighted 2x more than recall)

**Precision-Recall Trade-off:**
- Lower the threshold → higher recall, lower precision
- Raise the threshold → higher precision, lower recall
- Pick based on business cost of FP vs FN

**sklearn:** `precision_score`, `recall_score`, `f1_score`, `fbeta_score`

---

### 3.4 ROC-AUC — Area Under Receiver Operating Characteristic Curve

**ROC curve:** Plot of True Positive Rate (Recall) vs False Positive Rate at every threshold.

```
TPR = TP / (TP + FN)   (= Recall)
FPR = FP / (FP + TN)   (= 1 - Specificity)
```

**AUC interpretation:**
- 1.0 → perfect classifier
- 0.5 → random classifier
- < 0.5 → worse than random (invert predictions)

**Plain English:** Probability that the model ranks a random positive higher than a random negative.

**Pros:**
- ✅ Threshold-independent (good for ranking quality)
- ✅ Insensitive to class imbalance in moderate cases
- ✅ Standard metric for binary classification

**Cons:**
- ❌ Misleadingly optimistic on heavily imbalanced data (PR-AUC better)
- ❌ Equal weight to all thresholds — even useless extremes
- ❌ Doesn't reflect operating point

**Use when:** Balanced classes, comparing models' ranking quality.

**sklearn:** `roc_auc_score`

---

### 3.5 PR-AUC — Precision-Recall Area Under Curve

Plot of Precision vs Recall at every threshold.

**Baseline:** Equal to class prevalence (e.g., 1% positives → baseline = 0.01)

**Pros:**
- ✅ **Better than ROC-AUC for imbalanced datasets**
- ✅ Focuses on positive class performance
- ✅ Sensitive to model improvements on the rare class

**Cons:**
- ❌ Less intuitive than ROC
- ❌ No fixed baseline (changes with prevalence)

**Use when:** Heavy class imbalance (fraud, anomaly, rare disease).

**sklearn:** `average_precision_score`

---

### 3.6 Log Loss / Cross-Entropy

**Formula (binary):**
```
LogLoss = -1/n × Σ [y log(p) + (1-y) log(1-p)]
```

**Multi-class:**
```
LogLoss = -1/n × Σ Σ y_ij × log(p_ij)
```

**Plain English:** Penalizes confident wrong predictions heavily.

**Range:** 0 to ∞ (lower is better)

**Pros:**
- ✅ Differentiable — used for training neural networks
- ✅ Probabilistic — rewards well-calibrated confidence
- ✅ Sensitive to extreme errors

**Cons:**
- ❌ Hard to interpret on its own (no upper bound)
- ❌ Punishes confident wrong predictions disproportionately
- ❌ Requires probability outputs, not just class predictions

**Use when:** Training models; comparing well-calibrated probabilistic classifiers.

**sklearn:** `log_loss`

---

### 3.7 Brier Score

**Formula:**
```
Brier = 1/n × Σ (p - y)²
```

**Plain English:** Mean squared error between predicted probability and actual outcome (0/1).

**Range:** 0 to 1 (lower is better; perfect = 0)

**Pros:**
- ✅ Measures both calibration and discrimination
- ✅ Bounded and interpretable
- ✅ Less harsh than log loss

**Cons:**
- ❌ Less sensitive than log loss to extreme errors
- ❌ Not threshold-independent like AUC

**Use when:** Probabilistic forecasting, model calibration evaluation.

**sklearn:** `brier_score_loss`

---

### 3.8 MCC — Matthews Correlation Coefficient

**Formula:**
```
MCC = (TP × TN - FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Range:** -1 to +1
- +1 → perfect prediction
- 0 → random prediction
- -1 → inverse prediction

**Plain English:** Correlation between predictions and ground truth. Considers all 4 confusion matrix cells.

**Pros:**
- ✅ **Best single-number metric for imbalanced binary classification**
- ✅ Robust to class imbalance
- ✅ Symmetric — swapping positive/negative classes doesn't change result

**Cons:**
- ❌ Less intuitive than F1
- ❌ Undefined when any denominator term is 0

**Use when:** Imbalanced binary classification, need single robust score.

**sklearn:** `matthews_corrcoef`

---

### 3.9 Cohen's Kappa

**Formula:**
```
κ = (p_o - p_e) / (1 - p_e)

where p_o = observed agreement (accuracy)
      p_e = chance agreement
```

**Range:** -1 to 1
- > 0.8 → almost perfect
- 0.6 - 0.8 → substantial agreement
- 0.4 - 0.6 → moderate
- 0.2 - 0.4 → fair
- 0 - 0.2 → slight
- ≤ 0 → poor / chance

**Plain English:** Accuracy adjusted for the probability of agreement by chance.

**Use when:** Comparing annotator agreement, multi-class problems with imbalance.

**sklearn:** `cohen_kappa_score`

---

### 3.10 Balanced Accuracy

**Formula:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (TPR + TNR) / 2
```

**Plain English:** Mean of recall on each class.

**Pros:**
- ✅ Handles class imbalance better than accuracy
- ✅ Random baseline = 0.5

**Cons:**
- ❌ Treats classes equally (sometimes you want weighting)

**Use when:** Imbalanced binary or multi-class problems.

**sklearn:** `balanced_accuracy_score`

---

### 3.11 Specificity, Sensitivity, NPV, PPV

| Term | Formula | Other names |
|---|---|---|
| **Sensitivity (TPR)** | TP / (TP + FN) | Recall, True Positive Rate, Hit Rate |
| **Specificity (TNR)** | TN / (TN + FP) | True Negative Rate, Selectivity |
| **PPV** | TP / (TP + FP) | Precision |
| **NPV** | TN / (TN + FN) | Negative Predictive Value |
| **FPR** | FP / (FP + TN) | False Positive Rate, Fall-out, Type I error rate |
| **FNR** | FN / (FN + TP) | False Negative Rate, Miss rate, Type II error rate |
| **FDR** | FP / (FP + TP) | False Discovery Rate |

**Youden's J statistic:**
```
J = Sensitivity + Specificity - 1
```
**Use:** Finds optimal classification threshold (maximize J).

**Use when:** Medical diagnosis, screening — when distinguishing FP and FN matters separately.

---

### 3.12 Multi-Class Metrics

For K > 2 classes, single-class metrics need aggregation:

#### Macro-average
```
Macro F1 = mean(F1_per_class)
```
Treats all classes equally regardless of frequency.
**Use when:** Care about minority classes.

#### Micro-average
```
Micro F1 = global F1 computed from total TP, FP, FN
```
For multi-class, micro F1 = accuracy.
**Use when:** Care about overall correctness.

#### Weighted-average
```
Weighted F1 = Σ (F1_class × support_class) / total_support
```
Class-weighted by support.
**Use when:** Want overall view but acknowledge class sizes.

#### Top-K Accuracy
**Formula:**
```
Top-K Acc = % cases where true label is in model's top-K predicted classes
```
**Use when:** Many classes (e.g., ImageNet — Top-1 and Top-5 are standard).

#### Cohen's Kappa (multi-class)
Generalizes to K classes — accounts for chance agreement.

#### Multi-class Log Loss
Sum cross-entropy across all classes per example.

**sklearn:** `classification_report` gives all of these.

---

### 3.13 Multi-Label Metrics

When each instance can have multiple labels (not mutually exclusive).

#### Hamming Loss
```
Hamming Loss = fraction of incorrect labels / total labels
```
**Use when:** Per-label error matters.

#### Subset Accuracy (Exact Match)
```
Subset Acc = fraction of instances where ALL labels are correct
```
**Strictest metric** — even one wrong label = whole instance counted wrong.

#### Jaccard Index (Intersection over Union)
```
Jaccard = |y_true ∩ y_pred| / |y_true ∪ y_pred|
```
**Use when:** Set similarity matters (e.g., tag prediction).

#### Label Ranking Average Precision (LRAP)
Average precision over ranked labels.

#### Coverage Error
Average number of labels included to cover all true labels in rankings.

**sklearn:** `hamming_loss`, `jaccard_score`, `label_ranking_average_precision_score`

---

## 4. Regression Metrics

### 4.1 MAE — Mean Absolute Error
```
MAE = 1/n × Σ |y - ŷ|
```
**Pros:** Same units as data, robust to outliers, intuitive.
**Cons:** Not differentiable at zero (problem for gradient descent).
**sklearn:** `mean_absolute_error`

### 4.2 MSE — Mean Squared Error
```
MSE = 1/n × Σ (y - ŷ)²
```
**Pros:** Differentiable, penalizes large errors.
**Cons:** Not interpretable (squared units), sensitive to outliers.
**sklearn:** `mean_squared_error`

### 4.3 RMSE — Root Mean Squared Error
```
RMSE = √MSE
```
**Pros:** Same units as data, popular default.
**Cons:** Still outlier-sensitive.
**sklearn:** `mean_squared_error(squared=False)`

### 4.4 MAPE — Mean Absolute Percentage Error
```
MAPE = 100/n × Σ |y - ŷ| / |y|
```
**Pros:** Scale-free, interpretable.
**Cons:** Undefined when y=0; asymmetric (over vs under).
**sklearn:** `mean_absolute_percentage_error`

### 4.5 WMAPE — Weighted MAPE
```
WMAPE = Σ |y - ŷ| / Σ |y| × 100%
```
**Pros:** Handles intermittent data better than MAPE.
**Use when:** Volume-weighted accuracy needed.

### 4.6 sMAPE — Symmetric MAPE
```
sMAPE = 100/n × Σ |y - ŷ| / ((|y| + |ŷ|) / 2)
```
**Range:** 0 to 200%
**Use when:** Comparing across scales.

### 4.7 R² — Coefficient of Determination
```
R² = 1 - (Σ (y - ŷ)² / Σ (y - ȳ)²)
```
**Plain English:** Fraction of variance explained by the model.
**Range:** -∞ to 1 (1 = perfect, 0 = no better than mean, < 0 = worse than mean)
**Pros:** Scale-free, interpretable.
**Cons:** Misleading on non-linear data; always increases with more features.
**sklearn:** `r2_score`

### 4.8 Adjusted R²
```
Adj R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)

where p = number of features
```
**Use when:** Comparing models with different feature counts.

### 4.9 MedAE — Median Absolute Error
```
MedAE = median(|y - ŷ|)
```
**Pros:** Very robust to outliers.
**Use when:** Heavy-tailed error distribution.
**sklearn:** `median_absolute_error`

### 4.10 MSLE — Mean Squared Logarithmic Error
```
MSLE = 1/n × Σ (log(1 + y) - log(1 + ŷ))²
```
**Pros:** Penalizes under-prediction more than over-prediction; good for exponentially growing targets.
**Use when:** Target is heavily skewed / log-normal (e.g., revenue, population).
**sklearn:** `mean_squared_log_error`

### 4.11 Huber Loss
```
L = ½(y - ŷ)²              if |y - ŷ| ≤ δ
L = δ|y - ŷ| - ½δ²         if |y - ŷ| > δ
```
**Plain English:** MSE for small errors, MAE for large errors.
**Pros:** Combines benefits of MAE and MSE; outlier-robust but differentiable.
**Use when:** Outliers exist but you still want smooth gradients.

### 4.12 Quantile Loss / Pinball Loss
```
L_q = max(q(y - ŷ), (q-1)(y - ŷ))
```
where q ∈ (0, 1) is the target quantile.
**Use when:** Asymmetric cost — e.g., over-stocking vs under-stocking inventory.
**Common quantiles:** 0.1, 0.5 (median), 0.9.

### 4.13 Max Error
```
MaxError = max(|y - ŷ|)
```
**Use when:** Worst-case prediction matters (safety-critical systems).
**sklearn:** `max_error`

### 4.14 Explained Variance Score
```
EV = 1 - Var(y - ŷ) / Var(y)
```
Similar to R² but doesn't penalize bias.
**sklearn:** `explained_variance_score`

### 4.15 MASE — Mean Absolute Scaled Error
```
MASE = MAE / MAE_naive
```
**Use when:** Comparing forecast models against naive baseline.

---

## 5. Clustering Metrics

### When True Labels Known (External Metrics)

#### 5.1 ARI — Adjusted Rand Index
Measures similarity between two clusterings, corrected for chance.
**Range:** -1 to 1 (1 = perfect, 0 = random, < 0 = worse than random)
**sklearn:** `adjusted_rand_score`

#### 5.2 NMI — Normalized Mutual Information
```
NMI = 2 × I(X, Y) / (H(X) + H(Y))
```
**Range:** 0 to 1
**Use when:** Symmetric similarity between clusterings.
**sklearn:** `normalized_mutual_info_score`

#### 5.3 AMI — Adjusted Mutual Information
NMI adjusted for chance.
**sklearn:** `adjusted_mutual_info_score`

#### 5.4 Homogeneity, Completeness, V-measure
- **Homogeneity:** Each cluster contains only members of a single class
- **Completeness:** All members of a class are in the same cluster
- **V-measure:** Harmonic mean of homogeneity and completeness
**sklearn:** `homogeneity_score`, `completeness_score`, `v_measure_score`

#### 5.5 Fowlkes-Mallows Index
```
FMI = √(precision × recall)  (in pairwise clustering sense)
```
**sklearn:** `fowlkes_mallows_score`

---

### When True Labels Unknown (Internal Metrics)

#### 5.6 Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = mean distance to other points in same cluster
b(i) = mean distance to nearest other cluster
```
**Range:** -1 to 1 (higher = better)
**sklearn:** `silhouette_score`

#### 5.7 Calinski-Harabasz Index (Variance Ratio Criterion)
```
CH = (Between-cluster variance / Within-cluster variance) × (n - k) / (k - 1)
```
**Higher = better-defined clusters.**
**sklearn:** `calinski_harabasz_score`

#### 5.8 Davies-Bouldin Index
```
DB = 1/k × Σ max((s_i + s_j) / d_ij)
```
**Lower = better.** Zero = perfect clustering.
**sklearn:** `davies_bouldin_score`

#### 5.9 Inertia / WCSS
```
Inertia = Σ (distance to centroid)²
```
**Used in:** K-means objective function; elbow method for choosing K.

#### 5.10 Gap Statistic
Compares within-cluster dispersion to that expected under reference null distribution.

---

## 6. Ranking & Recommendation Metrics

### 6.1 Precision@K
```
P@K = (# relevant items in top K) / K
```

### 6.2 Recall@K
```
R@K = (# relevant items in top K) / (# all relevant items)
```

### 6.3 Hit Rate @ K
```
HR@K = 1 if any relevant item in top K else 0  (averaged over users)
```

### 6.4 MRR — Mean Reciprocal Rank
```
MRR = 1/|Q| × Σ (1 / rank_first_relevant)
```
**Use when:** Position of first correct answer matters (Q&A, search).

### 6.5 MAP — Mean Average Precision
```
AP = 1/R × Σ_k P@k × rel(k)
MAP = mean(AP) across queries
```
**Use when:** Multiple relevant items per query; care about order.

### 6.6 NDCG — Normalized Discounted Cumulative Gain
```
DCG@k = Σ_i rel_i / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k    (IDCG = perfect ranking DCG)
```
**Range:** 0 to 1
**Pros:** Handles graded (not just binary) relevance.
**Use when:** Search ranking, recommendation ordering.

### 6.7 Coverage
```
Coverage = (# unique items recommended) / (# total items)
```
**Use when:** Avoid recommending only popular items (long-tail).

### 6.8 Diversity
Average pairwise dissimilarity among recommended items.

### 6.9 Novelty
How many recommendations are not "popular" items.

### 6.10 Serendipity
Recommendations that are relevant AND unexpected.

### 6.11 ILS — Intra-List Similarity
Lower = more diverse list.

---

## 7. Computer Vision Metrics

### 7.1 IoU — Intersection over Union (Jaccard for boxes)
```
IoU = Area(Predicted ∩ True) / Area(Predicted ∪ True)
```
**Range:** 0 to 1
**Threshold:** Typically 0.5 for "match"

### 7.2 mAP — Mean Average Precision (Detection)
- For each class, compute AP from precision-recall curve
- Average across classes
- Common variants:
  - **mAP@0.5** — IoU threshold 0.5 (PASCAL VOC standard)
  - **mAP@0.75** — stricter
  - **mAP@[0.5:0.95]** — averaged over IoU thresholds 0.5 to 0.95 (COCO standard)

### 7.3 Average Recall (AR)
Recall averaged over IoU thresholds.

### 7.4 Pixel Accuracy (Segmentation)
```
Pixel Acc = correctly classified pixels / total pixels
```
**Cons:** Misleading on imbalanced classes (large background).

### 7.5 Mean IoU (Segmentation)
IoU per class, averaged.

### 7.6 Dice Coefficient / F1 (Segmentation)
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```
**Range:** 0 to 1
**Note:** Dice and F1 are mathematically equivalent.

### 7.7 Panoptic Quality (PQ)
Combines segmentation quality (SQ) + recognition quality (RQ).

### 7.8 SSIM — Structural Similarity Index
For image quality / reconstruction.
**Range:** -1 to 1 (1 = identical)

### 7.9 PSNR — Peak Signal-to-Noise Ratio
```
PSNR = 20 × log₁₀(MAX / RMSE)
```
For image reconstruction quality.

### 7.10 LPIPS — Learned Perceptual Image Patch Similarity
Deep-feature-based perceptual distance.

---

## 8. NLP Metrics

### 8.1 BLEU — Bilingual Evaluation Understudy
N-gram precision against reference translations with brevity penalty.
**Range:** 0 to 1 (often reported as 0-100)
**Use when:** Machine translation, generation evaluation.
**Cons:** Doesn't capture semantic similarity.

### 8.2 ROUGE — Recall-Oriented Understudy for Gisting Evaluation
- **ROUGE-N:** N-gram recall
- **ROUGE-L:** Longest common subsequence
- **ROUGE-W:** Weighted LCS
- **ROUGE-S:** Skip-bigram
**Use when:** Summarization evaluation.

### 8.3 METEOR
Combines unigram precision, recall, stem matching, synonyms, and word order.
**Better correlation with human judgment than BLEU.**

### 8.4 BERTScore
Uses BERT embeddings to compute similarity.
Considers semantic meaning, not just word overlap.

### 8.5 BLEURT
Pre-trained learned metric.

### 8.6 Perplexity
```
Perplexity = exp(cross_entropy)
```
Lower = better.
**Use when:** Language model evaluation.

### 8.7 WER — Word Error Rate (ASR)
```
WER = (S + D + I) / N

S = substitutions, D = deletions, I = insertions, N = total words in reference
```
Lower = better.

### 8.8 CER — Character Error Rate
WER but at character level.

### 8.9 Exact Match (EM) — for QA
Binary: 1 if prediction exactly matches reference, else 0.

### 8.10 F1 for QA
Token-level F1 between predicted and ground truth answer.

### 8.11 chrF / chrF++
Character n-gram F-score (handles morphologically rich languages).

### 8.12 COMET
Neural learned metric for translation quality.

### 8.13 Sacrebleu
Standardized BLEU implementation (reproducible).

---

## 9. Time Series Forecasting Metrics

See `metrics.md` for in-depth coverage. Key ones:

| Metric | Formula |
|---|---|
| **MAE** | Σ\|y - ŷ\| / n |
| **RMSE** | √(Σ(y - ŷ)² / n) |
| **MAPE** | Σ\|y - ŷ\|/y × 100% / n |
| **sMAPE** | Σ\|y - ŷ\| / ((\|y\| + \|ŷ\|)/2) × 100% / n |
| **WMAPE** | Σ\|y - ŷ\| / Σ\|y\| × 100% |
| **MASE** | MAE / MAE_naive |
| **Bias** | Σ(y - ŷ) / Σy × 100% |
| **Tracking Signal** | Σe / MAE |
| **CRPS** | Probabilistic forecast quality |
| **Pinball Loss** | Quantile forecast |

---

## 10. Anomaly Detection Metrics

### 10.1 Precision @ K
Of top K flagged anomalies, how many are true.

### 10.2 Recall @ K
Of all true anomalies, how many in top K flagged.

### 10.3 AUROC
Standard ROC-AUC.

### 10.4 AUPRC
Better for highly imbalanced anomaly data.

### 10.5 Detection Delay
Time between anomaly occurrence and detection.

### 10.6 False Alarm Rate
FP / (FP + TN). Critical for production alerting.

### 10.7 Numenta Anomaly Benchmark (NAB) Score
Domain-specific scoring with reward for early detection, penalty for false positives.

---

## 11. Probabilistic & Calibration Metrics

### 11.1 ECE — Expected Calibration Error
```
ECE = Σ |B_m| / n × |acc(B_m) - conf(B_m)|
```
Bin predictions by confidence; measure mismatch with accuracy.
**Use:** Detect overconfident or underconfident models.

### 11.2 MCE — Maximum Calibration Error
Max gap across bins.

### 11.3 Reliability Diagram
Plot of predicted confidence vs observed frequency.
Diagonal = perfect calibration.

### 11.4 Brier Score (decomposed)
```
Brier = Reliability - Resolution + Uncertainty
```

### 11.5 Spiegelhalter's Z-test
Statistical test for calibration.

### 11.6 Coverage Probability
% of true values inside predicted intervals.

### 11.7 Sharpness
Width of prediction intervals — narrower = sharper (but only useful if calibrated).

---

## 12. Generative Model Metrics

### 12.1 FID — Fréchet Inception Distance
Distance between feature distributions of generated and real images.
**Lower = better.**
**Use when:** Image generation (GANs, diffusion).

### 12.2 IS — Inception Score
Measures both image quality and diversity.
**Cons:** Replaced by FID in modern usage.

### 12.3 KID — Kernel Inception Distance
Unbiased variant of FID, better for small samples.

### 12.4 Precision and Recall for Generative Models
- **Precision:** Quality of generated samples
- **Recall:** Diversity / coverage of real distribution

### 12.5 Perceptual Path Length (PPL)
For GANs (especially StyleGAN) — smoothness of latent space.

### 12.6 LPIPS
Perceptual similarity using deep features.

### 12.7 CLIPScore
For text-to-image: cosine similarity between CLIP embeddings of prompt and generated image.

### 12.8 Reconstruction Loss
For autoencoders: MSE between input and reconstruction.

### 12.9 KL Divergence
```
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
```
Used in VAE loss.

### 12.10 ELBO — Evidence Lower Bound
Used for VAE training.

---

## 13. Reinforcement Learning Metrics

### 13.1 Cumulative / Total Reward
Sum of rewards over an episode.

### 13.2 Discounted Return
```
G_t = Σ γ^k × r_{t+k+1}
```
γ = discount factor (typically 0.9 - 0.99).

### 13.3 Average Reward per Episode
Mean over multiple episodes.

### 13.4 Sample Efficiency
Episodes / interactions needed to reach a performance threshold.

### 13.5 Regret
```
Regret(T) = Σ (V*(s_t) - V_π(s_t))
```
Difference between optimal value and policy's value over T steps.

### 13.6 Asymptotic Performance
Final reward after convergence.

### 13.7 Win Rate (Games)
% games won (often against fixed opponent).

### 13.8 Episode Length
For tasks with implicit goal (survival, navigation).

### 13.9 Success Rate
Binary success / failure across episodes.

---

## 14. Survival Analysis Metrics

### 14.1 C-index — Concordance Index
Probability that for a random pair of subjects, the one with shorter actual time has higher predicted risk.
**Range:** 0.5 (random) to 1 (perfect)
**Equivalent to AUROC for survival.**

### 14.2 Integrated Brier Score (IBS)
Brier score integrated over time.

### 14.3 Time-dependent AUC
AUC evaluated at specific time points.

### 14.4 Log-rank Test Statistic
For comparing survival curves between groups.

---

## 15. Fairness & Bias Metrics

### 15.1 Demographic Parity (Statistical Parity)
```
P(ŷ = 1 | A = 0) = P(ŷ = 1 | A = 1)
```
Positive prediction rate equal across groups.
**Difference:** |P(ŷ=1|A=0) - P(ŷ=1|A=1)|

### 15.2 Equal Opportunity
```
P(ŷ = 1 | y = 1, A = 0) = P(ŷ = 1 | y = 1, A = 1)
```
True positive rate equal across groups.

### 15.3 Equalized Odds
TPR AND FPR equal across groups.

### 15.4 Disparate Impact
```
DI = P(ŷ = 1 | A = 0) / P(ŷ = 1 | A = 1)
```
**Rule of thumb:** Should be between 0.8 and 1.25 (80% rule).

### 15.5 Calibration by Group
```
P(y = 1 | ŷ_score = s, A = a)
```
Should be equal across groups for the same score.

### 15.6 Treatment Equality
FN / FP ratio equal across groups.

### 15.7 Counterfactual Fairness
Prediction unchanged if protected attribute counterfactually changed.

### 15.8 Theil Index, Generalized Entropy
Inequality measures applied to predictions.

**sklearn alternative:** `fairlearn` library.

---

## 16. Production Monitoring Metrics

### 16.1 Data Drift

#### PSI — Population Stability Index
```
PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```
**Interpretation:**
- < 0.1 → no significant shift
- 0.1 - 0.2 → moderate shift
- > 0.2 → significant shift

#### Kolmogorov-Smirnov (KS) Test
Compares two distributions; non-parametric.

#### Jensen-Shannon Divergence
Symmetric KL divergence.

#### Wasserstein Distance (Earth Mover's)
Distance between distributions accounting for spatial position.

### 16.2 Concept Drift
- **Sudden:** Abrupt change (e.g., COVID)
- **Gradual:** Slow shift over time
- **Recurring:** Seasonal patterns
- **Detect via:** Performance metric monitoring + statistical tests

### 16.3 Model Decay
Tracking metric degradation over time vs initial deployment baseline.

### 16.4 Prediction Drift
Distribution of predictions changes even if input distribution stable.

### 16.5 Feature Importance Drift
Top features changing over time.

### 16.6 Latency / Throughput
- p50 / p95 / p99 prediction time
- Predictions per second

### 16.7 Service Level Indicators (SLIs)
- Availability (uptime %)
- Error rate
- Prediction success rate

---

## 17. Business / ROI Metrics

### 17.1 Cost-Sensitive Accuracy
```
Cost = TP × benefit_TP + TN × benefit_TN
     + FP × cost_FP + FN × cost_FN
```
Optimize cost, not accuracy.

### 17.2 Profit Curve
Profit at each classification threshold.

### 17.3 Lift Chart
```
Lift = (Response Rate in Top K%) / (Overall Response Rate)
```
**Use when:** Marketing, churn prediction — who to target first.

### 17.4 Cumulative Gains Chart
Y-axis: cumulative % of positives captured
X-axis: % of population targeted

### 17.5 Expected Value Framework
```
EV = Σ P(outcome) × V(outcome)
```

### 17.6 Customer Lifetime Value (CLV) Accuracy
For CLV prediction models.

### 17.7 Marketing-Specific
- ROAS (Return on Ad Spend)
- CAC (Customer Acquisition Cost)
- Conversion rate uplift

### 17.8 Forecasting Business Impact
- OOS cost ($)
- Overstock cost ($)
- Service level achieved

---

## 18. Common Pitfalls

### 18.1 Train-Test Leakage
- ❌ Feature engineered using future data
- ❌ Test set used for hyperparameter tuning
- ✅ Always have separate train / validation / test sets

### 18.2 Class Imbalance Trap
- ❌ Reporting accuracy on 99/1 split
- ✅ Use PR-AUC, F1, MCC, Balanced Accuracy

### 18.3 Single Metric Over-Reliance
- ❌ Optimizing only AUC → poorly calibrated probabilities
- ❌ Optimizing only F1 → ignores threshold trade-offs
- ✅ Report multiple complementary metrics

### 18.4 Aggregating Without Cohort Slicing
- ❌ Portfolio MAPE 20% (looks good)
- ✅ Slice by tier, time, geography, segment

### 18.5 Confusing Random Baseline
- ❌ Saying "AUC 0.75 is good" without context
- ✅ Compare to chance: AUC 0.5, F1 = prevalence, etc.

### 18.6 Optimizing for Wrong Metric
- ❌ Optimizing MSE when business cares about MAPE
- ✅ Match training loss to business objective

### 18.7 Forgetting Calibration
- ❌ Confidence scores not meaningful (model says 0.9 → actually right 60%)
- ✅ Check reliability diagrams, ECE

### 18.8 Statistical vs Practical Significance
- ❌ p < 0.05 ≠ business value
- ✅ Combine statistics with effect size + business impact

### 18.9 Mixing In-Sample vs Out-of-Sample
- ❌ Reporting training metrics as model performance
- ✅ Always report held-out test metrics

### 18.10 Not Considering Operating Point
- ❌ AUC 0.95 but operating at fixed threshold gives 50% recall
- ✅ Choose threshold based on business cost

---

## 19. Quick Reference Cheat Sheet

### By Task

| Task | Primary Metric | Secondary |
|---|---|---|
| Binary Classification (balanced) | F1, ROC-AUC | Accuracy, MCC |
| Binary Classification (imbalanced) | PR-AUC, MCC | F2, Recall@Prec |
| Multi-class | Macro/Weighted F1 | Top-K Accuracy |
| Multi-label | Hamming Loss | Jaccard, Subset Accuracy |
| Regression (general) | RMSE, R² | MAE, MAPE |
| Regression (outliers) | MAE, MedAE | Huber Loss |
| Time Series Forecasting | WMAPE, MASE | Bias, RMSE |
| Probabilistic Forecasting | CRPS, Pinball | Calibration, Coverage |
| Object Detection | mAP@0.5, mAP@[.5:.95] | AR |
| Segmentation | Mean IoU, Dice | Pixel Accuracy |
| Clustering (no labels) | Silhouette | Davies-Bouldin |
| Clustering (labels known) | ARI, NMI | V-measure |
| Ranking / Recommendation | NDCG@K, MAP | MRR, P@K |
| Anomaly Detection | PR-AUC, P@K | Detection delay |
| NLP Translation | BLEU, BERTScore | METEOR, chrF |
| NLP Summarization | ROUGE-L | ROUGE-1, ROUGE-2 |
| NLP ASR | WER | CER |
| QA | Exact Match, F1 | — |
| Image Generation | FID | IS, LPIPS |
| RL | Cumulative Reward | Sample Efficiency |
| Fairness | Demographic Parity, Equal Opp | Disparate Impact |
| Calibration | ECE, Brier | Reliability Diagram |
| Production | PSI, KS test | Latency, Drift |

### Metric Properties

| Property | Metrics |
|---|---|
| Scale-free | MAPE, sMAPE, R², MASE, NDCG |
| Bounded [0,1] | Accuracy, F1, AUC, Dice, NDCG, Brier |
| Symmetric | MAE, RMSE, sMAPE, MCC |
| Robust to outliers | MAE, MedAE, Huber |
| Probabilistic | Log Loss, Brier, CRPS, ECE |
| Threshold-free | ROC-AUC, PR-AUC, MAP, NDCG |
| Imbalance-tolerant | F1, MCC, Balanced Acc, PR-AUC |

---

## 20. Implementation Quick Reference

### Python — sklearn
```python
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    confusion_matrix, classification_report, hamming_loss, jaccard_score,
    # Regression
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    median_absolute_error, mean_squared_log_error, r2_score,
    explained_variance_score, max_error,
    # Clustering
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score, homogeneity_score,
    completeness_score, v_measure_score, fowlkes_mallows_score,
    # Ranking
    ndcg_score, label_ranking_average_precision_score,
)
```

### Other libraries
| Library | Use |
|---|---|
| `torchmetrics` | PyTorch-native metrics |
| `keras.metrics` | TensorFlow / Keras |
| `evaluate` (HuggingFace) | NLP metrics |
| `pycm` | Confusion matrix analysis |
| `fairlearn` | Fairness metrics |
| `evidently` | Drift / production monitoring |
| `nannyml` | Post-deployment monitoring |
| `aif360` | IBM fairness toolkit |
| `mlflow` | Experiment tracking |
| `wandb` | Weights & Biases tracking |
| `lifelines` | Survival analysis |
| `recmetrics` | Recommendation metrics |
| `seqeval` | NER / sequence labeling |
| `sacrebleu` | Translation BLEU |
| `rouge-score` | Summarization ROUGE |
| `nltk` | NLP utilities + BLEU |
| `pytrec_eval` | Information retrieval |

---

## 21. Glossary

| Term | Definition |
|---|---|
| **Ground truth** | The actual correct label / value |
| **Baseline** | A simple reference (random, naive, majority class) |
| **In-sample / Out-of-sample** | Performance on training data / unseen data |
| **Overfitting** | High in-sample, low out-of-sample performance |
| **Underfitting** | Poor performance on both train and test |
| **K-fold CV** | Train/test K times, average metric |
| **Stratified split** | Preserves class proportions in splits |
| **Hyperparameter tuning** | Searching for best model config |
| **Class imbalance** | Skewed distribution of target classes |
| **Macro avg** | Unweighted mean across classes |
| **Micro avg** | Pool global TP/FP/FN, then compute |
| **Weighted avg** | Class-weighted by sample count |
| **Calibration** | Predicted probabilities match true frequencies |
| **Operating point** | Specific threshold choice |
| **Cohort** | Slice of data sharing an attribute |
| **Drift** | Distribution change over time |
| **Concept drift** | Relationship between X and y changes |
| **Data drift** | Distribution of X changes |
| **PSI** | Population Stability Index — drift metric |
| **Reliability diagram** | Calibration visualization |
| **Decision threshold** | Cutoff for classification |
| **Bootstrap** | Sampling with replacement for confidence intervals |
| **Bayesian credible interval** | Probability that parameter lies in range |
| **Confidence interval** | Frequentist uncertainty range |

---

## 22. Further Reading

### Books
- **Hastie, Tibshirani, Friedman** — *Elements of Statistical Learning*
- **Murphy** — *Probabilistic Machine Learning*
- **Bishop** — *Pattern Recognition and Machine Learning*
- **Goodfellow, Bengio, Courville** — *Deep Learning*
- **Hyndman, Athanasopoulos** — *Forecasting: Principles and Practice* (free at otexts.com/fpp3)

### Online Resources
- **scikit-learn documentation** — metrics module reference
- **Papers with Code** — paperswithcode.com (benchmarks per task)
- **Google ML Crash Course** — developers.google.com/machine-learning/crash-course
- **Distill.pub** — interactive ML explanations
- **Made With ML** — madewithml.com (production ML patterns)
- **HuggingFace evaluate** — huggingface.co/docs/evaluate

### Key Papers
- **Powers (2011)** — "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation"
- **Davis, Goadrich (2006)** — "The Relationship Between Precision-Recall and ROC Curves"
- **Hyndman, Koehler (2006)** — "Another look at measures of forecast accuracy" (MASE)
- **Naeini et al. (2015)** — "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
- **Saito, Rehmsmeier (2015)** — "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"

---

## 23. Workflow Templates

### Template 1: Binary Classification Evaluation

```python
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef)

# 1. Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 2. Threshold-dependent metrics
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")

# 3. Threshold-independent
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")

# 4. Calibration
print(f"Brier:   {brier_score_loss(y_test, y_proba):.4f}")

# 5. Cohort slice (e.g., by age group)
for group in df.group.unique():
    mask = df.group == group
    print(f"{group}: F1 = {f1_score(y_test[mask], y_pred[mask]):.3f}")
```

### Template 2: Regression Evaluation

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.1f}%")

# Bias
bias = (y_test - y_pred).mean()
print(f"Bias: {bias:.2f}")

# WMAPE
wmape = abs(y_test - y_pred).sum() / abs(y_test).sum() * 100
print(f"WMAPE: {wmape:.1f}%")
```

### Template 3: Comparing Models

```python
results = {}
for name, model in [('Logistic', lr), ('RandomForest', rf), ('XGBoost', xgb)]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        'AUC': roc_auc_score(y_test, proba),
        'PR-AUC': average_precision_score(y_test, proba),
        'F1': f1_score(y_test, pred),
        'MCC': matthews_corrcoef(y_test, pred),
        'Brier': brier_score_loss(y_test, proba),
    }
pd.DataFrame(results).T
```

---

## Summary — Top 10 Most Important Metrics to Know

If you remember nothing else, master these 10:

1. **Accuracy** — basic correctness (balanced data only)
2. **Precision & Recall** — error type matters
3. **F1 Score** — balanced view of P/R
4. **ROC-AUC** — ranking quality
5. **PR-AUC** — imbalanced ranking
6. **MAE / RMSE** — regression magnitude
7. **R²** — regression explained variance
8. **MAPE / WMAPE** — percentage error / weighted
9. **Log Loss** — probabilistic training loss
10. **Confusion Matrix** — base for all classification

Everything else is a variant or specialization of these.

---

*Prepared by Rohan Mistry — Data Visualization Tool Owner*
*MaxColors Project · Authoritative ML metrics reference*
*Use this document as the single source of truth for evaluating any ML model in the team's work*
