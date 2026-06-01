# MLOps.md

### The Complete Production ML Reference — How Trained Models Become Running Systems, Stay Healthy, and Get Retrained Without Breaking

> **Why this file exists:** A model in a notebook is worth nothing. A model in production that nobody monitors, retrains, or rolls back is worth *less* than nothing — it's a liability shipping predictions to real users with no awareness that it's degrading. This file is the systematic reference for the operational layer of ML: how to deploy, monitor, experiment, retrain, and shut down models responsibly.
>
> **How to read this file:** §4 (Deployment patterns), §6 (Monitoring), §10 (Retraining triggers) are operational core — read those first. §3 (Architecture) is the mental model the rest is built on. §13 (Cost) is the section most teams discover the hard way.

---

## Table of Contents

1. [Philosophy — What MLOps Actually Is](#1-philosophy--what-mlops-actually-is)
2. [The MLOps Maturity Model](#2-the-mlops-maturity-model)
3. [Production ML Architecture](#3-production-ml-architecture)
4. [Deployment Patterns](#4-deployment-patterns)
5. [Serving Infrastructure](#5-serving-infrastructure)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Online Experimentation — A/B Testing for ML](#7-online-experimentation--ab-testing-for-ml)
8. [Feature Stores in Production](#8-feature-stores-in-production)
9. [Model Registries & Versioning](#9-model-registries--versioning)
10. [Retraining — When, How, Why](#10-retraining--when-how-why)
11. [Rollback & Incident Response](#11-rollback--incident-response)
12. [CI/CD for ML](#12-cicd-for-ml)
13. [Cost Optimization](#13-cost-optimization)
14. [Security & Governance](#14-security--governance)
15. [LLM Ops — The Modern Twist](#15-llm-ops--the-modern-twist)
16. [The MLOps Workflow](#16-the-mlops-workflow)
17. [Implementation Quick Reference](#17-implementation-quick-reference)

---

## 1. Philosophy — What MLOps Actually Is

### The Three Laws of MLOps

```
LAW 1 — The Law of the Living Model
    A model is not an artifact. It is a system component with
    upstream dependencies (data, features, code) and downstream
    impact (predictions, decisions, business outcomes). The job
    is not to "deploy" the model — it is to keep this system
    healthy as the world changes around it.

LAW 2 — The Law of Production Parity
    Whatever produced the predictions at training time must
    produce them identically in production. Same code, same
    features, same logic, same versions. Every divergence is
    a silent bug compounding over time. Training-serving skew
    is the single most common cause of "looked great offline,
    died in production" failures.

LAW 3 — The Law of Detection Over Trust
    Models do not warn you when they degrade. Data pipelines
    do not warn you when they corrupt. Features do not warn
    you when they drift. If you have not built the systems
    that detect these failures, you will discover them via
    angry users, lost revenue, or a regulatory complaint.
    Build the detection first; trust comes from instrumentation,
    not optimism.
```

### Why MLOps Is Harder Than DevOps

A software service has one source of behavior change: the code. An ML service has at least four:

| Source of Change | Failure Mode |
|---|---|
| Code | Same as software engineering |
| Data | Silent quality regressions, schema evolution |
| Model | Retraining produces different behavior |
| World | Concept drift — yesterday's truth becomes today's bias |

Standard DevOps handles the first. MLOps must handle all four — which is why it's its own discipline, not a flavor of DevOps.

### The Three Layers of Failure

```
CODE FAILURES        — bugs, crashes, latency spikes
                       (DevOps tooling catches these)

DATA FAILURES        — pipeline drops, schema changes, drift
                       (data quality tooling catches these)

MODEL FAILURES       — performance degradation, fairness regression,
                       prediction distribution shift
                       (ML observability tooling catches these)
```

A team running production ML without all three observability layers is operating blind. Most "we don't know why our metrics dropped" stories come from missing one of the three.

### What MLOps Is Not

| It is NOT | It IS |
|---|---|
| A tool you buy | A set of practices |
| Just training pipelines | Full lifecycle from data to decommission |
| A single team's job | Cross-functional: data, ML, platform, ops |
| Done once at launch | Continuous, every retraining cycle |
| Optional for small teams | Smaller scale; lower complexity; same principles |

---

## 2. The MLOps Maturity Model

A practical framework for assessing where a team is and where it needs to go.

### Level 0 — Manual

```
Notebook → email → "can you push this to production?"
```

- Models trained in notebooks by hand
- Deployments are manual file copies
- No versioning of data, features, or models
- No monitoring after deployment
- Retraining happens when someone notices a problem

**Where most teams actually are**, even at companies that claim Level 3.

### Level 1 — Pipelined

- Training is a script that can run reproducibly
- Models are saved to a known location with versions
- Basic deployment automation (containerization)
- Logging in production, but no alerting on model behavior
- Retraining is scheduled (e.g., monthly) but manual

### Level 2 — Continuous Training

- Automated training pipelines triggered by data, schedule, or performance
- Feature stores (or at least consistent feature definitions in code)
- Model registry with stage transitions (dev → staging → production)
- Basic drift monitoring with alerts
- A/B testing infrastructure for new model versions

### Level 3 — Continuous Delivery

- CI/CD for ML: code changes trigger pipeline validation, testing, deployment
- Automated rollback on performance regression
- Comprehensive monitoring: data, features, predictions, business metrics
- Self-service deployment for ML teams
- Cost tracking per model

### Level 4 — Adaptive

- Models retrain themselves based on detected drift
- Multi-armed bandits / online learning for adaptive serving
- Champion-challenger frameworks running continuously
- Automated fairness and safety audits in the pipeline
- Predictive capacity scaling

### Where to Aim

For most teams: **Level 2 is the sweet spot.** It's where the ROI of MLOps investment plateaus. Levels 3-4 require sustained platform investment that's only justified at scale (many models, many teams, regulatory demands).

The progression isn't optional skipping levels — Level 0 → Level 3 without going through 1 and 2 is how organizations end up with expensive ML platforms nobody uses.

---

## 3. Production ML Architecture

### 3.1 The Anatomy of a Production ML System

```
┌──────────────────────────────────────────────────────────────────┐
│                       OFFLINE / TRAINING                          │
│                                                                   │
│   Data Sources → ETL → Feature Engineering → Training → Registry  │
│        │           │           │                │           │     │
│        ▼           ▼           ▼                ▼           ▼     │
│   Data Lake   Warehouse   Feature Store   Experiment    Model    │
│                                            Tracker     Artifacts │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                          ONLINE / SERVING                         │
│                                                                   │
│   Request → API Gateway → Feature Lookup → Inference → Response   │
│       │          │              │              │           │     │
│       ▼          ▼              ▼              ▼           ▼     │
│    Logging   Auth/Quota    Online Feature   Model      Prediction │
│                              Store         Server        Cache    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY                             │
│                                                                   │
│   Metrics → Drift Detection → Performance Estimation → Alerting   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 The Four Critical Boundaries

Most production failures happen at these interfaces:

| Boundary | Common Failure |
|---|---|
| **Raw data → ETL** | Schema evolution unhandled |
| **ETL → Feature engineering** | Logic divergence between batch and stream |
| **Training → Serving** | Same feature computed differently in two places |
| **Predictions → Decisions** | Score interpreted incorrectly downstream |

Most of MLOps as a discipline is about making these boundaries trustworthy.

### 3.3 Serving Modes

| Mode | Latency | Throughput | Example |
|---|---|---|---|
| **Batch** | Minutes-hours | Millions/run | Daily churn scoring |
| **Near real-time** | Seconds | Thousands/sec | Email categorization |
| **Real-time / Online** | Milliseconds | Hundreds-thousands/sec | Ad ranking, fraud check |
| **Edge** | Milliseconds | Per device | On-device recommendations, mobile NLP |

Choose mode based on **actual business need, not aspiration.** Real-time serving is 10-100x more expensive than batch. Many products genuinely need batch and get pushed into real-time because nobody questioned the assumption.

### 3.4 Online vs. Offline Features

```
OFFLINE FEATURES — computed in batch, e.g.:
  - user_lifetime_purchase_count
  - average_session_duration_last_30_days
  - n_items_in_cart_last_week
  Computed once daily, served from a key-value store at inference.

ONLINE FEATURES — computed at request time, e.g.:
  - current_cart_value (just changed)
  - device_type (in the request)
  - is_logged_in
  Computed from the request payload or recent state.

ON-DEMAND FEATURES — derived at request time from other features:
  - ratio of online to offline values
  - similarity between request and stored profile
  Cheap derived features, computed by the serving layer.
```

A well-designed feature store handles all three (see §8).

---

## 4. Deployment Patterns

> The single most important MLOps skill: **never deploy a model with no rollback path.**

### 4.1 The Deployment Strategy Decision Tree

```
NEW MODEL READY FOR PRODUCTION?
  │
  ├── No previous version → Soft launch with shadow mode
  │
  ├── Replacement for an existing model?
  │     ├── Confident in equivalence? → Direct replacement (rare; usually wrong)
  │     ├── Need empirical validation? → Shadow + Canary + A/B
  │     └── High-stakes domain (medical, financial)? → Champion-challenger w/ manual review
```

### 4.2 Shadow Deployment

The new model runs alongside production, gets the same inputs, but its predictions are **not used** — only logged.

```
Request → [Champion model] → Response sent to user
       └→ [Challenger model] → Predictions logged, never served
```

**Use for:**
- Initial validation of a new model on real traffic
- Comparing inference latency, error rates, output distributions
- Catching production-environment bugs before they affect users

**Run for:** Days to weeks, depending on traffic volume and confidence level.

**What to compare:**
- Prediction distribution similarity
- Disagreement rate between champion and challenger
- Latency profile (p50, p95, p99)
- Error rate (failed predictions)
- Resource consumption

**Critical caveat:** Shadow mode doesn't validate **outcomes** — only behavior. You don't know if the challenger's predictions would have led to better business outcomes because they weren't acted on. For that, you need A/B testing.

### 4.3 Canary Deployment

Route a small percentage of traffic to the new model. If metrics hold, gradually increase.

```
1% traffic → new model    (24h, check error rates)
5% traffic → new model    (48h, check latency)
25% traffic → new model   (72h, check business metrics)
50% traffic → new model
100% traffic → new model
```

**Use for:**
- Validating production performance at small scale
- Catching infrastructure issues before full rollout
- Building organizational confidence in a new model

**Critical:** Define **abort criteria** before starting. Examples:
- p99 latency > 50ms → abort
- Error rate > 0.1% → abort
- Business metric drops > 2% → abort

Without pre-defined abort criteria, canary becomes "let's see what happens" — exactly the wrong posture.

### 4.4 Blue-Green Deployment

Two identical production environments. New model deployed to "green" while "blue" serves. Switch traffic in one operation. Easy rollback by switching back.

```
BLUE (current)  ←── serving traffic
GREEN (new)     ←── idle, fully tested

Switch: BLUE → idle, GREEN → serving

Rollback if needed: GREEN → idle, BLUE → serving
```

**Use for:** Stateless serving infrastructure, fast rollback requirements, simpler than canary.

**Downside:** Doubles infrastructure cost during rollout window.

### 4.5 A/B Testing

The only deployment pattern that validates **business outcomes**. Split users randomly between champion and challenger, measure the business metric.

```
50% users → Champion model → measure conversion rate
50% users → Challenger model → measure conversion rate

Statistical test: did challenger improve the metric?
```

**Use for:** Final validation before full rollout, comparing fundamentally different approaches.

**Critical:** This is a real experiment with all the statistical pitfalls of online experimentation. See §7.

### 4.6 Multi-Armed Bandit Deployment

Adaptive variant of A/B testing — traffic dynamically reallocates toward the better-performing variant.

```
Initial: 50/50 split
After 1 day: 60% to better variant
After 1 week: 80% to better variant
Eventually: 100% to winner
```

**Use for:** Low-stakes optimization, high-traffic scenarios where speed of convergence matters more than statistical rigor.

**Don't use for:** Decisions with regulatory or fairness implications where you need clear A/B evidence.

### 4.7 Champion-Challenger

Permanent comparison framework. A "champion" serves production traffic. One or more "challengers" run in parallel (shadow or small A/B). Challengers are promoted to champion when they demonstrate sustained improvement.

**Use for:** Mature ML platforms with constant model iteration.

### 4.8 Direct Replacement

The new model fully replaces the old, all at once.

**Use only when:**
- You have very high confidence (extensive offline + shadow validation)
- The model is non-critical
- Rollback is easy

For most cases, direct replacement is a mistake. Default to canary.

### 4.9 Rollback Triggers

Every deployment must have explicit rollback triggers, defined **before** deployment:

| Trigger | Threshold |
|---|---|
| Error rate spike | > 2× baseline for 5 minutes |
| Latency degradation | p99 > SLO for 10 minutes |
| Prediction distribution drift | PSI > 0.3 vs training reference |
| Business metric drop | > 5% relative to control for 1 hour |
| Manual escalation | On-call decides |

Document who can pull the trigger and how (single command, not a 10-step process).

---

## 5. Serving Infrastructure

### 5.1 The Serving Stack

```
[Load Balancer]
       │
       ▼
[API Gateway] — auth, rate limiting, routing
       │
       ▼
[Inference Service] — model loading, prediction
       │
       ├─→ [Feature Store] — feature lookups
       ├─→ [Cache] — recent prediction reuse
       └─→ [Logging] — request/prediction/feature logs
```

### 5.2 Serving Framework Choices

| Framework | Best For |
|---|---|
| **TorchServe** | PyTorch models, multi-model serving |
| **TensorFlow Serving** | TensorFlow models, mature production |
| **NVIDIA Triton** | GPU inference, multi-framework, dynamic batching |
| **BentoML** | Python-native, framework-agnostic, easy packaging |
| **KServe** (Kubeflow) | Kubernetes-native, autoscaling |
| **Ray Serve** | Multi-model, Python-flexible, distributed |
| **vLLM** | LLM-specific, optimized for transformer inference |
| **Modal, Replicate** | Managed serverless inference |
| **Custom FastAPI/Flask** | Simple use cases, full control |

**Default for new projects:** Start with FastAPI + Docker if simple. Move to BentoML or KServe if you need more sophisticated features.

### 5.3 Latency Optimization

Real-world latency sources, in typical magnitude:

| Source | Typical |
|---|---|
| Network / load balancer | 1-5 ms |
| Feature lookup | 1-20 ms |
| Model inference | 1-200 ms |
| Postprocessing | 0.1-5 ms |
| Logging | 0.5-2 ms (async) |

**Optimizations:**
- **Model optimization** — quantization (int8/int4), pruning, distillation, ONNX/TensorRT
- **Batch inference** — dynamic batching at the serving layer
- **Caching** — cache predictions for repeated inputs
- **Feature pre-computation** — push offline features to in-memory stores
- **Co-location** — feature store near inference server
- **Async logging** — never block on logs
- **Connection pooling** — reuse connections to feature stores

### 5.4 Throughput Optimization

For high-throughput needs:

| Technique | Effect |
|---|---|
| **Dynamic batching** | Group concurrent requests, run as one batch |
| **Async serving** | Non-blocking request handling |
| **Horizontal scaling** | More replicas behind load balancer |
| **GPU sharing** | Multiple models on one GPU (carefully) |
| **Speculative decoding** (LLMs) | Faster draft model generates, target model verifies |

### 5.5 Hardware Choices

| Hardware | Best For |
|---|---|
| **CPU** | Linear models, small trees, light NN, low-traffic |
| **GPU** (T4, A10) | Mid-size NN inference, high throughput |
| **GPU** (A100, H100) | Large model inference, LLM serving |
| **Inference accelerators** (Inferentia, TPU) | Cost-optimized, framework-specific |
| **Edge devices** | On-device inference, latency-critical or offline |

**Rule of thumb:** Don't reach for GPU until you've optimized the CPU path. A well-optimized CPU inference is cheaper than GPU at low-to-moderate scale.

### 5.6 Containerization

Production serving lives in containers. Key practices:

- Multi-stage Dockerfiles (build stage with deps, slim runtime stage)
- Pin base image, all deps, model artifacts
- Health checks (`/healthz`, `/readyz`)
- Graceful shutdown handling
- Resource limits (CPU, memory) declared explicitly

```dockerfile
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY model/ /app/model/
COPY src/ /app/src/
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Monitoring & Observability

> If you can't observe it, you can't operate it. The most expensive lesson in production ML is the one learned by discovering, weeks later, that a model has been broken since the last deployment.

### 6.1 The Four Levels of Monitoring

```
LEVEL 1 — System Health (infrastructure)
    - Latency (p50, p95, p99)
    - Throughput (requests/sec)
    - Error rates (4xx, 5xx)
    - Resource utilization (CPU, memory, GPU)

LEVEL 2 — Data Health (inputs)
    - Schema conformance
    - Missing rate per feature
    - Feature distributions (PSI vs reference)
    - Feature freshness

LEVEL 3 — Model Health (predictions)
    - Prediction distribution
    - Score distribution (probabilities)
    - Class balance (classification)
    - Confidence calibration

LEVEL 4 — Business Health (outcomes)
    - Performance metric (with ground truth lag)
    - Business KPI impact
    - Fairness metrics across subgroups
    - User feedback signals
```

Most teams monitor Level 1 well, Level 2 partially, Level 3 rarely, Level 4 almost never. The gap between Level 3 and Level 4 is where models silently degrade.

### 6.2 What to Log

Every prediction request should log:

```json
{
  "request_id": "uuid",
  "timestamp": "2026-05-18T10:23:14.512Z",
  "model_version": "fraud_v2.4.1",
  "feature_versions": {"feature_store_v": "2026-05-18"},
  "input_features": {...},
  "prediction": 0.84,
  "predicted_class": 1,
  "latency_ms": 14.3,
  "user_segment": "premium",
  "experiment_id": "challenger_b",
  "downstream_action_taken": null
}
```

**Critical fields:**
- `model_version` — for slicing performance by version
- `feature_versions` — for diagnosing feature pipeline issues
- `experiment_id` — for A/B test slicing
- `downstream_action_taken` (joined later) — for outcome analysis

### 6.3 Drift Detection in Production

For each feature, continuously compare current distribution to a training reference.

**See `Data.md` §9 for detection methods. In production specifically:**

| Cadence | Check |
|---|---|
| Real-time | Schema, missing rate, range violations |
| Hourly | Univariate feature drift (KS, PSI) |
| Daily | Multivariate drift (domain classifier) |
| Weekly | Performance estimation, calibration |

**Alerting thresholds (typical):**
- PSI > 0.25 on a single feature → page someone
- PSI > 0.10 on 5+ features → page someone
- Schema violation → page someone
- Prediction distribution drift → page someone

### 6.4 Performance Estimation Without Ground Truth

Ground truth is often delayed (was this transaction actually fraud? days later). Until then, estimate performance.

**Methods:**
- **Confidence-based** — if confidence distribution shifts, performance probably has too
- **Calibration tracking** — if predicted probabilities diverge from observed rates on the subset where labels are available
- **Reference-set monitoring** — periodically label a small random sample to estimate ongoing performance
- **NannyML / similar tools** — algorithmic performance estimation under covariate shift

### 6.5 Logging Architecture

```
Application logs → Local agent → Buffered → Log aggregator → Storage + Analysis
                                                                    │
                                                                    ▼
                                                            Dashboards + Alerts
```

**Tools:**
- **Logs**: ELK / Loki / Datadog / Splunk
- **Metrics**: Prometheus + Grafana / Datadog
- **Traces**: Jaeger / Tempo / Datadog APM
- **ML observability**: Evidently / Arize / Fiddler / WhyLabs / Truera

### 6.6 Alerting Discipline

Bad alerting causes the alert system to be ignored. Rules:

- Every alert must be **actionable** (not "FYI, drift detected")
- Every alert must have a **runbook** (here's what to do)
- Every alert must have an **owner** (this team responds)
- Tune thresholds aggressively against false positives
- Distinguish "page someone at 3am" from "ticket for morning"

**Three tiers of alert:**
```
P1 — Production down, customers affected, page immediately
P2 — Degradation likely, ticket within hours
P3 — Anomaly detected, ticket within days
```

### 6.7 Dashboards That Matter

Every production model should have a dashboard showing:

1. **Top section — Health snapshot**
   - Current latency (p50, p95, p99)
   - Error rate
   - Requests/sec
   - Up/down status

2. **Middle section — Data health**
   - Feature drift heatmap (PSI per feature over time)
   - Missing rate trends
   - Prediction distribution histogram

3. **Bottom section — Outcomes**
   - Business metric (with control comparison if A/B)
   - Performance on labeled subset (with lag)
   - Subgroup performance comparison

If your team can't open this dashboard and answer "is the model healthy right now?" in 30 seconds, the dashboard isn't working.

---

## 7. Online Experimentation — A/B Testing for ML

> Offline metrics tell you whether a model is statistically better on historical data. Online experiments tell you whether it produces better business outcomes when actually deployed. These often disagree.

### 7.1 Why Offline-Online Gaps Exist

A challenger model can win offline and lose online for many reasons:

| Cause | Example |
|---|---|
| **Selection bias** | Training data biased by current model's decisions |
| **Feedback loops** | Recommendations affect what gets clicked, which trains the next model |
| **Temporal shift** | Offline test data is older than online traffic |
| **Hidden confounders** | Offline metric optimizes a proxy, not the actual KPI |
| **Latency tradeoffs** | More accurate but slower → worse UX → worse business metric |
| **Distribution shift** | Online traffic differs from training |
| **Goodhart's law** | When metric becomes target, it ceases to be a good measure |

**This is why A/B testing exists.** Offline performance is a hypothesis; online performance is evidence.

### 7.2 The A/B Test Setup

```
Population → Random split → Treatment (challenger) and Control (champion)
                                    ↓
                            Observe primary metric
                                    ↓
                            Statistical test → decision
```

**Critical components:**
- Randomization unit (user, session, request)
- Primary metric (the one decision rests on)
- Guardrail metrics (cannot regress)
- Sample size calculation (power analysis)
- Pre-defined duration / stopping criteria

### 7.3 Sample Size and Power

Before running, calculate the sample size needed to detect the minimum effect that matters.

```python
import statsmodels.stats.power as smp
effect_size = 0.02  # 2% relative improvement
alpha = 0.05
power = 0.8
baseline_rate = 0.05

sample_per_arm = smp.zt_ind_solve_power(
    effect_size=effect_size / baseline_rate,
    alpha=alpha,
    power=power,
    nobs1=None,
    ratio=1.0
)
print(f"Need {int(sample_per_arm):,} samples per arm")
```

**Common mistake:** Running underpowered experiments and concluding "no effect" when actually you couldn't have detected the real effect.

### 7.4 Randomization Pitfalls

| Problem | Description |
|---|---|
| **Same user, different bucket** | User gets one experience on phone, another on web — inconsistent |
| **Network effects** | User A in treatment affects User B's experience |
| **Carryover** | User's previous experience affects current behavior |
| **Selection on the outcome** | Bucketing depends on something downstream |

**Fix:** Randomize at the right unit. For user-facing products, randomize at user level with stable hashing.

### 7.5 Sequential Testing

Standard A/B tests have fixed duration. **Peeking at results and stopping early inflates false positive rates** — you'll find "significant" effects that aren't real.

**Solutions:**
- **Group sequential designs** — pre-planned interim analyses with adjusted thresholds
- **Always-valid p-values** — statistical methods that allow continuous monitoring (e.g., mSPRT)
- **Bayesian methods** — posterior probability, no peeking penalty

### 7.6 CUPED (Variance Reduction)

CUPED (Controlled-experiment Using Pre-Experiment Data) uses pre-experiment data to reduce variance in metric estimation, often achieving the same statistical power with 30-50% smaller sample sizes.

**Use when:** A/B test sample sizes are constrained, results are noisy, you have pre-experiment data per user.

### 7.7 Common A/B Mistakes

- **HARKing** (Hypothesizing After Results are Known) — running 20 metrics and reporting the one that won
- **Multiple comparisons** without correction (Bonferroni, Benjamini-Hochberg)
- **Stopping early on positive results** without sequential adjustment
- **Ignoring guardrails** — primary metric won, but latency doubled
- **Generalizing from small segments** — won in one country, assumed to win everywhere
- **Forgetting novelty effects** — initial behavioral shifts that don't last
- **Forgetting day-of-week effects** — running too short to capture weekly cycles

### 7.8 Interleaving (Specific to Ranking)

For ranking problems (search, recommendations), **interleaving** is more sensitive than A/B testing — you mix results from both models for the same user and observe which model's results get clicked more.

**Use when:** Comparing ranker models, need faster signal, can mix results without breaking UX.

### 7.9 Causal vs. Correlational

A/B tests give you **causal** estimates. Most offline metrics give correlational ones. Don't confuse the two.

When A/B testing isn't feasible (regulatory, ethical, or operational reasons), reach for:
- **Quasi-experimental methods** — difference-in-differences, regression discontinuity
- **Causal inference** — propensity scoring, instrumental variables
- **Synthetic control** — construct a counterfactual control group

---

## 8. Feature Stores in Production

> See `Features.md` §18 for the foundation. This section focuses on production operations.

### 8.1 Why Feature Stores Exist

Without a feature store, you have:
- **Training-serving skew** — same feature computed two different ways
- **Feature duplication** — every team reimplements the same features
- **Point-in-time bugs** — historical training data joined with current dimension values
- **No reuse** — can't share features across models
- **No monitoring** — can't track feature health centrally

A feature store solves these by providing a single source of truth for feature definitions, with offline (training) and online (serving) consistency.

### 8.2 The Core Capabilities

| Capability | Why |
|---|---|
| **Feature definition layer** | One place where each feature is defined, in code |
| **Offline store** | Bulk historical features for training |
| **Online store** | Low-latency feature serving for inference |
| **Point-in-time joins** | Historical correctness for training data assembly |
| **Versioning** | Track feature definition changes |
| **Monitoring** | Drift detection, freshness, quality |

### 8.3 Feature Definitions as Code

```python
# Feast example
from feast import Entity, Feature, FeatureView, FileSource
from datetime import timedelta

user = Entity(name="user_id", value_type=ValueType.INT64)

user_stats = FeatureView(
    name="user_stats",
    entities=["user_id"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="lifetime_purchase_count", dtype=ValueType.INT64),
        Feature(name="avg_purchase_amount", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_purchase", dtype=ValueType.INT64),
    ],
    source=FileSource(
        path="data/user_stats.parquet",
        event_timestamp_column="event_timestamp",
    ),
)
```

The same definition serves both batch (training) and online (serving) — guaranteeing parity.

### 8.4 Online Store Backends

| Store | Latency | Best For |
|---|---|---|
| **Redis** | < 1ms | Hot features, real-time |
| **DynamoDB** | 1-5ms | AWS-native, autoscaling |
| **BigTable** | 1-10ms | GCP, large scale |
| **Cassandra** | 5-20ms | High-write throughput |
| **In-memory** (local) | < 0.1ms | Edge inference |

Choose based on latency SLO and scale.

### 8.5 Materialization Patterns

**Batch materialization:**
```
Daily/hourly job → Read source → Compute features → Write to online store
```

**Stream materialization:**
```
Event stream → Real-time compute (Flink/Kafka Streams) → Write to online store
```

**On-demand materialization:**
```
Request → Compute feature at request time from current state
```

Most production systems blend all three.

### 8.6 Point-in-Time Joins

The critical capability for training data assembly. For each `(entity_id, event_timestamp)` in training data, retrieve features as they were at that exact timestamp.

```python
training_df = store.get_historical_features(
    entity_df=pd.DataFrame({
        "user_id": [101, 102, 103],
        "event_timestamp": [t1, t2, t3],
    }),
    features=[
        "user_stats:lifetime_purchase_count",
        "user_stats:days_since_last_purchase",
    ],
).to_df()
```

The feature store ensures the values returned are the ones that were known at the respective timestamps — not today's values.

**This single capability is what justifies a feature store's existence for most teams.**

### 8.7 Tool Landscape

| Tool | Notes |
|---|---|
| **Feast** | Open source, lightweight, framework-agnostic |
| **Tecton** | Commercial, sophisticated streaming features |
| **Hopsworks** | Open-core, end-to-end ML platform |
| **AWS SageMaker Feature Store** | Managed, AWS-native |
| **Vertex AI Feature Store** | GCP managed |
| **Databricks Feature Store** | If you're on Databricks |
| **Custom-built** | Most large tech companies have one |

For small teams: **start without a feature store.** Use shared SQL/Python libraries to define features. Move to Feast when feature reuse and PIT joins become real pain points.

---

## 9. Model Registries & Versioning

### 9.1 What a Registry Provides

A model registry is a central catalog of trained models with:

- Versioned artifacts
- Metadata (training data, metrics, hyperparameters)
- Stage transitions (dev → staging → production → archived)
- Lineage (which data, which code, which experiment)
- Access control and audit log
- Serving integration

### 9.2 The Model Lifecycle

```
DEVELOPMENT  — experimentation, not for production
       ↓
STAGING      — validated, candidate for production
       ↓
PRODUCTION   — currently serving traffic
       ↓
ARCHIVED     — replaced but preserved for rollback / audit
```

Stage transitions should be auditable. "Who promoted this to production?" should always have an answer.

### 9.3 What to Register With Each Model

```
MODEL REGISTRATION RECORD
─────────────────────────────────────────────
Model Name:          [Name]
Version:             [Semantic version or hash]
Stage:               [Development / Staging / Production / Archived]
Owner:               [Team]
Training Data:       [Version / hash]
Training Code:       [Git commit]
Hyperparameters:     [Config]
Test Metrics:        [Primary, guardrails]
Training Date:       [Timestamp]
Deployment Date:     [Timestamp, if applicable]
Approval Trail:      [Who approved, when]
Known Limitations:   [Subgroup gaps, edge cases]
Linked Experiments:  [Experiment IDs]
─────────────────────────────────────────────
```

(See `Audit.md` §22 for the full model card template.)

### 9.4 Tools

| Tool | Notes |
|---|---|
| **MLflow** | Open source, widely adopted, simple to start |
| **Weights & Biases** | Strong experiment tracking, registry features |
| **Neptune.ai** | Experiment tracking + registry |
| **Vertex AI Model Registry** | GCP managed |
| **SageMaker Model Registry** | AWS managed |
| **Comet** | Tracking + registry |

### 9.5 Semantic Versioning for Models

```
model_v{MAJOR}.{MINOR}.{PATCH}

MAJOR — breaking change (new schema, new features, new task)
MINOR — backwards-compatible improvement (retrained, better data)
PATCH — bug fix, small recalibration
```

Downstream consumers can pin major versions and safely auto-update minor/patch.

### 9.6 Reproducibility From the Registry

Given a model version, you should be able to reproduce it:

```
Given model fraud_v2.4.1:
  → Retrieve training data version (data_hash from registry)
  → Retrieve training code (git commit from registry)
  → Retrieve config (hyperparameters from registry)
  → Re-run training
  → Verify metrics match
```

If you can't do this, your registry is incomplete.

---

## 10. Retraining — When, How, Why

### 10.1 Why Retrain

| Reason | Trigger |
|---|---|
| **Data drift** | Input distribution has shifted; model assumptions stale |
| **Concept drift** | Relationship between inputs and target has changed |
| **Performance decay** | Measured performance below threshold |
| **New data available** | More signal to learn from |
| **Feature changes** | New features added or removed |
| **Bug fix** | Discovered issue requires retraining |
| **Regulatory requirement** | Periodic refresh mandated |

### 10.2 The Retraining Strategy Decision Tree

```
WHEN TO RETRAIN?

  Scheduled (e.g., monthly)
    ├── Pros: predictable, easy to operate
    ├── Cons: wasteful when stable; lagged when drift hits
    └── Use when: data is reasonably stable, no drift detection in place

  Trigger-based (on drift / performance threshold)
    ├── Pros: efficient, responsive to actual change
    ├── Cons: requires reliable triggers, more infrastructure
    └── Use when: monitoring is in place, drift is known to occur

  Continuous (online learning)
    ├── Pros: always current
    ├── Cons: stability risks, harder to debug, calibration drift
    └── Use when: high-throughput, well-understood domain, robust monitoring

  Hybrid (scheduled + on-trigger)
    ├── Pros: best of both
    └── Use when: production-mature with full monitoring stack
```

**Most teams should be on trigger-based with a scheduled fallback.**

### 10.3 Retraining Triggers

Define these **before** deployment, not after problems appear:

| Trigger | Example Threshold |
|---|---|
| Performance drop | Test set accuracy below baseline - 2% |
| PSI on key features | Sustained PSI > 0.25 for 3 days |
| Prediction distribution shift | KS test p < 0.001 vs. reference |
| Time-based | No retraining in 90 days |
| New data volume | > 20% new labeled data accumulated |
| Schema change | Any upstream schema modification |

### 10.4 The Retraining Pipeline

```
1. TRIGGER FIRES
       ↓
2. Pull fresh training data (versioned)
       ↓
3. Run data quality checks (block if fail)
       ↓
4. Train model with current config
       ↓
5. Evaluate against holdout
       ↓
6. Compare to current production model
       ↓
7. If improved (by guardrail margin) → register as challenger
       ↓
8. Shadow / canary deployment (per §4)
       ↓
9. If sustained improvement → promote to production
       ↓
10. Archive previous production model
```

**Critical:** Step 6 must include guardrail metrics, not just primary metric. A new model that improves accuracy by 1% but degrades fairness is not an improvement.

### 10.5 Champion-Challenger Pattern

Run continuously:
- **Champion** — current production model
- **Challenger(s)** — candidate replacements

Challengers are evaluated continuously (offline + shadow). When a challenger demonstrates sustained, statistically significant improvement, it's promoted.

This pattern lets the team experiment without manual deployment ceremony.

### 10.6 Concept Drift–Specific Retraining

Concept drift means the target relationship has changed. Pure retraining may not be enough:

- Old data may now be **misleading**, not just stale
- Consider weighting recent data more heavily
- Consider truncating training window (only last N months)
- Re-evaluate features — some may no longer be predictive

### 10.7 Retraining Pitfalls

| Pitfall | Description |
|---|---|
| **Retraining on production-affected data** | Predictions affected outcomes; training on those outcomes reinforces the model's biases |
| **Auto-retrain without quality checks** | Drift can be caused by upstream bugs; retraining propagates the bug |
| **Drift in training labels** | Label sources can change too; check label distribution |
| **Forgetting to retrain calibration** | Threshold may need adjustment with the new model |
| **Insufficient holdout time** | Can't evaluate temporal generalization with too-recent data |

### 10.8 The Feedback Loop Problem

When a model's predictions affect future training data:

```
Model predicts fraud → blocks transactions →
  blocked transactions never produce labels →
  retraining data is missing the highest-risk examples →
  next model is worse at detecting them
```

**Mitigations:**
- Hold out a small fraction of decisions for ground truth (e.g., random review)
- Use uncertainty-based sampling to label the most informative examples
- Detect distribution shift in training data vs. production data

---

## 11. Rollback & Incident Response

### 11.1 The Three Failure Modes

```
MODEL FAILURE     — predictions are wrong but system works
DATA FAILURE      — features are corrupted, predictions reflect bad inputs
SYSTEM FAILURE    — serving infrastructure broken, no predictions
```

Each requires a different response.

### 11.2 The Rollback Checklist

When an incident is suspected:

```
1. CONFIRM
   - Real issue or measurement artifact?
   - Single user / segment / system-wide?
   
2. CONTAIN
   - Roll back to previous model version if uncertain
   - Throttle if resource pressure
   - Bypass to fallback if model unavailable
   
3. DIAGNOSE
   - Logs, metrics, traces
   - Recent changes (code, data, model)
   - Feature health
   
4. REMEDIATE
   - Fix code / data / model
   - Validate fix in staging
   - Deploy with extra caution
   
5. DOCUMENT
   - Postmortem within 5 days
   - Root cause, contributing factors
   - Detection gap, prevention plan
```

### 11.3 Rollback Strategies

**Blue-green rollback** — flip back to the previous environment. Seconds.

**Canary rollback** — reduce traffic to new version to 0%. Minutes.

**Direct rollback** — redeploy the previous model artifact. Minutes to hours.

**Feature rollback** — disable specific features that may be causing issues. Useful when the model itself is fine but a feature pipeline broke.

**Always have a path back.** "Deploy and pray" is not a strategy.

### 11.4 Fallback Strategies

When the ML system itself can't serve:

| Fallback | Use |
|---|---|
| **Last good prediction** | Cached results, slightly stale |
| **Default prediction** | Conservative default (e.g., low-risk class) |
| **Rule-based fallback** | Hand-coded heuristics |
| **Previous model version** | Auto-failover to N-1 |
| **No prediction** | Allow the downstream system to handle it |

Production-grade ML systems must define fallback behavior **before** they're needed.

### 11.5 The Postmortem Template

```
INCIDENT: [Short description]
Date / Time:         [When detected]
Duration:            [How long until resolved]
Severity:            [P1 / P2 / P3]
Affected Models:     [List]
Affected Users:      [Count / segments]

Timeline:
  ─ [Time] — First symptom observed
  ─ [Time] — Alert fired
  ─ [Time] — On-call paged
  ─ [Time] — Root cause identified
  ─ [Time] — Mitigation deployed
  ─ [Time] — Resolved

Root Cause:
  [Detailed explanation]

Contributing Factors:
  [Things that made it worse or longer]

What Went Well:
  [Detection, response, communication]

What Went Wrong:
  [Gaps, delays, missing tooling]

Action Items:
  [Owner / Item / Due Date]
```

Postmortems are blameless. The goal is to find systemic weaknesses, not individuals to fault.

---

## 12. CI/CD for ML

### 12.1 What's Different from Software CI/CD

| Aspect | Software | ML |
|---|---|---|
| Tests | Unit + integration | + data tests + model tests + performance tests |
| Artifacts | Code | Code + data + models |
| Quality gate | Tests pass | Tests pass + metric thresholds met |
| Deployment | Update code | Update code OR data OR model |

### 12.2 The ML CI/CD Pipeline

```
Code commit
    ↓
Lint, type check
    ↓
Unit tests (code logic)
    ↓
Data validation tests (schemas, distributions)
    ↓
Feature pipeline tests (deterministic outputs)
    ↓
Model training (if applicable)
    ↓
Model validation (offline metrics meet thresholds)
    ↓
Model integration tests (serves correctly)
    ↓
Push to staging
    ↓
Smoke test in staging
    ↓
Manual approval (or auto for low-risk)
    ↓
Canary deployment to production
    ↓
Promote if metrics hold
```

### 12.3 Tests Specific to ML

**Data tests:**
```python
def test_no_target_leakage():
    train_aucs = single_feature_aucs(X_train, y_train)
    assert train_aucs.max() < 0.95, f"Possible leakage: {train_aucs.idxmax()}"

def test_feature_distributions_match_reference():
    for col in features:
        psi = compute_psi(reference[col], current[col])
        assert psi < 0.25, f"Drift detected in {col}: PSI={psi}"

def test_no_null_in_required_fields():
    for col in required_features:
        assert df[col].isna().sum() == 0
```

**Model tests:**
```python
def test_model_meets_minimum_performance():
    score = evaluate(model, X_test, y_test)
    assert score >= 0.85, f"Below threshold: {score}"

def test_no_fairness_regression():
    for group in protected_groups:
        score_group = evaluate(model, X_test[X_test.group == group], y_test[X_test.group == group])
        assert abs(score_group - overall_score) < 0.05

def test_inference_latency():
    p99 = benchmark_inference(model, sample_inputs)
    assert p99 < 50, f"Latency too high: {p99}ms"
```

**Integration tests:**
```python
def test_end_to_end_prediction():
    request = {...}
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    assert "prediction" in response.json()
```

### 12.4 GitOps for ML

Configuration, infrastructure, and deployment specs all live in Git. Changes go through PR review. The deployment system reconciles to whatever's in Git.

**Pattern:**
- Code repo: training pipeline, model code, tests
- Config repo: model configs, feature definitions, deployment specs
- Infra repo: terraform / k8s manifests

A PR to config triggers deployment. A PR to code triggers training.

### 12.5 Tools

| Tool | Use |
|---|---|
| **GitHub Actions / GitLab CI / Jenkins** | General CI/CD |
| **Argo Workflows / Tekton** | Kubernetes-native pipelines |
| **Kubeflow Pipelines** | ML-specific orchestration |
| **MLflow Recipes** | ML pipeline templates |
| **DVC** | Data and model version control |
| **CML** | CI/CD for ML (Iterative.ai) |

---

## 13. Cost Optimization

> The lesson every team learns the expensive way: a $50K/month inference bill can usually be cut to $5K/month with focused optimization. The optimizations are unglamorous but high-leverage.

### 13.1 Where Costs Live

| Cost Center | Driver |
|---|---|
| **Training compute** | GPU hours × cluster size × frequency |
| **Inference compute** | Requests/sec × cost per request |
| **Data storage** | Volume × tier × retention |
| **Data transfer** | Cross-region, egress |
| **Feature store** | Online store capacity, materialization compute |
| **Monitoring** | Logging volume, dashboard infrastructure |
| **LLM API calls** | Tokens × rate |

### 13.2 Training Cost Optimization

| Technique | Savings |
|---|---|
| **Spot / preemptible instances** | 60-90% off on-demand |
| **Right-sizing GPUs** | Match GPU to model size (A10 vs A100) |
| **Mixed precision (fp16/bf16)** | 2x speedup, half the memory |
| **Distributed training only when needed** | Communication overhead can dominate |
| **Checkpoint + resume** | Don't restart on failure |
| **Hyperparameter optimization with early stopping** | Don't run bad configs to completion |
| **Cache intermediate computations** | Avoid recomputing features for every experiment |

### 13.3 Inference Cost Optimization

| Technique | Effect |
|---|---|
| **Model quantization (int8, int4)** | 2-4x speedup, minimal accuracy loss for most tasks |
| **Distillation** | Smaller model trained to mimic large model |
| **Pruning** | Remove unimportant weights |
| **ONNX / TensorRT compilation** | 2-5x speedup vs naive PyTorch |
| **Dynamic batching** | Better GPU utilization |
| **Caching repeated inputs** | Avoid redundant inference |
| **CPU vs GPU appropriate** | Don't GPU what CPU can do cheaply |
| **Move to batch where possible** | Batch is 10-100x cheaper than real-time |
| **Reduce model size** | Smaller is cheaper, often acceptable accuracy |

### 13.4 The Hierarchy of Cost Wins

```
Tier 1 — High impact, low effort:
  - Move workloads to spot instances
  - Right-size machines (most are oversized)
  - Reduce retention on log/data storage
  - Cache aggressively

Tier 2 — High impact, moderate effort:
  - Quantize models to int8
  - Move from real-time to batch where possible
  - Compile models (ONNX, TensorRT)
  - Implement dynamic batching

Tier 3 — High impact, high effort:
  - Distill large models to smaller ones
  - Custom inference servers
  - Model architecture changes for efficiency
  - Hardware-specific optimization
```

Most teams have done little of Tier 1 before reaching for Tier 3.

### 13.5 LLM-Specific Cost Considerations

For LLM applications:

- **Choose the right model size.** GPT-4 for everything is wasteful; mix small + large.
- **Cache prompts.** Identical prompts return identical outputs (when temperature=0); cache them.
- **Reduce token usage.** Shorter prompts, summarize chat history, structured outputs.
- **Batch where possible.** Many providers offer batch APIs at 50% cost.
- **Self-host smaller models.** Once volume justifies, vLLM + Llama can be 10x cheaper than APIs.
- **Use fine-tuned smaller models** instead of prompted large ones for repetitive tasks.

### 13.6 The Cost Per Prediction Metric

Track:
```
Cost per prediction = (Compute cost + Storage cost + Network cost) / Predictions served
```

Most teams have no idea what this is. Computing it monthly creates accountability and surfaces optimization opportunities.

### 13.7 The Real Trade-off

There's almost always a triangle:

```
        Accuracy
         /\
        /  \
       /    \
Latency ─── Cost
```

Better in any one dimension usually trades off against the others. Production ML is the discipline of finding the right point in the triangle for each use case.

---

## 14. Security & Governance

### 14.1 Threat Categories

| Threat | Description |
|---|---|
| **Model theft** | Adversary extracts model via API queries |
| **Adversarial attacks** | Crafted inputs cause misclassification |
| **Data poisoning** | Malicious data injected into training |
| **Prompt injection** (LLM) | Inputs manipulate LLM behavior |
| **Training data extraction** | Memorized data recovered from model |
| **Membership inference** | Determine if a record was in training data |

### 14.2 Production Security Baselines

- [ ] Authentication on all serving endpoints
- [ ] Rate limiting per client
- [ ] Input validation (reject malformed, bounded length)
- [ ] Output filtering (PII redaction for LLMs)
- [ ] Audit log of all predictions
- [ ] Encrypted data in transit and at rest
- [ ] Secret management (no hardcoded credentials)
- [ ] Least-privilege access to model artifacts and training data
- [ ] Network segmentation between training and serving

### 14.3 Adversarial Robustness

For high-stakes models:
- **Adversarial training** — train on adversarial examples
- **Input preprocessing** — randomization, normalization
- **Defensive distillation** — train second model on softened predictions
- **Detection** — train a model to detect adversarial inputs

Most production models don't need explicit adversarial defenses, but high-stakes ones (security, finance, autonomous systems) do.

### 14.4 Governance

| Practice | Purpose |
|---|---|
| **Model approval process** | Senior review before production |
| **Bias audits** | Performance across protected attributes |
| **Documentation** | Model cards, decision logs |
| **Audit trail** | Every prediction logged |
| **Right to explanation** | Can you justify a specific prediction? |
| **Decommissioning** | Process for retiring models |

### 14.5 Regulatory Compliance (2025+ Landscape)

| Regulation | Scope |
|---|---|
| **EU AI Act** | Risk-based AI regulation, applies to EU markets |
| **GDPR Article 22** | Right not to be subject to solely automated decisions |
| **NYC Local Law 144** | Hiring algorithm audits |
| **Colorado AI Act** | Algorithmic discrimination in consequential decisions |
| **NIST AI RMF** | Voluntary risk management framework |

Compliance is becoming a real constraint, not a theoretical concern. Build documentation and audit trails into the pipeline from day one — they're expensive to retrofit.

---

## 15. LLM Ops — The Modern Twist

> LLMs share most MLOps principles with classical ML but introduce new operational concerns. This section covers what's different.

### 15.1 What Changes With LLMs

| Concern | Classical ML | LLM |
|---|---|---|
| **Training** | Done in-house | Often pretrained externally, fine-tuned |
| **Inference cost** | Cents per million | Dollars per million tokens |
| **Latency** | Predictable | Variable, depends on output length |
| **Determinism** | Usually deterministic | Often stochastic (temperature > 0) |
| **Output validation** | Score thresholds | Structured output validation, content moderation |
| **Prompt as feature** | N/A | Prompts ARE part of the system |
| **Evaluation** | Standard metrics | LLM-as-judge, eval sets, human review |
| **Failure modes** | Wrong prediction | Hallucination, prompt injection, jailbreak |

### 15.2 Prompt Management

Prompts are code. Version, test, and deploy them with the same rigor:

- **Prompt registry** — central repository, versioned
- **Prompt tests** — regression tests on eval sets
- **Prompt deployment** — staged rollout, A/B testable
- **Prompt monitoring** — track per-prompt performance

### 15.3 LLM Application Patterns

| Pattern | Description |
|---|---|
| **Zero-shot prompting** | Prompt the model directly, no examples |
| **Few-shot prompting** | Include examples in the prompt |
| **RAG (Retrieval-Augmented Generation)** | Retrieve context, ground generation in it |
| **Fine-tuning** | Adapt model weights to a specific task |
| **Agentic** | LLM decides actions, calls tools |
| **Function calling** | Structured output for tool invocation |

### 15.4 RAG Operations

A production RAG system has multiple components, each needing operational discipline:

```
Document ingestion → Chunking → Embedding → Vector store
                                                  ↓
Query → Embedding → Retrieval → Reranking → LLM context → Generation
```

Each component can fail independently:
- Stale documents (ingestion lag)
- Bad chunking (irrelevant retrievals)
- Embedding model drift
- Vector store latency
- Wrong retrievals (recall failure)
- Hallucination despite correct context

**Monitor each stage** — not just final output quality.

### 15.5 LLM Evaluation

Traditional ML metrics rarely apply directly. Common approaches:

| Method | Notes |
|---|---|
| **Exact match / similarity** | For factual tasks |
| **Reference-based** (ROUGE, BLEU) | For summarization, translation |
| **LLM-as-judge** | Another LLM scores outputs |
| **Human evaluation** | Gold standard but expensive |
| **Task-specific benchmarks** | MMLU, HumanEval, etc. |
| **Eval sets** | Custom test cases for your task |
| **A/B testing** | Online comparison |

**Build an eval set early.** Curated test cases with expected outputs are more valuable than any benchmark for production LLM work.

### 15.6 LLM Safety & Content Moderation

- **Input filtering** — reject prompts that violate policies
- **Output filtering** — scan generations for unsafe content
- **Prompt injection defense** — detect attempts to override system instructions
- **Jailbreak monitoring** — track adversarial usage patterns
- **PII detection / redaction** — both directions
- **Hallucination detection** — fact-checking layer

### 15.7 LLM Cost Patterns

| Cost lever | Effect |
|---|---|
| **Model choice** | 10-100x cost difference between tiers |
| **Token count** | Direct cost driver — minimize both directions |
| **Cache hit rate** | Repeated prompts are free if cached |
| **Batch APIs** | 50% discount typical, latency tradeoff |
| **Self-hosting** | Worth it past a volume threshold |
| **Routing** | Easy queries → small model, hard → large |

### 15.8 The LLM Operations Stack

| Layer | Tools |
|---|---|
| **Foundation models** | OpenAI, Anthropic, Google, AWS Bedrock, Llama, Mistral |
| **Hosting** | OpenAI API, Anthropic, Together, Fireworks, Replicate, vLLM self-host |
| **Frameworks** | LangChain, LlamaIndex, DSPy, Haystack |
| **Vector stores** | Pinecone, Weaviate, Chroma, pgvector, Qdrant |
| **Observability** | LangSmith, Langfuse, Helicone, Phoenix |
| **Evaluation** | Ragas, DeepEval, Promptfoo, OpenAI Evals |
| **Guardrails** | NeMo Guardrails, Guardrails AI, LlamaGuard |

---

## 16. The MLOps Workflow

### 16.1 The Full Lifecycle

```
1. SCOPE         — Business problem, success metric, constraints
2. DATA          — Acquire, profile, audit, version  (see Data.md)
3. FEATURES      — Engineer, validate, store  (see Features.md)
4. MODEL         — Train, evaluate, tune  (see Algorithms.md)
5. AUDIT         — Verify before deployment  (see Audit.md)
6. PACKAGE       — Container, dependencies, artifacts
7. DEPLOY        — Shadow → canary → A/B → production
8. MONITOR       — Data, features, model, business
9. RETRAIN       — Triggered by drift or schedule
10. ITERATE      — Continuous improvement
11. DECOMMISSION — Sunset when no longer needed
```

### 16.2 The MLOps Checklist (Production Readiness)

Before any model serves real traffic:

```
□ Model artifact versioned and registered
□ Training reproducible from registry metadata
□ Test set performance meets agreed threshold
□ Guardrail metrics (fairness, latency, cost) pass
□ Inference latency within SLO
□ Container builds and serves correctly
□ Health and readiness endpoints implemented
□ Feature parity verified between training and serving
□ Logging captures predictions and features
□ Drift monitoring is set up against training reference
□ Performance monitoring is set up (with ground truth lag)
□ Rollback path is tested
□ Fallback behavior is defined
□ Alerts are configured and routed
□ Runbook exists for common failure modes
□ Postmortem template is ready
□ Retraining trigger criteria are documented
□ Model card / documentation is complete
□ Security review passed
□ Compliance review passed (if applicable)
□ On-call rotation knows about this model
```

If any box is unchecked, the model isn't ready.

### 16.3 The Decommissioning Process

Models don't live forever. Standard process:

```
1. Announce sunset timeline (typically 90 days)
2. Notify downstream consumers
3. Migrate consumers to replacement
4. Reduce traffic gradually
5. Stop serving
6. Archive model + metadata
7. Update documentation
8. Retain audit logs per retention policy
```

The audit trail must survive decommissioning. "What did the model predict on 2024-03-15?" should be answerable years later for compliance.

---

## 17. Implementation Quick Reference

### Minimal Production Stack (Solo / Small Team)

```python
# Training: scikit-learn / lightgbm / pytorch
# Tracking: MLflow
# Serving: FastAPI + Docker
# Monitoring: Prometheus + Grafana + custom drift checks
# Storage: PostgreSQL + S3
# Orchestration: cron + scripts (until you outgrow it)
```

### Mid-Scale Stack (Multiple Models, Single Team)

```python
# Training: same + experiment tracking via MLflow/W&B
# Feature store: Feast (open source)
# Serving: BentoML or KServe on Kubernetes
# Monitoring: Evidently or Arize
# Orchestration: Airflow / Prefect / Dagster
# Storage: feature warehouse + S3 + Redis (online)
# Registry: MLflow Model Registry
```

### Large-Scale Stack (Many Models, Multiple Teams)

```python
# Training: managed platform (Vertex AI / SageMaker / Databricks)
# Feature store: Tecton / Hopsworks / custom
# Serving: managed (SageMaker Endpoints / Vertex) or Kubeflow
# Monitoring: Arize / Fiddler / WhyLabs / internal platform
# Orchestration: Argo + GitOps
# Registry: integrated with platform
# Compliance: integrated audit tooling
```

### FastAPI Inference Skeleton

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import time

app = FastAPI()
model = joblib.load("model.pkl")
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: str

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/readyz")
def ready():
    return {"status": "ready", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = time.time()
    try:
        features = preprocess(request)
        prediction = model.predict(features)[0]
        latency_ms = (time.time() - start) * 1000
        
        # Log for monitoring (async in production)
        logger.info({
            "event": "prediction",
            "input": request.dict(),
            "prediction": float(prediction),
            "latency_ms": latency_ms,
            "model_version": MODEL_VERSION,
        })
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=MODEL_VERSION,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Drift Detection Script

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

def daily_drift_check(reference_path, current_path, output_path):
    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(current_path)
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    
    result = report.as_dict()
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        # Trigger alert (PagerDuty, Slack, etc.)
        send_alert("Data drift detected in production features")
    
    report.save_html(output_path)
    return drift_detected
```

### CI Pipeline Snippet (GitHub Actions)

```yaml
name: ML Pipeline

on:
  pull_request:
    paths: ['model/**', 'features/**']

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint
        run: ruff check .
      - name: Data quality tests
        run: pytest tests/data/
      - name: Feature pipeline tests
        run: pytest tests/features/
      - name: Train model
        run: python train.py --config configs/ci.yaml
      - name: Validate model
        run: pytest tests/model/
      - name: Check metric thresholds
        run: python scripts/check_thresholds.py
      - name: Build container
        run: docker build -t model:${{ github.sha }} .
      - name: Integration tests
        run: pytest tests/integration/
```

---

## Summary — The 10 MLOps Principles That Actually Matter

If you remember nothing else, internalize these 10:

1. **A model is a system, not an artifact.** Operate the whole system, not just the binary.
2. **Production parity is non-negotiable.** Same code, features, logic in training and serving.
3. **Build detection before you need it.** Models degrade silently — instrumentation is the only warning.
4. **Never deploy without a rollback path.** Test the rollback before you need to use it.
5. **Canary, don't switch.** Every deployment starts at 1%, not 100%.
6. **Offline metrics are hypotheses; A/B tests are evidence.** They often disagree.
7. **Define retraining triggers before deployment.** Not after problems appear.
8. **Costs compound silently.** Track cost per prediction; optimize before it hurts.
9. **Logs and audit trails are non-optional.** Every prediction is auditable, indefinitely.
10. **Maturity is earned, not bought.** Tools don't replace practices.

The difference between a model that works in a notebook and a model that survives a year of production is not a better algorithm. It's discipline applied consistently across every layer of this file.

---

*`MLOps.md` — Version 1.0*
*Scope: Comprehensive reference for production ML operations — from deployment through decommissioning*
*Companion to `Metrics.md`, `Audit.md`, `Algorithms.md`, `Features.md`, `Data.md`*
*Use this as the single source of truth for "how do we run this model in production without it falling over?"*
