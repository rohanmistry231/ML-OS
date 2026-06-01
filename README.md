# Top AI / ML Learning Material

### A complete, opinionated reference library for machine learning and AI — every layer, end to end.

> **What this is:** Eight deep, cross-referenced reference files covering the full surface area of modern ML work — from the mathematics that powers it, through the data and features that feed it, the algorithms that fit it, the metrics that measure it, the audits that verify it, all the way to the production systems that operate it.
>
> **What this isn't:** A tutorial series. A blog. A summary. This is a working reference — written to be consulted by section when you hit a real problem, not read cover-to-cover.
>
> **Who it's for:** Practitioners building production ML. Engineers transitioning into ML. Senior ICs who want a single source of truth they can drop into any project. Teams that want a shared vocabulary across data, features, models, evaluation, and operations.

---

## The 8 Files

| # | File | What It Covers | Approx Lines |
|---|---|---|---|
| 1 | **[Math.md](./Math.md)** | The mathematics of ML — linear algebra, calculus, probability, statistics, optimization, information theory, numerical stability | ~1,885 |
| 2 | **[Data.md](./Data.md)** | The data layer — acquisition, profiling, splits, sampling, imbalance, labeling, drift, versioning, pipelines, privacy | ~1,673 |
| 3 | **[Features.md](./Features.md)** | Feature engineering — by data type, every encoding, target encoding done safely, leakage patterns, feature stores | ~1,772 |
| 4 | **[Algorithms.md](./Algorithms.md)** | Algorithm reference — every major family, when to use what, failure modes, from linear models through transformers | ~1,458 |
| 5 | **[Metrics.md](./Metrics.md)** | Evaluation metrics — classification, regression, ranking, vision, NLP, calibration, fairness, with decision trees | ~1,445 |
| 6 | **[Audit.md](./Audit.md)** | Pipeline auditing — every stage, every failure point, leakage hunting, the discipline that catches silent disasters | ~1,735 |
| 7 | **[MLOps.md](./MLOps.md)** | Production operations — deployment patterns, monitoring, A/B testing, retraining, rollback, cost, LLM ops | ~1,828 |
| 8 | **README.md** | This file — index, dependencies, reading paths, philosophy | — |

**Total: ~11,800 lines of opinionated, cross-referenced ML reference material.**

---

## The Mental Model

Each file owns one **layer** of an ML system. Together they form a complete stack:

```
   ┌──────────────────────────────────────────────────────┐
   │                       Math.md                        │  ← The language
   │      (linear algebra, calculus, probability, ...)    │
   └──────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────────────────────────────────┐
   │                       Data.md                        │  ← The foundation
   │   (acquisition, splits, quality, imbalance, drift)   │
   └──────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────────────────────────────────┐
   │                     Features.md                      │  ← The signal
   │       (engineering, encoding, leakage prevention)    │
   └──────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────────────────────────────────┐
   │                    Algorithms.md                     │  ← The fit
   │  (linear → trees → boosting → neural → transformers) │
   └──────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────────────────────────────────┐
   │                     Metrics.md                       │  ← The measure
   │       (how to score what you built, honestly)        │
   └──────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────────────────────────────────┐
   │                      Audit.md                        │  ← The verification
   │   (every stage gate, leakage hunts, severity tiers)  │
   └──────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────────────────────────────────┐
   │                     MLOps.md                         │  ← The operation
   │   (deploy, monitor, A/B, retrain, rollback, govern)  │
   └──────────────────────────────────────────────────────┘
```

Each layer assumes the one above is sound. **A model trained on un-audited data with leaky features will produce confidently wrong predictions no matter how sophisticated the algorithm.** The order matters.

---

## Reading Paths

### "I'm new to ML, where do I start?"
```
Math.md (Tier 1 sections only) → Data.md → Features.md → Algorithms.md → Metrics.md
```
Skip MLOps and Audit until you've trained your first few models. Add them when you're ready to ship.

### "I'm building my first production ML system"
```
Audit.md → Data.md → Features.md → MLOps.md
```
The audit file gives you the checklist. Data and features cover the highest-leverage work. MLOps gives you the deployment playbook.

### "I'm debugging a broken model"
```
Audit.md §16 (leakage) → Features.md §17 (leakage patterns) → Data.md §6 → Math.md §16 (failure modes)
```
The four files form a debugging gauntlet for almost every silent-failure mode in ML.

### "I'm choosing a model for a new problem"
```
Algorithms.md §2 (decision trees) → Metrics.md §1 (decision tree) → Features.md §2
```
Match algorithm to data shape, metric to problem type, features to algorithm.

### "I'm preparing for an ML interview"
```
Math.md → Algorithms.md → Metrics.md → Features.md → MLOps.md (LLM era questions)
```
Math first, because most "why" questions trace to math. Algorithms next for breadth. The rest for depth on common interview themes.

### "I want to understand modern AI deeply"
```
Math.md (all tiers) → Algorithms.md §10-11 → MLOps.md §15 → Math.md §12 (diffusion) and §15.5
```
The math sections on calculus, linear algebra, and SDEs unlock everything from transformers to diffusion models.

---

## How These Files Are Written

Every file follows the same structure, deliberately:

1. **Three Laws** at the top — the philosophical core that the rest derives from
2. **Decision trees** — for finding what you need fast
3. **Sections with depth** — pros, cons, when to use, when to avoid, failure modes
4. **Cross-references** — `Audit.md §16`, `Features.md §17`, etc. — the files are linked by section
5. **"10 principles" summary** at the end — if you remember nothing else
6. **Implementation quick reference** — actual code you can copy

The tone is **opinionated, not neutral.** When the field has converged on an answer (e.g., gradient boosting beats deep learning on tabular data), the files say so. When there's genuine disagreement, the files surface the tradeoffs.

The depth is **operational, not academic.** You won't find proofs. You will find: what fails, why it fails, how to detect the failure, and how to fix it.

---

## The Three Laws That Run Through All Eight Files

Each file has its own Three Laws — but a few principles repeat across all of them:

```
NOTHING IS GIVEN. EVERYTHING IS PRODUCED.
    Data is produced, labels are produced, features are produced,
    metrics are produced. Production means it can be wrong.
    Audit accordingly.

THE INFERENCE BOUNDARY IS SACRED.
    Every decision must use only information available at the
    moment of prediction. Violating this is leakage — the silent
    killer of ML systems.

DETECTION OVER TRUST.
    Models don't warn you when they degrade. Data doesn't warn
    you when it corrupts. Build the instrumentation, or fly blind.
```

If you internalize nothing else from these files, internalize these three.

---

## What Each File Won't Cover (Deliberate Scope Limits)

To keep the files dense and useful, each has hard scope limits:

| File | Doesn't Cover | Why |
|---|---|---|
| **Math.md** | Proofs, measure theory beyond ML relevance | Operational math, not academic math |
| **Data.md** | Specific database tools (Snowflake vs BigQuery) | Vendor-neutral principles |
| **Features.md** | Tutorial-style "Pandas 101" | Assumes basic data manipulation fluency |
| **Algorithms.md** | Algorithm internals (e.g., how LightGBM's histograms work in C++) | When-to-use focus, not implementation |
| **Metrics.md** | Custom domain metrics (e.g., RecSys diversity) | Covers the metric types; you instantiate for your domain |
| **Audit.md** | Specific compliance frameworks (SOC 2, etc.) | Domain-agnostic principles |
| **MLOps.md** | Specific cloud service walkthroughs | Patterns, not vendor docs |

For specific tool documentation, use the tool's docs. For the underlying principle of *why* you're using that tool, use these files.

---

## When to Add Files

These eight cover the **full surface area** of practical ML. Adding more files risks redundancy. Considered alternatives:

- ❌ `Leakage.md` — already lives correctly inside Audit.md §16 and Features.md §17
- ❌ `Baselines.md` — covered in Audit.md and Algorithms.md
- ❌ `Imbalance.md` — Data.md §7 covers it thoroughly
- ❌ `TimeSeries.md` — distributed across Algorithms.md §15, Features.md §9, Data.md §14
- ❌ `Interpretability.md` — Features.md §16 covers SHAP/permutation; Algorithms.md notes interpretability per algorithm
- ❌ `Foundations.md` — Math.md absorbed this

Possible future additions only if there's clear demand:
- `LLMs.md` — deeper than MLOps.md §15 (prompting strategies, RAG patterns, fine-tuning, agents)
- `Experiments.md` — A/B testing depth beyond MLOps.md §7
- `Causal.md` — causal inference for ML practitioners

But the **core resource is complete at 8 files.**

---

## How to Use These Files Day-to-Day

### Drop into every ML repo
Copy these files (or symlink them) into the `docs/` folder of every ML repo. Reference specific sections in PR descriptions: "Per Audit.md §6, this preprocessing must be inside the pipeline, not pre-split."

### Pre-commit reading
Before starting a new ML project, read:
- Audit.md §2 (problem definition checklist)
- Data.md §3 (split decisions)
- MLOps.md §16 (production readiness checklist)

### Onboarding new team members
Hand them this README. They navigate to whichever file matches their first task. The Three Laws in each file teach the team's epistemics; the implementation snippets teach the team's idioms.

### Stuck on a hard problem
The cross-references are the search engine. "Performance dropped after deployment" → MLOps.md §6 (monitoring) → Data.md §9 (drift) → Audit.md §13 (deployment readiness audit) → likely root cause found.

### Periodic re-reads
Once a quarter, re-read one file. They're written to reward re-reading — every section has tradeoffs you didn't notice the first time, because you didn't have the operational experience yet to recognize them.

---

## The Files As a System

Three things distinguish this set from any individual file or blog series:

1. **Consistency.** Same structure, same tone, same level of opinionated depth. Read one file and you know how to navigate the others.

2. **Cross-referencing.** The files link to each other by section. Following a leakage thread from Audit.md to Features.md to Data.md gives you the same problem from three operational angles.

3. **Completeness.** Every layer of an ML system has exactly one file that owns it. No gaps. No redundancy. If you have a question, exactly one file is the right place to look.

This is the difference between **a library** and **a pile of documents**. A library has architecture.

---

## Maintenance

Last refreshed: **May 2026**

This material is opinionated about the current state of the field — which is changing fast. The half-life of specific tool recommendations (e.g., feature store choices, LLM eval frameworks) is short. The half-life of the underlying principles (e.g., split before preprocessing, never deploy without rollback) is permanent.

When you find a stale recommendation, update it locally and propagate. The principles will keep working.

---

## Credits

Compiled by **Rohan Mistry**.

Built as a complete reference for serious practitioners — the kind of resource that should exist but usually doesn't, because writing it takes longer than building the models.

If you found value in these files: drop them into a teammate's repo. The compounding returns of shared vocabulary across a team is more valuable than any individual technique covered here.

---

*Read deeply. Apply ruthlessly. Audit relentlessly.*
