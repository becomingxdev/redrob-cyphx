# 🤖 REDROB CyphX — AI Candidate Ranking Engine

> **Redrob Intelligent Candidate Discovery & Ranking Challenge**
> Team: `cyphx` · Target Role: *Senior AI Engineer (Founding Team)*

A fully deterministic, CPU-only candidate ranking pipeline that scores and ranks candidates from a ~100K JSONL dataset against a structured Job Description — producing a top-100 submission CSV within the competition's 5-minute compute budget.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Scoring Components](#scoring-components)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Validation](#validation)
- [Compute Constraints](#compute-constraints)

---

## Overview

This system implements a multi-layered candidate ranking engine for the [Redrob Hackathon](https://redrob.io). It is designed to:

- **Read profiles deeply**, not just surface-level skill matching — detecting hidden fit, research-vs-production signals, and career trajectory quality.
- **Handle behavioral signals** (23 Redrob platform signals including login recency, recruiter response rate, notice period) as first-class ranking inputs.
- **Detect and demote honeypot candidates** — profiles with impossible or contradictory data designed to catch keyword-only rankers.
- **Run entirely on CPU** with no external API calls, within a ≤16 GB RAM, ≤5 min wall-clock budget.

---

## Architecture

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Feature Extraction                    │
│  Title · Skills · Experience · Education · Location     │
│  Career Trajectory                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   Evidence Layer                        │
│  Verifier · Consistency Analyzer · Confidence Calc      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Scoring Engines                        │
│  Title · Skills · Experience · Education · Behavior     │
│  Location · Penalties · Honeypot Detection              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Composite Score + Ranking                  │
│  Weighted combination → deterministic rank 1..N         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Reason Generation (Top-100 only)              │
│  Per-candidate natural-language justifications          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              model_output/submission.csv
               (Top 100 candidates ranked)
```

---

## Pipeline Stages

The `main.py` entrypoint runs four sequential stages:

| Stage | Description |
|-------|-------------|
| **1 / 4 — Scoring** | Feature extraction + full scoring for every candidate in the dataset |
| **2 / 4 — Ranking** | Deterministic sort by composite score with tie-breaking |
| **3 / 4 — Reason Generation** | Natural-language justifications for top 100 + 20 buffer candidates |
| **4 / 4 — CSV Export** | Write exactly 100 rows to `model_output/submission.csv` |

Progress is logged to stdout with heartbeats every 100 candidates (or every 5 seconds), including ETA.

---

## Scoring Components

Composite score = weighted sum of 6 components (after penalty and honeypot adjustments):

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Skills** | 30% | Required skill coverage, preferred skill bonus, domain-mismatch penalty |
| **Title** | 25% | Semantic match to "Senior AI Engineer", seniority level, career trajectory |
| **Experience** | 25% | Years of experience vs target (7 YoE), production vs research signals |
| **Behavior** | 10% | 23 Redrob platform signals: recency, response rate, availability, engagement |
| **Education** | 5% | Degree level and field relevance (tiebreaker only) |
| **Location** | 5% | City preference (Pune/Noida > Tier-1 India > relocation willingness) |

### Evidence & Confidence Layer

Scores are modulated by a cross-profile evidence layer:

- **EvidenceVerifier** — checks skill claims against multi-source corroboration (assessment scores, endorsements, GitHub activity).
- **ConsistencyAnalyzer** — flags internal inconsistencies (e.g., impossible tenure durations, overlapping roles).
- **ConfidenceCalculator** — produces a `[0, 1]` confidence weight applied to the final composite.

### Penalties & Honeypot Detection

- **Penalties** — applied for consulting-only careers (TCS/Infosys/etc.), domain mismatches (CV/speech/robotics without NLP), framework-only skill profiles (LangChain-only), and stale engagement.
- **Honeypot Detection** — candidates with impossible profiles (e.g., 8+ years at a 3-year-old company, implausibly large skill counts) are hard-capped at a composite score of 20.

---

## Project Structure

```
redrob-cyphx/
├── main.py                        # Pipeline entrypoint
├── validate_submission.py         # Local submission validator
│
├── config/
│   ├── jd_config.yaml             # JD parameters: title, YoE, skills, location, companies
│   ├── weights.yaml               # Component scoring weights
│   ├── evidence_rules.yaml        # Evidence verification rules
│   ├── skill_normalization.yaml   # Skill alias/normalization map
│   ├── skill_rules.yaml           # Skill extraction rules
│   └── title_rules.yaml           # Title normalization rules
│
├── data/
│   ├── candidates.jsonl           # Full dataset (~100K candidates)
│   ├── test.jsonl                 # Small test slice
│   ├── sample_candidates.json     # Sample for local dev/sandbox
│   ├── sample_submission.csv      # Format reference (not a quality ranking)
│   └── candidate_schema.json      # JSON schema for candidate profiles
│
├── docs/
│   ├── job_description.md         # Full JD text
│   ├── submission_spec.md         # Hackathon submission specification
│   ├── redrob_signals_doc.md      # Reference for the 23 behavioral signals
│   └── submission_metadata_template.yaml
│
├── scripts/
│   └── explore_dataset.py         # Dataset exploration utility
│
├── src/
│   ├── jd_config.py               # JD config loader (singleton)
│   ├── models/
│   │   └── candidate.py           # Candidate dataclass
│   ├── parser/
│   │   └── loader.py              # JSONL streaming loader
│   ├── features/
│   │   ├── title.py               # Title feature extractor
│   │   ├── skills.py              # Skills extractor (multi-source)
│   │   ├── experience.py          # Experience feature extractor
│   │   ├── education.py           # Education feature extractor
│   │   ├── location.py            # Location feature extractor
│   │   └── career.py              # Career trajectory extractor
│   ├── evidence/
│   │   ├── verifier.py            # Multi-source evidence verifier
│   │   ├── consistency.py         # Profile consistency analyzer
│   │   └── confidence.py          # Confidence score calculator
│   ├── scoring/
│   │   ├── title_score.py
│   │   ├── skill_score.py
│   │   ├── experience_score.py
│   │   ├── education_score.py
│   │   ├── behavior_score.py
│   │   ├── location_score.py
│   │   ├── penalties.py
│   │   ├── honeypot.py
│   │   └── composite.py
│   ├── output/
│   │   ├── ranking.py             # Deterministic ranking engine
│   │   └── csv_writer.py          # Submission CSV writer (top-100 only)
│   ├── reasoning/
│   │   └── generator.py           # Natural-language reason generator
│   └── utils/
│       └── banner.py              # Startup banner printer
│
└── model_output/
    └── submission.csv             # ← Generated output (top-100 ranking)
```

---

## Setup & Installation

### Prerequisites

- Python **3.11+**
- No GPU required — CPU only

### Install dependencies

```bash
pip install pyyaml
```

> All other dependencies use the Python standard library (`csv`, `json`, `pathlib`, `dataclasses`, `re`, `math`).

---

## Running the Pipeline

### Full run (entire `candidates.jsonl` dataset)

```bash
python main.py
```

Output is written to `model_output/submission.csv`.

### Validate submission locally

```bash
python validate_submission.py
```

Checks format, row count, rank uniqueness, score monotonicity, and candidate ID validity against the dataset.

---

## Configuration

All JD-specific parameters and scoring weights are externalized in `config/`. **No code changes are needed to adjust scoring behavior** — edit the YAML files instead.

### `config/jd_config.yaml`

| Key | Value | Purpose |
|-----|-------|---------|
| `target_title` | `Senior AI Engineer` | Title matching anchor |
| `target_yoe` | `7.0` | Ideal years of experience |
| `ideal_yoe_min / max` | `5.0 / 9.0` | Acceptable YoE range |
| `required_skills` | 22 skills | Core skill scoring (40pt lever) |
| `preferred_skills` | 19 skills | Nice-to-have bonus (15pt lever) |
| `anti_skills` | CV/speech/robotics | Domain mismatch signals |
| `preferred_cities` | Pune, Noida | Tier-1 location preference |
| `acceptable_cities` | Hyderabad, Bangalore, etc. | Tier-2 location |
| `it_services_companies` | TCS, Infosys, Wipro, etc. | Consulting-only penalty trigger |
| `top_tier_companies` | Google, Meta, Flipkart, etc. | Career quality bonus |

### `config/weights.yaml`

```yaml
title:      0.25
skills:     0.30
experience: 0.25
education:  0.05
behavior:   0.10
location:   0.05
```

---

## Output Format

The submission CSV (`model_output/submission.csv`) contains exactly **100 rows** (plus 1 header):

```csv
candidate_id,rank,score,reasoning
CAND_XXXXXXX,1,0.92,"Senior AI Engineer with 7 years at product companies; strong retrieval + embedding background; high engagement."
CAND_YYYYYYY,2,0.89,"6 years applied ML at Flipkart; shipped vector search at scale; Pune-based and open to work."
...
CAND_ZZZZZZZ,100,0.41,"Adjacent skills only; low behavioral engagement reduces effective availability."
```

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | string | `CAND_XXXXXXX` identifier |
| `rank` | int (1–100) | Rank position; 1 = best fit |
| `score` | float | Composite score (non-increasing with rank) |
| `reasoning` | string | 1–2 sentence justification referencing candidate-specific facts |

---

## Validation

Run the local validator before submitting:

```bash
python validate_submission.py
```

Checks enforced:
- Exactly 100 data rows
- Ranks 1–100, each appearing exactly once
- Scores non-increasing with rank
- All `candidate_id` values exist in `candidates.jsonl`
- No duplicate `candidate_id` values

---

## Compute Constraints

This pipeline is designed to operate within the hackathon's sandbox limits:

| Constraint | Limit | This pipeline |
|------------|-------|---------------|
| Runtime | ≤ 5 min | Rule-based, no model inference |
| RAM | ≤ 16 GB | Streaming JSONL loader, no full load |
| Compute | CPU only | No GPU usage anywhere |
| Network | Off | Zero external API calls |
| Disk | ≤ 5 GB | Only small YAML configs + output CSV |

---

## Tie-Breaking

When two candidates share the same composite score, ranks are broken deterministically by:

1. Higher **confidence** score
2. Higher **consistency** score
3. Higher **experience** score
4. **Candidate ID** ascending (lexicographic)

---

*Built for the Redrob Intelligent Candidate Discovery & Ranking Hackathon.*
