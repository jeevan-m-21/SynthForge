# SynthForge

**Privacy-Preserving Synthetic Tabular Data Platform**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/jeevan-m-21/SynthForge)
[![Automated Tests](https://img.shields.io/badge/Tests-127%2F127%20Passed-success)](https://github.com/jeevan-m-21/SynthForge)

SynthForge is an end-to-end platform for generating, validating, and stress-testing privacy-preserving synthetic tabular datasets. It combines statistical copula modeling and deep generative neural networks (TVAE/CTGAN) with mathematical Differential Privacy noise mechanisms, an automated 5-pillar Quality & Trustworthiness audit, and empirical adversarial privacy threat simulations. Users can evaluate synthetic data fidelity and privacy resilience through an evidence-based assessment pipeline before downstream export.

```
Generation ──► Quality Evaluation ──► Privacy Attack Simulation ──► Evidence-Based Assessment
```

---

## Why SynthForge?

Organizations across healthcare, finance, and enterprise operations face a fundamental data-sharing challenge: raw tabular datasets frequently contain sensitive, proprietary, or regulated records that cannot be freely distributed across research teams, cross-functional partners, or external machine learning pipelines.

Traditional anonymization approaches like column suppression or coarse binning frequently degrade multi-variable relationship structures or remain vulnerable to linkage and re-identification attacks.

Synthetic data bridges this divide by generating artificial records that approximate the underlying statistical distributions, cross-column correlations, and predictive utility of the source data without exposing individual real records. However, synthetic data is **not automatically private or useful by default**. SynthForge provides a transparent framework that decouples and quantifies both **Data Fidelity** and **Privacy Protection**, ensuring practitioners have empirical evidence and statistical validation before deploying synthetic data into downstream workflows.

---

## Product Overview

SynthForge guides users through a clear six-stage workflow:

```
Upload Dataset
      ↓
Generate Synthetic Data
      ↓
Evaluate Fidelity
      ↓
Test Privacy
      ↓
Review Evidence
      ↓
Download Dataset
```

1. **Upload Dataset**: Ingest a custom CSV or select bundled sample datasets. The platform validates structure, infers semantic datatypes, and extracts statistical moments.
2. **Generate Synthetic Data**: Select a generation engine (Fast Copula vs. High-Fidelity TVAE/CTGAN) and optionally configure Differential Privacy epsilon and delta parameters.
3. **Evaluate Fidelity**: Run automated statistical hypothesis tests (Kolmogorov-Smirnov and Chi-Square) and correlation matrix alignment checks.
4. **Test Privacy**: Execute empirical attack simulations—including Membership Inference, Distance-Based Re-Identification, Attribute Inference, and exact record collision checks.
5. **Review Evidence**: Inspect decoupled scorecards, interactive distribution overlays, correlation heatmaps, and ML utility benchmarks (TSTR vs. TRTR).
6. **Download Dataset**: Export the generated synthetic dataset directly to CSV for downstream model training, pipeline testing, or analytical sharing.

---

## Key Features

### Generation
- **Automated Ingestion & Schema Intelligence**: Automated schema profiling, datatype inference (numeric, categorical, datetime, text), missing value tracking, and statistical summary extraction for custom CSV uploads and bundled samples.
- **Dual Generative Engines**:
  - *Fast Generation*: High-speed statistical Gaussian Copula modeling (~seconds execution on standard sample datasets).
  - *High-Fidelity Generation*: Deep Variational Autoencoder (TVAE / CTGAN via SDV) capturing non-linear feature interactions and multi-modal distributions.
- **Differential Privacy (DP)**: Post-generation Gaussian and Laplace noise injection with calibrated IQR-based sensitivity bounds, categorical randomized response, Rényi Differential Privacy (RDP) accounting, and privacy budget tracking.
- **Configurable Scale**: Generate target row counts up to configurable application safety bounds (`MAX_SYNTH_ROWS`).

### Quality Evaluation
- **Unified 5-Pillar Trust Framework**: Comprehensive scoring across Structural, Statistical, Relationship, ML Utility, and Privacy dimensions.
- **Statistical Hypothesis Testing**: Two-sample Kolmogorov-Smirnov tests for continuous features and Chi-Square goodness-of-fit tests for categorical distributions.
- **Cross-Attribute Relationship Alignment**: Covariance preservation and real-versus-synthetic correlation difference heatmaps (Pearson & Cramér's V).
- **Machine Learning Utility Benchmarking (TSTR)**: Train on Synthetic, Test on Real (TSTR) evaluation against Train on Real, Test on Real (TRTR) baselines across Random Forest and Gradient Boosting classifiers.

### Privacy & Threat Analysis
- **Decoupled Scoring**: Independent evaluation of Data Fidelity and Privacy Protection to reflect real-world privacy-utility trade-offs.
- **Adversarial Privacy Attack Suite**:
  - *Membership Inference Attacks (MIA)*: Evaluates whether a binary classifier trained on nearest-neighbor distance features can distinguish training members from holdout records, visualized via attack ROC curves.
  - *Distance-Based Re-Identification (Re-ID)*: Measures nearest-neighbor distances in normalized feature space to identify records at risk under quasi-identifier matching.
  - *Attribute Inference Defense*: Measures adversary prediction advantage over majority-class baselines when attempting to reconstruct sensitive columns.
  - *Exact Collision Defense*: Performs deterministic SHA-256 row hashing to verify whether real training rows were duplicated verbatim in the synthetic dataset.

### User Experience
- **Interactive Visualizations**: Dynamic Chart.js distribution selectors, correlation heatmaps, and threat radar charts.
- **Live Generation Progress**: Step-by-step progress modal tracking model training, sampling, and post-processing.
- **Theme Support**: Seamless Light, Dark, and System theme switching with zero layout shift.
- **Fully Responsive**: Optimized for mobile (375px), tablet (768px), and widescreen desktop (1280px+) viewports.

---

## Product Showcase

### Dashboard
<!-- Add screenshot here -->

### Synthesis Workflow
<!-- Add screenshot here -->

### Quality Report
<!-- Add screenshot here -->

### Privacy & Threat Analysis
<!-- Add screenshot here -->

---

## How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ Ingestion &     │ ──► │ Schema Profiling &   │ ──► │ Generative Synthesis  │
│ Validation      │     │ Moment Extraction    │     │ (Copula / TVAE + DP)  │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ Verified CSV    │ ◄── │ Threat Simulation &  │ ◄── │ 5-Pillar Quality      │
│ Export          │     │ Attack Evaluation    │     │ & ML Utility Audit    │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘
```

1. **Ingestion & Validation**: The input CSV is validated for structural format, row/column boundaries, non-empty content, and file size limits.
2. **Schema Profiling**: The system infers semantic types, extracts marginal distribution parameters (means, variances, quantiles, frequencies), and checks for identifier columns.
3. **Generative Synthesis**: The chosen generator fits the data distribution. When Differential Privacy is enabled, calibrated noise and randomized response are applied with RDP budget tracking.
4. **5-Pillar Quality Audit**: The generated table undergoes statistical distribution tests, correlation matrix comparison, and automated downstream ML classification (TSTR vs. TRTR).
5. **Privacy Threat Simulation**: Empirical attack modules evaluate membership inference vulnerability, nearest-neighbor proximity, and exact record collisions.
6. **Export & Evidence Reporting**: The platform renders interactive scorecards, distribution comparisons, letter grades, and provides direct CSV export.

---

## Architecture

SynthForge is built with a decoupled architecture separating compute services from the presentation layer:

```
┌────────────────────────────────────────────────────────┐
│               Frontend Presentation Layer              │
│    Vanilla HTML5 / CSS3 (Design Tokens) / JavaScript   │
│           Interactive Visualizations (Chart.js)        │
└───────────────────────────┬────────────────────────────┘
                            │ REST API (JSON)
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI Backend Server                │
│                   (backend/main.py)                    │
├───────────────────────────┬────────────────────────────┤
│   Data & Schema Service   │  Synthetic Generators      │
│ (backend/services/data)   │ (Statistical Copula, TVAE) │
├───────────────────────────┼────────────────────────────┤
│   Quality Evaluator &     │  Privacy & Attack Engine   │
│   Statistical Validators  │  (DP, MIA, Re-ID, Attr)    │
├───────────────────────────┴────────────────────────────┤
│             Local Storage & Job Persistence            │
│            (storage/uploads, generated, models)        │
└────────────────────────────────────────────────────────┘
```

- **Frontend**: Vanilla HTML5, modern CSS with custom design tokens, and modular ES6 JavaScript with Chart.js loaded from CDN. Zero frontend build toolchain or bundler required.
- **Backend API**: FastAPI asynchronous REST application with Pydantic v2 schemas and modular endpoint routers.
- **Generative & Analytical Services**: Python-based pipeline utilizing SDV, PyTorch, Scikit-Learn, SciPy, NumPy, and Pandas.
- **Storage Layer**: File-backed storage with atomic write-and-replace persistence (`JSONStore`) for uploaded datasets, generated CSVs, trained models, and job metadata.

---

## Generation Modes

| Mode | Underlying Method | Key Strengths | Recommended Use Cases |
| :--- | :--- | :--- | :--- |
| **Fast Generation** | Statistical Gaussian Copula | • Fast execution (~seconds on standard datasets)<br>• Preserves 1D marginals & linear correlations<br>• Lightweight computational footprint | • Rapid prototyping & UI testing<br>• ETL data pipeline validation<br>• Fast exploratory mockups |
| **High-Fidelity Generation** | Deep Variational Autoencoder (TVAE / CTGAN) | • Captures non-linear feature interactions<br>• Preserves multi-modal feature clusters<br>• Strong ML utility retention | • Downstream ML model training<br>• Complex multi-variable analytics<br>• Research workflows |

---

## Quality & Trustworthiness

SynthForge evaluates synthetic data through an automated **5-Pillar Quality Framework**. Data Fidelity and Privacy Protection are scored as decoupled dimensions, reflecting the empirical trade-off between statistical fidelity and privacy noise.

```
                  ┌─────────────────────────────────────┐
                  │      5-Pillar Trust Framework       │
                  └──────────────────┬──────────────────┘
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               │                               │
┌────▼─────────────┐       ┌─────────▼──────────┐       ┌────────────▼────┐
│ 1. Structural    │       │ 2. Statistical     │       │ 3. Relationship │
│    Fidelity      │       │    Fidelity        │       │    Fidelity     │
└──────────────────┘       └────────────────────┘       └─────────────────┘
                 ┌───────────────────┴───────────────────┐
                 │                                       │
       ┌─────────▼──────────┐                 ┌──────────▼──────────┐
       │ 4. ML Utility      │                 │ 5. Privacy          │
       │    (TSTR vs TRTR)  │                 │    Protection       │
       └────────────────────┘                 └─────────────────────┘
```

1. **Structural Fidelity**: Verifies column preservation, datatype alignment, nullability rate preservation, and category overlap.
2. **Statistical Fidelity**: Evaluates marginal probability distributions per feature using two-sample Kolmogorov-Smirnov (KS) tests for continuous attributes and Chi-Square goodness-of-fit tests for discrete attributes.
3. **Relationship Fidelity**: Computes pairwise correlation matrices across all columns (Pearson for continuous, Cramér's V for categorical) and measures the Frobenius norm difference between real and synthetic correlation matrices.
4. **ML Utility (TSTR Benchmarking)**: Trains predictive models exclusively on synthetic data and tests them on held-out real data (**Train on Synthetic, Test on Real — TSTR**). Compares accuracy, F1-score, and ROC-AUC against baseline models trained and tested on real data (**Train on Real, Test on Real — TRTR**).
5. **Privacy Protection**: Quantifies empirical defense resilience against simulated attacks and verifies zero exact record collisions against the source training set.

---

## Privacy Attack Suite

SynthForge provides an empirical adversarial attack suite to stress-test generated datasets against practical privacy threats before data release:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Empirical Attack Suite                          │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Membership Inference     │ Evaluates whether a classifier trained on   │
│ Attack (MIA)             │ nearest-neighbor distance features can      │
│                          │ distinguish members from holdout records.   │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Distance-Based           │ Measures nearest-neighbor distance in       │
│ Re-Identification        │ normalized feature space to identify        │
│                          │ records at risk under quasi-identifiers.    │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Attribute Inference      │ Tests whether sensitive target attributes   │
│ Defense                  │ can be predicted with higher accuracy than  │
│                          │ majority-class baseline guessing.           │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Exact Collision          │ Performs deterministic SHA-256 row hashing  │
│ Defense                  │ to verify that zero real training records   │
│                          │ were duplicated verbatim in output.         │
└──────────────────────────┴─────────────────────────────────────────────┘
```

> **Note on Privacy Guarantees**: Adversarial simulations represent empirical stress tests under specific attack assumptions. They complement formal mathematical Differential Privacy bounds ($\epsilon, \delta$) by evaluating practical model outputs.

---

## Technical Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5 / CSS3 / Vanilla JavaScript | Responsive user interface with custom CSS design tokens and zero bundler dependencies |
| **Data Visualization** | Chart.js | Interactive distribution charts, correlation heatmaps, ROC curves, and threat radar |
| **Backend API** | FastAPI / Uvicorn | Asynchronous RESTful API server with auto-generated OpenAPI documentation |
| **Data Validation** | Pydantic v2 / Python-multipart | Request schema validation, type enforcement, and streamed file uploads |
| **Generative Modeling** | SDV (Synthetic Data Vault) / PyTorch | Statistical Copula and deep learning TVAE / CTGAN tabular synthesizers |
| **Statistical & ML Engine** | Scikit-Learn / SciPy / NumPy / Pandas | KS tests, Chi-Square, correlation analysis, TSTR classification, and metric computation |
| **Testing & Quality** | Python `unittest` / Requests | Automated unit, security, concurrency, reproducibility, and E2E integration test suites |

---

## Testing & Verification

SynthForge maintains comprehensive automated test coverage across backend services and integration flows:

- **13 / 13** End-to-End Integration Tests (`test_e2e.py`)
- **114 / 114** Backend Service & Module Unit Tests
- **127 / 127** Total Automated Tests Passing

### Running the End-to-End Integration Suite
```bash
python test_e2e.py
```

### Running the Complete Backend Test Suite
```bash
python -m unittest test_dataset_profiler.py test_phase3_generation.py test_phase4_generalization.py test_phase7a_security.py test_phase7b_concurrency.py test_phase7c_reproducibility.py test_phase7d_job_recovery.py test_quality_evaluator.py test_schema_intelligence.py
```

---

## Security & Reliability Controls

SynthForge implements defense-in-depth measures across ingestion, compute, and presentation:

- **HTML Escaping & Output Sanitization**: Contextual escaping of user-provided content in dynamic tables and report outputs to mitigate XSS risks.
- **Upload Validation & Boundary Constraints**: File extension verification (`.csv`), chunked stream size limits (`MAX_UPLOAD_SIZE_MB`), and empty file rejection.
- **Path Traversal Protection**: Filename sanitization, null byte removal, Windows reserved name handling, and restricted storage boundaries.
- **Log & Secret Sanitization**: Redaction of raw sample records, PII fields, and internal path artifacts from execution logs.
- **Bounded Execution**: Configurable timeouts and bounded training parameters to mitigate resource exhaustion during model training.
- **Stale Job Recovery**: State reconciliation on server startup that identifies and handles interrupted background jobs.

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager
- Git

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/jeevan-m-21/SynthForge.git
cd SynthForge

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch SynthForge with a single command:
```bash
python run.py
```

Access the application interfaces:
- **Interactive Web Application**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Reference (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Project Structure

```
SynthForge/
├── backend/
│   ├── api/
│   │   ├── routes.py            # FastAPI REST route definitions and request handlers
│   │   └── schemas.py           # Pydantic v2 request/response schemas
│   ├── models/
│   │   └── database.py          # Thread-safe JSON file-backed persistence store
│   ├── services/
│   │   ├── attack_simulator.py  # Empirical privacy threat simulators (MIA, Re-ID, Attribute Inference)
│   │   ├── data_service.py      # CSV data loading, ingestion, and file management
│   │   ├── dataset_profiler.py  # Feature-level profiling, moments, and semantic types
│   │   ├── federated_learning.py# Federated synthesizer aggregation manager
│   │   ├── generator.py         # Statistical copula and deep learning (TVAE/CTGAN) generation
│   │   ├── ml_validator.py      # Downstream ML utility validation (TSTR vs. TRTR)
│   │   ├── privacy_engine.py    # Differential privacy noise mechanisms & RDP accounting
│   │   ├── quality_evaluator.py # 5-Pillar quality scoring and executive scorecard
│   │   ├── schema_intelligence.py# Type inference, identifier detection, and metadata
│   │   └── statistical_validator.py # KS-test and Chi-Square goodness-of-fit validators
│   ├── utils/
│   │   ├── logging_config.py    # Structured logging and audit trail helpers
│   │   └── security.py          # Filename sanitization, hashing, and token generation
│   ├── config.py                # Environment configuration, bounds, and directory paths
│   └── main.py                  # FastAPI application entry point & static mounts
├── frontend/
│   ├── css/
│   │   └── styles.css           # CSS design tokens (Light/Dark/System) & responsive layout
│   ├── js/
│   │   ├── api.js               # Client API wrapper with error handling
│   │   ├── app.js               # UI controller, state management, and event handlers
│   │   └── charts.js            # Chart.js renderers, heatmaps, and theme integration
│   └── index.html               # Single-page interface (Dashboard, Synthesize, Quality, Privacy)
├── data/
│   ├── sample/                  # Bundled clinical healthcare dataset
│   └── phase4_fixtures/         # Multi-domain evaluation datasets (e-commerce, finance)
├── storage/                     # Runtime uploads, generated datasets, and reports
├── requirements.txt             # Production dependencies
├── run.py                       # Application startup script
└── test_e2e.py                  # End-to-end integration test suite
```

---

## Limitations & Ethical Considerations

- **Empirical vs. Mathematical Guarantees**: Differential Privacy provides provable statistical upper bounds on privacy loss ($\epsilon, \delta$). Empirical adversarial attack simulations test specific heuristic attack models and should be interpreted as empirical stress tests rather than absolute guarantees.
- **Source Data Bias Reflection**: Generative models learn and reproduce the underlying distributions of their training data. Any systemic biases, missingness patterns, or imbalances in the source data will be reflected in the synthesized output.
- **Resource & Convergence Constraints**: Deep generative neural networks (TVAE / CTGAN) require adequate sample size and training epochs to converge on complex non-linear tabular distributions.

---

## Roadmap & Future Work

Future conceptual enhancements planned for SynthForge:
- **Conditional Generation Constraints**: Enforce domain-specific business rules and hard relational constraints during synthesis.
- **DP-SGD for Neural Architectures**: Native integration of Differentially Private Stochastic Gradient Descent during TVAE / CTGAN neural network backpropagation.
- **Multi-Format Storage Connectors**: Direct export and ingestion for Apache Parquet, Apache Arrow, and SQL database endpoints.
- **Custom Metric Plugin Architecture**: Extensible interface allowing domain teams to register proprietary quality validators.

---

## Author

Created by **Jeevan M.**
GitHub: [@jeevan-m-21](https://github.com/jeevan-m-21)

---

## Explore SynthForge

Explore the repository, run SynthForge locally, and inspect the evaluation pipeline.
