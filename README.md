# SynthForge

**Privacy-Preserving Synthetic Tabular Data Platform**

SynthForge is a modern platform for generating, validating, and stress-testing privacy-preserving synthetic tabular datasets. It combines statistical copula modeling and deep generative neural networks (TVAE/CTGAN) with mathematical Differential Privacy, an automated 5-pillar Quality & Trustworthiness audit, and empirical adversarial privacy threat simulations.

---

## Why Synthetic Data?

Real-world datasets (in healthcare, finance, e-commerce, and logistics) frequently contain sensitive, private, or regulated information that cannot be shared freely across teams, partners, or third-party vendors.

Synthetic data provides a practical bridge by generating artificial records that preserve the statistical distributions, correlations, and analytical utility of the original dataset without exposing individual real records. However, synthetic data is not automatically private or useful by default: it requires rigorous, independent verification across both **Data Fidelity** and **Privacy Protection** before being deployed.

---

## Key Features

- **Automated Ingestion & Schema Intelligence**: Upload custom CSV files or explore bundled clinical benchmarks with automatic type inference, statistical moment extraction, and schema profiling.
- **Dual Generation Modes**:
  - **Fast Generation**: High-speed statistical copula distribution modeling (~1s execution).
  - **High-Fidelity Generation**: Deep Variational Autoencoder (TVAE) capturing non-linear relationships and rare clusters for advanced machine learning workflows.
- **Differential Privacy (DP)**: Built-in Gaussian and Laplace noise mechanisms with Rényi Differential Privacy (RDP) accounting, epsilon tracking, and per-column parallel composition.
- **Unified 5-Pillar Quality Audit**:
  1. *Structural Fidelity*: Column preservation, schema compatibility, and datatype alignment.
  2. *Statistical Fidelity*: Kolmogorov-Smirnov (KS) tests and Chi-Square distribution conformity with human-readable status indicators.
  3. *Relationship Fidelity*: Covariance alignment and real vs. synthetic correlation heatmaps.
  4. *Machine Learning Utility (TSTR)*: Train on Synthetic, Test on Real benchmarking comparing Random Forest and Gradient Boosting models against real-data baselines (TRTR) with accuracy parity and ROC curves.
  5. *Privacy Protection*: Empirical defense resilience and zero exact record collisions.
- **Dual-Dimension Executive Scorecard**: Decoupled Data Fidelity and Privacy Protection scores and grades, acknowledging that fidelity and privacy represent distinct trade-off dimensions.
- **Adversarial Privacy Attack Suite**: Empirical stress-testing against:
  - *Membership Inference Attacks (MIA)* with attack ROC curves.
  - *Distance-Based Re-Identification (Re-ID)* measuring percentage of records at risk.
  - *Attribute Inference* calculating adversary prediction advantage over random baselines.
  - *Exact Collision Defense* verifying zero training record memorization.
- **Modern User Experience**:
  - Interactive variable distribution dropdown selector.
  - Multi-step progress modal during generation.
  - Instant Light, Dark, and System theme switcher.
  - Full mobile (375px), tablet (768px), and desktop (1280px+) responsiveness.

---

## Application Flow

```
Home (Overview & Value Proposition)
  ↓
Synthesize (Upload CSV / Sample Data → Select Fast/High-Fidelity Mode → Configure Records)
  ↓
Generation Progress (Step-by-step progress tracking overlay)
  ↓
Quality Report (Trust Verdict → Fidelity & Privacy Scorecards → 5 Pillars → Deep Dive Evidence)
  ↓
Privacy & Threats (Adversarial Attack Simulation → Threat Radar → MIA Curve → Breakdown Table)
  ↓
Download Synthetic Dataset (Direct CSV export)
```

---

## Architecture

SynthForge follows a decoupled, lightweight architecture built for speed and transparency:

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

---

## Generation Modes

1. **Fast Generation (Recommended for Exploration)**
   - Uses statistical copula distribution modeling.
   - Preserves 1D marginal distributions, continuous moments, and linear correlation structures.
   - Ideal for rapid UI mockups, testing data pipelines, and quick analytical exploration.

2. **High-Fidelity Generation (Best for Analysis & ML)**
   - Uses deep learning tabular generative models (TVAE / CTGAN).
   - Captures complex non-linear feature interactions, conditional probability distributions, and multi-modal clusters.
   - Ideal for downstream machine learning model training and scientific research.

---

## Quality & Privacy Evaluation

### The 5 Evaluation Pillars
1. **Structural Fidelity**: Verifies that generated tables strictly adhere to source column schemas, nullability constraints, and data types.
2. **Statistical Fidelity**: Evaluates marginal probability distributions using 2-sample Kolmogorov-Smirnov tests (numeric) and Chi-Square goodness-of-fit tests (categorical).
3. **Relationship Fidelity**: Compares cross-attribute correlation matrices (Pearson/Cramér's V) to ensure multi-variable dependencies are preserved.
4. **ML Utility (TSTR)**: Trains predictive models on synthetic data and tests them against held-out real data (Train on Synthetic, Test on Real), benchmarking accuracy, F1-score, and ROC-AUC parity against real-data baselines (Train on Real, Test on Real).
5. **Privacy Protection**: Quantifies empirical defense against identity leakage and verifies that no real training rows were copied identically.

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager

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

Launch the platform with one command:
```bash
python run.py
```

- **Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Reference (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Automated Test Suites

Run the end-to-end integration test suite:
```bash
python test_e2e.py
```

Run the complete backend unit test suite:
```bash
python -m unittest test_dataset_profiler.py test_phase3_generation.py test_phase4_generalization.py test_phase7a_security.py test_phase7b_concurrency.py test_phase7c_reproducibility.py test_phase7d_job_recovery.py test_quality_evaluator.py test_schema_intelligence.py
```

---

## Repository Structure

```
SynthForge/
├── backend/
│   ├── api/             # FastAPI REST route definitions and Pydantic schemas
│   ├── models/          # Storage models and background job persistence
│   ├── services/        # Generative engines, DP noise, evaluators, validators
│   ├── utils/           # Structured JSON logging and security sanitizers
│   ├── config.py        # Global configuration and path definitions
│   └── main.py          # FastAPI application entry point & static file mount
├── frontend/
│   ├── css/
│   │   └── styles.css   # Theme tokens (Light/Dark/System) & responsive styles
│   ├── js/
│   │   ├── api.js       # Client-side API fetch client
│   │   ├── app.js       # Main controller, state management, and DOM bindings
│   │   └── charts.js    # Chart.js utilities, theme management, and heatmaps
│   └── index.html       # Clean 4-destination interface
├── data/
│   ├── sample/          # Bundled clinical healthcare dataset
│   └── phase4_fixtures/ # Multi-domain independent datasets (ecommerce, finance, etc.)
├── storage/             # Runtime uploads, generated datasets, and reports
├── requirements.txt     # Production dependencies
├── run.py               # Single-command startup script
└── test_e2e.py          # End-to-end integration test suite
```

---

## Limitations & Ethical Considerations

- **No Absolute Guarantee**: Differential Privacy provides mathematical upper bounds on information disclosure ($\epsilon, \delta$), but privacy protection is an empirical trade-off. Extreme privacy noise can degrade statistical fidelity.
- **Distribution Bounds**: Synthetic data generators learn distributions from the training set. Biases or gaps present in the source data will naturally be reflected in the synthesized output.
- **Model Tuning**: Deep learning synthesizers (TVAE/CTGAN) require sufficient training epochs and sample volume to converge on highly complex multi-modal distributions.

---

## Author

Created by **Jeevan M.**
GitHub: [@jeevan-m-21](https://github.com/jeevan-m-21)
