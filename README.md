# MediSynth.AI

**Privacy-Preserving Synthetic Healthcare Data Generation Platform**

A production-ready system for generating high-quality synthetic healthcare data with formal differential privacy guarantees, multi-hospital federated learning, and comprehensive privacy attack resistance.

## Quick Start

```bash
cd synth-health-guard
python run.py
```

Then open http://127.0.0.1:8000 in your browser.

## Features

| Feature | Description |
|:---|:---|
| **Synthetic Data Generation** | CTGAN, TVAE, or Statistical (Gaussian copula) generators |
| **Differential Privacy** | Gaussian/Laplace mechanisms with RDP accounting |
| **Privacy Budget Tracking** | Per-dataset epsilon tracking with exhaustion warnings |
| **Statistical Validation** | KS test, mean/variance, chi-squared, correlation matrix comparison |
| **ML Utility Validation** | TSTR benchmark with RandomForest + GradientBoosting |
| **Privacy Attack Simulation** | Membership inference, re-identification, attribute inference |
| **Federated Learning** | FedAvg with order-independent aggregation, DP on updates |
| **Dashboard** | Glassmorphism UI with charts, gauges, heatmaps |

## Architecture

```
Frontend (HTML/CSS/JS + Chart.js)
        |
    FastAPI Backend
        |
   +----+----+----+----+----+
   |    |    |    |    |    |
  Gen  DP   Stat  ML  Atk  FL
```

## API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| GET | `/api/health` | Health check |
| GET | `/api/data/sample` | Load sample healthcare data |
| POST | `/api/data/upload` | Upload CSV |
| POST | `/api/generate` | Generate synthetic data |
| GET | `/api/privacy/budget/{id}` | Privacy budget status |
| POST | `/api/validate/statistical` | Statistical validation |
| POST | `/api/validate/ml` | ML utility validation |
| POST | `/api/attacks/simulate` | Privacy attack simulation |
| POST | `/api/federated/create` | Create federation |
| POST | `/api/federated/add-hospital` | Add hospital |
| POST | `/api/federated/train` | Run federated training |
| POST | `/api/federated/generate` | Generate from federation |

Full interactive docs at http://127.0.0.1:8000/docs

## Differential Privacy Guarantees

- **Gaussian Mechanism**: Adds N(0, sigma^2) noise where sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
- **Laplace Mechanism**: Adds Lap(sensitivity/epsilon) noise
- **RDP Accounting**: Renyi DP for tight composition across queries
- **Per-column parallel composition**: Each column gets full epsilon budget
- **Privacy Budget**: Cumulative tracking with warnings at 50%, 75%, 90%

## Privacy Attack Resistance

- **Membership Inference**: AUC near 0.5 = attacker cannot distinguish members
- **Re-identification**: Distance-based assessment of synthetic-to-real proximity
- **Attribute Inference**: Prediction advantage over random baseline

## Federated Learning

- **FedAvg**: Weighted parameter averaging (order-independent: A+B == B+A)
- **No raw data sharing**: Only model statistics are exchanged
- **DP on updates**: Optional Gaussian noise on parameter updates

## Requirements

- Python 3.10+
- FastAPI, uvicorn, pandas, numpy, scipy, scikit-learn
- Optional: sdv (for CTGAN/TVAE)
