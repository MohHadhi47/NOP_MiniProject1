# Logarithmic Regret Bounding via UCB in Collaborative Filtering

A recommender system that balances **exploration and exploitation** using Upper Confidence Bound (UCB) optimization mapped to an Online Convex Optimization (OCO) framework — applied to the MovieLens 100K dataset.

> Theme 9 | Numerical Optimization | Jain Deemed-to-be University, CSE-AIML

---

## What it does

Standard collaborative filtering only exploits what it already knows. This project gives the recommender a sense of curiosity — items that haven't been seen much or have inconsistent ratings get an exploration bonus, so cold-start users and niche items are never permanently ignored.

**UCB Score:**
```
UCB_i(t) = predicted_rating + c * sqrt( variance/N + ln(t)/N )
```

**OCO Update (decaying step size):**
```
eta_t = eta_0 / sqrt(t)
p_u  ←  p_u - eta_t * gradient
```

**Regret Guarantee:** `O(sqrt(K * T * ln(T)))`

---

## Results (MovieLens 100K)

| Metric | UCB-OCO | SGD-MF Baseline |
|---|---|---|
| RMSE | 1.0049 | 0.9439 |
| MAE | 0.7813 | 0.7391 |
| Precision@10 | 0.0729 | 0.0927 |
| Recall@10 | 0.0657 | 0.0718 |
| NDCG@10 | 0.0870 | 0.1165 |
| Convergence Rate (α) | 1.12 | — |

---

## Project Structure

```
ucb-collaborative-filtering/
├── data/
│   └── load_data.py       # Auto-downloads MovieLens 100K
├── src/
│   ├── bandit.py          # UCB-OCO optimizer (core algorithm)
│   ├── mf.py              # Online training loop
│   ├── baseline.py        # SGD-MF baseline
│   ├── evaluate.py        # RMSE, MAE, Precision@K, Recall@K, NDCG@K
│   └── plots.py           # All visualizations
├── notebooks/
│   └── analysis.ipynb     # Interactive analysis + EDA
├── plots/                 # Generated figures
├── main.py                # Run the full pipeline
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

The dataset downloads automatically on first run. All plots are saved to `./plots/` and results to `results.csv`.

**Notebook:**
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Latent Factors | 20 |
| Initial Step Size (η₀) | 0.1 |
| L2 Regularization (λ) | 0.01 |
| Exploration Coefficient (c) | 0.1 |
| Epochs | 30 |

---

## Requirements

- Python 3.8+
- numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, tqdm, jupyter
