"""
Main pipeline: UCB-OCO Collaborative Filtering vs SGD-MF Baseline
Theme-9: Logarithmic Regret Bounding via UCB in Collaborative Filtering

Run:
    python main.py
"""

import numpy as np
import pandas as pd

from data.load_data import get_train_test_split, build_interaction_matrix
from src.mf import train_ucb_oco, predict_all
from src.baseline import SGDMatrixFactorization
from src.evaluate import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, convergence_rate
from src.plots import (
    plot_loss_convergence, plot_cumulative_regret,
    plot_metric_comparison, plot_rmse_comparison, plot_ucb_exploration
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
K = 10                  # top-K for ranking metrics
N_FACTORS = 20          # latent dimension
UCB_ETA0 = 0.1          # initial OCO step size
UCB_LAMBDA = 0.01       # L2 regularization (UCB-OCO)
UCB_C = 0.1             # exploration coefficient (lower = more exploitation in ranking)
SGD_LR = 0.005          # SGD learning rate
SGD_LAMBDA = 0.02       # L2 regularization (SGD-MF)
SGD_EPOCHS = 30         # training epochs for both models
RELEVANCE_THRESHOLD = 4.0  # rating >= 4 is "relevant"
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  UCB-OCO Collaborative Filtering | MovieLens 100K")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading MovieLens 100K dataset...")
    train_df, test_df, n_users, n_items = get_train_test_split(test_size=0.2, random_state=42)
    print(f"  Users: {n_users} | Items: {n_items}")
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    R_train = build_interaction_matrix(train_df, n_users, n_items)

    # 2. Train UCB-OCO model
    print("\n[2/5] Training UCB-OCO model...")
    ucb_model, ucb_losses = train_ucb_oco(
        train_df, n_users, n_items,
        n_factors=N_FACTORS, eta0=UCB_ETA0,
        lambda_reg=UCB_LAMBDA, c_ucb=UCB_C,
        n_epochs=SGD_EPOCHS
    )

    # 3. Train SGD-MF baseline
    print("\n[3/5] Training SGD-MF baseline...")
    sgd_model = SGDMatrixFactorization(
        n_users=n_users, n_items=n_items,
        n_factors=N_FACTORS, lr=SGD_LR,
        lambda_reg=SGD_LAMBDA, n_epochs=SGD_EPOCHS
    ).fit(train_df)

    # 4. Evaluate
    print("\n[4/5] Evaluating models...")

    # Rating prediction
    ucb_preds, actuals = predict_all(ucb_model, test_df)
    sgd_preds = np.array([sgd_model.predict(r.user_id, r.item_id) for r in test_df.itertuples(index=False)])

    ucb_rmse = rmse(ucb_preds, actuals)
    ucb_mae  = mae(ucb_preds, actuals)
    sgd_rmse = rmse(sgd_preds, actuals)
    sgd_mae  = mae(sgd_preds, actuals)

    # Ranking metrics
    ucb_p = precision_at_k(ucb_model, test_df, train_df, k=K, relevance_threshold=RELEVANCE_THRESHOLD)
    ucb_r = recall_at_k(ucb_model, test_df, train_df, k=K, relevance_threshold=RELEVANCE_THRESHOLD)
    ucb_n = ndcg_at_k(ucb_model, test_df, train_df, k=K, relevance_threshold=RELEVANCE_THRESHOLD)

    sgd_p = precision_at_k(sgd_model, test_df, train_df, k=K, relevance_threshold=RELEVANCE_THRESHOLD)
    sgd_r = recall_at_k(sgd_model, test_df, train_df, k=K, relevance_threshold=RELEVANCE_THRESHOLD)
    sgd_n = ndcg_at_k(sgd_model, test_df, train_df, k=K, relevance_threshold=RELEVANCE_THRESHOLD)

    # Convergence rate
    ucb_alpha = convergence_rate(ucb_losses)

    # Print results table
    print("\n" + "=" * 60)
    print(f"  {'Metric':<25} {'UCB-OCO':>12} {'SGD-MF':>12}")
    print("-" * 60)
    print(f"  {'RMSE':<25} {ucb_rmse:>12.4f} {sgd_rmse:>12.4f}")
    print(f"  {'MAE':<25} {ucb_mae:>12.4f} {sgd_mae:>12.4f}")
    print(f"  {f'Precision@{K}':<25} {ucb_p:>12.4f} {sgd_p:>12.4f}")
    print(f"  {f'Recall@{K}':<25} {ucb_r:>12.4f} {sgd_r:>12.4f}")
    print(f"  {f'NDCG@{K}':<25} {ucb_n:>12.4f} {sgd_n:>12.4f}")
    print(f"  {'Convergence Rate (alpha)':<25} {ucb_alpha:>12.4f} {'~0.5 (theory)':>12}")
    print("=" * 60)

    # Save results to CSV
    results = pd.DataFrame({
        "Metric": ["RMSE", "MAE", f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}", "Convergence Alpha"],
        "UCB-OCO": [ucb_rmse, ucb_mae, ucb_p, ucb_r, ucb_n, ucb_alpha],
        "SGD-MF":  [sgd_rmse, sgd_mae, sgd_p, sgd_r, sgd_n, None]
    })
    results.to_csv("results.csv", index=False)
    print("\n  Results saved to results.csv")

    # 5. Plots
    print("\n[5/5] Generating plots...")
    plot_loss_convergence(ucb_losses, sgd_model.loss_history)
    plot_cumulative_regret(ucb_losses)
    plot_metric_comparison(
        {"precision": ucb_p, "recall": ucb_r, "ndcg": ucb_n},
        {"precision": sgd_p, "recall": sgd_r, "ndcg": sgd_n},
        k=K
    )
    plot_rmse_comparison(ucb_rmse, sgd_rmse, ucb_mae, sgd_mae)
    plot_ucb_exploration(ucb_model, top_n=30)

    print("\nDone. Plots saved to ./plots/")


if __name__ == "__main__":
    main()
