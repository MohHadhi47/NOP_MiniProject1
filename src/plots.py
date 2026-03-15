"""
Visualization utilities for convergence, regret, and metric comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_loss_convergence(ucb_losses, sgd_losses, window=500):
    """Smoothed per-step loss for UCB-OCO vs SGD-MF."""
    fig, ax = plt.subplots(figsize=(9, 4))

    # Smooth UCB losses with rolling mean
    ucb_smooth = np.convolve(ucb_losses, np.ones(window) / window, mode="valid")
    steps_ucb = np.arange(len(ucb_smooth))

    # SGD losses are per-epoch
    epochs = np.arange(1, len(sgd_losses) + 1)

    ax2 = ax.twinx()
    ax.plot(steps_ucb, ucb_smooth, color="steelblue", lw=1.5, label="UCB-OCO (per step, smoothed)")
    ax2.plot(epochs, sgd_losses, color="tomato", lw=2, marker="o", ms=4, label="SGD-MF (per epoch)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("UCB-OCO Loss (MSE)", color="steelblue")
    ax2.set_ylabel("SGD-MF Epoch Loss (MSE)", color="tomato")
    ax.set_title("Convergence: UCB-OCO vs SGD-MF")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, "convergence.png")


def plot_cumulative_regret(ucb_losses, label="UCB-OCO"):
    """Log-scale cumulative regret plot with O(sqrt(T)) reference line."""
    fig, ax = plt.subplots(figsize=(9, 4))
    T = len(ucb_losses)
    t = np.arange(1, T + 1)
    cum_regret = np.cumsum(ucb_losses)

    # Theoretical O(sqrt(T * ln(T))) reference
    ref = cum_regret[0] * np.sqrt(t * np.log(t + 1))
    ref = ref / ref[0] * cum_regret[0]  # normalize to same start

    ax.plot(t, cum_regret, color="steelblue", lw=1.5, label=label)
    ax.plot(t, ref, color="gray", lw=1, ls="--", label=r"$O(\sqrt{T \ln T})$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Round t (log scale)")
    ax.set_ylabel("Cumulative Loss (log scale)")
    ax.set_title("Cumulative Regret Bound (Log-Log Scale)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, "cumulative_regret.png")


def plot_metric_comparison(metrics_ucb, metrics_sgd, k=10):
    """Bar chart comparing Precision@K, Recall@K, NDCG@K."""
    labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]
    ucb_vals = [metrics_ucb["precision"], metrics_ucb["recall"], metrics_ucb["ndcg"]]
    sgd_vals = [metrics_sgd["precision"], metrics_sgd["recall"], metrics_sgd["ndcg"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, ucb_vals, width, label="UCB-OCO", color="steelblue")
    bars2 = ax.bar(x + width / 2, sgd_vals, width, label="SGD-MF", color="tomato")

    ax.set_ylabel("Score")
    ax.set_title(f"Recommendation Quality Metrics (K={k})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(ucb_vals), max(sgd_vals)) * 1.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    _save(fig, "metric_comparison.png")


def plot_rmse_comparison(rmse_ucb, rmse_sgd, mae_ucb, mae_sgd):
    """Side-by-side RMSE and MAE comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, vals in zip(
        axes,
        ["RMSE", "MAE"],
        [(rmse_ucb, rmse_sgd), (mae_ucb, mae_sgd)]
    ):
        bars = ax.bar(["UCB-OCO", "SGD-MF"], vals, color=["steelblue", "tomato"], width=0.4)
        ax.set_title(metric)
        ax.set_ylim(0, max(vals) * 1.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rating Prediction Error Comparison")
    _save(fig, "rmse_mae_comparison.png")


def plot_ucb_exploration(model, top_n=30):
    """Visualize UCB exploration bonus vs exploitation for top-N items."""
    fig, ax = plt.subplots(figsize=(10, 4))
    items = np.arange(top_n)
    mu = model.mu_hat[:top_n]
    var = model.M2[:top_n] / model.N[:top_n]
    exploration = model.c_ucb * np.sqrt(var / model.N[:top_n] + np.log(model.t + 1) / model.N[:top_n])

    ax.bar(items, mu, label="Exploitation (mean reward)", color="steelblue", alpha=0.8)
    ax.bar(items, exploration, bottom=mu, label="Exploration bonus", color="orange", alpha=0.8)
    ax.set_xlabel("Item ID")
    ax.set_ylabel("UCB Score")
    ax.set_title(f"UCB Decomposition: Exploitation vs Exploration (Top {top_n} Items)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "ucb_exploration.png")
