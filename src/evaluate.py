"""
Evaluation metrics:
    - RMSE, MAE
    - Precision@K, Recall@K, NDCG@K
    - Cumulative Regret over time
"""

import numpy as np
import pandas as pd


def rmse(preds, actuals):
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def mae(preds, actuals):
    return float(np.mean(np.abs(preds - actuals)))


def precision_at_k(model, test_df, train_df, k=10, relevance_threshold=4.0):
    """
    Precision@K: fraction of top-K recommended items that are relevant.
    Relevant = rating >= relevance_threshold in test set.
    """
    # Build ground truth: user -> set of relevant test items
    relevant = test_df[test_df["rating"] >= relevance_threshold].groupby("user_id")["item_id"].apply(set).to_dict()
    # Build seen items per user from training set
    seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    precisions = []
    for user_id, rel_items in relevant.items():
        excl = seen.get(user_id, set())
        recs = model.recommend(user_id, k=k, exclude_seen=excl)
        hits = len(set(recs) & rel_items)
        precisions.append(hits / k)

    return float(np.mean(precisions))


def recall_at_k(model, test_df, train_df, k=10, relevance_threshold=4.0):
    relevant = test_df[test_df["rating"] >= relevance_threshold].groupby("user_id")["item_id"].apply(set).to_dict()
    seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls = []
    for user_id, rel_items in relevant.items():
        if len(rel_items) == 0:
            continue
        excl = seen.get(user_id, set())
        recs = model.recommend(user_id, k=k, exclude_seen=excl)
        hits = len(set(recs) & rel_items)
        recalls.append(hits / len(rel_items))

    return float(np.mean(recalls))


def ndcg_at_k(model, test_df, train_df, k=10, relevance_threshold=4.0):
    """Normalized Discounted Cumulative Gain @ K."""
    relevant = test_df[test_df["rating"] >= relevance_threshold].groupby("user_id")["item_id"].apply(set).to_dict()
    seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    ndcgs = []
    for user_id, rel_items in relevant.items():
        if len(rel_items) == 0:
            continue
        excl = seen.get(user_id, set())
        recs = model.recommend(user_id, k=k, exclude_seen=excl)

        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item in enumerate(recs)
            if item in rel_items
        )
        ideal_hits = min(len(rel_items), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def compute_cumulative_regret(losses_ucb, losses_sgd):
    """
    Proxy cumulative regret: cumulative sum of per-step squared errors.
    UCB-OCO should grow sub-linearly (O(sqrt(T))) vs SGD.
    """
    cum_ucb = np.cumsum(losses_ucb)
    # SGD losses are per-epoch; repeat each epoch loss across its steps for alignment
    return cum_ucb


def convergence_rate(loss_history):
    """
    Estimate empirical convergence rate by fitting log(loss) ~ -alpha * log(t).
    Returns alpha (should be ~0.5 for O(1/sqrt(t)) convergence).
    """
    t = np.arange(1, len(loss_history) + 1, dtype=float)
    log_t = np.log(t)
    log_loss = np.log(np.clip(loss_history, 1e-10, None))
    # Linear regression: log_loss = a - alpha * log_t
    alpha = -np.polyfit(log_t, log_loss, 1)[0]
    return float(alpha)
