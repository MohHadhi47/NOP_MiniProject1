"""
Online Matrix Factorization trainer using the UCB-OCO optimizer.
Streams training data row-by-row to simulate an online learning setting.
"""

import numpy as np
from src.bandit import UCBBanditOCO


def train_ucb_oco(train_df, n_users, n_items, n_factors=20, eta0=0.05,
                  lambda_reg=0.01, c_ucb=1.5, n_epochs=20, random_state=42):
    """
    Train the UCB-OCO model by streaming each rating as an online update.
    Multiple epochs allow the latent factors to converge while the UCB
    bandit statistics accumulate across all passes.

    Returns the trained model and per-epoch average loss history.
    """
    model = UCBBanditOCO(
        n_users=n_users, n_items=n_items, n_factors=n_factors,
        eta0=eta0, lambda_reg=lambda_reg, c_ucb=c_ucb, random_state=random_state
    )

    epoch_losses = []
    rng = np.random.RandomState(random_state)

    for epoch in range(n_epochs):
        # Re-shuffle each epoch to simulate i.i.d. online stream
        idx = rng.permutation(len(train_df))
        shuffled = train_df.iloc[idx].reset_index(drop=True)
        step_losses = []

        # Reset round counter per epoch so eta_t = eta0/sqrt(t) stays effective
        model.t = 1

        for row in shuffled.itertuples(index=False):
            loss = model.partial_fit(row.user_id, row.item_id, row.rating)
            step_losses.append(loss)

        avg = float(np.mean(step_losses))
        epoch_losses.append(avg)
        print(f"  [UCB-OCO] Epoch {epoch+1}/{n_epochs} | MSE: {avg:.4f}")

    return model, epoch_losses


def predict_all(model, test_df):
    """Generate predictions for all test interactions."""
    preds, actuals = [], []
    for row in test_df.itertuples(index=False):
        preds.append(model.predict(row.user_id, row.item_id))
        actuals.append(row.rating)
    return np.array(preds), np.array(actuals)
