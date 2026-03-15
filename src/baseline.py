"""
Baseline: Standard SGD Matrix Factorization (no bandit exploration).

Update rule (batch epoch-based):
    u_u <- u_u + lr * (err * v_i - lambda * u_u)
    v_i <- v_i + lr * (err * u_u - lambda * v_i)

This is the classic Funk SVD / SGD-MF used as the comparison baseline.
"""

import numpy as np
from tqdm import tqdm


class SGDMatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.005,
                 lambda_reg=0.02, n_epochs=20, random_state=42):
        rng = np.random.RandomState(random_state)
        self.U = rng.normal(0, 0.1, (n_users, n_factors))
        self.V = rng.normal(0, 0.1, (n_items, n_factors))
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.loss_history = []

    def fit(self, train_df):
        data = train_df[["user_id", "item_id", "rating"]].values
        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            epoch_loss = 0.0
            for user_id, item_id, rating in data:
                user_id, item_id = int(user_id), int(item_id)
                pred = self.U[user_id] @ self.V[item_id]
                err = rating - pred
                # SGD update
                self.U[user_id] += self.lr * (err * self.V[item_id] - self.lambda_reg * self.U[user_id])
                self.V[item_id] += self.lr * (err * self.U[user_id] - self.lambda_reg * self.V[item_id])
                epoch_loss += err ** 2
            avg_loss = epoch_loss / len(data)
            self.loss_history.append(avg_loss)
            print(f"  [SGD-MF] Epoch {epoch+1}/{self.n_epochs} | MSE: {avg_loss:.4f}")
        return self

    def predict(self, user_id, item_id):
        return float(self.U[user_id] @ self.V[item_id])

    def recommend(self, user_id, k=10, exclude_seen=None):
        scores = self.U[user_id] @ self.V.T
        if exclude_seen:
            scores[list(exclude_seen)] = -np.inf
        return np.argsort(scores)[::-1][:k]
