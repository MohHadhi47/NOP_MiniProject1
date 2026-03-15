"""
UCB-OCO Bandit Optimizer for Collaborative Filtering.

Mathematical Framework:
-----------------------
Each item i is treated as an arm in a Multi-Armed Bandit (MAB).
The UCB score for arm i at round t is:

    UCB_i(t) = mu_i(t) + c * sqrt( sigma_i^2(t) / N_i(t) + ln(t) / N_i(t) )

where:
    mu_i(t)     = empirical mean reward (predicted rating) for item i
    sigma_i^2(t)= variance of user-item interaction (subgradient bound)
    N_i(t)      = number of times item i has been recommended
    c           = exploration coefficient
    t           = current round

Online Convex Optimization (OCO) Integration:
----------------------------------------------
The latent factor update follows an online subgradient step on the
instantaneous squared loss:

    L_t(U, V) = (r_ui - u_u^T v_i)^2 + lambda * (||u_u||^2 + ||v_i||^2)

Subgradient update:
    u_u <- u_u - eta_t * grad_u L_t
    v_i <- v_i - eta_t * grad_v L_t

with decaying step size eta_t = eta_0 / sqrt(t) to guarantee O(sqrt(T)) regret
for the online learner, and the UCB exploration bonus ensures logarithmic
cumulative regret over the bandit horizon.

Regret Bound:
    R(T) <= O(sqrt(K * T * ln(T)))  [UCB regret]
    R_OCO(T) <= O(sqrt(T))          [OCO subgradient regret]
"""

import numpy as np


class UCBBanditOCO:
    """
    UCB-guided Online Convex Optimization for Matrix Factorization.

    Parameters
    ----------
    n_users : int
    n_items : int
    n_factors : int       - latent dimension
    eta0 : float          - initial OCO step size
    lambda_reg : float    - L2 regularization
    c_ucb : float         - UCB exploration coefficient
    """

    def __init__(self, n_users, n_items, n_factors=20, eta0=0.05,
                 lambda_reg=0.01, c_ucb=1.5, random_state=42):
        rng = np.random.RandomState(random_state)
        scale = 0.1
        self.U = rng.normal(0, scale, (n_users, n_factors))  # user latent factors
        self.V = rng.normal(0, scale, (n_items, n_factors))  # item latent factors

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.eta0 = eta0
        self.lambda_reg = lambda_reg
        self.c_ucb = c_ucb

        # Bandit statistics per item
        self.N = np.ones(n_items, dtype=np.float64)        # pull counts (init=1 avoids div/0)
        self.mu_hat = np.zeros(n_items, dtype=np.float64)  # empirical mean reward
        self.M2 = np.ones(n_items, dtype=np.float64)       # sum of squared deviations (Welford)

        self.t = 1          # global round counter
        self.regret_log = []  # cumulative regret over rounds

    # ------------------------------------------------------------------
    # UCB Score
    # ------------------------------------------------------------------
    def _ucb_scores(self, user_id):
        """
        Compute UCB scores for all items for a given user.

        UCB_i = predicted_rating_i + c * sqrt( var_i / N_i + ln(t) / N_i )
        """
        pred = self.U[user_id] @ self.V.T  # shape: (n_items,)
        var_i = self.M2 / self.N           # running variance estimate
        exploration = self.c_ucb * np.sqrt(var_i / self.N + np.log(self.t + 1) / self.N)
        return pred + exploration

    # ------------------------------------------------------------------
    # OCO Subgradient Update
    # ------------------------------------------------------------------
    def _oco_update(self, user_id, item_id, rating):
        """
        Online subgradient step on squared loss with L2 regularization.

        eta_t = eta0 / sqrt(t)  -- ensures O(sqrt(T)) OCO regret
        """
        eta_t = self.eta0 / np.sqrt(self.t)

        pred = self.U[user_id] @ self.V[item_id]
        err = pred - rating

        # Subgradients
        grad_u = 2 * err * self.V[item_id] + 2 * self.lambda_reg * self.U[user_id]
        grad_v = 2 * err * self.U[user_id] + 2 * self.lambda_reg * self.V[item_id]

        self.U[user_id] -= eta_t * grad_u
        self.V[item_id] -= eta_t * grad_v

        return err ** 2  # instantaneous loss

    # ------------------------------------------------------------------
    # Bandit Statistics Update (Welford online algorithm)
    # ------------------------------------------------------------------
    def _update_bandit_stats(self, item_id, reward):
        self.N[item_id] += 1
        delta = reward - self.mu_hat[item_id]
        self.mu_hat[item_id] += delta / self.N[item_id]
        delta2 = reward - self.mu_hat[item_id]
        self.M2[item_id] += delta * delta2  # Welford M2 accumulator

    # ------------------------------------------------------------------
    # Single Training Step
    # ------------------------------------------------------------------
    def partial_fit(self, user_id, item_id, rating):
        """Process one observed (user, item, rating) interaction."""
        loss = self._oco_update(user_id, item_id, rating)
        self._update_bandit_stats(item_id, rating)
        self.t += 1
        return loss

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def recommend(self, user_id, k=10, exclude_seen=None):
        """
        Recommend top-k items for a user using UCB scores.

        Parameters
        ----------
        exclude_seen : set of item_ids already rated by the user
        """
        scores = self._ucb_scores(user_id)
        if exclude_seen:
            scores[list(exclude_seen)] = -np.inf
        return np.argsort(scores)[::-1][:k]

    # ------------------------------------------------------------------
    # Predict rating
    # ------------------------------------------------------------------
    def predict(self, user_id, item_id):
        return float(self.U[user_id] @ self.V[item_id])

    # ------------------------------------------------------------------
    # Cumulative Regret Computation
    # ------------------------------------------------------------------
    def compute_instantaneous_regret(self, user_id, chosen_item, best_item, rating_matrix):
        """
        Pseudo-regret: difference between best possible reward and chosen reward.
        r_t = max_i E[reward_i] - E[reward_chosen]
        """
        best_reward = rating_matrix[user_id, best_item] if rating_matrix[user_id, best_item] > 0 else 0
        chosen_reward = rating_matrix[user_id, chosen_item] if rating_matrix[user_id, chosen_item] > 0 else 0
        return max(0.0, best_reward - chosen_reward)
