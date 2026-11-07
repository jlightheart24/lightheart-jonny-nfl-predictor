"""Bayesian logistic regression (Gaussian prior, Laplace approximation) in plain numpy.

P(home team wins) = sigmoid(w . x), where x[0] = 1 is the home-field intercept.

numpy 2.x on macOS (Accelerate BLAS) emits spurious divide/overflow warnings from small
matmuls, so they are silenced inside np.errstate and results are checked for finiteness.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class BayesLogit:
    def __init__(self, prior_mean, prior_sd):
        self.m0 = np.asarray(prior_mean, float)
        self.P0 = np.diag(1.0 / np.asarray(prior_sd, float) ** 2)  # prior precision
        self.mean = self.m0.copy()
        self.cov = np.linalg.inv(self.P0)

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 50):
        """MAP estimate by Newton's method; covariance from the Hessian at the mode."""
        with np.errstate(all="ignore"):
            self._fit(X, y, iters)
        if not (np.all(np.isfinite(self.mean)) and np.all(np.isfinite(self.cov))):
            raise FloatingPointError("posterior is not finite")
        return self

    def _fit(self, X, y, iters):
        w = self.mean.copy()
        for _ in range(iters):
            p = _sigmoid(X @ w)
            grad = X.T @ (p - y) + self.P0 @ (w - self.m0)
            H = X.T @ (X * (p * (1 - p))[:, None]) + self.P0
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        p = _sigmoid(X @ w)
        H = X.T @ (X * (p * (1 - p))[:, None]) + self.P0
        self.mean, self.cov = w, np.linalg.inv(H)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Posterior predictive P(y=1), using the probit approximation to integrate out w."""
        with np.errstate(all="ignore"):
            mu = X @ self.mean
            var = np.einsum("ij,jk,ik->i", X, self.cov, X)
            p = _sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("prediction is not finite")
        return p

    def sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))
