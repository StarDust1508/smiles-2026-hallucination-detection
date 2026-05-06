"""
probe.py — Hallucination probe classifier (v3: linear probe + improved features).

Pipeline applied to the input feature vector:

    raw features (~4560 dims: 5×896 mean-pool + ~80 geometric)
        → VarianceThreshold        (remove near-constant features)
        → StandardScaler           (zero-mean, unit-variance per feature)
        → PCA  (n_components=128)  (compress to a learnable subspace)
        → LogisticRegression       (L2-penalised, class-balanced)
        → tuned threshold          (max F1 on official validation slice)

v3 changes vs v2:
  - PCA 128 instead of 64 (richer features from mean-pool over response tokens)
  - VarianceThreshold preprocessing (removes dead / near-constant features)
  - C=1.0 (lighter regularisation — the new features carry more signal)

Why a linear probe:
  With ~482 training samples per fold and ~4560 raw features, any non-linear
  model with enough capacity to fit the training set perfectly will memorise
  noise. The "linear probe" is the standard tool in the interpretability
  literature (Alain & Bengio 2016; Belinkov 2022) precisely because of this.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

PCA_COMPONENTS = 128
# Inverse regularisation strength. C=1.0 gave the best average validation
# AUROC with the v3 feature set (mean-pool over response tokens).
LR_C = 1.0
LR_MAX_ITER = 3000
# Minimum variance threshold: features with variance below this are removed.
# Normalised to unit variance after scaling, so 0.01 = essentially constant.
VAR_THRESHOLD = 0.01
SEED = 42


class HallucinationProbe(nn.Module):
    """Linear probe: VarianceThreshold → StandardScaler → PCA(128) → LogisticRegression.

    Subclasses ``nn.Module`` for compatibility with the evaluation pipeline,
    but contains no torch parameters — all learning is delegated to sklearn.
    """

    def __init__(self) -> None:
        super().__init__()
        self._var_thresh: VarianceThreshold | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit variance threshold, scaler, PCA, and logistic regression."""
        # 0. Remove near-constant features.
        self._var_thresh = VarianceThreshold(threshold=VAR_THRESHOLD)
        X_var = self._var_thresh.fit_transform(X)

        # 1. Standardise feature columns.
        X_scaled = self._scaler.fit_transform(X_var)

        # 2. PCA — n_components capped at min(128, n_samples-1, n_features).
        n_components = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        X_reduced = self._pca.fit_transform(X_scaled)

        # 3. Class-balanced L2 logistic regression.
        self._clf = LogisticRegression(
            C=LR_C,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=LR_MAX_ITER,
            random_state=SEED,
        )
        self._clf.fit(X_reduced, y.astype(int))
        return self

    # ------------------------------------------------------------------
    # Decision threshold tuning
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold to maximise F1 on the validation set."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_var = self._var_thresh.transform(X)
        X_scaled = self._scaler.transform(X_var)
        return self._pca.transform(X_scaled) if self._pca is not None else X_scaled

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self._transform(X)
        if self._clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        return self._clf.predict_proba(X_reduced)

    # ------------------------------------------------------------------
    # nn.Module compatibility shim
    # ------------------------------------------------------------------
    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError(
            "HallucinationProbe v3 is an sklearn pipeline; use predict_proba()."
        )
