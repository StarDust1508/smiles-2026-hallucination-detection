"""
probe.py — Hallucination probe classifier (v7).

Pipeline matched to the v7 feature design:

    raw features (2700 dims = 2688 hidden + 12 alignment)
        → StandardScaler
        → XGBoost (or LogisticRegression fallback)
        → tuned decision threshold

Why XGBoost is well-suited here, even though we kept hidden states:

  The v7 alignment features (Jaccard, BLEU, cosine sim, attention grounding)
  carry a **dense, non-linear** signal: e.g. Jaccard < 0.1 + cos sim < 0.3 is
  a much stronger hallucination signal than either feature alone.  Tree
  ensembles capture such interactions natively.  At the same time, XGBoost
  with strong regularisation does not collapse the high-dimensional hidden
  states the way PCA did in v5.

Regularisation tuning notes (vs v6):

  * max_depth = 2 (was 3) — alignment signal is mostly low-order
    interactions; deeper trees mainly memorise hidden-state noise.
  * n_estimators = 200 (was 300) — diminishing returns past ~200 with
    early stopping unavailable here.
  * reg_alpha / reg_lambda = 2.0 (was 1.0) — push more features to zero
    weight, leaning on the strongest 5-10 alignment signals.
  * min_child_weight = 5 (NEW) — refuse to split when a leaf would have
    fewer than 5 samples, killing pure noise splits on small folds.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# XGBoost — tuned for v7 alignment-rich feature space.
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 2
XGB_LEARNING_RATE = 0.05
XGB_REG_ALPHA = 2.0
XGB_REG_LAMBDA = 2.0
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.6
XGB_MIN_CHILD_WEIGHT = 5

# LogisticRegression fallback (used only when XGBoost is not installed).
LR_C = 0.5
LR_MAX_ITER = 3000

SEED = 42


class HallucinationProbe(nn.Module):
    """Scaler + XGBoost (or class-balanced LogReg) + threshold tuning."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._clf: XGBClassifier | LogisticRegression | None = None
        self._threshold: float = 0.5
        self._use_xgb = HAS_XGBOOST

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)

        if self._use_xgb:
            n_neg = int((y == 0).sum())
            n_pos = int((y == 1).sum())
            spw = n_neg / max(n_pos, 1)

            self._clf = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                reg_alpha=XGB_REG_ALPHA,
                reg_lambda=XGB_REG_LAMBDA,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                min_child_weight=XGB_MIN_CHILD_WEIGHT,
                scale_pos_weight=spw,
                eval_metric="auc",
                random_state=SEED,
                verbosity=0,
            )
            self._clf.fit(X_scaled, y.astype(int))
        else:
            self._clf = LogisticRegression(
                C=LR_C,
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=LR_MAX_ITER,
                random_state=SEED,
            )
            self._clf.fit(X_scaled, y.astype(int))

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold to maximise F1 on the validation slice."""
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X_scaled = self._scaler.transform(X)
        return self._clf.predict_proba(X_scaled)

    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError(
            "HallucinationProbe v7 delegates to sklearn/xgboost; "
            "use predict_proba()."
        )
