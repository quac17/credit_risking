"""
Sinh nhãn xếp hạng 5 mức (0..4) (mô tả trong README.md dự án):
- TARGET=1 (nợ xấu) → hạng cao nhất (lớp 4).
- TARGET=0: chia theo quantile xác suất nợ (LogisticRegression) thành 4 nhóm → lớp 0..3.
Ngưỡng quantile chỉ học trên tập `df_fit` (train) để tránh rò rỉ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CreditRatingLabeler:
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self._lr: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._quantiles: np.ndarray | None = None  # ba ngưỡng cho 4 nhóm trong TARGET=0

    def fit(self, X_num: np.ndarray, y_binary: np.ndarray) -> CreditRatingLabeler:
        """
        X_num: (n, d) đã là số, không có NaN.
        y_binary: TARGET 0/1.
        """
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X_num)
        self._lr = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        self._lr.fit(Xs, y_binary)
        p = self._lr.predict_proba(Xs)[:, 1]
        mask0 = y_binary == 0
        if mask0.sum() < 10:
            raise ValueError("Quá ít mẫu TARGET=0 để cắt quantile.")
        p0 = p[mask0]
        # Bốn nhóm: thấp → cao rủi ro trong nhóm "tốt": quantile 0.25, 0.5, 0.75
        self._quantiles = np.quantile(p0, [0.25, 0.5, 0.75])
        return self

    def transform(self, X_num: np.ndarray, y_binary: np.ndarray | None) -> np.ndarray:
        assert self._lr is not None and self._scaler is not None and self._quantiles is not None
        Xs = self._scaler.transform(X_num)
        p = self._lr.predict_proba(Xs)[:, 1]
        n = len(p)
        labels = np.zeros(n, dtype=np.int64)
        if y_binary is None:
            # Inference: không có TARGET — gán hạng chỉ từ p (heuristic)
            # p cao → hạng cao hơn (gần nợ xấu)
            q = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            thr = np.quantile(p, q[1:-1])
            labels = np.digitize(p, thr)
            return np.clip(labels, 0, self.num_classes - 1)

        mask1 = y_binary == 1
        labels[mask1] = self.num_classes - 1

        mask0 = ~mask1
        p0 = p[mask0]
        q1, q2, q3 = self._quantiles
        # p0 thấp → lớp 0 (rủi ro thấp nhất trong nhóm tốt)
        sub = np.zeros(mask0.sum(), dtype=np.int64)
        sub[p0 <= q1] = 0
        sub[(p0 > q1) & (p0 <= q2)] = 1
        sub[(p0 > q2) & (p0 <= q3)] = 2
        sub[p0 > q3] = 3
        labels[mask0] = sub
        return labels

    def fit_transform(self, X_num: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
        self.fit(X_num, y_binary)
        return self.transform(X_num, y_binary)
