"""
Tiền xử lý thống nhất: fit trên tập train, transform cho validate/test.
Không import torch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from deep_credit_rating.common import config as cfg
from deep_credit_rating.common.labels import CreditRatingLabeler
from deep_credit_rating.common.preprocess import TabularPreprocessor, infer_feature_columns


def fit_training_pipeline(df: pd.DataFrame) -> tuple[
    TabularPreprocessor,
    StandardScaler,
    CreditRatingLabeler,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Fit preprocessor, labeler (toàn bộ train), StandardScaler (numeric cho MLP).
    Trả về mảng đã scale và nhãn 5 lớp cho toàn bộ df.
    """
    if "TARGET" not in df.columns:
        raise ValueError("Tập train phải có cột TARGET.")
    cat_cols, num_cols = infer_feature_columns(df, cfg.DROP_COLS, cfg.DEFAULT_CAT_COLS)
    pre = TabularPreprocessor(list(cat_cols), list(num_cols))
    df_t = pre.fit_transform(df)
    X_cat = df_t[pre.cat_cols].values.astype(np.int64)
    X_num = df_t[pre.num_cols].values.astype(np.float32)
    y_bin = df["TARGET"].values.astype(np.int64)
    labeler = CreditRatingLabeler(num_classes=cfg.NUM_CLASSES)
    y_labels = labeler.fit_transform(X_num, y_bin)
    scaler = StandardScaler()
    X_num_sc = scaler.fit_transform(X_num)
    return pre, scaler, labeler, X_cat, X_num_sc, y_labels, y_bin


def transform_eval(
    df: pd.DataFrame,
    pre: TabularPreprocessor,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X_num_raw: numeric sau preprocessor (cho CreditRatingLabeler).
    X_num_sc: sau StandardScaler (cho MLP).
    """
    df_t = pre.transform(df)
    X_cat = df_t[pre.cat_cols].values.astype(np.int64)
    X_num_raw = df_t[pre.num_cols].values.astype(np.float32)
    X_num_sc = scaler.transform(X_num_raw)
    return X_cat, X_num_sc, X_num_raw
