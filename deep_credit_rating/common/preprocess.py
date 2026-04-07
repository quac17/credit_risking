from __future__ import annotations

import numpy as np
import pandas as pd


def fix_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Thay 365243 (lỗi ghi nhận) bằng NaN — thống nhất với pipeline dự án."""
    if "DAYS_EMPLOYED" in df.columns:
        df = df.copy()
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    return df


class TabularPreprocessor:
    """
    Học median (số) và từ điển (phân loại) trên tập train;
    áp dụng cho val/test.
    """

    def __init__(self, cat_cols: list[str], num_cols: list[str]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self._medians: dict[str, float] = {}
        self._cat_vocab: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        df = fix_days_employed_anomaly(df)
        for c in self.num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            self._medians[c] = float(np.nanmedian(s.values))
        for c in self.cat_cols:
            vals = df[c].astype(str).fillna("__nan__")
            uniq = sorted(vals.unique().tolist())
            self._cat_vocab[c] = {v: i + 1 for i, v in enumerate(uniq)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = fix_days_employed_anomaly(df.copy())
        out = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            out[c] = s.fillna(self._medians[c]).astype(np.float32)
        for c in self.cat_cols:
            vocab = self._cat_vocab[c]
            raw = df[c].astype(str).fillna("__nan__")
            out[c] = raw.map(lambda x: vocab.get(x, 0)).astype(np.int64)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def num_numeric(self) -> int:
        return len(self.num_cols)

    def cat_cardinalities(self) -> list[int]:
        return [len(self._cat_vocab[c]) + 1 for c in self.cat_cols]


def infer_feature_columns(df: pd.DataFrame, drop: tuple[str, ...], cat_hint: tuple[str, ...]) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in drop]
    cat = [c for c in cols if c in cat_hint or df[c].dtype == object]
    num = [c for c in cols if c not in cat]
    return cat, num
