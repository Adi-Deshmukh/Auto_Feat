# backend/app/services/preprocessing.py
from __future__ import annotations
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



class Preprocessor:
    """
    A production-ready preprocessing service that:
    * Handles datetime → year/month/day/dayofweek
    * Builds a single scikit-learn pipeline for features
    * Encodes the target (classification) or casts to numeric (regression)
    * Allows fit-once / transform-many
    """

    def __init__(self, problem_type: str, target_column: str):
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'")

        self.problem_type = problem_type
        self.target_column = target_column

        self._feature_pipeline: ColumnTransformer | None = None
        self._target_encoder: LabelEncoder | None = None
        self._datetime_cols: list[str] = []


    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].copy()

        # 1. Extract datetime components *before* pipeline construction
        X, self._datetime_cols = self._decompose_datetimes(X)

        # 2. Build & fit the feature pipeline
        self._feature_pipeline = self._build_feature_pipeline(X)
        X_arr = self._feature_pipeline.fit_transform(X)

        # 3. Re-create DataFrame with proper column names
        X_processed = self._array_to_dataframe(X_arr, X)

        # 4. Process target
        y_processed = self._process_target(y)

        return X_processed, y_processed

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self._feature_pipeline is None:
            raise RuntimeError("Call fit_transform first.")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].copy()

        # Re-apply the same datetime decomposition
        X, _ = self._decompose_datetimes(X, fit=False)

        X_arr = self._feature_pipeline.transform(X)
        X_processed = self._array_to_dataframe(X_arr, X)

        y_processed = self._process_target(y, fit=False)
        return X_processed, y_processed

    # DateTime decomposition

    @staticmethod
    def _decompose_datetimes(
        X: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, list[str]]:
        datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        X_out = X.copy()
        new_cols = []

        for col in datetime_cols:
            prefix = f"{col}_"
            X_out[f"{prefix}year"] = X_out[col].dt.year
            X_out[f"{prefix}month"] = X_out[col].dt.month
            X_out[f"{prefix}day"] = X_out[col].dt.day
            X_out[f"{prefix}dow"] = X_out[col].dt.dayofweek
            new_cols.extend([f"{prefix}year", f"{prefix}month", f"{prefix}day", f"{prefix}dow"])
            X_out = X_out.drop(columns=col)

        return X_out, datetime_cols if fit else datetime_cols
    
    # Imputer and Scaler

    def _build_feature_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            [
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )
        
    
# getting names
    def _array_to_dataframe(self, arr: np.ndarray, original_X: pd.DataFrame) -> pd.DataFrame:
        num_cols = original_X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = original_X.select_dtypes(include=["object", "category"]).columns.tolist()

        onehot_names = (
            self._feature_pipeline.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(cat_cols)
            .tolist()
        )
        columns = num_cols + onehot_names
        return pd.DataFrame(arr, columns=columns, index=original_X.index)

    def _process_target(self, y: pd.Series, fit: bool = True) -> pd.Series:
        if self.problem_type == "classification":
            if fit:
                self._target_encoder = LabelEncoder()
                y_out = self._target_encoder.fit_transform(y)
            else:
                y_out = self._target_encoder.transform(y)
            return pd.Series(y_out, index=y.index, name=y.name)

        # regression → coerce to float, fill NaN with 0 (or median, etc.)
        return pd.to_numeric(y, errors="coerce").fillna(0)