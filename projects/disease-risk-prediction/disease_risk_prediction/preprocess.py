"""Handle data preprocessing for the Disease Risk Prediction project."""

from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
from category_encoders.one_hot import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sklearn.set_config(transform_output="pandas")


def preprocess_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """Encode categorical features and scale numerical features."""
    categorical_cols = [
        "sex",
        "marital",
        "veteran3",
        "employ1",
        "smoke100",
        "flushot7",
        "pneuvac4",
    ]

    df[categorical_cols] = df[categorical_cols].astype("category")

    numerical_cols = [
        "educa",  # Ordinal.
        "income3",  # Ordinal.
        "children",
        "wtkg3",
        "htm4",
        "physhlth",
        "menthlth",
        "genhlth",  # Ordinal.
        "checkup1",  # Ordinal.
        "state_latitude",
        "state_longitude",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(
                    drop_invariant=True,  # Drop columns with 0 variance.
                    use_cat_names=True,
                ),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )

    power_cols = ["children", "physhlth", "menthlth"]
    log_cols = ["wtkg3", "htm4"]
    y_cols = [
        col for col in df.columns if col not in [*numerical_cols, *categorical_cols]
    ]

    deskewer = sklearn.compose.ColumnTransformer(
        transformers=[
            (
                "log",
                sklearn.preprocessing.FunctionTransformer(
                    np.log1p,
                    validate=True,
                    feature_names_out="one-to-one",
                ),
                log_cols,
            ),
            (
                "power",
                sklearn.preprocessing.PowerTransformer(method="yeo-johnson"),
                power_cols,
            ),
            (
                "passthrough",
                "passthrough",
                [
                    col
                    for col in df.columns
                    if col not in [*power_cols, *log_cols, *y_cols]
                ],
            ),
        ],
        verbose_feature_names_out=False,  # Don't prepend to the feature names.
    )

    pipeline = Pipeline(
        [
            ("deskewer", deskewer),
            ("transformer", preprocessor),
        ],
    )

    X = df.drop(columns=y_cols)
    ys = df[y_cols]

    X = pd.DataFrame(
        data=pipeline.fit_transform(X),
        columns=pipeline.get_feature_names_out(),
    )

    return (
        X,
        ys,
        pipeline,
    )
