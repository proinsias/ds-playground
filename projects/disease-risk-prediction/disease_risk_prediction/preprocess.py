"""Handle data preprocessing for the Disease Risk Prediction project."""

from typing import Tuple

import pandas as pd
import sklearn
from category_encoders.one_hot import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

sklearn.set_config(transform_output="pandas")


def preprocess_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Encode categorical features and scale numerical features."""
    categorical_cols = [
        "gender",
        "smoking_status",
        "alcohol_use",
        "physical_activity",
        "blood_pressure",
        "cholesterol_level",
        "diabetes_history",
    ]

    numerical_cols = ["age", "bmi"]

    # Define the column transformer
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
    )

    # Apply the transformations
    X = df.drop(columns=["disease_outcome"])
    y = df["disease_outcome"]
    X = preprocessor.fit_transform(X)

    return (
        X,
        y,
        preprocessor,
    )
