"""Handle data preprocessing for the Disease Risk Prediction project."""

import numpy as np
import pandas as pd
import sklearn
from category_encoders.one_hot import OneHotEncoder
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import disease_risk_prediction.constants as c
from disease_risk_prediction.data import HealthDataValidator

sklearn.set_config(transform_output="pandas")


class VIFFeatureDropper(
    BaseEstimator,
    TransformerMixin,
):
    """
    Drop features with high Variance Inflation Factor (VIF) from input DataFrame.

    Parameters:
    - threshold (float): The VIF threshold above which features will be dropped.
    - n_jobs (int): Number of CPU cores to use for parallel VIF computation.

    Attributes:
    - high_vif_features (list): The list of features identified with high VIF.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        n_jobs: int | None = -1,
    ) -> None:
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.high_vif_features: list[str] = []
        self.feature_names_in_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "VIFFeatureDropper":
        """
        Identify features with VIF above the threshold.

        Parameters:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Target variable (ignored).

        Returns:
        - self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")

        self.feature_names_in_ = X.columns.to_list()

        # Calculate VIF for each feature.
        vif_values = Parallel(
            n_jobs=self.n_jobs,
        )(delayed(variance_inflation_factor)(X.values, i) for i in range(X.shape[1]))

        vif_data = pd.DataFrame(
            data={
                "feature": X.columns,
                "VIF": vif_values,
            }
        )

        # Identify high-VIF features.
        self.high_vif_features = vif_data[vif_data["VIF"] > self.threshold][
            "feature"
        ].tolist()
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Drop high-VIF features from the DataFrame.

        Parameters:
        - X (pd.DataFrame): Feature matrix.

        Returns:
        - pd.DataFrame: Reduced feature matrix.
        """
        if not self.high_vif_features:
            raise NotFittedError(
                "This VIFFeatureDropper instance is not fitted yet. Call 'fit' first."
            )

        return X.drop(columns=self.high_vif_features, errors="ignore")

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """
        Get the list of output feature names after dropping high-VIF features.

        Parameters:
        - input_features (list[str]): Original feature names.

        Returns:
        - list[str]: Remaining feature names.
        """
        if input_features is None:
            input_features = self.feature_names_in_

        return [
            feature
            for feature in input_features
            if feature not in self.high_vif_features
        ]


def get_preprocess_pipeline() -> Pipeline:
    """Get pre-processing pipeline."""

    scaler_encoder = ColumnTransformer(
        n_jobs=-1,
        transformers=[
            ("numerical", StandardScaler(), c.NUMERICAL_COLS),
            (
                "categorical",
                OneHotEncoder(
                    drop_invariant=True,  # Drop columns with 0 variance.
                    use_cat_names=True,
                    verbose=1,
                ),
                c.CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,  # Don't prepend to the feature names.
        verbose=True,
    )

    deskewer = sklearn.compose.ColumnTransformer(
        n_jobs=-1,
        transformers=[
            (
                "log",
                sklearn.preprocessing.FunctionTransformer(
                    np.log1p,
                    validate=True,
                    feature_names_out="one-to-one",
                ),
                c.LOG_COLS,
            ),
            (
                "power",
                sklearn.preprocessing.PowerTransformer(method="yeo-johnson"),
                c.POWER_COLS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,  # Don't prepend to the feature names.
        verbose=True,
    )

    return Pipeline(
        [
            (
                "data_validator",
                HealthDataValidator(),
            ),
            ("deskewer", deskewer),
            ("scaler_encoder", scaler_encoder),
            ("variance_filter", VarianceThreshold()),
            ("vif_filter", VIFFeatureDropper()),
        ],
        verbose=True,
    )


def get_disease_df(
    X: pd.DataFrame,
    y: pd.DataFrame,
    disease_col: str,
) -> pd.DataFrame:
    """Create standardized DataFrame for specified disease."""
    mask = ~y.isna()

    disease_df = pd.concat(
        objs=[
            X[mask],
            y[mask],
        ],
        axis="columns",
    ).rename(
        columns={disease_col: "target"},
    )

    disease_df["disease"] = disease_col

    disease_df = disease_df.drop_duplicates()

    return disease_df


def get_training_df(
    X: pd.DataFrame,
    ys: pd.DataFrame,
) -> pd.DataFrame:
    """Create unified, standardized DataFrame for all diseases."""
    data_path = c.DATA_DIR / "training.feather"

    if data_path.exists():
        return pd.read_feather(data_path)

    ys = ys.loc[X.index]

    # FIXME:
    # training_dfs = [
    #     get_disease_df(
    #         X, ys, col, values["keep_values"], values["ones"], values["zeros"]
    #     )
    #     for col, values in c.TRAINING_PREP_DATA.items()
    # ]

    # training_df = pd.concat(training_dfs, ignore_index=True)

    # training_df.to_feather(data_path)

    # if len(training_df) != c.NUM_RECORDS_VALID_2023:
    #     logger.error(
    #         f"DataFrame has {len(training_df)} (not {c.NUM_RECORDS_VALID_2023}) rows!!!",
    #     )

    # return training_df
