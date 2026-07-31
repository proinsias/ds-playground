# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""Training."""

# %% [markdown]
# ## Imports

# %%
from collections import Counter

import disease_risk_prediction.constants as c
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.compose
import sklearn.dummy
import sklearn.preprocessing
import xgboost as xgb
from disease_risk_prediction.data import HealthTrainingDataValidator, fetch_health_data
from disease_risk_prediction.preprocess import (
    VIFFeatureDropper,
    get_preprocess_pipeline,
    get_training_df,
)
from disease_risk_prediction.train import build_model, get_X_y_df
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from IPython.core.interactiveshell import InteractiveShell
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# %%
matplotlib.use("nbagg")
# prettier plots
plt.style.use("ggplot")
# larger plots - two different ways.
matplotlib.rc("figure", figsize=(15, 10))
plt.rcParams["figure.dpi"] = 90

# larger fonts
sns.set_context("notebook", font_scale=1.5)


# %%
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
pd.set_option("display.max_rows", 500)
pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 80
# Adjust the number of columns profiled and displayed by the `info()` method.
pd.options.display.max_info_columns = 150
# Adjust the number of decimals to be displayed in a DataFrame.
pd.options.display.precision = 15
# Adjust the display format in a DataFrame.
# pd.options.display.float_format = '{:.2f}%'.format
# Prints and parses dates with the year first.
pd.options.display.date_yearfirst = True

InteractiveShell.ast_node_interactivity = "all"


# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Load training datasets

# %%
training_df = get_training_df(None, None)  # Load landed file.

# %%
print(training_df.shape)
training_df.head()

# %%
disease = "asthms1"
X, y, disease_df = get_X_y_df(training_df, disease)


# %% [markdown]
# ## Drop features with low signal with target


# %%
class MutualInfoThresholdSelector(SelectorMixin, BaseEstimator):
    """
    Feature selector that removes features with mutual information (MI) below a given threshold.

    Parameters:
    - threshold (float): Minimum MI score a feature must have to be kept.
    - discrete_features (bool or 'auto'): Whether features are discrete.
    - random_state (int, optional): Random state for reproducibility.

    Attributes:
    - mi_scores_ (np.ndarray): Mutual information scores for each feature.
    """

    # FIXME: threshold is very low, but I don't want to throw away too many features just yet.
    def __init__(
        self,
        threshold: float = 0.001,
        discrete_features: str | bool = "auto",
        random_state: int | None = None,
    ):
        """Store the MI threshold, feature-type hint, and random state used by `fit`."""
        self.threshold = threshold
        self.discrete_features = discrete_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInfoThresholdSelector":
        """
        Compute mutual information scores and determine which features to keep.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target array.

        Returns:
        - self: Fitted selector.
        """
        self.mi_scores_ = mutual_info_classif(
            X,
            y,
            discrete_features=self.discrete_features,
            random_state=self.random_state,
        )
        return self

    def _get_support_mask(self) -> np.ndarray:
        """
        Generate a boolean mask indicating which features to keep.

        Returns:
        - np.ndarray: Boolean array where True indicates a feature is kept.
        """
        return self.mi_scores_ >= self.threshold

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce feature matrix to selected features.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Reduced feature matrix.
        """
        return X[:, self._get_support_mask()]


# %%
X_vif = VIFFeatureDropper().fit_transform(X)

# %%
mi = MutualInfoThresholdSelector()
X_mi = mi.fit_transform(X_vif.to_numpy(), y)

# %%
print(X_vif.shape, X_mi.shape)

# %% [markdown]
# ## Sample targets

# %% [markdown]
# ### Undersampling

# %%
X_rus, y_rus = RandomUnderSampler(
    sampling_strategy="auto",
    random_state=c.RANDOM_STATE,
).fit_resample(X, y)

print(f"Class distribution before RandomUnderSampler: {Counter(y)}")
print(f"Class distribution after RandomUnderSampler: {Counter(y_rus)}")

# %% [markdown]
# ### Oversampling

# %%
X_smote, y_smote = SMOTE(
    sampling_strategy="auto",
    random_state=c.RANDOM_STATE,
).fit_resample(X, y)

print(f"Class distribution before SMOTE: {Counter(y)}")
print(f"Class distribution after SMOTE: {Counter(y_smote)}")

# %% [markdown]
# ### Under- and over-sampling

# %%
# %%time
X_smt, y_smt = SMOTETomek(
    sampling_strategy="auto",
    random_state=c.RANDOM_STATE,
    n_jobs=-1,
).fit_resample(X, y)

print(f"Class distribution before SMT: {Counter(y)}")
print(f"Class distribution after SMT: {Counter(y_smt)}")


# %%
def compare_training(X, y):
    """Train a dummy baseline, XGBoost, and the Keras model on the same split and print each classification report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=c.RANDOM_STATE,
        stratify=y,
    )

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weight_dict = dict(enumerate(class_weights))

    dummy = sklearn.dummy.DummyClassifier(strategy="prior", random_state=c.RANDOM_STATE)

    _ = dummy.fit(
        X_train,
        y_train,
    )

    print(classification_report(y_test, dummy.predict(X_test)))

    xmodel = xgb.XGBClassifier(
        random_state=c.RANDOM_STATE,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    )

    _ = xmodel.fit(
        X_train,
        y_train,
    )

    print(classification_report(y_test, xmodel.predict(X_test)))

    model = build_model(X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
    )

    print(
        classification_report(y_test, pd.DataFrame(model.predict(X_test)).astype(int)),
    )


# %% [markdown]
# If your classes aren't well separated, you'll likely see poor model performance — so increasing class separability can make a huge difference. Here are some practical strategies:
#
# Feature Engineering:
#     1.  Create Interaction Terms:
# Combine features to capture relationships that single features miss.
#
# ```python
# X['new_feature'] = X['feature1'] * X['feature2']
# ```
#
# https://levelup.gitconnected.com/4-python-libraries-for-automated-feature-engineering-that-you-should-use-in-2023-54bccecb1683
# https://github.com/feature-engine/feature_engine
# https://github.com/cod3licious/autofeat
# https://github.com/alteryx/featuretools/
#
#     2.  Polynomial Features:
# Capture non-linear relationships by adding squared or cubic terms.
#
# ```python
# from sklearn.preprocessing import PolynomialFeatures
#
# poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
# X_poly = poly.fit_transform(X)
# ```
#
#     7.  Class Weights:
# Penalize misclassification of the minority class more heavily.
#
# ```python
# model.fit(X, y, class_weight='balanced')
# ```

# %% [markdown]
# ## Build wide feature/target data for per-disease training

# %%
health_df = fetch_health_data()
preprocessor = get_preprocess_pipeline()
X = preprocessor.fit_transform(health_df)
ys = HealthTrainingDataValidator().fit_transform(health_df)

# %% [markdown]
# ## Train asthma

# %%
ys["asthms1"].value_counts()

# %%
y_asthma = ys["asthms1"]
mask = y_asthma.notna()

X_asthma = X[mask]
y_asthma = y_asthma[mask].astype(int)

# %%
y_asthma.value_counts()

# %% jupyter={"source_hidden": true}
compare_training(X_asthma, y_asthma)

# %% [markdown]
# ## Train arthritis

# %%
ys["drdxar2"].value_counts()

# %%
y_arthritis = ys["drdxar2"]
mask = y_arthritis.notna()

X_arthritis = X[mask]
y_arthritis = y_arthritis[mask].astype(int)

# %%
y_arthritis.value_counts()

# %%
compare_training(X_arthritis, y_arthritis)

# %% [markdown]
# ## Train mi/chd

# %%
ys["michd"].value_counts()

# %%
y_michd = ys["michd"]
mask = y_michd.notna()

X_michd = X[mask]
y_michd = y_michd[mask].astype(int)

# %%
y_michd.value_counts()

# %%
compare_training(X_michd, y_michd)

# %% [markdown]
# ## Train depressive disorder

# %%
ys["addepev3"].value_counts()

# %%
y_addepev3 = ys["addepev3"]
mask = y_addepev3.notna()

X_addepev3 = X[mask]
y_addepev3 = y_addepev3[mask].astype(int)

# %% jupyter={"source_hidden": true}
y_addepev3.value_counts()

# %%
compare_training(X_addepev3, y_addepev3)

# %% [markdown]
# ## Train diabetes

# %%
ys["diabete4"].value_counts()

# %%
y_diabete4 = ys["diabete4"]
mask = y_diabete4.notna()

X_diabete4 = X[mask]
y_diabete4 = y_diabete4[mask].astype(int)

# %%
y_diabete4.value_counts()

# %%
compare_training(X_diabete4, y_diabete4)

# %% [markdown]
# ## Train high blood pressure

# %%
ys["rfhype6"].value_counts()

# %%
y_rfhype6 = ys["rfhype6"]
mask = y_rfhype6.notna()

X_rfhype6 = X[mask]
y_rfhype6 = y_rfhype6[mask].astype(int)

# %%
y_rfhype6.value_counts()

# %%
compare_training(X_rfhype6, y_rfhype6)

# %% [markdown]
# ## Train high cholestrol

# %%
ys["rfchol3"].value_counts()

# %%
y_rfchol3 = ys["rfchol3"]
mask = y_rfchol3.notna()

X_rfchol3 = X[mask]
y_rfchol3 = y_rfchol3[mask].astype(int)

# %%
y_rfchol3.value_counts()

# %%
compare_training(X_rfchol3, y_rfchol3)

# %% [markdown]
# ## Train kidney disease

# %%
ys["chckdny2"].value_counts()

# %%
y_chckdny2 = ys["chckdny2"]
mask = y_chckdny2.notna()

X_chckdny2 = X[mask]
y_chckdny2 = y_chckdny2[mask].astype(int)

# %%
y_chckdny2.value_counts()

# %%
compare_training(X_chckdny2, y_chckdny2)

# %% [markdown]
# ## Train lung disease

# %%
ys["chccopd3"].value_counts()

# %%
y_chccopd3 = ys["chccopd3"]
mask = y_chccopd3.notna()

X_chccopd3 = X[mask]
y_chccopd3 = y_chccopd3[mask].astype(int)

# %%
y_chccopd3.value_counts()

# %%
compare_training(X_chccopd3, y_chccopd3)

# %% [markdown]
# ## Train stroke

# %%
ys["cvdstrk3"].value_counts()

# %%
y_cvdstrk3 = ys["cvdstrk3"]
mask = y_cvdstrk3.notna()

X_cvdstrk3 = X[mask]
y_cvdstrk3 = y_cvdstrk3[mask].astype(int)

# %%
y_cvdstrk3.value_counts()

# %%
compare_training(X_cvdstrk3, y_cvdstrk3)

# %% [markdown]
# ## Train cancer

# %%
ys["cancer"].value_counts()

# %%
y_cancer = ys["cancer"]
mask = y_cancer.notna()

X_cancer = X[mask]
y_cancer = y_cancer[mask].astype(int)

# %%
y_cancer.value_counts()

# %%
compare_training(X_cancer, y_cancer)

# %%

# %%

# %%

# %% [markdown]
# ## Notes

# %% [markdown]
# Best keras:
#
# - addepev3 - 0.71 -> is this also where i have plenty of 1s?
# - rfhype6
# - arthritis
# - X_diabete4
