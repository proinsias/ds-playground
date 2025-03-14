#!/usr/bin/env python
# coding: utf-8
# %%
"""Exploratory Data Analysis."""

# %% [markdown]
# ## Imports

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn.preprocessing
import sklearn.compose
import ydata_profiling
from category_encoders.one_hot import OneHotEncoder
from IPython.core.interactiveshell import InteractiveShell
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import disease_risk_prediction.constants as c
from disease_risk_prediction.data import (
    fetch_health_data,
    HealthDataValidator,
    HealthTrainingDataValidator,
)
from disease_risk_prediction.preprocess import (
    get_preprocess_pipeline,
)


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
# ## Load data

# %%
health_df = fetch_health_data()


# %%
health_df.shape
health_df.head()

# %% [markdown]
# ## Validate

# %%
valid_health_df = HealthDataValidator().fit_transform(health_df)

# %%
valid_health_df.shape
valid_health_df.head()

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Overall

# %%
valid_health_df.describe()

# %%
profile = ydata_profiling.ProfileReport(valid_health_df, title="Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
temp_df = valid_health_df[
    [
        "state_latitude",
        "state_longitude",
    ]
].drop_duplicates()

fig = px.scatter_geo(
    temp_df,
    lat="state_latitude",
    lon="state_longitude",
    scope="usa",
    title="US States Latitude/Longitude Locations",
)

fig.show()

# %% [markdown]
# ### Outliers / skew

# %%
log_transformer = sklearn.preprocessing.FunctionTransformer(
    np.log1p,
    validate=True,
    feature_names_out="one-to-one",
)
power_transformer = sklearn.preprocessing.PowerTransformer(method="yeo-johnson")
power_cols = ["children", "physhlth", "menthlth"]
log_cols = ["wtkg3", "htm4"]
passthrough_cols = [
col for col in valid_health_df.columns if col not in [*power_cols, *log_cols]
]

preprocessor = sklearn.compose.ColumnTransformer(
[
    ("log", log_transformer, log_cols),
    ("power", power_transformer, power_cols),
    ("num", "passthrough", passthrough_cols),
],
verbose_feature_names_out=False,  # Don't prepend to the feature names.
)

temp_df = pd.DataFrame(
data=preprocessor.fit_transform(valid_health_df),
columns=preprocessor.get_feature_names_out(),
)

# %%
ax = valid_health_df["children"].hist(bins=50)
ax.set_yscale("log")

# %%
temp_df["physhlth"].hist(bins=50)

# %%
ax = valid_health_df["physhlth"].hist(bins=50)
ax.set_yscale("log")

# %%
temp_df["physhlth"].hist(bins=50)

# %%
ax = valid_health_df["menthlth"].hist(bins=50)
ax.set_yscale("log")

# %%
temp_df["menthlth"].hist(bins=50)

# %%
ax = valid_health_df["wtkg3"].hist(bins=50)
ax.set_yscale("log")

# %%
temp_df["wtkg3"].hist(bins=50)

# %%
ax = valid_health_df["htm4"].hist(bins=50)
ax.set_yscale("log")

# %%
temp_df["htm4"].hist(bins=50)

# %% [markdown]
# ## Multicollinearity

# %% [markdown]
# ### corr

# %%
X = valid_health_df.drop(columns=c.Y_COLS)


# %%
def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.5) -> None:
    """Plots a heatmap of highly correlated features."""
    corr_matrix = df.corr().abs()
    high_corr = corr_matrix[corr_matrix > threshold]

    plt.figure(figsize=(10, 8));
    sns.heatmap(high_corr, cmap='coolwarm', annot=True);
    plt.title('Correlation Heatmap (Threshold > {:.1f})'.format(threshold));
    plt.show();

plot_correlation_heatmap(X)

# %% [markdown]
# ### VIF

# %%
# %%time

preprocessor = ColumnTransformer(
    n_jobs=-1,
    transformers=[
        ("num", StandardScaler(), c.NUMERICAL_COLS),
        (
            "cat",
            OneHotEncoder(
                drop_invariant=True,  # Drop columns with 0 variance.
                use_cat_names=True,
            ),
            c.CATEGORICAL_COLS,
        ),
    ],
    verbose_feature_names_out=False,  # Don't prepend to the feature names.
)

X_preproc = preprocessor.fit_transform(X)

n_jobs = -1
vif_values = Parallel(
    n_jobs=n_jobs,
)(delayed(variance_inflation_factor)(X_preproc.values, i) for i in range(X_preproc.shape[1]))

vif_data = pd.DataFrame(
    data={
        "feature": X_preproc.columns,
        "VIF": vif_values,
    }
)

# %%
vif.sort_values(by='VIF').reset_index(drop=True)

# %%
vif['VIF'].describe()

# %%
vif['VIF'].hist(bins=50);

# %%
threshold1 = 5
high_vif_features1 = vif[vif["VIF"] > threshold1]["feature"].tolist()
X_reduced1 = X_preproc.drop(columns=high_vif_features1)

# %%
vif_values1 = Parallel(
    n_jobs=n_jobs,
)(delayed(variance_inflation_factor)(X_reduced1.values, i) for i in range(X_reduced1.shape[1]))

vif1 = pd.DataFrame(
    data={
        "feature": X_reduced1.columns,
        "VIF": vif_values1,
    }
)

# %%
vif1.sort_values(by='VIF').reset_index(drop=True)

# %%
vif1['VIF'].hist(bins=50);

# %% [markdown]
# ## Pre-process data

# %%
preprocessor = get_preprocess_pipeline()

# %%
# %%time
X = preprocessor.fit_transform(health_df)

# %%
# %%time
ys = HealthTrainingDataValidator().fit_transform(health_df)

# %%
# FIXME: Check distributions!

# %%
ys.head()

# %%
# FIXME: get_training_df instead!

preprocessed_df = pd.concat(
    objs=[
        X,
        ys.loc[X.index],
    ],
    axis="columns",
)

# %%
preprocessed_df.head()

# %% [markdown]
# ## EDA of training data

# %%
profile = ydata_profiling.ProfileReport(training_df, title="Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
