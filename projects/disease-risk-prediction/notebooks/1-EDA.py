#!/usr/bin/env python
# coding: utf-8
# %%
"""Exploratory Data Analysis."""

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
from IPython.core.interactiveshell import InteractiveShell

from disease_risk_prediction.data import (
    fetch_health_data,
    validate_health_data,
)
from disease_risk_prediction.preprocess import (
    preprocess_training_data,
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
valid_health_df = validate_health_data(health_df)

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
log_transformer = sklearn.preprocessing.FunctionTransformer(np.log1p, validate=True)
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
)

# %%
ax = valid_health_df["children"].hist(bins=50)
ax.set_yscale("log")

# %%
temp_df["children"].hist(bins=50)

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
# ## Pre-process data

# %%
X, ys, _ = preprocess_training_data(valid_health_df)

# %%
preprocessed_df = pd.concat(
    objs=[
        X,
        ys,
    ],
    axis="columns",
)

# %%
preprocessed_df.head()

# %% [markdown]
# ## EDA of pre-processed data

# %%
profile = ydata_profiling.ProfileReport(preprocessed_df, title="Profiling Report")

# %%
profile.to_notebook_iframe()
