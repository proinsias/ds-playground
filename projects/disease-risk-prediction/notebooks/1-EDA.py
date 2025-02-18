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
import ydata_profiling
from IPython.core.interactiveshell import InteractiveShell

from disease_risk_prediction.data import (
    fetch_health_data,
    validate_health_data,
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

# %%

# %%
# FIXME:

# Children outliers
# wtkg3 outliers, skew?
# htm4 outliers, skew?
# skew: physhlth, menthlth
