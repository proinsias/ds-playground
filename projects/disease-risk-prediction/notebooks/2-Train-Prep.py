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
"""Training dataset preparation."""

# %% [markdown]
# ## Imports

# %%
import gc
import warnings

import hdbscan
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import ppscore as pps
import seaborn as sns
import sklearn.preprocessing
import sklearn.compose
import sklearn.dummy
import umap
import umap.plot
import xgboost as xgb
import ydata_profiling
from dbscan import DBSCAN
from IPython.core.interactiveshell import InteractiveShell
from openTSNE import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.metrics import adjusted_rand_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from yellowbrick.features import joint_plot, Manifold, pca_decomposition
from yellowbrick.features.manifold import manifold_embedding
from yellowbrick.features.pcoords import parallel_coordinates
from yellowbrick.features.radviz import radviz

import disease_risk_prediction.constants as c
from disease_risk_prediction.data import (
    fetch_health_data,
    HealthTrainingDataValidator,
)
from disease_risk_prediction.preprocess import (
    get_preprocess_pipeline,
    get_training_df,
)
from disease_risk_prediction.train import build_model, get_X_y_df


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
training_df = get_training_df(None)

# %%
training_df.shape
training_df.head()

# %%

# %%

# %%

# %%
disease = 'asthms1'
X, y, disease_df = get_X_y_df(training_df, disease)

# %% [markdown]
# ## Find most balanced targets

# %%
data = {
    'disease': [],
    'perc_0': [],
}

for disease, groupdf in training_df.groupby('disease'):
    s = groupdf['target'].value_counts(normalize=True)
    data['disease'].append(disease)
    data['perc_0'].append(int(100 * s[0]))

pd.DataFrame(data=data).assign(
    diff_perc_0=lambda x: abs(x.perc_0 - 50)
).sort_values(by='diff_perc_0')


# %% [markdown]
# ## Class separability

# %% [markdown]
# ### radvix

# %%
def radvix_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    _ = radviz(
        X,
        y,
        classes=[
            f"Negative {disease}",
            f"Positive {disease}",
        ],
    )


# %%
for disease in training_df['disease'].unique():
    radvix_disease(training_df, disease)


# %% [markdown]
# ### parallel_coordinates

# %%
def parallel_coordinates_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    _ = parallel_coordinates(
        X,
        y,
        classes=[
            f"Negative {disease}",
            f"Positive {disease}",
        ],
        fast=True,
        sample=0.2,
        shuffle=True,
        random_state=c.RANDOM_STATE,
    )


# %%
parallel_coordinates_disease(training_df, disease)


# %% [markdown]
# ### f_classif

# %%
def f_classif_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    f_values, _ = f_classif(X, y)

    # Higher Fisher’s Score → better class separability for that feature.

    ax = pd.Series(f_values).hist();
    ax.set_xlabel('Fisher score');
    ax.set_ylabel('Number of features');
    ax.set_title(f'Distribution of Fisher sources for {disease}');
    plt.show();


# %%
for disease in training_df['disease'].unique():
    f_classif_disease(training_df, disease)


# %% [markdown]
# ### mutual_info_classif

# %% [markdown]
# Could also use:
#
# https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html

# %%
def mutual_info_classif_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    m_values = mutual_info_classif(
        X=X,
        y=y,
        random_state=c.RANDOM_STATE,
        n_jobs=-1,
    )

    ax = pd.Series(m_values).hist(bins=50);
    ax.set_xlabel('Mutual information');
    ax.set_ylabel('Number of features');
    ax.set_title(f'Distribution of mutual information for {disease}');
    plt.show();


# %%
for disease in training_df['disease'].unique():
    mutual_info_classif_disease(training_df, disease)


# %% [markdown]
# ### Pearson correlation

# %% [markdown]
# Could also use:
#
# https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html

# %%
def corr_disease(training_df, disease):
    _, _, disease_df = get_X_y_df(training_df, disease)
    
    corr_df = disease_df.corr()
    ax = corr_df.loc[corr_df.index != 'target', 'target'].hist();
    ax.set_xlabel('Pearson correlation');
    ax.set_ylabel('Number of features');
    ax.set_title(f'Distribution of Pearson correlation for {disease}');
    plt.show();


# %%
for disease in training_df['disease'].unique():
    corr_disease(training_df, disease)


# %% [markdown]
# ### pscore

# %%
def ppscore_disease(training_df, disease):
    _, _, disease_df = get_X_y_df(training_df, disease)

    print(f'{disease}:')
    
    display(
        pps.predictors(disease_df, y="target")['ppscore'].value_counts()
    )


# %%
for disease in training_df['disease'].unique():
    ppscore_disease(training_df, disease)


# %% [markdown]
# ### kmeans

# %% [markdown]
# Apply unsupervised clustering and see if the clusters align with class labels.
#
# A higher Adjusted Rand Index means the clustering aligns well with class labels — indicating good separability.

# %%
def kmeans_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    kmeans = KMeans(n_clusters=2, random_state=c.RANDOM_STATE)
    clusters = kmeans.fit_predict(X)
    score = adjusted_rand_score(y, clusters)
    print(f"Adjusted rand index for {disease}: {score:.2f}")


# %%
for disease in training_df['disease'].unique():
    kmeans_disease(training_df, disease)


# %% [markdown]
# ### sklearn dbscan

# %%
def sklearn_dbscan_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    model = sklearn.cluster.DBSCAN(eps=1.25, n_jobs=-1)
    clusters = model.fit_predict(X)
    score = adjusted_rand_score(y, clusters)
    print(f"Adjusted rand index for {disease}: {score:.2f}")


# %%
# %%time

for disease in training_df['disease'].unique():
    sklearn_dbscan_disease(training_df, disease)


# %% [markdown]
# ### dbscan

# %%
def dbscan_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)

    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
    
    clusters, _ = DBSCAN(X, eps=0.3, min_samples=10)
    
    score = adjusted_rand_score(y, clusters)
    print(f"Adjusted rand index for {disease}: {score:.2f}")

    gc.collect();


# %%
# # Caused kernel issues.
#
# try:
#     for disease in training_df['disease'].unique():
#         dbscan_disease(training_df, disease)
# except Exception as e:
#     print(e)

# %% [markdown]
# ### hdbscan

# %%
def hdbscan_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)

    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
    
    model = hdbscan.HDBSCAN(
        core_dist_n_jobs=-1,
        min_cluster_size=10,
    )
    clusters = model.fit_predict(X)
    
    score = adjusted_rand_score(y, clusters)
    print(f"Adjusted rand index for {disease}: {score:.2f}")

    model = None
    gc.collect();


# %%
# %%time

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

    for disease in training_df['disease'].unique():
        hdbscan_disease(training_df, disease)


# %% [markdown]
# ### pca

# %%
# yellowbrick PCA didn't work for me.

def pca_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X).to_numpy()  # NB To avoid issue with X being a DataFrame.
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5);
    plt.xlabel('PC1');
    plt.ylabel('PC2');
    plt.title(f'PCA Visualization of Class Separability for {disease}');
    plt.show();


# %%
for disease in training_df['disease'].unique():
    pca_disease(training_df, disease)


# %% [markdown]
# ### pca 3d

# %%
# yellowbrick PCA didn't work for me - actually I think I just needed to add to_numpy().

def pca_3d_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    # Apply PCA with 3 components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X).to_numpy()

    # Create an interactive 3D scatter plot with Plotly
    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=y,
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Class'},
        title=f'3D PCA Visualization of Class Separability for {disease}',
        opacity=0.5,
    );

    fig.show();



# %%
# # Didn't give me anything more useful.

# for disease in training_df['disease'].unique():
#     pca_3d_disease(training_df, disease)

# %% [markdown]
# ### Rank1D / Rank2D

# %% [markdown]
# This didn't work for me:
#
# https://www.scikit-yb.org/en/latest/api/features/rankd.html

# %% [markdown]
# ### openTSNE

# %%
# # Had a similar 'TypeError: '(slice(None, None, None), 0)' is an invalid key'
# # that I believe is related to X being DataFrame, even though I used to_numpy!

# embeddings = TSNE(
#     metric="euclidean",
#     n_jobs=-1,
#     random_state=c.RANDOM_STATE,
#     verbose=True,
# ).fit(X.to_numpy())

# fig, ax = plt.subplots();

# ax.scatter(embeddings[:, 0], embeddings[:, 1], c = [colors[i] for i in y], s=5);
# ax.set_title("openTSNE Visualisation", fontsize=20, weight="bold");

# plt.show();

# %% [markdown]
# ### tsne

# %%
# # Still having issue with 'slice' exception.

# # %%time

# selectk = SelectKBest(
#     k=5,
#     score_func=f_classif,
# )

# X_k = selectk.fit_transform(X, y)

# X_k_sample, _, y_sample, _ = train_test_split(
#     X_k,
#     y,
#     train_size=1000,  # Out of 221397.
#     random_state=c.RANDOM_STATE,
#     shuffle=True,
#     stratify=y,
# )

# viz = Manifold(manifold="tsne", classes=["no disease", "disease"])

# viz.fit_transform(X_k_sample.to_numpy(), y_sample.to_numpy())
# viz.show()

# %% [markdown]
# ### umap

# %%
def umap_disease(training_df, disease):
    X, y, _ = get_X_y_df(training_df, disease)
    
    train_size = 10 ** 4  # Out of 221397.
    
    selectk = SelectKBest(
        k=5,
        score_func=f_classif,
    )
    
    X_k = selectk.fit_transform(X, y)
    
    X_k_sample, _, y_sample, _ = train_test_split(
        X_k,
        y,
        train_size=train_size,  # Out of 221397.
        random_state=c.RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
    
    mapper = umap.UMAP().fit(
        X_k_sample,
        y_sample,
    )

    fig, ax = plt.subplots();
    
    umap.plot.points(mapper, labels=y_sample, ax=ax)

    plt.show();


# %%

# %%

# %%
# %%time

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

    for disease in training_df['disease'].unique():
        print(disease)
        umap_disease(training_df, disease)

# %% [markdown]
# ### joint_plot

# %%
visualizer = joint_plot(X, y, columns="menthlth")

# %% [markdown]
# ### violinplot

# %%
sns.violinplot(x=y, y=X["menthlth"]);
plt.title('Feature Distribution by Class');
plt.show();
