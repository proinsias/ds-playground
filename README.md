# Heart Disease (heart-disease)

Sandbox machine learning classification project using the
University of California Irvine (UCI) Heart Disease dataset.

This is my playground where I try out new tools and approaches.

<!-- FIXME: Add badges from template. -->

## Table of contents

<!-- toc -->

-   [Project organization](#project-organization)
-   [Updating the table of contents of this file](#updating-the-table-of-contents-of-this-file)

<!-- tocstop -->

## Additional documentation

Full documentation can be found here: [./docs](./docs).

## Project organization

<!-- FIXME: Convert to mermaid? -->

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── heart_disease      <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Contributions

-   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/proinsias/heart-disease?quickstart=1)
-   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/betatim/binderlyzer/master)

## Acknowledgments

The authors of the databases have requested:

      ...that any publications resulting from the use of the data include the
      names of the principal investigator responsible for the data collection
      at each institution.  They would be:

       1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
       2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
       3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
       4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
      Robert Detrano, M.D., Ph.D.

## Updating the table of contents of this file

We use [markdown-toc](https://github.com/jonschlinkert/markdown-toc)
to automatically generate the table of contents for this file.
You can update the TOC using:

```bash
# npm install --global markdown-toc
markdown-toc -i README.md
```
