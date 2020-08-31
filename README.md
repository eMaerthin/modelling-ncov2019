# modelling-ncov2019
[![Build Status](https://travis-ci.org/MOCOS-COVID19/modelling-ncov2019.svg?branch=master)](https://travis-ci.org/MOCOS-COVID19/modelling-ncov2019)

## useful links:
### Technical stuff:
* [our slack space](https://modellingncov2019.slack.com/)
* [kanban board for programming team](https://trello.com/b/nZAEFbG0/kanban-board-for-programming-team)
* [Blogpost on data science project](https://towardsdatascience.com/the-data-science-workflow-43859db0415)
* [Blogpost on TDD in data science](https://towardsdatascience.com/tdd-datascience-689c98492fcc)
* [Blogpost on implementing git in data science](https://towardsdatascience.com/implementing-git-in-data-science-11528f0fb4a7)
#### Python components
* [NetworkX - Software for complexe networks](https://networkx.github.io/)
* [Stellargraph - Machine Learning on graphs](https://github.com/stellargraph/stellargraph)
### Tracking the epidemics
* [Timeline of the 2019 Wuhan coronavirus outbreak](https://en.wikipedia.org/wiki/Timeline_of_the_2019%E2%80%9320_Wuhan_coronavirus_outbreak)
* [Tracking coronavirus: Map, data and timeline](https://bnonews.com/index.php/2020/02/the-latest-coronavirus-cases/)

## Project Organization
------------

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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
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
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

### Git Large File Storage
[git lfs](https://git-lfs.github.com/) should be used to store big files.
Please follow [instructions](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) to set up git-lfs on your end.
As of now the following paths are tracked with git-lfs:
- `data/*/*.zip`
- `data/*/*.csv`
- `data/*/*.xlsx`
- `references/*.pdf`
- `notebooks/*.ipynb`

If you need to track different paths, please add them using `git lfs track [path-to-be-tracked]`.
This will append new lines to `.gitattributes` file.
