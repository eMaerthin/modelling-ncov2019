# modelling-ncov2019
## README for reproducing results presented in the "Mitigation and herd immunity strategy for COVID-19 is likely to fail"
### System requirements
#### All software dependencies and operating systems (including version numbers)
Operating system: The software was tested on macOS Catalina 10.15.4, macOS Mojave 10.14.6 and CentOS Linux 7 (Core).
Python 3.7.3 is required.
There is a list of python libraries required for running the code enlisted in `requirements.txt` in the repository (see [requirements.txt](https://github.com/MOCOS-COVID19/modelling-ncov2019/blob/batch-poland-grid/requirements.txt)).

### Installation guide
#### Instructions
- make sure [GIT LFS is installed](https://developer.lsst.io/v/DM-7552/tools/git_lfs.html)
- clone the repository initiated to branch used for producing the paper <code>git clone --single-branch --branch batch-poland-grid git@github.com:MOCOS-COVID19/modelling-ncov2019.git</code>

(Tip: This is a live codebase - There are some changes everyday from the time of writing the paper including completely new model written in julia language which is faster and has more components. Some of the changes are not backward compatible, so we recommend you to use the branch <code>batch-poland-gid</code> specified in the command above to reproduce paper's results.)

- enter the root of the project and initialize git lfs `git lfs install`
- extract wroclaw dataset to directory `data`: `tar -xf data/raw/wroclaw_population.tar.gz -C data/`
- create virtual env `python3 -m venv venv`
- activate virtual env `source venv/bin/activate`
- install python libraries required for running the code `pip install -r requirements.txt`

#### Typical install time on a "normal" desktop computer
- under 5 minutes

### Demo
#### Instructions to run demo
TODO
#### Expected output
TODO
#### Expected run time for demo on a "normal" desktop computer
TODO

### Instructions for use
#### How to run the software on your data
#### Reproduction instructions
##### Steps to reproduce Figure 4
1. `mkdir data/poland-dir`
2. `tar -xf data/raw/poland_population.tar.gz -C data/poland-dir`
3. ``

(We encourage you to include instructions for reproducing all the quantitative results in the manuscript.)

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
- `data/*/*.tar.gz`
- `references/*.pdf`
- `notebooks/*.ipynb`

If you need to track different paths, please add them using `git lfs track [path-to-be-tracked]`.
This will append new lines to `.gitattributes` file.
