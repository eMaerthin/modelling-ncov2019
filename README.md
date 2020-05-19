# modelling-ncov2019
## README for reproducing results presented in the "Mitigation and herd immunity strategy for COVID-19 is likely to fail"
### System requirements
#### All software dependencies and operating systems (including version numbers)
Operating system: The software was tested on macOS Catalina 10.15.4, macOS Mojave 10.14.6 and CentOS Linux 7 (Core).
Python 3.6.8 or later is required (was tested on Python 3.6.8 and Python 3.7.3).
There is a list of python libraries required for running the code enlisted in `requirements.txt` in the repository (see [requirements.txt](https://github.com/MOCOS-COVID19/modelling-ncov2019/blob/batch-poland-grid/requirements.txt)).

### Installation guide
#### Instructions
- make sure [GIT LFS is installed](https://developer.lsst.io/v/DM-7552/tools/git_lfs.html)
- clone the repository initiated to branch used for producing the paper `git clone --single-branch --branch grid_SD_detection git@github.com:MOCOS-COVID19/modelling-ncov2019.git` or `git clone --single-branch --branch grid_SD_detection https://github.com/MOCOS-COVID19/modelling-ncov2019.git`

(Tip: This is a live codebase - There are some changes everyday from the time of writing the paper including completely new model written in julia language which is faster and has more components. Some of the changes are not backward compatible, so we recommend you to use the branch <code>batch-poland-gid</code> specified in the command above to reproduce paper's results.)

- enter the root of the project and initialize git lfs `git lfs install`
- extract wroclaw dataset to directory `data/raw/wroclaw_population`:
  1. `mkdir data/raw/wroclaw_population`
  2. `tar -xf data/raw/wroclaw_population.tar.gz -C data/raw/wroclaw_population`
- create virtual env `python3 -m venv venv`
- activate virtual env `source venv/bin/activate`
- install python src `pip install -e .`
- install python libraries required for running the code `pip install -r requirements.txt`

#### Typical install time on a "normal" desktop computer
- under 5 minutes

### Demo
In order to run a simulation we need a json file and two files
with individual level attributes of a population and household
assignment.
Population file has as a header specifying which kind of attributes
persons have and as many rows as the size of population.
There are three obligatory attributes that should exist in every
generated population: `gender`, `sex` and `household_id`.
If possible, one can add more attributes like average time spent
while commuting, social activity score indicating how social
the person is, the classification of work the person does etc.

#### Instructions to run demo
In order to run the demo we prepared a Wroclaw population and
demo json file.
Assuming you already did steps from installation guide please
ensure the environment is already activated, go to root path
of the project and run the demo by typing in your commandline:

<code>python3 src/models/infection_model.py --params-path experiments/demo/demo.json --df-individuals-path data/raw/wroclaw_population/population_experiment.csv --df-households-path data/raw/wroclaw_population/households_experiment.csv run-simulation</code>


#### Expected output
Demo is configured to output updated statistics every simulated "24h" including per-day-increase, number of active cases, number of deaths, cumulative number of affected cases.

In the end a successful demo should create a folder `outputs/demo/` inside the root directory.

Inside the folder a unique subfolder is created and inside that folder all pictures and times are stored.

The demo represents the free growth of epidemics - the outcome should be similar to the picture from Figure 1 from the paper.

One of resulted files should be `paper_bins.png` and `paper_summary.png`
which were used respectively as Figure 1 left and respectively Figure 1 right panel.

#### Expected run time for demo on a "normal" desktop computer

The demo should finish in 15 minutes. The time can differ depending on the RAM available.

### Instructions for use
#### How to run the software on your data
#### Reproduction instructions

##### Steps to reproduce Figure 1
1. `mkdir data/raw/wroclaw_population`
2. `tar -xf data/raw/wroclaw_population.tar.gz -C data/raw/wroclaw_population`
3. `bash experiment/figure1_experiment/run.sh`

#####

##### Steps to reproduce Figure 4
1. `mkdir data/raw/poland_population`
2. `tar -xf data/raw/poland_population.tar.gz -C data/raw/poland_population`
3. go to `experiments/figure4_experiment` and run `bash create_input_folders.sh` to create simulation folders.
4. go back to root directory of the repository and run `bash experiments/figure4_experiment/run.sh` to run set of 260 experiments. This will last several hours on "normal" desktop.
5. run `bash experiments/figure4_experiment/create_csv.sh` to create final csv file based on outcomes from the simulation.
6. run `python3 experiments/figure4_experiment/summarize.py` to generate two parts of Figure 4 based on final csv file.

##### Steps to reproduce Figure 5
1. `mkdir data/raw/wroclaw_population`
2. `tar -xf data/raw/wroclaw_population.tar.gz -C data/raw/wroclaw_population`
3. go to `experiments/figure5_experiment` and run `bash create_input_folders.sh` to create simulation folders.
4. `bash experiments/figure5_experiment/run.sh` to run 231 experiments. This will last around few hours on "normal" desktop.
5. run `bash experiments/figure5_experiment/create_csv.sh` to create final csv file based on outcomes from the simulation.
6. run `python3 experiments/figure5_experiment/summarize.py` to generate two parts of Figure 5 based on final csv file.

## Project Organization

    ├── .gitattributes
    ├── .gitignore
    ├── config.py
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
    ├── experiments        <- Directory with configs to recreate paper results
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
    ├── test_environment.py
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
