# cover_type_prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Cover (tree species) prediction using sklearn data

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation and supporting materials
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks for analysis and exploration.
│   └── 1v1-KW-exploratory-analysis.ipynb  <- Exploratory analysis of the Covertype dataset
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         cover_type_prediction and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── cover_type_prediction   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cover_type_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    ├── modeling/               <- Model training scripts
    │   ├── v1/                 <- Version 1: Gradient Boosting Classifier (scikit-learn)
    │   │   └── train.py        <- Trains and evaluates a GBM model on the Covertype dataset
    │   ├── v2/                 <- Version 2: Support Vector Machine (scikit-learn)
    │   │   └── train.py        <- Trains and evaluates an SVM model (with downsampling and continuous features)
    │   └── __init__.py         <- Module initialization
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    └── ...                    <- Other source files
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Modeling
--------

- **v1/train.py**: Trains a Gradient Boosting Classifier on the Covertype dataset, evaluates accuracy and prints a classification report.
- **v2/train.py**: Trains a Support Vector Machine on selected continuous features, with downsampling for efficiency. Also prints evaluation metrics.

## Notebooks

- **1v1-KW-exploratory-analysis.ipynb**: Jupyter notebook for exploratory data analysis of the Covertype dataset, feature inspection, and initial insights.

## Usage

- To train models, run the respective scripts in `cover_type_prediction/modeling/v1/train.py` or `cover_type_prediction/modeling/v2/train.py`.
- For data exploration, open the notebook in the `notebooks` directory.

## Requirements

See `requirements.txt` for dependencies.

