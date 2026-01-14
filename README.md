# Tennis Match Outcome Predictor

This project is an end-to-end machine learning pipeline for predicting the outcome of professional tennis matches using historical ATP data.
It is designed as a showcase project for applying machine learning methods to a real-world, noisy, time-dependent dataset.

The focus is on:
- clean data processing pipelines,
- leakage-free feature engineering,
- time-aware evaluation,
- and interpretable tree-based models.

Current models include:
- Decision Trees
- Random Forests
- XGBoost

with different feature configurations:
- baseline (no historical performance features),
- previous-match statistics,
- ELO ratings,
- and combinations of both.

------------------------------------------------------------

Repository Structure

```
tennis-tester/
│
├── data/
│   ├── raw/            # raw ATP match CSV files (not tracked)
│   └── processed/      # processed datasets (not tracked)
│
├── models/             # trained models (not tracked)
├── results/            # evaluation results, metrics, plots (not tracked)
│
├── 1.combine_years.py
├── 2.preprocessing.py
├── 3.feature_engineering.py
├── 4.preparing_data_for_tree_models.py
├── 5.tree_models.py
│
├── requirements.txt
├── .gitignore
└── README.txt
```

Only the code and folder structure are tracked in GitHub.
Datasets, trained models, and experimental results are generated locally.

------------------------------------------------------------

Data

The project uses historical ATP match data.

Source:
https://github.com/JeffSackmann/tennis_atp

Download the yearly match files, for example:
- atp_matches_2019.csv
- atp_matches_2020.csv
- atp_matches_2021.csv
  \\...

Place them in:
data/raw/

The repository does not include the data files themselves.

------------------------------------------------------------

Pipeline Overview

The project follows a clear, numbered pipeline.

1. Combine yearly datasets

Run:
python 1.combine_years.py

This reads all atp_matches_YYYY.csv files from data/raw/ and creates:
data/processed/combined_STARTYEAR-ENDYEAR.csv

------------------------------------------------------------

2. Preprocessing

Handled in code by:
from preprocessing import preprocessing

This step:
- cleans missing values,
- encodes categorical variables,
- builds target labels,
- randomizes player position to avoid dataset bias.

------------------------------------------------------------

3. Feature Engineering

Handled in code by:
from feature_engineering import *

This step adds:
- previous-match statistics,
- ELO ratings computed in a leakage-free way.

------------------------------------------------------------

4. Time-aware splitting

Handled in code by:
from preparing_data_for_tree_models import *

This creates chronological train/test splits that respect match order.

------------------------------------------------------------

5. Model training and evaluation

Run:
python 5.tree_models.py

This runs experiments for:
- baseline model,
- previous-match features,
- ELO features,
- combined features.

Outputs:
- trained models → models/
- evaluation metrics and results → results/

------------------------------------------------------------

Installation

Install the dependencies with:
pip install -r requirements.txt

Main dependencies:
- pandas
- numpy
- scikit-learn
- xgboost

------------------------------------------------------------

Design Principles

No data leakage:
Only information available before a match starts is used as model input.
Score-derived or post-match statistics are explicitly excluded.

Time-aware evaluation:
All model testing respects the chronological order of matches.

Reproducible project structure:
All file paths are relative to the repository:
data/raw/
data/processed/
models/
results/

Separation of concerns:
- raw data → never modified
- processed data → reproducible
- models/results → generated locally and not committed

------------------------------------------------------------

Status

This is an ongoing project.
Planned future improvements include:
- improved categorical encoding strategies,
- performance optimization of historical feature computation,
- probabilistic calibration and evaluation,
- comparison with additional model families.

------------------------------------------------------------

Motivation

This project was developed as a personal machine learning portfolio project to demonstrate:
- applied ML engineering,
- careful experimental design,
- and the translation of mathematical thinking into practical data science.
