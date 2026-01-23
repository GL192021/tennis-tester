# Tennis Match Outcome Predictor

## Project Overview

This project is an end-to-end machine learning pipeline for predicting the outcome of professional tennis matches using historical ATP data.  
It is designed as a showcase project for applying machine learning methods to a real-world, noisy, and time-dependent dataset, following the structure of a **quantitative sports prediction system**.

Rather than treating match prediction as a simple classification problem, the project focuses on:
- Historical performance modeling  
- Dynamic rating systems (Elo and surface-Elo)  
- Head-to-Head dynamics  
- Time-aware validation  
- Strict control of data leakage  

The emphasis is on:
- Clean and reproducible data processing pipelines  
- Leakage-free feature engineering  
- Time-aware model evaluation  
- Interpretable tree-based models  
  - Current models include:
    - Decision Trees  
    - Random Forests  
    - XGBoost 

Tested under different feature configurations:
- Baseline (no historical performance features)  
- Previous-match statistics  
- Elo ratings  
- Combinations of all the above  

---

## Design Principles

This project is designed as a **quantitative sports prediction system**, not a toy machine-learning example.  
It explicitly enforces the following principles:

- **Temporal causality**  
  All features are computed using only information that was available before each match was played.  
  The dataset is ordered chronologically, and all historical statistics (previous match stats, rolling averages, Elo, H2H) are updated only after a match has occurred.

- **Feature leakage prevention**  
  No information derived from the current match outcome (such as final score or winner-dependent statistics) is used to build predictive features.  
  This prevents the model from learning from future information and ensures that evaluation metrics reflect true predictive performance.

- **Symmetry between players**  
  The model does not assume that “player 1 is the winner.”  
  Player positions are randomly swapped during preprocessing, and the target variable is adjusted accordingly, preventing positional bias.

- **Realistic evaluation protocol**  
  Time-aware cross-validation is used, where training always occurs on older matches and testing is performed on newer matches.  
  This mirrors how a real-world prediction system would operate.

- **Performance modeling instead of raw memorization**  
  Player form is represented using:
  - Previous match statistics  
  - Rolling averages over recent matches  
  - Surface-specific performance  
  - Elo and surface-Elo ratings  
  - Head-to-Head dynamics  

This design ensures that the project behaves like a real quantitative sports analytics system rather than a static machine learning experiment.


------------------------------------------------------------


## Repository Structure

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


## Data

The project uses historical ATP match data.

Source:
https://github.com/JeffSackmann/tennis_atp

Download the yearly match files, for example:

atp_matches_2019.csv  
atp_matches_2020.csv  
atp_matches_2021.csv  
...

Place them in:
data/raw/

The repository does not include the data files themselves.

------------------------------------------------------------


## Pipeline Overview

The project follows a clear, numbered pipeline.

### 1. Combine yearly datasets

Run:
```python
1.combine_years.py
```

This reads all atp_matches_YYYY.csv files from data/raw/ and creates:
data/processed/combined_STARTYEAR-ENDYEAR.csv

------------------------------------------------------------


### 2. Preprocessing

Handled in code by:
```python
from preprocessing import preprocessing
```

This step:
- cleans missing values,
- encodes categorical variables,
- builds target labels,
- randomizes player position to avoid dataset bias.

------------------------------------------------------------


### 3. Feature Engineering

Handled in code by:
```python
from feature_engineering import *
```

This step adds:
- previous-match statistics,
- ELO ratings computed in a leakage-free way,
- H2H statistics,
- surface specific statistics.

------------------------------------------------------------


### 4. Time-aware splitting

Handled in code by:
```python
from preparing_data_for_tree_models import *
```

This creates chronological train/test splits that respect match order.

------------------------------------------------------------


### 5. Model training and evaluation

Run:
```python
5.tree_models.py
```

This runs experiments for:
- baseline model,
- previous-match features,
- ELO features,
- combined features.

Outputs:
- trained models → models/
- evaluation metrics and results → results/

------------------------------------------------------------


## Installation

Install the dependencies with:
```python
pip install -r requirements.txt
```

Main dependencies:
- pandas
- numpy
- scikit-learn
- xgboost

Python version:
- Python 3.10 or higher

------------------------------------------------------------


## Design Principles

No data leakage:
Only information available before a match starts is used as model input.
Score-derived or post-match statistics are explicitly excluded.

Time-aware evaluation:
All model testing respects the chronological order of matches.

Reproducible project structure:
All file paths are relative to the repository:
```
data/raw/
data/processed/
models/
results/
```

Separation of concerns:
- raw data → never modified
- processed data → reproducible
- models/results → generated locally and not committed

------------------------------------------------------------


## Status

This is an ongoing project.
Planned future improvements include:
- improved categorical encoding strategies,
- performance optimization of historical feature computation,
- probabilistic calibration and evaluation,
- comparison with additional model families.

------------------------------------------------------------


## Motivation

This project was developed as a personal machine learning portfolio project to demonstrate:
- applied ML engineering,
- careful experimental design,
- and the translation of mathematical thinking into practical data science.
