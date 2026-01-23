import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
from preprocessing import preprocessing
from feature_engineering import *
from preparing_data_for_tree_models import make_time_splits_descending, run_test, summarize_results
from pathlib import Path



#-------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"





#------------------------------------------
# Load the combined dataset
#------------------------------------------

matches_path = DATA_PROCESSED / "combined_2020-2024.csv"
matches = pd.read_csv(matches_path)


preprocessed_matches_50, Y_50 = preprocessing(matches, 0.5, random_state=42)



splits = make_time_splits_descending(
    preprocessed_matches_50,
    n_splits=20,
    test_size_frac=0.10,
    min_train_frac=0.50,
    step_frac=0.05
)


# 1) baseline
res_baseline = run_test(
    df_sorted_desc=preprocessed_matches_50,
    y_all=Y_50,
    splits=splits,
    use_prev_diffs=False,
    use_elo=False,
    model=DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
)
summarize_results("BASELINE (no prev diffs, no elo)", res_baseline)

# 2) prev diffs only
res_prev = run_test(
    df_sorted_desc=preprocessed_matches_50,
    y_all=Y_50,
    splits=splits,
    use_prev_diffs=True,
    use_elo=False,
    model=DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
)
summarize_results("PREV DIFFS ONLY", res_prev)

# 3) elo only
res_elo = run_test(
    df_sorted_desc=preprocessed_matches_50,
    y_all=Y_50,
    splits=splits,
    use_prev_diffs=False,
    use_elo=True,
    model=DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
)
summarize_results("ELO ONLY", res_elo)

# 4) prev diffs + elo
res_both = run_test(
    df_sorted_desc=preprocessed_matches_50,
    y_all=Y_50,
    splits=splits,
    use_prev_diffs=True,
    use_elo=True,
    model=DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
)

summarize_results("PREV DIFFS + ELO", res_both)
