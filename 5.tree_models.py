import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
from preprocessing import preprocessing
from feature_engineering import *
from preparing_data_for_tree_models import sliding_time_split
from pathlib import Path



#-------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"





#------------------------------------------
# Load the combined dataset
#------------------------------------------

matches_path = DATA_PROCESSED / "combined_2020-2024.csv"
matches = pd.read_csv(matches_path)










def run_test(
    df_sorted_desc: pd.DataFrame,
    y_all: list | np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    use_prev_stats: bool = True,
    use_av_prev_stats: bool = False,
    n_prev_matches: list[int] | None = None,
    use_overall_stats: bool = False,
    use_elo: bool = True,
    use_h2h: bool = True,
    surface: bool = True,
    model=None
):
    if model is None:
        model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            random_state=42
        )

    y_all = np.asarray(y_all)

    results = []

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df_sorted_desc.iloc[train_idx].copy()
        test_df  = df_sorted_desc.iloc[test_idx].copy()

        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]

        # combined: [test newer] + [train older]
        combined = pd.concat([test_df, train_df], axis=0, ignore_index=True)
        y_combined = np.concatenate([y_test, y_train], axis=0)


        X_combined = build_features_for_trees(
            data=combined,
            y=y_combined,
            use_prev_stats=use_prev_stats,
            use_av_prev_stats=use_av_prev_stats,
            n_prev_matches_list=n_prev_matches,
            use_overall_stats=use_overall_stats,
            use_elo=use_elo,
            use_h2h=use_h2h,
            surface=surface
        )

        n_test = len(test_df)
        X_test  = X_combined.iloc[:n_test].copy()
        X_train = X_combined.iloc[n_test:].copy()

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            p_test = model.predict_proba(X_test)[:, 1]
        else:
            p_test = model.predict(X_test).astype(float)

        yhat = (p_test >= 0.5).astype(int)

        auc = np.nan
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, p_test)

        ll = log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6))
        acc = accuracy_score(y_test, yhat)

        results.append({
            "split_id": split_id,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "auc": auc,
            "logloss": ll,
            "accuracy": acc
        })

    return pd.DataFrame(results)



def summarize_results(name, df):
    print(f"\n{name}")
    print(df)
    print("Mean AUC:", df["auc"].mean())
    print("Mean LogLoss:", df["logloss"].mean())
    print("Mean Acc:", df["accuracy"].mean())
    print("\n====================================")















preprocessed_matches_50, Y_50 = preprocessing(matches, 0.5, random_state=42)



splits = sliding_time_split(
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

