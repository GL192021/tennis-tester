import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from feature_engineering import *

def sliding_time_split(
    df, 
    n_splits: int, 
    test_size_frac: float, 
    min_train_size_frac: float , 
    step_size_frac: float, 
    max_train_size_frac: float = None, 
    train_test_dist: int = 0
):
    """
    df is assumed ordered newest -> oldest (index 0 is the latest match).
    Each split uses:
      test: a block of newer rows
      train: all older rows below that block (expanding older history)
    """

    n = len(df)
    min_train_size = max(1, int(round(min_train_size_frac*n)))
    step_size = max(1, int(round(step_size_frac*n)))

    splits = []

    test_start = 0

    j = 10
    i = step_size_frac/j

    while len(splits) < n_splits:
        test_end = test_start + int(round(test_size_frac*n))
        train_start = test_end + 1+ train_test_dist
        train_end = n if max_train_size_frac is None else train_start+int(round(n*max_train_size_frac))

        train_size = n-train_start

        if train_size < min_train_size or test_end >= n-2:
            j -= 1
            if i == 1:
                break
            test_start = int(n*i)
            test_end = test_start + int(round(test_size_frac * n))
            train_start = test_end + train_test_dist
            train_end = n if max_train_size_frac is None else train_start + int(round(n * max_train_size_frac))

            i = step_size_frac/max(1, j)

        test_indx = test_start, test_end
        train_indx = train_start, train_end

        splits.append((train_indx, test_indx))

        test_start += step_size

    return splits


# def random_time_split(df, n_splits, test_size_frac, min_train_size_frac, train_test_dist=0):



# MATCH STATS: STATS OF CURRENT MATCH THAT WE CAN ACTUALLY USE, E.G. RANK OF PLAYERS
CURRENT_NUMERIC_DIFFS = [
    # demographics / ranking
    ("p1_age", "p2_age"),
    ("p1_ht", "p2_ht"),
    ("p1_rank", "p2_rank"),
    ("p1_rank_points", "p2_rank_points"),
    ("p1_seed", "p2_seed"),
]


# Binary "same" features (encoded): we want to separate them from "current+diffs", as they could introduce hurtful bias /unecessary noise
#   e.g. if GRE=9, ITA = 24 and ESP = 32, then taking a difference would either create a false boundary line or ad noise in the sense:
#       if both ITA and ESP, both "won" over GRE, then this could create a false boundary point "if country_player > country_player', then player beats player'" (especially is also ESP "won" over ITA), which is total nonsense;
#       now, if for example ITA won over GRE and ESP lost to GRE, this would just confuse the model and consume resources for no reason.
SAME_FEATURES = [
    ("p1_ioc_enc", "p2_ioc_enc", "same_ioc"),
    ("p1_entry_enc", "p2_entry_enc", "same_entry"),
    ("p1_hand_enc", "p2_hand_enc", "same_hand"),
]


#PREV STATS: STATS FROM CURRENT MATCH THAT WE CANNOT USE IF WE WANT TO BUILD A PREDICTOR, E.G. SCORE OF CURRENT MATCH
PREV_STAT_COLS = [
    "sets_ratio_enc",
    "tie_breaks_enc",
    "minutes",
    "ace",
    "df",
    "svpt",
    "1stIn",
    "1stWon",
    "2ndWon",
    "SvGms",
    "bpSaved",
    "bpFaced",
]

OVERALL_STATS = [
    "sets_ratio_enc", "tie_breaks_enc", "best_of", "round_enc", "minutes",
    "p1_ioc_enc", "p1_entry_enc",
    "p1_ace", "p1_df", "p1_svpt", "p1_1stIn", "p1_1stWon", "p1_2ndWon",
    "p1_SvGms", "p1_bpSaved", "p1_bpFaced",
]

# TOURNAMENT'S FEATURES
TOURN_FEATURES = [
    "surface_enc",
    "tourney_level_enc",
    "round_enc",
    "best_of",
    "draw_size",
    "tourney_name_enc",
    "best_of"
]


# ELO Features
ELO_FEATURES = ["elo_p1_pre", "elo_p2_pre", "elo_diff", "elo_p1_win_prob"]


#H2H
H2H_FEATURES = ["H2H"]


def build_features_for_trees(
    data: pd.DataFrame,
    y,
    use_prev_stats: bool = True,
    use_av_prev_stats: bool = False,
    n_prev_matches_list: list[int] | None = None,
    use_overall_stats: bool = False,
    use_elo: bool = True,
    use_h2h: bool = True,
    surface: bool = True
) -> pd.DataFrame:
    """
    Returns a dataframe with all player stats differences ready for tree-based models.
    Each feature is the difference between p1 and p2 stats.
    """
    df = data.copy()
    df_features = pd.DataFrame(index=df.index)

    # --- CURRENT (NON-HISTORICAL) FEATURES ---
    for (c1, c2) in CURRENT_NUMERIC_DIFFS:
        if c1 in df.columns and c2 in df.columns:
            df_features[f"{c1}_minus_{c2}"] = df[c1] - df[c2]

    for col in TOURN_FEATURES:
        if col in df.columns:
            df_features[col] = df[col]

    # --- SINGLE PREVIOUS MATCH STATS ---
    if use_prev_stats:
        prev_stats_df = single_prev_match_stats(df)
        for stat in PREV_STAT_COLS:
            diff_col = f"single_prev_{stat}_diff"
            df_features[diff_col] = prev_stats_df[f"p1_single_prev_{stat}"] - prev_stats_df[f"p2_single_prev_{stat}"]

    # --- ELO ---
    if use_elo:
        elo_df = elo(df, y)
        df_features["elo_diff"] = elo_df["elo_diff"]
        df_features["elo_p1_win_prob"] = elo_df["elo_p1_win_prob"]


    # --- H2H ---
    if use_h2h:
        h2h_df = h2h(df, y)
        df_features["h2h_diff"] = h2h_df["h2h_p1_p2_pre"] - h2h_df["h2h_p2_p1_pre"]

    # --- OVERALL STATS ---
    if use_overall_stats:
        overall_df = overall_sats(df)
        for stat in PREV_STAT_COLS:
            diff_col = f"overall_{stat}_diff"
            df_features[diff_col] = overall_df[f"p1_overall_{stat}"] - overall_df[f"p2_overall_{stat}"]

    # --- AVERAGE PREVIOUS MATCHES ---
    if use_av_prev_stats and n_prev_matches_list is not None:
        for n in n_prev_matches_list:
            avrg_df = avrg_stats_from_multiple_prev_matches(df, prev_match_num=n)
            for stat in PREV_STAT_COLS:
                diff_col = f"av_{n}_{stat}_diff"
                df_features[diff_col] = avrg_df[f"p1_av_{n}_prev_{stat}"] - avrg_df[f"p2_av_{n}_prev_{stat}"]

    # --- SURFACE STATS ---
    if surface:
        if use_prev_stats:
            prev_surf_df = single_prev_match_stats_surface(df)
            for stat in PREV_STAT_COLS:
                diff_col = f"single_prev_surface_{stat}_diff"
                df_features[diff_col] = prev_surf_df[f"p1_single_prev_surface_{stat}"] - prev_surf_df[f"p2_single_prev_surface_{stat}"]

        if use_elo:
            surf_elo_df = elo_surface(df, y)
            df_features["surface_elo_diff"] = surf_elo_df["surface_elo_diff"]
            df_features["surface_elo_p1_win_prob"] = surf_elo_df["surface_elo_p1_win_prob"]


        if use_h2h:
            h2h_surf_df = h2h_surface(df, y)
            df_features["h2h_surface_diff"] = h2h_surf_df["h2h_p1_p2_pre_surface"] - h2h_surf_df["h2h_p2_p1_pre_surface"]

        if use_overall_stats:
            overall_surf_df = overall_sats_surface(df)
            for stat in PREV_STAT_COLS:
                diff_col = f"overall_surface_{stat}_diff"
                df_features[diff_col] = overall_surf_df[f"p1_overall_surface_{stat}"] - overall_surf_df[f"p2_overall_surface_{stat}"]

        if use_av_prev_stats and n_prev_matches_list is not None:
            for n in n_prev_matches_list:
                avrg_surf_df = avrg_stats_from_multiple_prev_matches_surface(df, prev_match_num=n)
                for stat in PREV_STAT_COLS:
                    diff_col = f"av_{n}_surface_{stat}_diff"
                    df_features[diff_col] = avrg_surf_df[f"p1_av_{n}_prev_surface_{stat}"] - avrg_surf_df[f"p2_av_{n}_prev_surface_{stat}"]

    # Final check
    if len(df_features) == len(df):
        return df_features
    else:
        raise ValueError("Feature dataframe length mismatch!")








