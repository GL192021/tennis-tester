import numpy as np
import pandas as pd










## COLLECTING PREVIOUS MATCH'S STATS FOR EACH PLAYER
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




''' PREVIOUS MATCH STATS '''
def single_prev_match_stats(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)

    returns:
    df with columns:
        p1_single_prev_sets_ratio_enc, p1_single_prev_tie_breaks_enc, ..., p1_single_prev_bpFaced, p2_single_prev_sets_ratio_enc, p2_single_prev_tie_breaks_enc,...
    and len(df) = len(data)
    (i.e. for each CURRENT match, we take the stats of each player's immediate preceding match (if it exists))
    """
    df = data.copy()

    n = len(df)

    prev_stats_dict = {}

    out_df = pd.DataFrame(index=df.index)


    for i in range(n):
        j = n-i-1

        row = df.iloc[j]

        p1 = row.p1_id
        p2 = row.p2_id

        # prev_stats_pre[j, 0] = prev_stats_dict.get(p1, np.nan)
        # prev_stats_pre[j, 1] = prev_stats_dict.get(p2, np.nan)
        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            prev = prev_stats_dict.get(player)

            for stat in PREV_STAT_COLS:
                col = f"{prefix}_single_prev_{stat}"
                # NaN if no previous stats
                out_df.loc[j, col] = np.nan if prev is None else prev[stat]

            out_df.loc[j,f"{prefix}_single_prev_position"] = np.nan if prev is None else prev["position"]

        # Store current match stats for future matches
        prev_stats_dict[p1] = {
            "sets_ratio_enc": row.sets_ratio_enc,
            "tie_breaks_enc": row.tie_breaks_enc,
            "minutes": row.minutes,
            "ace": row.p1_ace,
            "df": row.p1_df,
            "svpt": row.p1_svpt,
            "1stIn": row.p1_1stIn,
            "1stWon": row.p1_1stWon,
            "2ndWon": row.p1_2ndWon,
            "SvGms": row.p1_SvGms,
            "bpSaved": row.p1_bpSaved,
            "bpFaced": row.p1_bpFaced,
            "position": j
        }

        prev_stats_dict[p2] = {
            "sets_ratio_enc": row.sets_ratio_enc,
            "tie_breaks_enc": row.tie_breaks_enc,
            "minutes": row.minutes,
            "ace": row.p2_ace,
            "df": row.p2_df,
            "svpt": row.p2_svpt,
            "1stIn": row.p2_1stIn,
            "1stWon": row.p2_1stWon,
            "2ndWon": row.p2_2ndWon,
            "SvGms": row.p2_SvGms,
            "bpSaved": row.p2_bpSaved,
            "bpFaced": row.p2_bpFaced,
            "position": j
        }

    return out_df


#
# ''' SURFACE PREVIOUS MATCH STATS '''
# def single_prev_match_stats_surface(data):
#     """
#     dataset must be in chronological order: future -> past
#     (not necessarily consecutive matches)
#     """
#     df = data.copy()
#
#     n = len(df)
#
#     prev_stats_dict_same_surface = {}
#
#     prev_stats_pre_same_surface = np.empty((n, 2), dtype=object)
#
#
#     for i in range(n):
#         j = n-i-1
#
#         row = df.iloc[j]
#
#         surface = row.surface
#
#         p1 = row.p1_id
#         p2 = row.p2_id
#
#         prev_stats_pre_same_surface[j, 0] = prev_stats_dict_same_surface.get((p1,surface), np.nan)
#         prev_stats_pre_same_surface[j, 1] = prev_stats_dict_same_surface.get((p2,surface), np.nan)
#
#         prev_stats_dict_same_surface[(p1,surface)] = row[['sets_ratio_enc', 'tie_breaks_enc', 'minutes', 'p1_ace', 'p1_df',
#                                                           'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms',
#                                                           'p1_bpSaved', 'p1_bpFaced']]
#
#         prev_stats_dict_same_surface[(p2,surface)] = row[['sets_ratio_enc', 'tie_breaks_enc', 'minutes', 'p2_ace', 'p2_df',
#                                                           'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms',
#                                                           'p2_bpSaved', 'p2_bpFaced']]
#
#         avg_stats.index = [
#             f"{prefix}_av_{stat}_{prev_match_num}_matches"
#             for stat in avg_stats.index
#         ]
#
#     df_prev_stats_same_surface = pd.DataFrame({
#         "p1_single_prev_stats_same_surface": prev_stats_pre_same_surface[:,0],
#         "p2_single_prev_stats_same_surface": prev_stats_pre_same_surface[:,1],
#     })
#
#
#     return df_prev_stats_same_surface




''' SURFACE PREVIOUS MATCH STATS '''
def single_prev_match_stats_surface(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)

    returns:
    df with columns:
        p1_single_prev_[SURFACE]_sets_ratio_enc, p1_single_prev_[SURFACE]_tie_breaks_enc, ..., p1_single_prev_[SURFACE]_bpFaced, p2_single_prev_[SURFACE]_sets_ratio_enc, p2_single_prev_[SURFACE]_tie_breaks_enc,...
    and len(df) = len(data)
    (i.e. for each CURRENT match, we take the stats of each player's immediate preceding match of the same surface as the current surface (if it exists))
    """
    df = data.copy()

    n = len(df)

    prev_stats_dict = {}

    out_df = pd.DataFrame(index=df.index)


    for i in range(n):
        j = n-i-1

        row = df.iloc[j]

        surface = row.surface

        p1 = row.p1_id
        p2 = row.p2_id

        # prev_stats_pre[j, 0] = prev_stats_dict.get(p1, np.nan)
        # prev_stats_pre[j, 1] = prev_stats_dict.get(p2, np.nan)
        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            prev = prev_stats_dict.get((player, surface))

            for stat in PREV_STAT_COLS:
                col = f"{prefix}_single_prev_surface_{stat}"
                # NaN if no previous stats
                out_df.loc[j, col] = np.nan if prev is None else prev[stat]

            out_df.loc[j,f"{prefix}_single_prev_surface_position"] = np.nan if prev is None else prev["position"]


        # Store current match stats for future matches
        prev_stats_dict[(p1, surface)] = {
            "sets_ratio_enc": row.sets_ratio_enc,
            "tie_breaks_enc": row.tie_breaks_enc,
            "minutes": row.minutes,
            "ace": row.p1_ace,
            "df": row.p1_df,
            "svpt": row.p1_svpt,
            "1stIn": row.p1_1stIn,
            "1stWon": row.p1_1stWon,
            "2ndWon": row.p1_2ndWon,
            "SvGms": row.p1_SvGms,
            "bpSaved": row.p1_bpSaved,
            "bpFaced": row.p1_bpFaced,
            "position": j
        }

        prev_stats_dict[(p2, surface)] = {
            "sets_ratio_enc": row.sets_ratio_enc,
            "tie_breaks_enc": row.tie_breaks_enc,
            "minutes": row.minutes,
            "ace": row.p2_ace,
            "df": row.p2_df,
            "svpt": row.p2_svpt,
            "1stIn": row.p2_1stIn,
            "1stWon": row.p2_1stWon,
            "2ndWon": row.p2_2ndWon,
            "SvGms": row.p2_SvGms,
            "bpSaved": row.p2_bpSaved,
            "bpFaced": row.p2_bpFaced,
            "position": j
        }

    return out_df





''' AVERAGE STATS OF n PREVIOUS MATCHES '''
def avrg_stats_from_multiple_prev_matches(data, prev_match_num=5):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)

    returns:
    df with columns:
        p1_av_{n}_prev_sets_ratio_enc, p1_av_{n}_prev_tie_breaks_enc, ..., p1_av_{n}_prev_bpFaced, p2_av_{n}_prev_sets_ratio_enc, p2_av_{n}_prev_tie_breaks_enc,...
    and len(df) = len(data)
    (i.e. for each CURRENT match, we take the stats of each player's immediate n preceding matches (if they exist))
    """
    data = data.copy()
    prev_df = single_prev_match_stats(data)
    n = len(data)

    df_av_stats = pd.DataFrame(index=data.index)

    for i in range(n):

        row = data.iloc[i]

        p1 = row.p1_id
        p2 = row.p2_id

        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            prev_matches = pd.DataFrame(columns=[f"{prefix}_av_{prev_match_num}_prev_{stat}" for stat in PREV_STAT_COLS])
            current_pos = i
            j=0
            while j < prev_match_num:
                prev_match_idx = prev_df.loc[current_pos, f"{prefix}_single_prev_position"]
                if pd.isna(prev_match_idx):
                    break
                for stat in PREV_STAT_COLS:
                    single_col = f"{prefix}_single_prev_{stat}"
                    col = f"{prefix}_av_{prev_match_num}_prev_{stat}"
                    prev_matches.loc[j, col] = prev_df.loc[current_pos, single_col]
                j += 1
                current_pos = int(prev_match_idx)

                df_av_stats.loc[i, prev_matches.columns] = prev_matches.mean()

    return df_av_stats

    


''' SURFACE AVERAGE STATS OF n PREVIOUS MATCHES '''
def avrg_stats_from_multiple_prev_matches_surface(data, prev_match_num=5):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)

    returns:
    df with columns:
        p1_avrg_{n}_prev_[SURFACE]_sets_ratio_enc, p1_avrg_{n}_prev_[SURFACE]_tie_breaks_enc, ..., p1_avrg_{n}_prev_[SURFACE]_bpFaced, p2_avrg_{n}_prev_[SURFACE]_sets_ratio_enc, p2_avrg_{n}_prev_[SURFACE]_tie_breaks_enc,...
    and len(df) = len(data)
    (i.e. for each CURRENT match, we take the stats of each player's immediate preceding matches of the same surface as the current surface (if they exist))
    """
    data = data.copy()
    prev_df = single_prev_match_stats_surface(data)
    n = len(data)

    df_av_stats_surface = pd.DataFrame(index=data.index)

    for i in range(n):

        row = data.iloc[i]

        p1 = row.p1_id
        p2 = row.p2_id

        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            prev_matches = pd.DataFrame(columns=[f"{prefix}_av_{prev_match_num}_prev_surface_{stat}" for stat in PREV_STAT_COLS])
            current_pos = i
            j=0
            while j < prev_match_num:
                prev_match_idx = prev_df.loc[current_pos, f"{prefix}_single_prev_surface_position"]
                if pd.isna(prev_match_idx):
                    break
                for stat in PREV_STAT_COLS:
                    single_col = f"{prefix}_single_prev_surface_{stat}"
                    col = f"{prefix}_av_{prev_match_num}_prev_surface_{stat}"
                    prev_matches.loc[j, col] = prev_df.loc[current_pos, single_col]
                j += 1
                current_pos = int(prev_match_idx)

                df_av_stats_surface.loc[i, prev_matches.columns] = prev_matches.mean()

    return df_av_stats_surface






''' OVERALL PLAYER STATS '''
def overall_sats(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)

    returns:
    df with columns:
        p1_overall_sets_ratio_enc, p1_overall_tie_breaks_enc, ..., p1_overall__bpFaced, p2_overall_sets_ratio_enc, p2_overall_tie_breaks_enc,...
    and len(df) = len(data)
    (i.e. for each CURRENT match, we take the stats of each player's ALL preceding matches (if they exist))
    """
    data = data.copy()
    prev_df = single_prev_match_stats(data)
    n = len(data)

    df_overall_stats = pd.DataFrame(index=data.index)

    for i in range(n):

        row = data.iloc[i]

        p1 = row.p1_id
        p2 = row.p2_id

        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            prev_matches = pd.DataFrame()
            current_pos = i
            j=0
            while True:
                prev_match_idx = prev_df.loc[current_pos, f"{prefix}_single_prev_position"]
                if pd.isna(prev_match_idx):
                    break
                for stat in PREV_STAT_COLS:
                    single_col = f"{prefix}_single_prev_{stat}"
                    col = f"{prefix}_overall_{stat}"
                    prev_matches.loc[j, col] = prev_df.loc[current_pos, single_col]
                j += 1
                current_pos = int(prev_match_idx)

                df_overall_stats.loc[i, prev_matches.columns] = prev_matches.mean()

    return df_overall_stats

''' OVERALL PLAYER SURFACE STATS '''
def overall_sats_surface(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)

    returns:
    df with columns:
        p1_overall_[SURFACE]_sets_ratio_enc, p1_overall_[SURFACE]_tie_breaks_enc, ..., p1_overall_[SURFACE]_bpFaced, p2_overall_[SURFACE]_sets_ratio_enc, p2_overall_[SURFACE]_tie_breaks_enc,...
    and len(df) = len(data)
    (i.e. for each CURRENT match, we take the stats of each player's ALL preceding matches of the same surface as the current surface (if they exist))
    """
    data = data.copy()
    prev_df = single_prev_match_stats_surface(data)
    n = len(data)

    df_overall_stats_surface = pd.DataFrame(index=data.index)

    for i in range(n):

        row = data.iloc[i]

        p1 = row.p1_id
        p2 = row.p2_id

        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            prev_matches = pd.DataFrame()
            current_pos = i
            j=0
            while True:
                prev_match_idx = prev_df.loc[current_pos, f"{prefix}_single_prev_surface_position"]
                if pd.isna(prev_match_idx):
                    break
                for stat in PREV_STAT_COLS:
                    single_col = f"{prefix}_single_prev_surface_{stat}"
                    col = f"{prefix}_overall_surface_{stat}"
                    prev_matches.loc[j, col] = prev_df.loc[current_pos, single_col]
                j += 1
                current_pos = int(prev_match_idx)

                df_overall_stats_surface.loc[i, prev_matches.columns] = prev_matches.mean()

    return df_overall_stats_surface




# ''' CONSECUTIVE MATCHES '''
# # possibly FATIGUE
# #implementation: matches in the last 7, 15 and 30 days
# #                   put in the minutes and/or sets they played
#
# '''MOMENTUM'''
# #consecutive wins or win ratio of n last matches
# #implementation: win ratio
#
# ''' DAYS SINCE LAST MATCH'''
# # a bit confusing stat: possibly INJURY but also could be long break -> refreshed  and/or  period of not many
# #                       tournaments (somewhere after the US open)



''' H2H '''
def h2h(data, labels):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches).
    labels must be aligned to dataset rows (same row order), where:
      labels[i] = 0 => p1 won
      labels[i] = 1 => p2 won

    Returns dataset_h2h_surface with:
    "h2h_p1_p2_pre", "h2h_p2_p1_pre":  h2h up to any given moment
    """
    df = data.copy()

    h2h = {}

    h2h_p1_p2_pre = np.empty(len(df), dtype=int)
    h2h_p2_p1_pre = np.empty(len(df), dtype=int)

    n = len(df)

    initital_h2h = 0

    for i in range(n):
        j = n-i-1

        row = df.iloc[j]

        p1 = row.p1_id
        p2 = row.p2_id

        h2h_p1_p2 = h2h.get((p1, p2), initital_h2h)
        h2h_p2_p1 = h2h.get((p2, p1), initital_h2h)

        # 1) FEATURES = pre-match h2h (strictly from previous matches)
        h2h_p1_p2_pre[j] = h2h_p1_p2
        h2h_p2_p1_pre[j] = h2h_p2_p1

        # 2) update ONLY AFTER h2h features are recorded
        delta = 1 if labels[j] == 0 else -1

        h2h[(p1, p2)] = h2h_p1_p2 + delta
        h2h[(p2, p1)] = h2h_p2_p1 - delta


    df_h2h = pd.DataFrame({
        "h2h_p1_p2_pre": h2h_p1_p2_pre,
        "h2h_p2_p1_pre": h2h_p2_p1_pre,
    })

    return df_h2h


''' SURFACE H2H'''
def h2h_surface(data, labels):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches).
    labels must be aligned to dataset rows (same row order), where:
      labels[i] = 0 => p1 won
      labels[i] = 1 => p2 won
    
    Returns dataset_h2h_surface with:
      "h2h_p1_p2_pre_surface", "h2h_p2_p1_pre_surface": surface h2h up to any given moment
    """
    df = data.copy()

    h2h_surface = {}

    h2h_p1_p2_pre_surface = np.empty(len(df), dtype=int)
    h2h_p2_p1_pre_surface = np.empty(len(df), dtype=int)

    n = len(df)

    initital_h2h = 0

    for i in range(n):
        j = n-i-1

        row = df.iloc[j]
        
        surface = row.surface

        p1 = row.p1_id
        p2 = row.p2_id

        h2h_p1_p2_surface = h2h_surface.get((p1, p2, surface), initital_h2h)
        h2h_p2_p1_surface = h2h_surface.get((p2, p1, surface), initital_h2h)

        # 1) FEATURES = pre-match h2h (strictly from previous matches)
        h2h_p1_p2_pre_surface[j] = h2h_p1_p2_surface
        h2h_p2_p1_pre_surface[j] = h2h_p2_p1_surface

        # 2) update ONLY AFTER h2h features are recorded
        delta = 1 if labels[j] == 0 else -1

        h2h_surface[(p1, p2, surface)] = h2h_p1_p2_surface + delta
        h2h_surface[(p2, p1, surface)] = h2h_p2_p1_surface - delta


    df_h2h_surface = pd.DataFrame({
        "h2h_p1_p2_pre_surface": h2h_p1_p2_pre_surface,
        "h2h_p2_p1_pre_surface": h2h_p2_p1_pre_surface,
    })

    return df_h2h_surface








''' ELO '''
def elo(dataset, y,  initial_elo=1500, k=32, scale=400):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches).
    y must be aligned to df rows (same row order), where:
      y[i] = 0 => p1 won
      y[i] = 1 => p2 won

    Returns dataset_elo with:
      elo_p1_pre, elo_p2_pre, elo_diff, elo_p1_win_prob: elo up to any given moment
    """
    df = dataset.copy()

    def expected(r1, r2):
        return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / scale))

    elo = {}

    elo_p1_pre = np.empty(len(df), dtype=float)
    elo_p2_pre = np.empty(len(df), dtype=float)
    elo_prob = np.empty(len(df), dtype=float)

    n = len(df)

    for i in range(n):
        j = n-i-1
        row = df.iloc[j]

        p1 = row.p1_id
        p2 = row.p2_id

        # Looks for key p1 in the dictionary elo
        # If p1 exists, returns elo[p1]
        # If p1 does not exist, returns initial_elo
        # Does not modify the dictionary
        # Alternatively:   r1 = elo[p1] if p1 in elo else initial_elo
        r1 = elo.get(p1, initial_elo)
        r2 = elo.get(p2, initial_elo)

        # 1) FEATURES = pre-match ratings (strictly from previous matches)
        elo_p1_pre[j] = r1
        elo_p2_pre[j] = r2
        p = expected(r1, r2)
        elo_prob[j] = p

        # 2) update ONLY AFTER features are recorded
        S = 1.0 - float(y[j])  # p1 win indicator
        elo[p1] = r1 + k * (S - p)
        elo[p2] = r2 + k * ((1.0 - S) - (1.0 - p))

    df_elo = pd.DataFrame({
        "elo_p1_pre": elo_p1_pre,
        "elo_p2_pre": elo_p2_pre,
        "elo_diff": elo_p1_pre - elo_p2_pre,
        "elo_p1_win_prob": elo_prob
    })

    return df_elo


''' SURFACE ELO '''
def elo_surface(dataset, y,  initial_elo=1500, k=32, scale=400):
    """
    dataset must be in chronological order (future -> past)
    (not necessarily consecutive matches).
    y must be aligned to dataset rows (same row order), where:
      y[i] = 0 => p1 won
      y[i] = 1 => p2 won

    Returns dataset_elo with:
    surface_elo_p1_pre, surface_elo_p2_pre, surface_elo_diff, surface_elo_p1_win_prob: surface elo up to any given moment
    """
    df = dataset.copy()

    def expected(r1, r2):
        return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / scale))

    surface_elo = {}

    surface_elo_p1_pre = np.empty(len(df), dtype=float)
    surface_elo_p2_pre = np.empty(len(df), dtype=float)
    surface_elo_prob = np.empty(len(df), dtype=float)

    n = len(df)

    for i in range(n):
        j = n-i-1
        row = df.iloc[j]

        p1 = row.p1_id
        p2 = row.p2_id

        surface = row.surface

        r1 = surface_elo.get((p1, surface), initial_elo)
        r2 = surface_elo.get((p2, surface), initial_elo)

        # 1) FEATURES = pre-match ratings (strictly from previous matches)
        surface_elo_p1_pre[j] = r1
        surface_elo_p2_pre[j] = r2
        p = expected(r1, r2)
        surface_elo_prob[j] = p

        # 2) update ONLY AFTER features are recorded
        S = 1.0 - float(y[j])  # p1 win indicator
        surface_elo[(p1, surface)] = r1 + k * (S - p)
        surface_elo[(p2, surface)] = r2 + k * ((1.0 - S) - (1.0 - p))

    df_surface_elo = pd.DataFrame({
        "surface_elo_p1_pre": surface_elo_p1_pre,
        "surface_elo_p2_pre": surface_elo_p2_pre,
        "surface_elo_diff": surface_elo_p1_pre - surface_elo_p2_pre,
        "surface_elo_p1_win_prob": surface_elo_prob
    })

    return df_surface_elo








