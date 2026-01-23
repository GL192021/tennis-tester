import numpy as np
import pandas as pd










## COLLECTING PREVIOUS MATCH'S STATS FOR EACH PLAYER
''' previous matche's stats '''
def single_prev_match_stats(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)
    """
    df = data.copy()

    n = len(df)

    prev_stats_dict = {}

    prev_stats_pre = np.empty((n, 2), dtype=object)


    for i in range(n):
        j = n-i-1

        row = df.iloc[j]

        p1 = row.p1_id
        p2 = row.p2_id

        prev_stats_pre[j, 0] = prev_stats_dict.get(p1, np.nan)
        prev_stats_pre[j, 1] = prev_stats_dict.get(p2, np.nan)

        prev_stats_dict[p1] = row[['sets_ratio_enc', 'tie_breaks_enc', 'minutes', 'p1_ace', 'p1_df',
                                    'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms',
                                    'p1_bpSaved', 'p1_bpFaced']]

        prev_stats_dict[p2] = row[['sets_ratio_enc', 'tie_breaks_enc', 'minutes', 'p2_ace', 'p2_df',
                                    'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms',
                                    'p2_bpSaved', 'p2_bpFaced']]


    df_prev_stats = pd.DataFrame({
        "p1_single_prev_stats": prev_stats_pre[:,0],
        "p2_single_prev_stats": prev_stats_pre[:,1],
    })


    return df_prev_stats


''' Surface previous matche's stats '''
def single_prev_match_stats_same_surface(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)
    """
    df = data.copy()

    n = len(df)

    prev_stats_dict_same_surface = {}

    prev_stats_pre_same_surface = np.empty((n, 2), dtype=object)


    for i in range(n):
        j = n-i-1

        row = df.iloc[j]

        surface = row.surface

        p1 = row.p1_id
        p2 = row.p2_id

        prev_stats_pre_same_surface[j, 0] = prev_stats_dict_same_surface.get((p1,surface), np.nan)
        prev_stats_pre_same_surface[j, 1] = prev_stats_dict_same_surface.get((p2,surface), np.nan)

        prev_stats_dict_same_surface[(p1,surface)] = row[['sets_ratio_enc', 'tie_breaks_enc', 'minutes', 'p1_ace', 'p1_df',
                                                          'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms',
                                                          'p1_bpSaved', 'p1_bpFaced']]

        prev_stats_dict_same_surface[(p2,surface)] = row[['sets_ratio_enc', 'tie_breaks_enc', 'minutes', 'p2_ace', 'p2_df',
                                                          'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms',
                                                          'p2_bpSaved', 'p2_bpFaced']]


    df_prev_stats_same_surface = pd.DataFrame({
        "p1_single_prev_stats_same_surface": prev_stats_pre_same_surface[:,0],
        "p2_single_prev_stats_same_surface": prev_stats_pre_same_surface[:,1],
    })


    return df_prev_stats_same_surface



''' avergae stat of n previous matches '''
def avrg_stats_from_multiple_prev_matches(data, prev_match_num=5):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)
    """

    prev_df = single_prev_match_stats(data)
    n = len(data)

    rows_out = []

    for i in range(n):
        row_features = {}

        for player_col, prefix in [
            ("p1_single_prev_stats", "p1"),
            ("p2_single_prev_stats", "p2"),
        ]:
            prev_matches = []
            idx = i

            while len(prev_matches) < prev_match_num:
                if idx < 0:
                    break

                stats = prev_df.iloc[idx][player_col]

                if isinstance(stats, pd.Series):
                    prev_matches.append(stats)

                idx -= 1

            if len(prev_matches) == 0:
                # fill NaNs later once we know column names
                row_features[prefix] = np.nan
            else:
                avg_stats = (
                    pd.concat(prev_matches, axis=1)
                      .mean(axis=1)
                )

                # rename columns
                avg_stats.index = [
                    f"{prefix}_av_{stat}_{prev_match_num}_matches"
                    for stat in avg_stats.index
                ]

                row_features[prefix] = avg_stats

        rows_out.append(row_features)

    # --- build final DataFrame ---
    df_out = pd.DataFrame(index=data.index)

    for i, row in enumerate(rows_out):
        for prefix in ["p1", "p2"]:
            val = row[prefix]
            if isinstance(val, pd.Series):
                df_out.loc[i, val.index] = val.values

    return df_out

    


''' Surface avergae stat of n previous matches '''
def avrg_stats_from_multiple_prev_matches(data, prev_match_num=5):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)
    """
    data = data.copy()
    
    surface_prev_df = single_prev_match_stats_same_surface(data)
    n = len(data)

    rows_out = []

    for i in range(n):
        row_features = {}

        for player_col, prefix in [
            ("p1_single_prev_stats_same_surface", "p1"),
            ("p2_single_prev_stats_same_surface", "p2"),
        ]:
            prev_matches = []
            idx = i

            while len(prev_matches) < prev_match_num:
                if idx < 0:
                    break

                stats = surface_prev_df.iloc[idx][player_col]

                if isinstance(stats, pd.Series):
                    prev_matches.append(stats)

                idx -= 1

            if len(prev_matches) == 0:
                # fill NaNs later once we know column names
                row_features[prefix] = np.nan
            else:
                avg_stats = (
                    pd.concat(prev_matches, axis=1)
                      .mean(axis=1)
                )

                # rename columns
                avg_stats.index = [
                    f"{prefix}_av_{stat}_{prev_match_num}_matches"
                    for stat in avg_stats.index
                ]

                row_features[prefix] = avg_stats

        rows_out.append(row_features)

    # --- build final DataFrame ---
    df_out = pd.DataFrame(index=data.index)

    for i, row in enumerate(rows_out):
        for prefix in ["p1", "p2"]:
            val = row[prefix]
            if isinstance(val, pd.Series):
                df_out.loc[i, val.index] = val.values

    return df_out






''' OVERALL PLAYER STATS '''
def overall_sats(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)
    """
    prev_stats_dict = single_prev_match_stats(data)
    data = data.copy()

    overall_stats = {}
    for indx, row in data.iterrows():
        for player in [row["p1_id"], row["p2_id"]]:

            prev_matches_lst_for_each_player = []
            prev_matches_found = 0
            key = f"prev_p_{player}__for_row{indx}"

            while key in prev_stats_dict:
                idx_2, prev_stats = prev_stats_dict[key]
                idx_2 = int(idx_2)

                if idx_2 is None or pd.isna(idx_2) or idx_2 == "N":
                    break

                prev_matches_lst_for_each_player.append(prev_stats)
                prev_matches_found += 1
                key = f"prev_p_{player}__for_row{int(idx_2)}"

            if prev_matches_found == 0:
                overall_stats[f"prev_p_{player}__for_row{indx}"] = np.nan
            else:
                overall_stats[f"prev_p_{player}__for_row{indx}"] = (
                    pd.concat(prev_matches_lst_for_each_player, axis=1)
                      .mean(axis=1)
                )

    return overall_stats

''' OVERALL PLAYER SURFACE STATS '''
def overall_sats_same_surface(data):
    """
    dataset must be in chronological order: future -> past
    (not necessarily consecutive matches)
    """
    prev_stats_dict_same_surface = single_prev_match_stats_same_surface(data)
    data = data.copy()

    overall_stats_same_surface = {}
    for indx, row in data.iterrows():
        for player in [row["p1_id"], row["p2_id"]]:

            prev_matches_lst_for_each_player = []
            prev_matches_found = 0
            key = f"prev_p_{player}__for_row{indx}"

            while key in prev_stats_dict_same_surface:
                idx_2, prev_stats = prev_stats_dict_same_surface[key]
                idx_2 = int(idx_2)

                if idx_2 is None or pd.isna(idx_2) or idx_2 == "N":
                    break

                prev_matches_lst_for_each_player.append(prev_stats)
                prev_matches_found += 1
                key = f"prev_p_{player}__for_row{int(idx_2)}"

            if prev_matches_found == 0:
                overall_stats_same_surface[f"prev_p_{player}__for_row{indx}"] = np.nan
            else:
                overall_stats_same_surface[f"prev_p_{player}__for_row{indx}"] = (
                    pd.concat(prev_matches_lst_for_each_player, axis=1)
                      .mean(axis=1)
                )

    return overall_stats_same_surface




''' CONSECUTIVE MATCHES '''
# possibly FATIGUE
#implementation: matches in the last 7, 15 and 30 days
#                   put in the minutes and/or sets they played

'''MOMENTUM'''
#consecutive wins or win ratio of n last matches
#implementation: win ratio

''' DAYS SINCE LAST MATCH'''
# a bit confusing stat: possibly INJURY but also could be long break -> refreshed  and/or  period of not many
#                       tournaments (somewhere after the US open)



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
def add_elo_feature(dataset, y,  initial_elo=1500, k=32, scale=400):
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
def add_surface_elo_feature(dataset, y,  initial_elo=1500, k=32, scale=400):
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








