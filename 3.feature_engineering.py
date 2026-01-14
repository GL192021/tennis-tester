import numpy as np
import pandas as pd










## COLLECTING PREVIOUS MATCH'S STATS FOR EACH PLAYER

def single_prev_match_stats(data):
    """
    :param train_set: takes as an input the name of the dataset where now we have p1 & p2, and the first player is not always
           the winner.
    :return: it returns (and saves) a new DataFrame that now also has the previous matches stats for each player-so we have a DF with some extra columns
             (automatically, if the previous match does not exist (first tournament or RR rounds) it puts NaN for the previous match's stats.)
    """
    data_2 = data.copy()

    prev_stats_dict = {}
    for indx, row in data_2.loc[:].iterrows():
        for player in [row["p1_id"], row["p2_id"]]:
            prev_stats = np.nan, np.nan
            for indx2, row2 in data_2.loc[indx+1:].iterrows():
                if (row2["p1_id"] == player or row2["p2_id"] == player) and indx2 > indx and f"prev_p_{player}__for_row{indx}" not in prev_stats_dict:
                    if row2["p1_id"] == player:
                        prev_stats = indx2, row2[['sets_ratio_enc', 'tie_breaks_enc', 'best_of', 'round_enc', 'minutes', 'p1_hand_enc', 'p1_ioc_enc', 'p1_entry_enc', 'p1_ace', 'p1_df',
                                                                     'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms',
                                                                     'p1_bpSaved', 'p1_bpFaced']]
                    elif row2["p2_id"] == player:
                        prev_stats = indx2, row2[['sets_ratio_enc', 'tie_breaks_enc', 'best_of', 'round_enc', 'minutes', 'p2_hand_enc', 'p2_ioc_enc', 'p2_entry_enc', 'p2_ace', 'p2_df',
                                                                     'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms',
                                                                     'p2_bpSaved', 'p2_bpFaced']]
                    break
            prev_stats_dict[f"prev_p_{player}__for_row{indx}"] = prev_stats


    return prev_stats_dict






def single_prev_match_stats_same_surface(data):
        data = data.copy()

        prev_stats_dict = {}
        for indx, row in data.iloc[:].iterrows():
            surface = row["surface"]
            for player in [row["p1_id"], row["p2_id"]]:
                prev_stats = np.nan, np.nan
                for indx2, row2 in data.iloc[indx+1:].iterrows():
                    surface_2 = row2["surface"]
                    if (row2["p1_id"] == player or row2["p2_id"] == player) and surface == surface_2 and indx2 > indx and f"prev_p_{player}__for_row{indx}" not in prev_stats_dict:
                        if row2["p1_id"] == player:
                            prev_stats = indx2, row2[['sets_ratio_enc', 'tie_breaks_enc', 'best_of', 'round_enc', 'minutes', 'p1_hand_enc', 'p1_ioc_enc', 'p1_entry_enc', 'p1_ace', 'p1_df',
                                                                         'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms',
                                                                         'p1_bpSaved', 'p1_bpFaced']]
                        elif row2["p2_id"] == player:
                            prev_stats = indx2, row2[['sets_ratio_enc', 'tie_breaks_enc', 'best_of', 'round_enc', 'minutes', 'p2_hand_enc', 'p2_ioc_enc', 'p2_entry_enc', 'p2_ace', 'p2_df',
                                                                         'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms',
                                                                         'p2_bpSaved', 'p2_bpFaced']]
                        break
                prev_stats_dict[f"prev_p_{player}__for_row{indx}"] = prev_stats

        return prev_stats_dict





def avrg_stats_from_multiple_prev_matches(data, prev_match_num = 5):
    prev_stats_dict = single_prev_match_stats(data)

    data = data.copy()
    data_2 = data

    avrg_prev_stats_dict = {}
    for indx, row in data_2.iloc[:].iterrows():
        for player in [row["p1_id"], row["p2_id"]]:
            prev_matches_lst_for_each_player = []
            prev_matches_found = 0
            key = f"prev_p_{player}__for_row{indx}"
            # print(player)
            # print(indx)
            while prev_matches_found < prev_match_num:
                val = prev_stats_dict[key]
                idx_2 = val[0]
                # print(idx_2, type(idx_2))
                prev_stats = val[1]
                prev_matches_lst_for_each_player.append(val)
                if idx_2 is None or pd.isna(idx_2) or idx_2 == "N":
                    break
                idx_2 = int(idx_2)
                prev_matches_lst_for_each_player.append(prev_stats)
                prev_matches_found += 1
                key = f"prev_p_{player}__for_row{idx_2}"
            avrg_prev_stats_dict[f"prev_p_{player}__for_row{indx}"] = prev_matches_lst_for_each_player

    return avrg_prev_stats_dict





def avrg_stats_from_multiple_prev_matches_same_surface(train_set, prev_match_num=5):
    prev_stats_same_surf_dict = single_prev_match_stats_same_surface(train_set)

    data = train_set.copy()
    data_2 = data

    avrg_prev_stats_same_surface_dict = {}
    for indx, row in data_2.loc[:].iterrows():
        for player in [row["p1_id"], row["p2_id"]]:
            prev_matches_lst_for_each_player = []
            prev_matches_found = 0
            key = f"prev_p_{player}__for_row{indx}"
            while prev_matches_found < prev_match_num:
                val = prev_stats_same_surf_dict[key]
                idx_2 = val[0]
                prev_stats = val[1]
                prev_matches_lst_for_each_player.append(prev_stats_same_surf_dict[key])
                if idx_2 is None or pd.isna(idx_2) or idx_2 == "N":
                    break
                idx_2 = int(idx_2)
                prev_matches_lst_for_each_player.append(prev_stats)
                prev_matches_found += 1
                key = f"prev_p_{player}__for_row{idx_2}"
            avrg_prev_stats_same_surface_dict[f"prev_p_{player}__for_row{indx}"] = prev_matches_lst_for_each_player

    return avrg_prev_stats_same_surface_dict


''' OVERALL PLAYER STATS '''
''' OVERALL PLAYER SURFACE STATS '''




''' FATIGUE '''
''' AND INJURY'''
#CONSECUTIVE MATCHES, DAYS SINCE LAST MATCH etc

''' H2H '''
''' SURFACE H2H'''


''' ELO '''
def add_elo_feature(dataset, y,  initial_elo=1500, k=32, scale=400):
    """
    dataset must be in chronological order (future -> past).
    y must be aligned to df rows (same row order), where:
      y[i] = 0 => p1 won
      y[i] = 1 => p2 won

    Returns dataset_elo with:
      elo_p1_pre, elo_p2_pre, elo_diff, elo_p1_win_prob
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
