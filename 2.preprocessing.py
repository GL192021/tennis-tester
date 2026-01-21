import pandas as pd
from sklearn.preprocessing import LabelEncoder






##### CREATING NUMERICAL VALUES FOR NON-NUMERIC FEATURES
## we use rounds to order the matches chronologically too, so we make a specific mapping for that. for the rest we will
## use the LabelEncoder
round_mapping = {
    "RR": 0,
    "BR": 7,
    "F": 8,
    "SF": 6,
    "QF": 5,
    "R16": 4,
    "R32": 3,
    "R64": 2,
    "R128": 1
}

special_tournaments = ["Tour Finals", "Next Gen Finals"]



def enumerating_score(data):
    data = data.copy()

    sets_ratio_list = []
    game_ratio_list = []
    tie_breaks_list = []
    for idx, row in data.iterrows():
        best_of = row["best_of"]
        score = str(row["score"])  # ensure it's a string
        tie_breaks = score.count("(")
        tie_breaks_list.append(tie_breaks)

        retirement_bool = 'RET' in score
        # games_num = 0
        # set_weight = (w_games - l_games) / total_games_in_set

        if retirement_bool:
            final_set_bool = '0-0' in score
            if final_set_bool:
                sets_played = score.count(" ") - 1
            else:
                sets_played = score.count(" ")
        else:
            sets_played = score.count(" ") + 1  # number of sets




        score_enc = sets_played / best_of
        sets_ratio_list.append(score_enc)

    data["sets_ratio_enc"] = sets_ratio_list
    data["tie_breaks_enc"] = tie_breaks_list

    return data





def enumerating_features(df):
    df = df.copy()

    # Initialize LabelEncoders
    le_tourney = LabelEncoder()
    le_surface = LabelEncoder()
    le_level = LabelEncoder()

    #shared labels across each matche's two players -> I need to create a labeling system that it is concistent for both players
    le_hand = LabelEncoder()
    le_ioc = LabelEncoder()
    le_entry = LabelEncoder()

    def fit_shared_encoder(series1, series2):
        le = LabelEncoder()
        le.fit(pd.concat([series1, series2]))
        return le

    le_ioc = fit_shared_encoder(df["winner_ioc"], df["loser_ioc"])
    df["winner_ioc_enc"] = le_ioc.transform(df["winner_ioc"])
    df["loser_ioc_enc"] = le_ioc.transform(df["loser_ioc"])

    le_hand = fit_shared_encoder(df["winner_hand"], df["loser_hand"])
    le_entry = fit_shared_encoder(df["winner_entry"], df["loser_entry"])


    # Encode features
    df["tourney_name_enc"] = le_tourney.fit_transform(df["tourney_name"])
    df["surface_enc"] = le_surface.fit_transform(df["surface"])
    df["tourney_level_enc"] = le_level.fit_transform(df["tourney_level"])
    df["winner_entry_enc"] = le_entry.transform(df["winner_entry"])
    df["loser_entry_enc"] = le_entry.transform(df["loser_entry"])
    df["winner_hand_enc"] = le_hand.transform(df["winner_hand"])
    df["loser_hand_enc"] = le_hand.transform(df["loser_hand"])
    df["winner_ioc_enc"] = le_ioc.transform(df["winner_ioc"])
    df["loser_ioc_enc"] = le_ioc.transform(df["loser_ioc"])

    df["round_enc"] = df["round"].map(round_mapping)



    return df





def sorting_matches(data, special_tournaments=special_tournaments, round_mapping=round_mapping):
    '''
    sort in the follwing order: last tournament at top, second to last below etc
    inside each tournament it sorts: F first row, SF second row, QF third row etc
    with this order, the single previous match of each player, is in the first below row from the current that this player appears
    '''

    # helper function
    def sort_tournament_group(df, include_groups=False):
        if df["is_special"].iloc[0]:
            # reverse based on original order
            return df.sort_values("_orig_idx", ascending=True)
        else:
            return df.sort_values("round_ordering", ascending=False)


    data = data.copy()
    data["round_ordering"] = data["round"].map(round_mapping)

    # keep original order
    data["_orig_idx"] = data.index

    # mark tournaments as special
    data["is_special"] = data["tourney_name"].isin(special_tournaments)

    # first sort by date ascending (chronological order) - global ordering
    date_sorted_data = data.sort_values(["tourney_date"], ascending=False)

    # group by tournament but preserve global order and locally order each tournament
    sorted_data = date_sorted_data.groupby(["tourney_name", "tourney_date"], group_keys=False, sort=False).apply(sort_tournament_group)

    # drop helper column
    sorted_data = sorted_data.drop(columns=["is_special"])

    # the after sorting rows have as index their current position, and not their initial one.
    sorted_data = sorted_data.reset_index(drop=True)


    return sorted_data







def interchanging_players_position(data, percent, random_state=None):
    """
    :param data: is the csv with our data as we downloaded them
    :return: the csv, but now we have swapped for a random 'percent' of the rows, the position of player_1 -> who is always the winner in raw_data
                                                                                              and player_2 -> who is always the loser in raw data
             a list of the labels of the reversed csv
             a DataFrame of the before the swapping of the sample rows
             a DataFrame of the after the swapping of the sample rows
             a list of the  indexes of the rows changed (this does not change with the player_reversion process)
    things to pay attention to:
    1) remember to keep track who the actual winner is. we do that by creating the label list before interchanging
    2) creating a copy of the input data is always a good idea, not to mess up the original dataset
    """

    data_copy = data.copy()
    data_copy["old_row_index"] = data_copy.index  # store old labels
    data_copy = data_copy.reset_index(drop=True)

    ## changing the winner/loser to player_1/player_2 for all columns in each row!:
    rename_map = {}
    for col in data_copy.columns:
        if col.startswith("winner_"):
            rename_map[col] = col.replace("winner_", "p1_")
        elif col.startswith("loser_"):
            rename_map[col] = col.replace("loser_", "p2_")
        elif col.startswith("w_"):
            rename_map[col] = col.replace("w_", "p1_")
        elif col.startswith("l_"):
            rename_map[col] = col.replace("l_", "p2_")


    data_copy = data_copy.rename(columns=rename_map)

    if random_state:
        before_swapping_rows_samples = data_copy.sample(frac=percent, replace=False, random_state=random_state)
    else:
        before_swapping_rows_samples = data_copy.sample(frac=percent, replace=False)

    # rows_sample_indices = [ind for ind in before_swapping_rows_samples.index]
    rows_sample_indices = before_swapping_rows_samples.index.tolist()

    old_rows_sample_indices = before_swapping_rows_samples["old_row_index"].tolist()


    ### creating the labels : 0 if p1 won. 1 if p2 won. the unswapped dataset, always produces label=0.
    labels = [0 for _ in range(len(data_copy))]
    for i in rows_sample_indices:
        labels[i] = 1

    pairs = []

    for col in data_copy.columns:
        if col.startswith("p1_"):
            suffix = col[3:]  # remove "p1_"
            p2_col = "p2_" + suffix  # expected matching p2 column
            if p2_col in data_copy.columns:
                pairs.append((col, p2_col))

    swapped_data = data_copy.copy()

    for r in rows_sample_indices:
        for p1_col, p2_col in pairs:
            swapped_data.loc[r, p1_col], swapped_data.loc[r, p2_col] = (
                swapped_data.loc[r, p2_col],
                swapped_data.loc[r, p1_col]
            )


    after_swapping_rows_samples = swapped_data.iloc[rows_sample_indices]

    ## keeping the new datasets clean from the old incidices. after all we keep them separately in their own csv
    swapped_data = swapped_data.drop(columns=["old_row_index"])
    before_swapping_rows_samples = before_swapping_rows_samples.drop(columns=["old_row_index"])
    after_swapping_rows_samples = after_swapping_rows_samples.drop(columns=["old_row_index"])


    return swapped_data, labels, before_swapping_rows_samples, after_swapping_rows_samples, rows_sample_indices, old_rows_sample_indices



## wrap up function
def preprocessing(data, percent, random_state=None, special_tournaments=special_tournaments, round_mapping=round_mapping):
    data = data.copy()

    enumerated_score_data = enumerating_score(data)
    all_enumerated_data = enumerating_features(enumerated_score_data)

    sorted_data = sorting_matches(all_enumerated_data, special_tournaments, round_mapping)

    swapped_data, labels, _, _, _, _ = interchanging_players_position(sorted_data, percent, random_state)



    return swapped_data, labels


