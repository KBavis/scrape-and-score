from db import fetch_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from config import props


# constant
relevant_props = {
    "QB": [
        "rushing_attempts_over_under",
        "rushing_yards_over_under",
        "anytime_touchdown_scorer",
        "passing_yards_over_under",
        "passing_touchdowns_over_under",
        "passing_attempts_over_under",
        "fantasy_points_over_under",
    ],
    "RB": [
        "rushing_attempts_over_under",
        "rushing_yards_over_under",
        "anytime_touchdown_scorer",
        "receiving_yards_over_under",
        "receptions_over_under",
        "fantasy_points_over_under",
    ],
    "WR": [
        "anytime_touchdown_scorer",
        "receiving_yards_over_under",
        "receptions_over_under",
        "fantasy_points_over_under",
    ],
    "TE": [
        "anytime_touchdown_scorer",
        "receiving_yards_over_under",
        "receptions_over_under",
        "fantasy_points_over_under",
    ],
}


"""
Main functionality of module to kick of data fetching and pre-procesing 

Args:
   None 

Returns:
   qb_data, rb_data, wr_data, te_data (tuple(pd.DataFrame, ...)): tuple containing pre-processed data
"""


def pre_process_data():
    sns.set_theme()

    df = get_data()

    filtered_df = filter_data(df)

    qb_data, rb_data, te_data, wr_data = split_data_by_position(filtered_df)

    (
        transformed_qb_data,
        transformed_rb_data,
        transformed_wr_data,
        transformed_te_data,
    ) = transform_data(qb_data, rb_data, wr_data, te_data)
    tranformed_data = [
        transformed_qb_data,
        transformed_rb_data,
        transformed_wr_data,
        transformed_te_data,
    ]

    positions = ["QB", "RB", "WR", "TE"]
    for i, df in enumerate(tranformed_data):
        df.drop(
            columns=[
                "player_id",
                "fantasy_points",
                "off_rush_rank",
                "off_pass_rank",
                "def_rush_rank",
                "def_pass_rank",
            ],
            inplace=True,
        )

        # generate ratios to avoid multicollinearity
        tranformed_data[i] = apply_feature_engineering(df, positions[i])

    validated_qb_data = validate_ols_assumptions(tranformed_data[0], "QB")
    validated_rb_data = validate_ols_assumptions(tranformed_data[1], "RB")
    validated_wr_data = validate_ols_assumptions(tranformed_data[2], "WR")
    validated_te_data = validate_ols_assumptions(tranformed_data[3], "TE")

    base_cols = [
        "log_fantasy_points",
        "log_ratio_rank",
    ]

    preprocessed_qb_data = validated_qb_data[
        base_cols + ["qb_composite_score", "is_favorited"]
    ]
    preprocessed_rb_data = validated_rb_data[
        base_cols + ["rb_composite_score", "is_favorited"]
    ]
    preprocessed_wr_data = validated_wr_data[
        base_cols + ["wr_te_composite_score"]
    ]  # is_favorited is statisically insignficant for WRs
    preprocessed_te_data = validated_te_data[
        base_cols + ["wr_te_composite_score", "is_favorited"]
    ]

    # # return tuple of pre-processed data
    return (
        preprocessed_qb_data,
        preprocessed_rb_data,
        preprocessed_wr_data,
        preprocessed_te_data,
    )


"""
Functionality to validate multicollinearity & other OLS assumptions 


Args:
   df (pd.DataFrame): data frame to validate 
   position (str): position to validate

Returns:
   updated_df (pd.DataFrame): updated df (if there weren't anym then return original)
"""


def validate_ols_assumptions(df: pd.DataFrame, position: str):
    extra_cols = []
    if position == "QB":
        extra_cols.append("qb_composite_score")
    elif position == "RB":
        extra_cols.append("rb_composite_score")
    elif position == "WR" or position == "TE":
        extra_cols.append("wr_te_composite_score")

    variables = df[["log_ratio_rank", "is_favorited"] + extra_cols]

    vif = pd.DataFrame()

    vif["VIF"] = [
        variance_inflation_factor(variables.values, i)
        for i in range(variables.shape[1])
    ]
    vif["Features"] = variables.columns

    features_to_remove = vif[vif["VIF"] > 5]["Features"].tolist()
    if features_to_remove:
        logging.info(
            f"Removing the following columns due to high VIFs: [{features_to_remove}]"
        )
    else:
        logging.info(
            "No features need to be removed to validate OLS assumptions; returning original DataFrame"
        )

    return df.drop(columns=features_to_remove)


""" 
Apply manual feature engineering in order to avoid multicollinearity regarding features 

Args:
    df (pd.DataFrame): data frame to apply feature engineering to
    
Returns:
    feature_engineered_df (pd.DataFrame): updated data frame 
"""


def apply_feature_engineering(df: pd.DataFrame, position: str):
    logging.info(
        f"Applying feautre engineering for the following position {position} and DataFrame: \n\n {df.head}"
    )

    combined_weight_names = []

    # calculate fantasy potential for player
    weight_names = [
        "log_avg_fantasy_points",
        "fantasy_points_over_under_ratio",
    ]
    combined_weight_names += weight_names
    compute_weighted_sum(df, weight_names, "fantasy_potential")

    # calculate game context for player
    weight_names = [
        "game_over_under",
        "anytime_touchdown_scorer_ratio",
    ]
    combined_weight_names += weight_names
    compute_weighted_sum(df, weight_names, "game_context")

    # calculate expected rushing volume if position applicable
    if position == "RB" or position == "QB":
        weight_names = [
            "rushing_yards_over_under_ratio",
            "rushing_attempts_over_under_ratio",
        ]
        combined_weight_names += weight_names
        compute_weighted_sum(df, weight_names, "expected_rushing_volume")

    # calculate receiving volume if position applicable
    if position == "WR" or position == "RB" or position == "TE":
        weight_names = [
            "receiving_yards_over_under_ratio",
            "receptions_over_under_ratio",
        ]
        combined_weight_names += weight_names
        compute_weighted_sum(df, weight_names, "expected_receiving_volume")

    # calculate passing volume if posiiton applicable
    if position == "QB":
        weight_names = [
            "passing_touchdowns_over_under_ratio",
            "passing_attempts_over_under_ratio",
            "passing_yards_over_under_ratio",
        ]
        combined_weight_names += weight_names
        compute_weighted_sum(df, weight_names, "expected_passing_volume")

    # combine calculated volumes based on position
    calculate_total_expected_volume(df, position)
    calculate_composite_scores(df, position)

    feature_engineered_df = df.drop(columns=combined_weight_names)
    return feature_engineered_df


""" 
Calculate composite scores for players based on expected volume, game context, and fantasy potential according to Vegas 

Args:
    df (pd.DataFrame); data frame to calculate composite scores for 

Returns:
    None
"""


def calculate_composite_scores(df: pd.DataFrame, position: str):
    cols_to_drop = []
    if position == "QB":
        weights = ["total_expected_volume_qb", "fantasy_potential", "game_context"]
        cols_to_drop += weights
        compute_weighted_sum(df, weights, "qb_composite_score")
    elif position == "RB":
        weights = ["total_expected_volume_rb", "fantasy_potential", "game_context"]
        cols_to_drop += weights
        compute_weighted_sum(df, weights, "rb_composite_score")
    else:
        weights = ["expected_receiving_volume", "fantasy_potential", "game_context"]
        cols_to_drop += weights
        compute_weighted_sum(df, weights, "wr_te_composite_score")

    df.drop(columns=cols_to_drop, inplace=True)


""" 
Functionality to calculate the total exepcted volume in given data 

Args:
    df (pd.DataFrame): data frame to account for 
    position (str): player position 

Returns:
    None
"""


def calculate_total_expected_volume(df: pd.DataFrame, position: str):
    if position == "TE" or position == "WR":
        return

    weight_names = ["expected_rushing_volume"]

    if position == "QB":
        outcome_column = "total_expected_volume_qb"

        weight_names += [
            "expected_passing_volume",
            "pass_weight",
            "rush_weight",
            "is_dual_threat",
        ]

        prop_name = f"weights.total_expected_volume_qb."

        # base weights for pocket passers
        base_pass_weight = props.get_config(prop_name + "expected_passing_volume")

        # weights for dual thread
        dual_threat_pass_weight = props.get_config(
            prop_name + "dual_threat_passing_volume"
        )

        # determine if QB is dual threat (i.e above 75 percentile)
        df["is_dual_threat"] = df["expected_rushing_volume"] > df[
            "expected_rushing_volume"
        ].quantile(0.25)

        # apply weights conditionally on if QB is dual thread
        df["pass_weight"] = np.where(
            df["is_dual_threat"], dual_threat_pass_weight, base_pass_weight
        )
        df["rush_weight"] = 1 - df["pass_weight"]

        # calculate final expectation
        df[outcome_column] = (
            df["expected_rushing_volume"] * df["rush_weight"]
            + df["expected_passing_volume"] * df["pass_weight"]
        )
    else:
        weight_names.append("expected_receiving_volume")
        compute_weighted_sum(df, weight_names, "total_expected_volume_rb")

    df.drop(columns=weight_names, inplace=True)


""" 
Utility function to apply weighted sum computation to a data frame 

Args:
    df (pd.DataFrame): data frame to apply weighted sum to
    weights (list): list of weights to apply 
    outcome_col (str): outcome col in data 
"""


def compute_weighted_sum(df: pd.DataFrame, weights: list, outcome_col: str):
    weight_mapping = retrieve_weights(outcome_col, weights)
    df[outcome_col] = sum(
        weight_mapping[weight_name] * df[weight_name] for weight_name in weights
    )


""" 
Retrieve weights corresponding to a particular outcome 

Args:
    outcome (str): the outcome value that these weights will be used to calculate 
    weight_names (list): list of weight names to retrive from props 

Returns: 
    weight_mappings (dict): mapping of a weight name to its correspondng value 
    
"""


def retrieve_weights(outcome: str, weight_names: list):
    return {
        weight: props.get_config(f"weights.{outcome}.{weight}")
        for weight in weight_names
    }


"""
Split dataframes into position specific data due to fantasy points 
vary signficantly by position and invoke functionality to get ranking ratios

Args:
   df (pd.DataFrame): dataframe to split 

Returns
   qb_data, rb_data, te_data, wr_data (tuple): split dataframes by position
"""


def split_data_by_position(df: pd.DataFrame):
    # split data by position
    qb_data = df[df["position"] == "QB"]
    rb_data = df[df["position"] == "RB"]
    wr_data = df[df["position"] == "WR"]
    te_data = df[df["position"] == "TE"]

    # drop un-needed position column
    new_qb_data = qb_data.drop("position", axis=1)
    new_rb_data = rb_data.drop("position", axis=1)
    new_wr_data = wr_data.drop("position", axis=1)
    new_te_data = te_data.drop("position", axis=1)

    qb_data_with_relevant_prop_ratios = get_relevant_props_ratios(new_qb_data, "QB")
    rb_data_with_relevant_prop_ratios = get_relevant_props_ratios(new_rb_data, "RB")
    wr_data_with_relevant_prop_ratios = get_relevant_props_ratios(new_wr_data, "WR")
    te_data_with_relevant_prop_ratios = get_relevant_props_ratios(new_te_data, "TE")

    return get_rankings_ratios(
        qb_data_with_relevant_prop_ratios,
        rb_data_with_relevant_prop_ratios,
        wr_data_with_relevant_prop_ratios,
        te_data_with_relevant_prop_ratios,
    )


""" 
Generate relevant player props ratios that reward low COSTS and higher LINES 

NOTE: Do not use this functionality for props such as interceptions 

TODO: Refactor this functionality to use helper function so it can be used by prediction.py

Args:
    df (pd.DataFrame): position specific data 

Returns:
    updated (pd.DataFrame): updated data frame with new ratio columns 
"""


def get_relevant_props_ratios(df: pd.DataFrame, position: str):
    selected_props = relevant_props[position]

    # TODO: make this a config
    k = 100

    # calculate relevant ratios
    for prop in selected_props:
        cost_col = f"{prop}_(over)_cost"
        line_col = f"{prop}_(over)_line"
        ratio_col = f"{prop}_ratio"

        if prop == "anytime_touchdown_scorer":
            cost_col = f"{prop}_cost"
            line_col = f"{prop}_line"

        if cost_col in df.columns and line_col in df.columns:
            df["adjusted_cost"] = df[cost_col].apply(
                lambda x: abs(x) if x < 0 else x + k
            )
            df[ratio_col] = df[line_col] / df["adjusted_cost"]

            # drop adjusted cost column each time
            df.drop(columns=["adjusted_cost"], inplace=True)

    # remove un-needed columns
    selected_props_cols = [f"{prop}_ratio" for prop in selected_props]
    keep_columns = [
        "player_id",
        "fantasy_points",
        "off_rush_rank",
        "off_pass_rank",
        "def_rush_rank",
        "def_pass_rank",
        "game_over_under",
        "spread",
        "is_favorited",
        "avg_fantasy_points",
    ] + selected_props_cols

    # return expected column without N/A values
    return df[keep_columns].dropna()


"""
Filter out outliers from data 

Args:
   df (pd.DataFrame): dataframe to filter out records 

Returns:
   filtered_df (pd.DataFrame): data frame with records filtered
"""


def filter_data(df: pd.DataFrame):
    logging.info(
        "Removing all records where fantasy points equal 0 of avg fantasy points less than 5"
    )
    df = df.dropna(
        subset=["fantasy_points", "avg_fantasy_points"]
    )  # remove rows with N/A fantasy points

    non_zero_data = df[(df["fantasy_points"] > 0) & (df["avg_fantasy_points"] > 5)]

    upper_fantasy_points_outliers = non_zero_data["fantasy_points"].quantile(0.99)
    lower_fantasy_points_outliers = non_zero_data["fantasy_points"].quantile(0.01)
    logging.info(
        f"Upper 99% Fantasy Points Outliers: [{upper_fantasy_points_outliers}]"
    )
    logging.info(f"Lower 1% Fantasy Points Outliers: [{lower_fantasy_points_outliers}]")

    # remove top 99% and bottom 1% of fantasy points
    no_fantasy_points_outliers = non_zero_data[
        (non_zero_data["fantasy_points"] < upper_fantasy_points_outliers)
        & (non_zero_data["fantasy_points"] > lower_fantasy_points_outliers)
    ]

    upper_avg_fantasy_points_outliers = no_fantasy_points_outliers[
        "avg_fantasy_points"
    ].quantile(0.99)
    lower_avg_fantasy_points_outliers = no_fantasy_points_outliers[
        "avg_fantasy_points"
    ].quantile(0.01)

    logging.info(
        f"Upper 99% Avg Fantasy Points Outliers: [{upper_avg_fantasy_points_outliers}]"
    )
    logging.info(
        f"Lower 1% Avg Fantasy Points Outliers: [{lower_avg_fantasy_points_outliers}]"
    )

    # remove top 99% and bottom 1% of avg_fantasy_points
    no_outlier_data = no_fantasy_points_outliers[
        (
            no_fantasy_points_outliers["avg_fantasy_points"]
            < upper_avg_fantasy_points_outliers
        )
        & (
            no_fantasy_points_outliers["avg_fantasy_points"]
            > lower_avg_fantasy_points_outliers
        )
    ]

    return no_outlier_data


"""
Remove any skewed nature from our features 

Args:
   qb_data (pd.DataFrame): qb dataframe to transform 
   rb_data (pd.DataFrame): rb dataframe to transform 
   wr_data (pd.DataFrame): wr dataframe to transform 
   te_data (pd.DataFrame): te dataframe to transform 
Returns:
   None
"""


def transform_data(
    qb_data: pd.DataFrame = None,
    rb_data: pd.DataFrame = None,
    wr_data: pd.DataFrame = None,
    te_data: pd.DataFrame = None,
):
    for df in [qb_data, rb_data, wr_data, te_data]:
        if df is None:
            continue

        logged_avg_fantasy_points = np.log1p(df["avg_fantasy_points"])

        # only account for fantasy points if passed as input
        if "fantasy_points" in df.columns:
            logged_fantasy_points = np.log1p(df["fantasy_points"])

        if "rush_ratio_rank" in df.columns:
            logged_ratio_rank = np.log1p(df["rush_ratio_rank"])
        else:
            logged_ratio_rank = np.log1p(df["pass_ratio_rank"])

        df["log_avg_fantasy_points"] = logged_avg_fantasy_points
        df["log_ratio_rank"] = logged_ratio_rank

        # only add log fantasy points if passed in input data
        if "fantasy_points" in df.columns:
            df["log_fantasy_points"] = logged_fantasy_points

    return qb_data, rb_data, wr_data, te_data


"""
Calculate ratio of relevant defensive ranking to relevant offensive ranking 

Args:
   df (pd.DataFrame): dataframe to generate relevant ranks for 
   is_rushing (bool): boolean to determine if we are calculating this for rushing or for passing

Returns:
   updated_df (pd.DataFrame): dataframe with correct rank ratios
"""


def get_rankings_ratios(
    qb_data: pd.DataFrame,
    rb_data: pd.DataFrame,
    wr_data: pd.DataFrame,
    te_data: pd.DataFrame,
):
    # Ensure modifications are applied to a copy, preventing SettingWithCopyWarning
    rb_data = rb_data.copy()
    qb_data = qb_data.copy()
    wr_data = wr_data.copy()
    te_data = te_data.copy()

    # Compute rush ratio rank for RBs
    rb_data.loc[:, "rush_ratio_rank"] = (
        rb_data["def_rush_rank"] / rb_data["off_rush_rank"]
    )

    # Compute pass ratio rank for QBs, WRs, and TEs
    for df in [qb_data, wr_data, te_data]:
        df.loc[:, "pass_ratio_rank"] = df["def_pass_rank"] / df["off_pass_rank"]

    return qb_data, rb_data, wr_data, te_data


"""
Addition of avg_fantasy_points feature so historical data is included in model 

Args:
   df (pd.DataFrame): dataframe to add avg_fantasy_points to 

Returns:
   df (pd.DataFrame): updated data frame 
"""


def include_averages(df: pd.DataFrame):
    player_avg_points = df.groupby("player_id")["fantasy_points"].mean()
    df["avg_fantasy_points"] = df["player_id"].map(player_avg_points)
    return df


"""
Plot a series in order to determine if outliers exists 

Args:
   series (pd.Series): series to plot 
   pdf_name (str): file name for pdf

Returns:
   None
"""


def plot(series: pd.Series, pdf_name: str):
    sns.displot(
        series, kind="hist", kde=True, bins=10, color="skyblue", edgecolor="white"
    )

    relative_dir = "./data/distributions"
    file_name = f"{pdf_name}.pdf"
    os.makedirs(relative_dir, exist_ok=True)
    file_path = os.path.join(relative_dir, file_name)

    plt.savefig(file_path)
    plt.close()


"""
Create scatter plots for features vs dependent variable

Args:
   data (pd.DataFrame): dataframe to yank tdata from 
   independet_var (str): value to plot against fantasy points 
   position (str); position of player 

Returns:
   None
"""


def create_plot(data: pd.DataFrame, independent_var: str, position: str):
    plt.figure(figsize=(8, 6))

    plt.scatter(data[independent_var], data["log_fantasy_points"])
    plt.title(f"Log Fantasy Points vs. {independent_var}")
    plt.xlabel(independent_var)
    plt.ylabel("Log Fantasy Points")

    relative_dir = "./data/scatter"
    file_name = f"{position}_{independent_var}_plot.pdf"
    os.makedirs(relative_dir, exist_ok=True)
    file_path = os.path.join(relative_dir, file_name)

    plt.savefig(file_path)
    plt.close()


"""
Parse player props retrieved from DB 

Args:
    df (pd.DataFrame): data frame containing player props 

Returns:
    df (pd.DataFrame): data frame with relevant player props columns
"""


def parse_player_props(df: pd.DataFrame):

    parsed_data = []

    for _, row in df.iterrows():
        week_props = row["props"]

        row_data = {}

        for prop in week_props:
            label = prop["label"].lower().replace(" ", "_").replace("/", "_")
            cost = prop["cost"]
            line = prop["line"]

            row_data[f"{label}_cost"] = cost
            row_data[f"{label}_line"] = line

        parsed_data.append(row_data)

    parsed_df = pd.DataFrame(parsed_data)

    df = df.drop(columns=["props"])
    return pd.concat([df, parsed_df], axis=1)


"""
Functionality to retrieve pandas df, containing independent & dependent variable(s)

Args:
   None

Returns:
   df (pd.DataFrame): data frame containing inputs/outputs for linear regression model
"""


def get_data():
    df = fetch_data.fetch_independent_and_dependent_variables_for_mult_lin_regression()
    dfs_with_player_props = parse_player_props(df)
    updated_df = include_averages(dfs_with_player_props)
    return updated_df
