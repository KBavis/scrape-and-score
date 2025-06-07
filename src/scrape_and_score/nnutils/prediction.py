import torch
import logging
from torch import nn
import pandas as pd
from scrape_and_score.service import player_service


def generate_predictions(
    position: str,
    week: int,
    season: int,
    model: nn.Module,
    players: pd.Series,
    X: torch.tensor,
):
    """
    Generate top-40 predictions for a particular position

    Args:
        position (str): relevant position predictions are geenrated for
        week (int): relevant week this prediction is being generated for
        season (int); relevant season this prediciton is generated for
        model (nn.Module): neural network model trained for this particular position
        players (pd.Series): relevant list of player IDs to correspond to each prediction
        X (torch.tensor): tensor to utilzie to generate predictions
    """

    logging.info(
        f"Generating top-40 fantasy point predictions for the {position} position for Week {week} of the {season} NFL Season"
    )

    # determine relevant device to have tensors on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using the following device: {device}")

    # move tensors to correct device
    X = X.to(device)
    model = model.to(device)
    model.eval()

    # generate predictions
    with torch.no_grad():
        preds = model(X).squeeze(1)

    # reset players index
    players = players.reset_index(drop=True)

    # iterate through and generate predictions for each player
    predictions = {}
    for i, pred in enumerate(preds):

        # extract player name this prediction is being made for
        player_name = player_service.get_player_name_by_id(int(players[i]))
        predictions[player_name] = pred.item()

    sorted_dict = dict(
        sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    )

    results = []
    for i, (key, value) in enumerate(sorted_dict.items()):
        if i >= 40:
            break
        results.append(f"{i+1}. {key}: {value:.2f}")

    return results


def log_predictions(predictions: dict, week: int, season: int):
    """
    Log out relevant top-40 predictions

    Args:
        predictions (list): list of predictions
    """

    for position, ranks in predictions.items():

        print(
            f"""-----------------------------------------\nTop 40 {position} Predictions – Week {week}, {season} Season\n-----------------------------------------"""
        )

        for rank in ranks:
            print(f"{rank}")

        print("\n")