import pytest
import torch
import pandas as pd
from unittest.mock import patch
from torch import nn

from scrape_and_score.nnutils.prediction import generate_predictions, log_predictions


class DummyModel(nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)  


@pytest.fixture
def dummy_input():
    return torch.ones((10, 5)) # dummy tensor (10 players, 5 features each)


@pytest.fixture
def dummy_players():
    return pd.Series([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])


@patch("scrape_and_score.nnutils.prediction.player_service.get_player_name_by_id")
def test_generate_predictions(mock_get_name, dummy_input, dummy_players):

    # arrange & setup mocks
    mock_get_name.side_effect = lambda player_id: f"Player_{player_id}"
    model = DummyModel()
    position = "RB"
    week = 5
    season = 2024

    # act
    predictions = generate_predictions(position, week, season, model, dummy_players, dummy_input)

    # assertions
    assert len(predictions) == 10 # ensure only 10 predictions generated for 10 players
    assert predictions[0].startswith("1.")
    assert "Player_101" in predictions[0]


@patch("scrape_and_score.nnutils.prediction.player_service.get_player_name_by_id")
def test_generate_predictions_limit_top_40(mock_get_name):
    
    # arrange & setup mocks
    mock_get_name.side_effect = lambda player_id: f"Player_{player_id}"
    dummy_players = pd.Series(range(1, 101)) # 100 player IDs
    dummy_input = torch.ones((100, 5)) # 100 players w/ 5 features each
    model = DummyModel()

    predictions = generate_predictions("QB", 3, 2024, model, dummy_players, dummy_input)

    # validate only top 40 account for
    assert len(predictions) == 40
    assert predictions[0].startswith("1.")
    assert predictions[-1].startswith("40.")


@patch("builtins.print")
def test_log_predictions(mock_print):
    dummy_predictions = {
        "QB": [
            "1. Player_A: 25.6",
            "2. Player_B: 24.3",
        ],
        "RB": [
            "1. Player_X: 22.1",
            "2. Player_Y: 21.0",
        ]
    }

    log_predictions(dummy_predictions, week=4, season=2023)

    calls = [str(call.args[0]) for call in mock_print.call_args_list]

    # validate headers are present
    assert any("Top 40 QB Predictions" in c for c in calls)
    assert any("Top 40 RB Predictions" in c for c in calls)

    # validate rankings are present
    assert any("1. Player_A: 25.6" in c for c in calls)
    assert any("2. Player_B: 24.3" in c for c in calls)
    assert any("1. Player_X: 22.1" in c for c in calls)
    assert any("2. Player_Y: 21.0" in c for c in calls)