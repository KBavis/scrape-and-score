from scrape_and_score.workflows.workflows import (
    predict,
    upcoming,
    historical,
    results,
    linear_regression,
    neural_network,
)
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch, mock_open, call
from pytest import raises
import pandas as pd
import numpy as np


def test_predict_workflows_raises_exception_no_model_passed():

    with raises(Exception) as e:
        predict(1, 2024, "")

    assert (
        str(e.value)
        == "Only 'nn' or 'lin_reg' are valid aguments to invoke 'predict' workflow"
    )


def test_predict_workflows_raises_exception_lin_reg_passed():

    with raises(Exception) as e:
        predict(1, 2024, "lin_reg")

    assert (
        str(e.value)
        == "Linear regression prediction functionality is currently not implemented"
    )


def test_predict_workflow_invokes_necessary_functions():

    # arrange & mock
    mock_df = pd.DataFrame(
        [
            {"position_RB": 1, "rb_feature": 100, "player_id": 1},
            {"position_WR": 1, "wr_feature": 100, "player_id": 2},
            {"position_TE": 1, "te_feature": 100, "player_id": 3},
            {"position_QB": 1, "qb_feature": 100, "player_id": 4},
        ]
    )
    mock_features = {
        "RB": ["rb_feature"],
        "WR": ["wr_feature"],
        "TE": ["te_feature"],
        "QB": ["qb_feature"],
    }

    with patch(
        "scrape_and_score.workflows.workflows.nn_preprocess.preprocess"
    ) as mock_preprocess, patch(
        "scrape_and_score.workflows.workflows.utils.get_position_features"
    ) as mock_get_features, patch(
        "scrape_and_score.workflows.workflows.nn_preprocess.add_missing_features"
    ) as mock_add_missing_features, patch(
        "scrape_and_score.workflows.workflows.nn_preprocess.scale_and_transform"
    ) as mock_scale_and_transform, patch(
        "scrape_and_score.workflows.workflows.prediction.generate_predictions"
    ) as mock_gen_predictions, patch(
        "scrape_and_score.workflows.workflows.prediction.log_predictions"
    ) as mock_log_predictions:

        # arrange mocks
        mock_preprocess.return_value = mock_df
        mock_get_features.return_value = mock_features
        mock_add_missing_features.side_effect = lambda x, y: x
        mock_scale_and_transform.return_value = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        mock_gen_predictions.return_value = []
        mock_log_predictions.return_value = None

        # act
        predict(1, 2024, "nn")

        # assert
        assert mock_preprocess.call_count == 1
        assert mock_get_features.call_count == 1
        assert mock_add_missing_features.call_count == 4
        assert mock_scale_and_transform.call_count == 4
        assert mock_log_predictions.call_count == 1


@patch("scrape_and_score.workflows.workflows.utils.is_player_records_persisted")
@patch("scrape_and_score.workflows.workflows.our_lads.scrape_and_persist_upcoming")
@patch("scrape_and_score.workflows.workflows.utils.is_player_demographics_persisted")
@patch(
    "scrape_and_score.workflows.workflows.pfr.scrape_and_persist_player_demographics"
)
@patch("scrape_and_score.workflows.workflows.espn.scrape_upcoming_games")
@patch("scrape_and_score.workflows.workflows.utils.generate_game_mapping")
@patch("scrape_and_score.workflows.workflows.utils.filter_completed_games")
@patch("scrape_and_score.workflows.workflows.utils.add_stubbed_player_game_logs")
@patch("scrape_and_score.workflows.workflows.rotowire.scrape_upcoming")
@patch("scrape_and_score.workflows.workflows.football_db.scrape_upcoming")
@patch(
    "scrape_and_score.workflows.workflows.betting_pros.fetch_upcoming_player_odds_and_game_conditions"
)
def test_upcoming_invokes_necessary_functions(
    mock_fetch_odds,
    mock_football_db_scrape,
    mock_rotowire_scrape,
    mock_add_stubbed,
    mock_filter_completed,
    mock_generate_mapping,
    mock_espn_scrape,
    mock_pfr_demographics,
    mock_is_demo_persisted,
    mock_our_lads_scrape,
    mock_is_records_persisted,
):

    # arrange & setup mocks
    week = 1
    season = 2024
    fake_game_mapping = [
        {"team_ids": [1, 2], "player_ids": [10, 20]},
        {"team_ids": [3, 4], "player_ids": [30, 40]},
    ]
    all_team_ids = [1, 2, 3, 4]
    all_player_ids = [10, 20, 30, 40]
    mock_generate_mapping.return_value = fake_game_mapping
    mock_filter_completed.return_value = fake_game_mapping
    mock_is_records_persisted.return_value = True
    mock_is_demo_persisted.return_value = False

    # act
    upcoming(week, season)

    # assert
    mock_is_records_persisted.assert_called_once_with(season)
    mock_our_lads_scrape.assert_called_once_with(season, week, is_update=True)
    mock_is_demo_persisted.assert_called_once_with(season)
    mock_pfr_demographics.assert_called_once_with(season)
    mock_espn_scrape.assert_called_once_with(season, week)
    mock_generate_mapping.assert_called_once_with(season, week)
    mock_filter_completed.assert_called_once_with(fake_game_mapping)

    mock_add_stubbed.assert_called_once_with(all_player_ids, week, season)
    mock_rotowire_scrape.assert_called_once_with(week, season, all_team_ids)
    mock_football_db_scrape.assert_called_once_with(week, season, all_player_ids)
    mock_fetch_odds.assert_called_once_with(week, season, all_player_ids)


@patch("scrape_and_score.workflows.workflows.our_lads.scrape_and_persist")
@patch("scrape_and_score.workflows.workflows.pfr.scrape_historical")
@patch(
    "scrape_and_score.workflows.workflows.player_game_logs_service.calculate_fantasy_points"
)
@patch(
    "scrape_and_score.workflows.workflows.team_game_logs_service.calculate_all_teams_rankings"
)
@patch("scrape_and_score.workflows.workflows.rotowire.scrape_all")
@patch("scrape_and_score.workflows.workflows.betting_pros.fetch_historical_odds")
@patch(
    "scrape_and_score.workflows.workflows.pfr.fetch_teams_and_players_seasonal_metrics"
)
@patch("scrape_and_score.workflows.workflows.pfr.scrape_player_advanced_metrics")
@patch("scrape_and_score.workflows.workflows.football_db.scrape_historical")
def test_historical_invokes_necessary_functions(
    mock_football_db_scrape,
    mock_pfr_scrape_adv,
    mock_pfr_seasonal_metrics,
    mock_betting_fetch,
    mock_rotowire_scrape_all,
    mock_team_rankings,
    mock_player_fp,
    mock_pfr_scrape,
    mock_our_lads_scrape,
):

    # arrnage & setup mocks
    start_year = 2020
    end_year = 2022
    expected_betting_calls = [((year,),) for year in range(start_year, end_year + 1)]
    expected_calls = [((year,),) for year in range(start_year, end_year)]
    mock_player_fp.return_value = None
    mock_team_rankings.return_value = None
    mock_betting_fetch.return_value = None

    # act
    historical(start_year, end_year)

    # assert
    mock_our_lads_scrape.assert_called_once_with(start_year, end_year)
    mock_pfr_scrape.assert_called_once_with(start_year, end_year)
    mock_player_fp.assert_called_once_with(False, start_year, end_year)

    actual_calls = mock_team_rankings.call_args_list  # account for multiple invocations
    assert actual_calls == expected_calls

    mock_rotowire_scrape_all.assert_called_once_with(start_year, end_year)

    actual_betting_calls = (
        mock_betting_fetch.call_args_list
    )  # account for multiple invocations
    assert actual_betting_calls == expected_betting_calls

    mock_pfr_seasonal_metrics.assert_called_once_with(start_year, end_year)
    mock_pfr_scrape_adv.assert_called_once_with(start_year, end_year)
    mock_football_db_scrape.assert_called_once_with(start_year, end_year)


def test_results_invokes_necessary_functions_when_week_18():
    week = 18
    season = 2024

    with patch(
        "scrape_and_score.workflows.workflows.pfr.update_game_logs_and_insert_advanced_metrics"
    ) as mock_update_logs, patch(
        "scrape_and_score.workflows.workflows.player_game_logs_service.calculate_fantasy_points"
    ) as mock_calc_fp, patch(
        "scrape_and_score.workflows.workflows.team_game_logs_service.calculate_all_teams_rankings"
    ) as mock_calc_team_rank, patch(
        "scrape_and_score.workflows.workflows.rotowire.update_recent_betting_records"
    ) as mock_update_betting, patch(
        "scrape_and_score.workflows.workflows.props.get_config"
    ) as mock_get_config, patch(
        "scrape_and_score.workflows.workflows.utils.are_team_seasonal_metrics_persisted"
    ) as mock_team_metrics_persisted, patch(
        "scrape_and_score.workflows.workflows.utils.are_player_seasonal_metrics_persisted"
    ) as mock_player_metrics_persisted, patch(
        "scrape_and_score.workflows.workflows.pfr.fetch_teams_and_players_seasonal_metrics"
    ) as mock_fetch_metrics:

        # arrange
        mock_get_config.return_value = 18
        mock_team_metrics_persisted.return_value = False
        mock_player_metrics_persisted.return_value = False

        # act
        results(week, season)

        # assert
        mock_update_logs.assert_called_once_with(week, season)
        mock_calc_fp.assert_called_once_with(season, season, week=week)
        mock_calc_team_rank.assert_called_once_with(season, week)
        mock_update_betting.assert_called_once_with(week, season)
        mock_fetch_metrics.assert_called_once_with(season, season)


def test_results_invokes_necessary_functions_when_not_week_18():
    week = 16
    season = 2024

    with patch(
        "scrape_and_score.workflows.workflows.pfr.update_game_logs_and_insert_advanced_metrics"
    ) as mock_update_logs, patch(
        "scrape_and_score.workflows.workflows.player_game_logs_service.calculate_fantasy_points"
    ) as mock_calc_fp, patch(
        "scrape_and_score.workflows.workflows.team_game_logs_service.calculate_all_teams_rankings"
    ) as mock_calc_team_rank, patch(
        "scrape_and_score.workflows.workflows.rotowire.update_recent_betting_records"
    ) as mock_update_betting, patch(
        "scrape_and_score.workflows.workflows.props.get_config"
    ) as mock_get_config, patch(
        "scrape_and_score.workflows.workflows.utils.are_team_seasonal_metrics_persisted"
    ) as mock_team_metrics_persisted, patch(
        "scrape_and_score.workflows.workflows.utils.are_player_seasonal_metrics_persisted"
    ) as mock_player_metrics_persisted, patch(
        "scrape_and_score.workflows.workflows.pfr.fetch_teams_and_players_seasonal_metrics"
    ) as mock_fetch_metrics:

        # arrange
        mock_get_config.return_value = 18
        mock_team_metrics_persisted.return_value = False
        mock_player_metrics_persisted.return_value = False

        # act
        results(week, season)

        # assert
        mock_update_logs.assert_called_once_with(week, season)
        mock_calc_fp.assert_called_once_with(season, season, week=week)
        mock_calc_team_rank.assert_called_once_with(season, week)
        mock_update_betting.assert_called_once_with(week, season)
        assert mock_fetch_metrics.call_count == 0


def test_linear_regression_invokes_necessary_functions():
    with patch(
        "scrape_and_score.workflows.workflows.linreg_preprocess.pre_process_data"
    ) as mock_preprocess, patch(
        "scrape_and_score.workflows.workflows.LinReg"
    ) as mock_LinReg:

        # arrange
        mock_preprocess.return_value = ("qb_data", "rb_data", "wr_data", "te_data")

        mock_linreg_instance = MagicMock()
        mock_LinReg.return_value = mock_linreg_instance

        # act
        linear_regression()

        # assert
        mock_preprocess.assert_called_once()
        mock_LinReg.assert_called_once_with("qb_data", "rb_data", "wr_data", "te_data")
        mock_linreg_instance.create_regressions.assert_called_once()
        mock_linreg_instance.test_regressions.assert_called_once()


def test_neural_network_loads_nns_when_available():
    should_train = False
    start_time = datetime.now()

    with patch(
        "scrape_and_score.workflows.workflows.os.path.exists", return_value=True
    ), patch("scrape_and_score.workflows.workflows.torch.load") as mock_torch_load:

        neural_network(should_train, start_time)

        # enusre each nn loaded
        assert mock_torch_load.call_count == 4
        mock_torch_load.assert_any_call("models/nn/rb_model.pth", weights_only=False)
        mock_torch_load.assert_any_call("models/nn/qb_model.pth", weights_only=False)
        mock_torch_load.assert_any_call("models/nn/wr_model.pth", weights_only=False)
        mock_torch_load.assert_any_call("models/nn/te_model.pth", weights_only=False)


@patch("scrape_and_score.workflows.workflows.NeuralNetwork")
@patch("scrape_and_score.workflows.workflows.FantasyDataset")
@patch("scrape_and_score.workflows.workflows.DataLoader")
@patch(
    "scrape_and_score.workflows.workflows.nn_preprocess.feature_selection",
    return_value=["feature1", "feature2"],
)
@patch("scrape_and_score.workflows.workflows.nn_preprocess.preprocess")
@patch("scrape_and_score.workflows.workflows.optimization.optimization_loop")
@patch("scrape_and_score.workflows.workflows.post_training.feature_importance")
@patch("scrape_and_score.workflows.workflows.open", new_callable=mock_open)
@patch("scrape_and_score.workflows.workflows.os.makedirs")
@patch("scrape_and_score.workflows.workflows.os.path.exists", return_value=False)
@patch("scrape_and_score.workflows.workflows.torch.save")
def test_neural_network_invokes_necessary_functions(
    mock_torch_save,
    mock_path_exists,
    mock_makedirs,
    mock_open_file,
    mock_feature_importance,
    mock_optimization_loop,
    mock_preprocess,
    mock_feature_selection,
    mock_dataloader,
    mock_dataset,
    mock_nn,
):
    df = pd.DataFrame(
        {
            "position_RB": [1, 1, 1, 1],
            "position_QB": [0, 0, 0, 0],
            "position_WR": [0, 0, 0, 0],
            "position_TE": [0, 0, 0, 0],
            "feature1": [0.5, 0.7, 0.3, 0.6],
            "feature2": [1.0, 0.9, 1.1, 1.2],
            "fantasy_points": [10, 15, 5, 8],
        }
    )
    mock_preprocess.return_value = df

    mock_dataset.return_value = MagicMock()
    mock_dataloader.return_value = MagicMock()
    mock_nn.return_value.to.return_value = MagicMock()
    mock_optimization_loop.return_value = None

    neural_network(should_train=True, start_time=datetime.now())

    assert mock_feature_selection.called
    assert mock_optimization_loop.called
    assert mock_torch_save.called
    assert mock_feature_importance.called


@patch("scrape_and_score.workflows.workflows.os.makedirs")
@patch("scrape_and_score.workflows.workflows.os.path.exists", return_value=True)
@patch("scrape_and_score.workflows.workflows.open", new_callable=mock_open)
@patch("scrape_and_score.workflows.workflows.post_training.feature_importance")
@patch("scrape_and_score.workflows.workflows.optimization.optimization_loop")
@patch(
    "scrape_and_score.workflows.workflows.nn_preprocess.feature_selection",
    return_value=["feature1", "feature2"],
)
@patch("scrape_and_score.workflows.workflows.nn_preprocess.preprocess")
@patch("scrape_and_score.workflows.workflows.DataLoader", return_value=MagicMock())
@patch("scrape_and_score.workflows.workflows.FantasyDataset", return_value=MagicMock())
@patch("scrape_and_score.workflows.workflows.NeuralNetwork")
@patch("scrape_and_score.workflows.workflows.torch.save")
def test_writes_selected_features_to_file(
    mock_torch_save,
    mock_nn,
    mock_dataset,
    mock_dataloader,
    mock_preprocess,
    mock_feature_selection,
    mock_optimization_loop,
    mock_feature_importance,
    mock_open_file,
    mock_path_exists,
    mock_makedirs,
):

    # arrange & setup mocks
    test_time = datetime(2025, 6, 7, 12, 0, 0)
    df = pd.DataFrame(
        {
            "position_RB": [1, 1],
            "position_QB": [0, 0],
            "position_WR": [0, 0],
            "position_TE": [0, 0],
            "feature1": [0.1, 0.2],
            "feature2": [1.1, 1.2],
            "fantasy_points": [10, 11],
        }
    )
    mock_preprocess.return_value = df
    mock_model_instance = MagicMock()
    mock_nn.return_value.to.return_value = mock_model_instance
    expected_filename = (
        f"data/inputs/RB_inputs_{test_time.strftime('%Y%m%d_%H%M%S')}.txt"
    )

    # act
    neural_network(should_train=True, start_time=test_time)

    # assert
    mock_open_file.assert_any_call(expected_filename, "w")
    file_handle = mock_open_file()
    file_handle.write.assert_has_calls([call("feature1\n"), call("feature2\n")])
