import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from scrape_and_score.nnutils import post_training


@pytest.fixture
def mock_dataset():
    class MockDataset:
        def __init__(self):
            self.X = torch.rand((300, 5))
            self.df = pd.DataFrame(
                {
                    "feature1": np.random.rand(300),
                    "feature2": np.random.rand(300),
                    "feature3": np.random.rand(300),
                    "feature4": np.random.rand(300),
                    "feature5": np.random.rand(300),
                    "fantasy_points": np.random.rand(300),
                }
            )

    return MockDataset()


@pytest.fixture
def mock_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    return model


@patch("scrape_and_score.nnutils.post_training.shap.DeepExplainer")
@patch("scrape_and_score.nnutils.post_training.save_model_feature_significance_table")
def test_feature_importance(
    mock_save_table, mock_deep_explainer, mock_dataset, mock_model
):
    # arrange
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.shap_values.return_value = np.random.rand(200, 5)
    mock_deep_explainer.return_value = mock_explainer_instance

    # act
    post_training.feature_importance(
        model=mock_model, training_data_set=mock_dataset, position="QB", device="cpu"
    )

    # assert
    mock_deep_explainer.assert_called_once()
    mock_explainer_instance.shap_values.assert_called_once()
    mock_save_table.assert_called_once()
    df_arg, position_arg = mock_save_table.call_args[0]
    assert isinstance(df_arg, pd.DataFrame)
    assert position_arg == "QB"
    assert "Feature" in df_arg.columns
    assert "SHAP_Importance" in df_arg.columns


@patch("pandas.DataFrame.to_csv")
@patch("scrape_and_score.nnutils.post_training.datetime")
@patch("scrape_and_score.nnutils.post_training.os.makedirs")
def test_save_model_feature_significance_table_creates_file(
    mock_makedirs, mock_datetime, mock_to_csv
):

    # arrange & setup mocks
    df = pd.DataFrame(
        {"Feature": ["feature1", "feature2"], "SHAP_Importance": [0.5, 0.3]}
    )
    position = "WR"
    mock_datetime.now.return_value.strftime.return_value = "2025-06-07_12-00-00"
    expected_path = (
        "data/shap_analysis/WR_feature_summary_table_2025-06-07_12-00-00.csv"
    )

    # act
    post_training.save_model_feature_significance_table(df, position)

    # assert
    mock_makedirs.assert_called_once_with("data/shap_analysis", exist_ok=True)
    mock_to_csv.assert_called_once_with(expected_path, index=False)
