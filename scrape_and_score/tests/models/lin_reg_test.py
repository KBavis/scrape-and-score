from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from models.lin_reg import LinReg


@patch("models.lin_reg.train_test_split")
def test_create_training_and_testing_split(mock_train_test_split):
    mock_train_test_split.side_effect = (
        lambda inputs, targets, test_size, random_state: (
            inputs[:2],
            inputs[2:],
            targets[:2],
            targets[2:],
        )
    )

    qb_data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "log_fantasy_points": [9, 10, 11, 12],
        }
    )

    rb_data = qb_data.copy()
    wr_data = qb_data.copy()
    te_data = qb_data.copy()

    lin_reg = LinReg(qb_data, rb_data, wr_data, te_data)
    split_data = lin_reg.create_training_and_testing_split()

    assert set(split_data.keys()) == {"QB", "RB", "WR", "TE"}
    for position in ["QB", "RB", "WR", "TE"]:
        assert "x_train" in split_data[position]
        assert "x_test" in split_data[position]
        assert "y_train" in split_data[position]
        assert "y_test" in split_data[position]
        assert len(split_data[position]["x_train"]) == 2
        assert len(split_data[position]["x_test"]) == 2
        assert len(split_data[position]["y_train"]) == 2
        assert len(split_data[position]["y_test"]) == 2


@patch("models.lin_reg.LinearRegression")
@patch("models.lin_reg.LinReg.create_training_and_testing_split")
def test_test_regressions(mock_create_split, mock_linear_regression):
    mock_split_data = {
        "QB": {
            "x_train": np.array([[1, 2], [3, 4]]),
            "x_test": np.array([[5, 6], [7, 8]]),
            "y_train": pd.Series([9, 10]),
            "y_test": pd.Series([11, 12]),
        },
        "RB": {
            "x_train": np.array([[1, 2], [3, 4]]),
            "x_test": np.array([[5, 6], [7, 8]]),
            "y_train": pd.Series([9, 10]),
            "y_test": pd.Series([11, 12]),
        },
        "WR": {
            "x_train": np.array([[1, 2], [3, 4]]),
            "x_test": np.array([[5, 6], [7, 8]]),
            "y_train": pd.Series([9, 10]),
            "y_test": pd.Series([11, 12]),
        },
        "TE": {
            "x_train": np.array([[1, 2], [3, 4]]),
            "x_test": np.array([[5, 6], [7, 8]]),
            "y_train": pd.Series([9, 10]),
            "y_test": pd.Series([11, 12]),
        },
    }
    mock_create_split.return_value = mock_split_data

    mock_regression = MagicMock()
    mock_regression.predict.return_value = np.array(
        [11.0, 12.0]
    )  # Consistent return value for all positions
    mock_linear_regression.return_value = mock_regression

    qb_data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "log_fantasy_points": [9, 10, 11, 12],
        }
    )
    rb_data = qb_data.copy()
    wr_data = qb_data.copy()
    te_data = qb_data.copy()

    lin_reg = LinReg(qb_data, rb_data, wr_data, te_data)
    lin_reg.create_regressions()
    lin_reg.test_regressions()

    assert (
        mock_regression.predict.call_count == 8
    )  # four for training, four for testing
