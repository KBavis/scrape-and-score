from unittest.mock import MagicMock
from unittest.mock import patch
from data import preprocess 
import pandas as pd
import numpy as np

@patch('data.preprocess.include_averages')
@patch('data.preprocess.fetch_data.fetch_independent_and_dependent_variables_for_mult_lin_regression')
def test_get_data_calls_expected_functions(mock_fetch, mock_include_averages):
    mock_data = [{"player_id": "12"}, {"player_id" : "13"}]
    mock_fetch.return_value = pd.DataFrame(data = mock_data)
    mock_include_averages.return_value = mock_data

    preprocess.get_data()

    mock_fetch.assert_called_once() 
    mock_include_averages.assert_called_once() 


def test_include_averages_returns_expected_df(): 
    data = [
        {"player_id": 1, "fantasy_points": 7},
        {"player_id": 1, "fantasy_points": 8},
        {"player_id": 1, "fantasy_points": 9}, 
        {"player_id": 2, "fantasy_points": 4},
        {"player_id": 2, "fantasy_points": 5},
        {"player_id": 2, "fantasy_points": 6}
    ]
    mock_df = pd.DataFrame(data=data, columns=['player_id', 'fantasy_points'])
    

    actual_df = preprocess.include_averages(mock_df)
    added_col = actual_df['avg_fantasy_points']

    assert isinstance(added_col, pd.Series)
    assert added_col.to_list() == [8,8,8,5,5,5] 


def test_get_rankings_ratios_returns_expected_tuples(): 
    mock_qb_data = pd.DataFrame(data=
        [
            {"def_pass_rank": 32, "off_pass_rank": 1},
            {"def_pass_rank": 1, "off_pass_rank": 32}
        ])
    mock_wr_data = pd.DataFrame(data=
        [
            {"def_pass_rank": 32, "off_pass_rank": 1},
            {"def_pass_rank": 1, "off_pass_rank": 32}
        ])
    mock_te_data = pd.DataFrame(data=
        [
            {"def_pass_rank": 32, "off_pass_rank": 1},
            {"def_pass_rank": 1, "off_pass_rank": 32}
        ])
    mock_rb_data = pd.DataFrame(data=
        [
            {"def_rush_rank": 32, "off_rush_rank": 1},
            {"def_rush_rank": 1, "off_rush_rank": 32}
        ])

    actual_qb_data, actual_rb_data, actual_wr_data, actual_te_data = preprocess.get_rankings_ratios(mock_qb_data, mock_rb_data, mock_wr_data, mock_te_data)

    assert actual_rb_data['rush_ratio_rank'].to_list() == [32.0, 0.03125]
    assert actual_qb_data['pass_ratio_rank'].to_list() == [32.0, 0.03125]
    assert actual_wr_data['pass_ratio_rank'].to_list() == [32.0, 0.03125]
    assert actual_te_data['pass_ratio_rank'].to_list() == [32.0, 0.03125]


def test_transform_data_returns_expected_tuples(): 
    qb_df = pd.DataFrame(data=[{"avg_fantasy_points": 15.0, "pass_ratio_rank": 32.0, "fantasy_points": 17.7}])
    te_df = pd.DataFrame(data=[{"avg_fantasy_points": 16.0, "pass_ratio_rank": 16.0, "fantasy_points": 18.7}])
    wr_df = pd.DataFrame(data=[{"avg_fantasy_points": 17.0, "pass_ratio_rank": 1.0, "fantasy_points": 19.7}])
    rb_df = pd.DataFrame(data=[{"avg_fantasy_points": 15.0, "rush_ratio_rank": 32.0, "fantasy_points": 18.7}])

    qb_data, rb_data, wr_data, te_data = preprocess.transform_data(qb_df,rb_df, wr_df, te_df)

    assert qb_data['log_avg_fantasy_points'].to_list() == [np.log1p(15.0)]
    assert qb_data['log_fantasy_points'].to_list() == [np.log1p(17.7)]
    assert qb_data['log_ratio_rank'].to_list() == [np.log1p(32.0)]


    assert rb_data['log_avg_fantasy_points'].to_list() == [np.log1p(15.0)]
    assert rb_data['log_fantasy_points'].to_list() == [np.log1p(18.7)]
    assert rb_data['log_ratio_rank'].to_list() == [np.log1p(32.0)]
    
    assert wr_data['log_avg_fantasy_points'].to_list() == [np.log1p(17.0)]
    assert wr_data['log_fantasy_points'].to_list() == [np.log1p(19.7)]
    assert wr_data['log_ratio_rank'].to_list() == [np.log1p(1.0)]

    assert te_data['log_avg_fantasy_points'].to_list() == [np.log1p(16.0)]
    assert te_data['log_fantasy_points'].to_list() == [np.log1p(18.7)]
    assert te_data['log_ratio_rank'].to_list() == [np.log1p(16.0)]

@patch('data.preprocess.get_rankings_ratios')
def test_split_data_by_position(mock_get_rankings):
    data = [{"player_id": 1, "position": "QB"}, {"player_id": 2, "position": "RB"}, {"player_id": 3, "position": "TE"}, {"player_id": 4, "position": "WR"}]
    mock_df = pd.DataFrame(data=data)

    mock_get_rankings.side_effect = lambda w,x,y,z: (w,x,y,z) 

    qb_data, rb_data, wr_data, te_data = preprocess.split_data_by_position(mock_df)
    tuples = [qb_data, rb_data, wr_data, te_data]

    for df in tuples:
        assert 'position' not in df.columns 
        assert len(df.columns) == 1
        assert 'player_id' in df.columns 

def test_filter_data():
    data = {
        "fantasy_points": [15, 25, 35, 45, 55],
        "avg_fantasy_points": [6.5, 7.5, 8.5, 9.5, 10.5],
        "player_name": ["Player A", "Player B", "Player C", "Player D", "Player E"]
    }
    df = pd.DataFrame(data)
    
    filtered_df = preprocess.filter_data(df)
    
    assert len(filtered_df) > 0, "Filtered DataFrame should not be empty"
    assert len(filtered_df) < len(df), "Filtered DataFrame should have fewer rows than input"


@patch('data.preprocess.variance_inflation_factor', side_effect = [6.0, 4.0])
def test_validate_ols_assmptions_removes_vif_over_5(mock_vif): 
    data = { 
        "log_avg_fantasy_points": [1.2, 1.5, 1.7],
        "log_ratio_rank": [0.8, 1.1, 1.4]
    } 
    df = pd.DataFrame(data=data)

    result_df = preprocess.validate_ols_assumptions(df)

    assert "log_avg_fantasy_points" not in result_df.columns
    assert "log_ratio_rank" in result_df.columns

