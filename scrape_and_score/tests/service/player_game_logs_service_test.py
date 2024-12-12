from service import player_game_logs_service
from unittest.mock import patch 
import pandas as pd


@patch('service.player_game_logs_service.get_player_id_by_name')
@patch('service.player_game_logs_service.get_qb_game_log_tuples')
@patch('service.player_game_logs_service.insert_data.insert_qb_player_game_logs')
def test_insert_multiple_player_game_logs_calls_expected_functions_when_position_is_qb(mock_insert_qb_game_logs, mock_get_qb_game_log_tuples, mock_get_player_id_by_name): 
   mock_df = pd.DataFrame(data = [{'Hi'}])
   depth_charts = []
   player_metrics = [{'player': 'Anthony Richardson', 'position': 'QB', 'player_metrics': mock_df}]
   
   mock_get_player_id_by_name.return_value = 5
   mock_get_qb_game_log_tuples.return_value = (5,5)
   mock_insert_qb_game_logs.return_value = None
   
   player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts)
   
   mock_insert_qb_game_logs.assert_called_once()
   mock_get_player_id_by_name.assert_called_once()
   mock_get_qb_game_log_tuples.assert_called_once() 
   
   
   
@patch('service.player_game_logs_service.get_player_id_by_name')
@patch('service.player_game_logs_service.get_rb_game_log_tuples')
@patch('service.player_game_logs_service.insert_data.insert_rb_player_game_logs')
def test_insert_multiple_player_game_logs_calls_expected_functions_when_position_is_rb(mock_insert_rb_game_logs, mock_get_rb_game_log_tuples, mock_get_player_id_by_name):
   mock_df = pd.DataFrame(data = [{'Hi'}])
   depth_charts = []
   player_metrics = [{'player': 'Jonathon Taylor', 'position': 'RB', 'player_metrics': mock_df}]
   
   mock_get_player_id_by_name.return_value = 5
   mock_get_rb_game_log_tuples.return_value = (5,5)
   mock_insert_rb_game_logs.return_value = None
   
   player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts)
   
   mock_insert_rb_game_logs.assert_called_once()
   mock_get_player_id_by_name.assert_called_once()
   mock_get_rb_game_log_tuples.assert_called_once()  


@patch('service.player_game_logs_service.get_player_id_by_name')
@patch('service.player_game_logs_service.get_wr_or_te_game_log_tuples')
@patch('service.player_game_logs_service.insert_data.insert_wr_or_te_player_game_logs')
def test_insert_multiple_player_game_logs_calls_expected_functions_when_position_is_wr_or_te(mock_insert_wr_game_logs, mock_get_wr_game_log_tuples, mock_get_player_id_by_name): 
   mock_df = pd.DataFrame(data = [{'Hi'}])
   depth_charts = []
   player_metrics = [{'player': 'Josh Downs', 'position': 'WR', 'player_metrics': mock_df}]
   
   mock_get_player_id_by_name.return_value = 5
   mock_get_wr_game_log_tuples.return_value = (5,5)
   mock_insert_wr_game_logs.return_value = None
   
   player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts)
   
   mock_insert_wr_game_logs.assert_called_once()
   mock_get_player_id_by_name.assert_called_once()
   mock_get_wr_game_log_tuples.assert_called_once()  
 
   
@patch('service.player_game_logs_service.get_player_id_by_name')
def test_insert_multiple_player_game_logs_raises_exception_when_unknown_posiiton_found(mock_get_player_id):
   mock_df = pd.DataFrame(data = [{'Hi'}])
   depth_charts = []
   invalid_position = 'OL'
   player_metrics = [{'player': 'Josh Downs', 'position': invalid_position, 'player_metrics': mock_df}]
   mock_get_player_id.return_value = 5
   
   try:
      player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts)
   except Exception as e:
      assert str(e) == f"Unknown position '{invalid_position}'; unable to fetch game log tuples"


@patch('service.player_game_logs_service.insert_data.insert_wr_or_te_player_game_logs')
@patch('service.player_game_logs_service.insert_data.insert_rb_player_game_logs')
@patch('service.player_game_logs_service.insert_data.insert_qb_player_game_logs')
@patch('service.player_game_logs_service.get_player_id_by_name')
def test_insert_multiple_player_game_logs_skips_players_with_empty_metrics(mock_get_player_id, mock_insert_qb, mock_insert_rb, mock_insert_wr):
   mock_df = pd.DataFrame(data = [])
   depth_charts = []
   player_metrics = [{'player': 'Josh Downs', 'position': 'WR', 'player_metrics': mock_df}]
   mock_get_player_id.return_value = 5
   
   player_game_logs_service.insert_multiple_players_game_logs(player_metrics, depth_charts)
   
   assert mock_get_player_id.call_count == 0
   assert mock_insert_qb.call_count == 0
   assert mock_insert_rb.call_count == 0
   assert mock_insert_wr.call_count == 0
   
   
@patch('service.player_game_logs_service.fetch_data.fetch_one_player_game_log')  
def test_is_player_game_logs_empty_returns_true_when_game_log_not_persisted(mock_fetch_one_player_game_log):
   mock_fetch_one_player_game_log.return_value = None 
   assert player_game_logs_service.is_player_game_logs_empty() == True


@patch('service.player_game_logs_service.fetch_data.fetch_one_player_game_log')  
def test_is_player_game_logs_empty_returns_false_when_game_log_is_persisted(mock_fetch_one_player_game_log): 
   mock_fetch_one_player_game_log.return_value = None 
   assert player_game_logs_service.is_player_game_logs_empty() == True

@patch('service.player_game_logs_service.service_util.extract_day_from_date', return_value = '12')
@patch('service.player_game_logs_service.service_util.extract_year_from_date', return_value = '2024')
@patch('service.player_game_logs_service.service_util.extract_home_team_from_game_location', return_value = True)
@patch('service.player_game_logs_service.team_service.get_team_id_by_name', return_value = '18')
@patch('service.player_game_logs_service.service_util.get_team_name_by_pfr_acronym', return_value = 'Colts')
def test_get_qb_game_log_tuples_returns_expected_tuples(mock_get_team_name, mock_get_team_id, mock_get_home_team, mock_extract_year, mock_extract_day):
   player_id = 1
   expected_tuples = [
      (
         player_id, 1, '12', '2024', True, '18', 'W 24-10', 24, 10, 20, 30, 250, 2, 1, 98.5, 2, 5, 30, 1
      )
   ]
   data = {
      'week': [1],
      'date': ['2024-09-10'],
      'game_location': ['@'],
      'opp': ['IND'],
      'result': ['W 24-10'],
      'team_pts': [24],
      'opp_pts': [10],
      'cmp': [20],
      'att': [30],
      'pass_yds': [250],
      'pass_td': [2],
      'int': [1],
      'rating': [98.5],
      'sacked': [2],
      'rush_att': [5],
      'rush_yds': [30],
      'rush_td': [1]
   }
   df = pd.DataFrame(data)
   
   actual_tuples = player_game_logs_service.get_qb_game_log_tuples(df, player_id)
   
   assert actual_tuples == expected_tuples


@patch('service.player_game_logs_service.service_util.extract_day_from_date', return_value = '12')
@patch('service.player_game_logs_service.service_util.extract_year_from_date', return_value = '2024')
@patch('service.player_game_logs_service.service_util.extract_home_team_from_game_location', return_value = True)
@patch('service.player_game_logs_service.team_service.get_team_id_by_name', return_value = '18')
@patch('service.player_game_logs_service.service_util.get_team_name_by_pfr_acronym', return_value = 'Colts')
def test_get_qb_game_log_tuples_calls_expected_functions(mock_get_team_name, mock_get_team_id, mock_get_home_team, mock_extract_year, mock_extract_day):
   player_id = 1
   data = {
      'week': [1],
      'date': ['2024-09-10'],
      'game_location': ['@'],
      'opp': ['IND'],
      'result': ['W 24-10'],
      'team_pts': [24],
      'opp_pts': [10],
      'cmp': [20],
      'att': [30],
      'pass_yds': [250],
      'pass_td': [2],
      'int': [1],
      'rating': [98.5],
      'sacked': [2],
      'rush_att': [5],
      'rush_yds': [30],
      'rush_td': [1]
   }
   df = pd.DataFrame(data)

   player_game_logs_service.get_qb_game_log_tuples(df, player_id)

   mock_extract_day.assert_called_once()
   mock_extract_year.assert_called_once()
   mock_get_home_team.assert_called_once()
   mock_get_team_name.assert_called_once()
   mock_get_team_id.assert_called_once()

@patch('service.player_game_logs_service.service_util.extract_day_from_date', return_value='15')
@patch('service.player_game_logs_service.service_util.extract_year_from_date', return_value='2024')
@patch('service.player_game_logs_service.service_util.extract_home_team_from_game_location', return_value=False)
@patch('service.player_game_logs_service.team_service.get_team_id_by_name', return_value='20')
@patch('service.player_game_logs_service.service_util.get_team_name_by_pfr_acronym', return_value='Packers')
def test_get_rb_game_log_tuples_returns_expected_tuples(mock_get_team_name, mock_get_team_id, mock_get_home_team, mock_extract_year, mock_extract_day):
   player_id = 2
   data = {
      'week': [2],
      'date': ['2024-09-17'],
      'game_location': [''],
      'opp': ['GB'],
      'result': ['L 17-21'],
      'team_pts': [17],
      'opp_pts': [21],
      'rush_att': [15],
      'rush_yds': [70],
      'rush_td': [1],
      'tgt': [5],
      'rec': [3],
      'rec_yds': [25],
      'rec_td': [0]
   }
   df = pd.DataFrame(data)

   expected_tuples = [
      (
         player_id, 2, '15', '2024', False, '20', 'L 17-21', 17, 21, 15, 70, 1, 5, 3, 25, 0
      )
   ]

   result = player_game_logs_service.get_rb_game_log_tuples(df, player_id)
      
   assert result == expected_tuples


@patch('service.player_game_logs_service.service_util.extract_day_from_date')
@patch('service.player_game_logs_service.service_util.extract_year_from_date')
@patch('service.player_game_logs_service.service_util.extract_home_team_from_game_location')
@patch('service.player_game_logs_service.team_service.get_team_id_by_name')
@patch('service.player_game_logs_service.service_util.get_team_name_by_pfr_acronym')
def test_get_rb_game_log_tuples_calls_expected_functions(mock_get_team_name, mock_get_team_id, mock_get_home_team, mock_extract_year, mock_extract_day):
   player_id = 2
   data = {
      'week': [2],
      'date': ['2024-09-17'],
      'game_location': [''],
      'opp': ['GB'],
      'result': ['L 17-21'],
      'team_pts': [17],
      'opp_pts': [21],
      'rush_att': [15],
      'rush_yds': [70],
      'rush_td': [1],
      'tgt': [5],
      'rec': [3],
      'rec_yds': [25],
      'rec_td': [0]
   }
   df = pd.DataFrame(data)

   player_game_logs_service.get_rb_game_log_tuples(df, player_id)

   mock_extract_day.assert_called_once()
   mock_extract_year.assert_called_once()
   mock_get_home_team.assert_called_once()
   mock_get_team_name.assert_called_once()
   mock_get_team_id.assert_called_once()

@patch('service.player_game_logs_service.service_util.extract_day_from_date', return_value='22')
@patch('service.player_game_logs_service.service_util.extract_year_from_date', return_value='2024')
@patch('service.player_game_logs_service.service_util.extract_home_team_from_game_location', return_value=True)
@patch('service.player_game_logs_service.team_service.get_team_id_by_name', return_value='30')
@patch('service.player_game_logs_service.service_util.get_team_name_by_pfr_acronym', return_value='Bears')
def test_get_wr_or_te_game_log_tuples_returns_expected_tuples(mock_get_team_name, mock_get_team_id, mock_get_home_team, mock_extract_year, mock_extract_day):
   player_id = 3
   data = {
      'week': [3],
      'date': ['2024-09-24'],
      'game_location': ['@'],
      'opp': ['CHI'],
      'result': ['W 30-27'],
      'team_pts': [30],
      'opp_pts': [27],
      'tgt': [8],
      'rec': [6],
      'rec_yds': [85],
      'rec_td': [1],
      'snap_pct': [90]
   }
   df = pd.DataFrame(data)

   expected_tuples = [
      (
         player_id, 3, '22', '2024', True, '30', 'W 30-27', 30, 27, 8, 6, 85, 1, 90
      )
   ]

   result = player_game_logs_service.get_wr_or_te_game_log_tuples(df, player_id)

   assert result == expected_tuples

@patch('service.player_game_logs_service.service_util.extract_day_from_date')
@patch('service.player_game_logs_service.service_util.extract_year_from_date')
@patch('service.player_game_logs_service.service_util.extract_home_team_from_game_location')
@patch('service.player_game_logs_service.team_service.get_team_id_by_name')
@patch('service.player_game_logs_service.service_util.get_team_name_by_pfr_acronym')
def test_get_wr_or_te_game_log_tuples_calls_expected_functions(mock_get_team_name, mock_get_team_id, mock_get_home_team, mock_extract_year, mock_extract_day):
   player_id = 3
   data = {
      'week': [3],
      'date': ['2024-09-24'],
      'game_location': ['@'],
      'opp': ['CHI'],
      'result': ['W 30-27'],
      'team_pts': [30],
      'opp_pts': [27],
      'tgt': [8],
      'rec': [6],
      'rec_yds': [85],
      'rec_td': [1],
      'snap_pct': [90]
   }
   df = pd.DataFrame(data)

   player_game_logs_service.get_wr_or_te_game_log_tuples(df, player_id)

   mock_extract_day.assert_called_once()
   mock_extract_year.assert_called_once()
   mock_get_home_team.assert_called_once()
   mock_get_team_name.assert_called_once()
   mock_get_team_id.assert_called_once()