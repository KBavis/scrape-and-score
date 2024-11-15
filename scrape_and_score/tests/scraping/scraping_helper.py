from unittest.mock import MagicMock

'''
Mock function to map each 'data-stat' attribute to its respective mock return value for common metrics 
'''
def mock_find_common_metrics(tag, attrs): 
   data_stat_map = {
      'game_date': MagicMock(text='2023-09-12'),
      'week_num': MagicMock(text='1'),
      'team': MagicMock(text='DAL'),
      'game_location': MagicMock(text='@'),
      'opp': MagicMock(text='NYG'),
      'game_result': MagicMock(text='W 20-10')
   }
   
   # return mock based on specific data-stat value   
   return data_stat_map.get(attrs['data-stat'])

'''
Mock function to map each 'data-stat' attribute to its respective mock return value for WR specific metrics 
'''
def mock_find_wr_metrics(tag, attrs): 
   data_stat_map = {
      'targets': MagicMock(text='9'),
      'rec': MagicMock(text='6'),
      'rec_yds': MagicMock(text='118'),
      'rec_td': MagicMock(text='2'),
      'off_pct': MagicMock(text='67%')
   }
   
   return data_stat_map.get(attrs['data-stat'])

'''
Mock function to map each 'data-stat' attribute to its respective mock return value for WR specific metrics 
'''
def mock_find_rb_metrics(tag, attrs): 
   data_stat_map = {
      'rush_att': MagicMock(text='9'),
      'rush_yds': MagicMock(text='68'),
      'rush_td': MagicMock(text='2'),
      'targets': MagicMock(text='2'),
      'rec': MagicMock(text='2'),
      'rec_yds': MagicMock(text='41'),
      'rec_td': MagicMock(text='0')
   }
   
   return data_stat_map.get(attrs['data-stat'])

'''
Mock function to map each 'data-stat' attribute to its respective mock return value for QB specific metrics 
'''
def mock_find_qb_metrics(tag, attrs): 
   data_stat_map = {
      'pass_cmp': MagicMock(text='24'),      
      'pass_att': MagicMock(text='36'),       
      'pass_yds': MagicMock(text='315'),      
      'pass_td': MagicMock(text='3'),         
      'pass_int': MagicMock(text='1'),        
      'pass_rating': MagicMock(text='98.7'),  
      'pass_sacked': MagicMock(text='2'),     
      'rush_att': MagicMock(text='5'),        
      'rush_yds': MagicMock(text='23'),       
      'rush_td': MagicMock(text='1')          
   }
   
   return data_stat_map.get(attrs['data-stat'])

'''
Setup mocks necessary for get_game_log() 

Args: status(str)
         - Status to set 
'''
def setup_game_log_mocks(status):    
   mock_soup = MagicMock()
   mock_tbody = MagicMock()
   mock_tr = MagicMock()
   mock_element = MagicMock()
   mock_element.text = status
   mock_soup.find.return_value = mock_tbody
   mock_tbody.find_all.return_value = [mock_tr]
   mock_tr.find_all.return_value = [MagicMock(), mock_element]
   
   return mock_soup


'''
Mock the functionality of add_common_game_log_metrics() with hard-coded values 

'''
def mock_add_common_game_log_metrics(data, tr):
   data['date'].append('2024-11-10')
   data['week'].append(10)
   data['team'].append('Team A')
   data['game_location'].append('@')
   data['opp'].append('Team B')
   data['result'].append('W')
   data['team_pts'].append(24)
   data['opp_pts'].append(17)

'''
Mock the functionality of add_wr_game_log_metrics() with hard-coded values 
'''   
def mock_add_wr_game_log_metrics(data, tr):   
   data['tgt'].append(7)
   data['rec'].append(5)
   data['rec_yds'].append(102)
   data['rec_td'].append(1)
   data['snap_pct'].append(67.7)
   

'''
Mock the BeautifulSoup instance passed to get_href

Args:
   invalid_date (bool) - flag to determine if dates should be invalid
   invalid_a_tag (bool) - flag to determine if a tag should be invalid
'''
def setup_get_href_mocks(invalid_date, invalid_a_tag): 
   mock_soup = MagicMock() 
   mock_div_players = MagicMock()
   
   mock_player_one = MagicMock() 
   mock_player_two = MagicMock() 
   mock_player_one.find.return_value = { 'href' : 'my-href.htm' } if not invalid_a_tag else None 
   
   mock_player_one.text = 'Arbitrary Invalid-Date' if invalid_date else 'QB 2023-2024'
   mock_player_two.text = 'Arbitrary Invalid-Date' if invalid_date else 'QB 2023-2024'
   mock_soup.find.return_value = mock_div_players
   mock_div_players.find_all.return_value = [mock_player_one, mock_player_two]
   
   return mock_soup