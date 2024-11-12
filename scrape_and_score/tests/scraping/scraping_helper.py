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

def setup_game_log_mocks():    
   mock_soup = MagicMock()
   mock_tbody = MagicMock()
   mock_tr = MagicMock()
   mock_element = MagicMock()
   mock_element.text = ''
   mock_soup.find.return_value = mock_tbody
   mock_tbody.find_all.return_value = [mock_tr]
   mock_tr.find_all.return_value = [MagicMock(), mock_element]
   
   return mock_soup