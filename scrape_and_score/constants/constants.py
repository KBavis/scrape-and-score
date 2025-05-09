"""
Module to store all of our applications constants utilized by our packages
"""

VALID_POSITIONS = ["QB", "RB", "WR", "TE"]

TEAM_HREFS = {
    "Arizona Cardinals": "crd",
    "Baltimore Colts": "clt",
    "St. Louis Cardinals": "crd",
    "Boston Patriots": "nwe",
    "Chicago Bears": "chi",
    "Green Bay Packers": "gnb",
    "New York Giants": "nyg",
    "Detroit Lions": "det",
    "Washington Commanders": "was",
    "Washington Football Team": "was",
    "Washington Redskins": "was",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "Los Angeles Chargers": "sdg",
    "San Francisco 49ers": "sfo",
    "Houston Oilers": "oti",
    "Cleveland Browns": "cle",
    "Indianapolis Colts": "clt",
    "Dallas Cowboys": "dal",
    "Kansas City Chiefs": "kan",
    "Los Angeles Rams": "ram",
    "Denver Broncos": "den",
    "New York Jets": "nyj",
    "New England Patriots": "nwe",
    "Las Vegas Raiders": "rai",
    "Tennessee Titans": "oti",
    "Tennessee Oilers": "oti",
    "Phoenix Cardinals": "crd",
    "Los Angeles Raiders": "rai",
    "Buffalo Bills": "buf",
    "Minnesota Vikings": "min",
    "Atlanta Falcons": "atl",
    "Miami Dolphins": "mia",
    "New Orleans Saints": "nor",
    "Cincinnati Bengals": "cin",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tam",
    "Carolina Panthers": "car",
    "Jacksonville Jaguars": "jax",
    "Baltimore Ravens": "rav",
    "Houston Texans": "htx",
    "Oakland Raiders": "rai",
    "San Diego Chargers": "sdg",
    "St. Louis Rams": "ram",
    "Boston Patriots": "nwe",
}

MONTHS = {"September": 9, "October": 10, "November": 11, "December": 12, "January": 1}

LOCATIONS = {
    "Boston": {"latitude": 42.3656, "longitude": 71.0096, "airport": "BOS"},
    "Phoenix": {"latitude": 33.4352, "longitude": 112.0101, "airport": "PHX"},
    "Chicago": {"latitude": 41.9803, "longitude": 87.9090, "airport": "ORD"},
    "Green Bay": {"latitude": 44.4923, "longitude": 88.1278, "airport": "GRB"},
    "New York": {"latitude": 40.6895, "longitude": 74.1745, "airport": "EWR"},
    "Detroit": {"latitude": 42.2162, "longitude": 83.3554, "airport": "DTW"},
    "Washington DC": {"latitude": 38.9531, "longitude": 77.4565, "airport": "IAD"},
    "Philadelphia": {"latitude": 39.9526, "longitude": 75.1652, "airport": "PHL"},
    "Pittsburgh": {"latitude": 40.4919, "longitude": 80.2352, "airport": "PIT"},
    "Los Angeles": {"latitude": 33.9416, "longitude": 118.4085, "airport": "LAX"},
    "San Francisco": {"latitude": 37.3639, "longitude": 121.9289, "airport": "SJC"},
    "Cleveland": {"latitude": 41.4058, "longitude": 81.8539, "airport": "CLE"},
    "Indianapolis": {"latitude": 39.7169, "longitude": 86.2956, "airport": "IND"},
    "Dallas": {"latitude": 32.8998, "longitude": 97.0403, "airport": "DFW"},
    "Kansas City": {"latitude": 39.3036, "longitude": 94.7093, "airport": "MCI"},
    "Denver": {"latitude": 39.8564, "longitude": 104.6764, "airport": "DEN"},
    "Providence": {"latitude": 41.7235, "longitude": 71.4270, "airport": "PVD"},
    "Las Vegas": {"latitude": 36.0840, "longitude": 115.1537, "airport": "LAS"},
    "Nashville": {"latitude": 36.1263, "longitude": 86.6774, "airport": "BNA"},
    "Buffalo": {"latitude": 42.9397, "longitude": 78.7295, "airport": "BUF"},
    "Minneapolis": {"latitude": 44.8848, "longitude": 93.2223, "airport": "MSP"},
    "Atlanta": {"latitude": 33.6407, "longitude": 84.4277, "airport": "ATL"},
    "Miami": {"latitude": 26.0742, "longitude": 80.1506, "airport": "FLL"},
    "New Orleans": {"latitude": 29.9911, "longitude": 90.2592, "airport": "MSY"},
    "Cincinnati": {"latitude": 39.0508, "longitude": 84.6673, "airport": "CVG"},
    "Seattle": {"latitude": 47.4480, "longitude": 122.3088, "airport": "SEA"},
    "Tampa Bay": {"latitude": 27.9772, "longitude": 82.5311, "airport": "TPA"},
    "Charlotte": {"latitude": 35.2144, "longitude": 80.9473, "airport": "CLT"},
    "Jacksonville": {"latitude": 30.4941, "longitude": 81.6879, "airport": "JAX"},
    "Baltimore": {"latitude": 39.1774, "longitude": 76.6684, "airport": "BWI"},
    "Houston": {"latitude": 29.9902, "longitude": 95.3368, "airport": "IAH"},
    "Oakland": {"latitude": 37.7126, "longitude": 122.2197, "airport": "OAK"},
    "San Diego": {"latitude": 32.7338, "longitude": 117.1933, "airport": "SAN"},
    "St. Louis": {"latitude": 38.7499, "longitude": 90.3748, "airport": "STL"},
}

CITIES = {
    "Arizona Cardinals": "Phoenix",
    "Chicago Bears": "Chicago",
    "Green Bay Packers": "Green Bay",
    "New York Giants": "New York",
    "Detroit Lions": "Detroit",
    "Washington Commanders": "Washington DC",
    "Washington Football Team": "Washington DC",
    "Washington Redskins": "Washington DC",
    "Philadelphia Eagles": "Philadelphia",
    "Pittsburgh Steelers": "Pittsburgh",
    "Los Angeles Chargers": "Los Angeles",
    "San Francisco 49ers": "San Francisco",
    "Houston Oilers": "Houston",
    "Cleveland Browns": "Cleveland",
    "Indianapolis Colts": "Indianapolis",
    "Dallas Cowboys": "Dallas",
    "Kansas City Chiefs": "Kansas City",
    "Los Angeles Rams": "Los Angeles",
    "Denver Broncos": "Denver",
    "New York Jets": "New York",
    "New England Patriots": "Providence",
    "Las Vegas Raiders": "Las Vegas",
    "Oakland Raiders": "Oakland",
    "Tennessee Titans": "Nashville",
    "Tennessee Oilers": "Nashville",
    "Phoenix Cardinals": "Phoenix",
    "Los Angeles Raiders": "Los Angeles",
    "Buffalo Bills": "Buffalo",
    "Minnesota Vikings": "Minneapolis",
    "Atlanta Falcons": "Atlanta",
    "Miami Dolphins": "Miami",
    "New Orleans Saints": "New Orleans",
    "Cincinnati Bengals": "Cincinnati",
    "Seattle Seahawks": "Seattle",
    "Tampa Bay Buccaneers": "Tampa Bay",
    "Carolina Panthers": "Charlotte",
    "Jacksonville Jaguars": "Jacksonville",
    "Baltimore Ravens": "Baltimore",
    "Houston Texans": "Houston",
    "Oakland Raiders": "Oakland",
    "San Diego Chargers": "San Diego",
    "St. Louis Rams": "St. Louis",
    "Baltimore Colts": "Baltimore",
    "St. Louis Cardinals": "St. Louis",
    "Boston Patriots": "Boston",
}

TRAINING_CONFIGS = {
    "RB": {
        "Batch Size": 256,
        "Learning Rate": 2e-4
    }, 
    "WR": {
        "Batch Size": 512,
        "Learning Rate": 3e-4
    },
    "TE": {
        "Batch Size": 150,
        "Learning Rate": 2e-4
    },
    "QB": {
        "Batch Size": 64,
        "Learning Rate": 1e-4
    }
}

# Ensure that relevant features are included by our model via manual selection 

COMMON_FEATURES = [
    # team game logs 
    'rest_days',
    # player weekly features
    'avg_wkly_offensive_snaps', 
    'avg_wkly_fantasy_points',
    'avg_wkly_snap_pct',
    # players team offensive weekly features
    'avg_wkly_points_for',
    'avg_wkly_points_allowed',
    'avg_wkly_off_tot_yds',
    'avg_wkly_off_pass_rate',
    'avg_wkly_off_total_off_plays',
    'avg_wkly_off_yds_per_play',
    'avg_wkly_time_of_poss',
    'avg_wkly_def_time_of_possession',
    'avg_wkly_off_tot_yds',
    # player team defensive weekly features
    'avg_wkly_def_tot_yds',
    'avg_wkly_def_tot_off_plays',
    'avg_wkly_def_yds_per_play',
    'avg_wkly_def_thrd_down_conv',
    'avg_wkly_def_time_of_possession',
    # opposing team offensive stats 
    'avg_opp_wkly_points_for',
    'avg_opp_wkly_points_allowed',
    # cyclical features
    'week_sin', 
    'week_cos',
    # player betting odds
    'anytime_touchdown_scorer_cost',
    # player injuries 
    'official_game_status',
    'wednesday_practice_status',
    'thursday_practice_status', 
    'friday_practice_status', 
    # weather 
    'precip_probability',
    'temperature',
    'weather_status_clear_day',
    'weather_status_clear_night',
    'weather_status_cloudy',
    'weather_status_fog',
    'weather_status_partly_cloudy_day',
    'weather_status_partly_cloudy_night',
    'weather_status_rain',
    'weather_status_snow',
    'weather_status_unknown',
    'weather_status_wind',
    # depth chart position,
    'depth_chart_position',
    # player demographics 
    'age',
    'height',
    'weight',
    # betting odds
    'anytime_touchdown_scorer_cost',
    'is_favorited',
    'game_over_under',
    'spread',
    # surface
    'surface_turf',
    'surface_grass',
    'prev_year_player_fumbles',
    'is_home_team',
    # injuries
    'injury_hamstring', 'injury_ankle', 'injury_concussion',
    # turnovers 
    'avg_wkly_turnovers'
]

QB_FEATURES = COMMON_FEATURES + [
    # betting odds
    'passing_yards_over_under_cost',
    'passing_yards_over_under_line',
    'passing_touchdowns_over_under_cost',
    'passing_touchdowns_over_under_line',
    'passing_completions_over_under_cost',
    'passing_completions_over_under_line',
    'rushing_yards_over_under_cost',
    'rushing_yards_over_under_line',
    'rushing_attempts_over_under_cost',
    'rushing_attempts_over_under_line',
    'most_rushing_yards_cost',
    'interception_over_under_cost',
    'interception_over_under_line',
    # weekly stats
    'avg_wkly_rush_yds',
    'avg_wkly_rush_tds',
    'avg_wkly_rush_broken_tackles',
    'avg_wkly_rush_yds_before_contact_per_att',
    'avg_wkly_rush_yds_after_contact_per_att',
    'avg_wkly_rating',
    'avg_opp_wkly_def_pass_yds',
    'avg_opp_wkly_def_sacked',
    'avg_wkly_pass_yds_per_att',
    'avg_wkly_completed_air_yards',
    'avg_wkly_completions', 
    'avg_wkly_pass_tds',
    'avg_opp_wkly_def_int',
    'avg_wkly_pass_pressured_pct',
    'avg_wkly_pass_yds_per_scramble',
    'avg_wkly_pass_poor_throws_pct',
    'avg_wkly_attempts',
    'avg_wkly_interceptions',
    'avg_wkly_pass_yds_per_att',
    'avg_wkly_completed_air_yards_per_cmp',
    'avg_wkly_completed_air_yards_per_att',
    'avg_wkly_intended_air_yards_per_pass_attempt',
    'avg_wkly_completed_air_yards_per_cmp',
    'avg_wkly_off_yds_gained_per_pass_att'
]


RB_FEATURES = COMMON_FEATURES + [
    # betting odds 
    "receiving_yards_over_under_cost",
    "receiving_yards_over_under_line",
    "receptions_over_under_cost",
    "receptions_over_under_line",
    'most_receiving_yards_cost',
    'rushing_yards_over_under_cost',
    'rushing_yards_over_under_line',
    'rushing_attempts_over_under_cost',
    'rushing_attempts_over_under_line',
    'most_rushing_yards_cost',
    # weekly features
    "avg_wkly_rec_yds",
    'avg_wkly_rush_broken_tackles',
    'avg_opp_wkly_def_rush_yds',
    'avg_opp_wkly_def_rush_tds',
    'avg_wkly_rush_attempts',
    'avg_wkly_rush_yds_before_contact_per_att',
    'avg_wkly_rush_yds_after_contact_per_att',
    'avg_wkly_rec_yds_after_catch_per_rec',
    'avg_wkly_rec_yds_before_catch_per_rec',
    'avg_wkly_avg_depth_of_target',
    'avg_wkly_off_rush_tds',
    'avg_wkly_tgt_share', 
    'avg_wkly_yds_per_touch'
]

TE_FEATURES = COMMON_FEATURES + [
    # bettings odds
    "receiving_yards_over_under_cost",
    "receiving_yards_over_under_line",
    "receptions_over_under_cost",
    "receptions_over_under_line",
    'most_receiving_yards_cost',
    # weekly stats
    'avg_wkly_targets',
    'avg_opp_wkly_def_pass_tds',
    'avg_wkly_rec_yds',
    'avg_wkly_rec_yds_after_catch_per_rec',
    'avg_wkly_rec_yds_before_catch_per_rec',
    'avg_wkly_rec_first_downs',
    'avg_wkly_rec_tds',
    'avg_opp_wkly_def_pass_tds',
    'avg_wkly_tgt_share', 
    'avg_wkly_yds_per_touch',
    'avg_wkly_rec_qbr_when_targeted',
    'avg_wkly_rec_drop_pct',
    'avg_wkly_avg_depth_of_target',
]

WR_FEATURES = COMMON_FEATURES + [
    # betting odds
    "receiving_yards_over_under_cost",
    "receiving_yards_over_under_line",
    "receptions_over_under_cost",
    'receptions_over_under_line',
    'most_receiving_yards_cost',
    # weekly stats
    'avg_wkly_intended_air_yards',
    'avg_opp_wkly_def_pass_yds',
    'avg_wkly_rec_yds_after_catch_per_rec',
    'avg_wkly_targets',
    'avg_wkly_rec_drop_pct',
    'avg_opp_wkly_def_pass_yds',
    'avg_wkly_tgt_share', 
    'avg_wkly_yds_per_touch',
    'avg_wkly_rec_qbr_when_targeted',
    'avg_wkly_avg_depth_of_target',
]
