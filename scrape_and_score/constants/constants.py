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

MARKET_ID_MAPPING = {
    71: "Player To Score The Last Touchdown",
    253: "Fantasy Points Over/Under",
    75: "Most Receiving Yards",
    105: "Receiving Yards Over/Under",
    104: "Receptions Over/Under",
    66: "First Touchdown Scorer",
    78: "Anytime Touchdown Scorer",
    107: "Rushing Yards Over/Under",
    106: "Rushing Attempts Over/Under",
    101: "Interception Over/Under",
    103: "Passing Yards Over/Under",
    333: "Passing Attempts Over/Under",
    102: "Passing Touchdowns Over/Under",
    76: "Most Rushing Yards",
    100: "Passing Completions Over/Under",
    73: "Most Passing Touchdowns",
    74: "Most Passing Yards",
}


RELEVANT_PROPS = {
    "QB": [
        "rushing_attempts_over_under",
        "rushing_yards_over_under",
        "anytime_touchdown_scorer",
        "passing_yards_over_under",
        "passing_touchdowns_over_under",
        "passing_attempts_over_under",
        "fantasy_points_over_under",
    ],
    "RB": [
        "rushing_attempts_over_under",
        "rushing_yards_over_under",
        "anytime_touchdown_scorer",
        "receiving_yards_over_under",
        "receptions_over_under",
        "fantasy_points_over_under",
    ],
    "WR": [
        "anytime_touchdown_scorer",
        "receiving_yards_over_under",
        "receptions_over_under",
        "fantasy_points_over_under",
    ],
    "TE": [
        "anytime_touchdown_scorer",
        "receiving_yards_over_under",
        "receptions_over_under",
        "fantasy_points_over_under",
    ],
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


QUERY = """
    WITH PlayerProps AS (
        SELECT 
            pbo.player_name,
            pbo.week,
            pbo.season,
            jsonb_agg(
                json_build_object(
                    'label', pbo.label,
                    'line', pbo.line,
                    'cost', pbo.cost
                )
            ) AS props
        FROM 
            player_betting_odds pbo
        GROUP BY
            pbo.player_name, pbo.week, pbo.season
    )
	  SELECT
         p.player_id,
         p.position,
         pgl.fantasy_points,

         -- Team Rest Days 
         tgl.rest_days,
         tgl.home_team,

         -- Player Injuries 
         pi.injury_loc as injury_locations, 
         pi.wed_prac_sts as wednesday_practice_status, 
         pi.thurs_prac_sts as thursday_practice_status, 
         pi.fri_prac_sts as friday_practice_status, 
         pi.off_sts as official_game_status,

		 -- Game Conditions
		 gc.weather_icon as weather_status,
		 gc.temperature,
		 gc.game_time, 
		 gc.month, 
		 gc.precip_probability,
		 gc.precip_type,
		 gc.wind_speed,
		 gc.wind_bearing,
		 gc.surface,

		 -- Player Weekly Aggregate Metrics 
		 pam.avg_pass_first_downs AS avg_wkly_pass_first_downs,
		 pam.avg_pass_first_downs_per_pass_play AS avg_wkly_pass_first_downs_per_pass_play,
		 pam.avg_intended_air_yards AS avg_wkly_intended_air_yards,
		 pam.avg_intended_air_yards_per_pass_attempt AS avg_wkly_intended_air_yards_per_pass_attempt,
		 pam.avg_completed_air_yards AS avg_wkly_completed_air_yards,
		 pam.avg_completed_air_yards_per_cmp AS avg_wkly_completed_air_yards_per_cmp,
		 pam.avg_completed_air_yards_per_att AS avg_wkly_completed_air_yards_per_att,
		 pam.avg_pass_yds_after_catch AS avg_wkly_pass_yds_after_catch,
		 pam.avg_pass_yds_after_catch_per_cmp AS avg_wkly_pass_yds_after_catch_per_cmp,
		 pam.avg_pass_drops AS avg_wkly_pass_drops,
		 pam.avg_pass_drop_pct AS avg_wkly_pass_drop_pct,
		 pam.avg_pass_poor_throws AS avg_wkly_pass_poor_throws,
		 pam.avg_pass_poor_throws_pct AS avg_wkly_pass_poor_throws_pct,
		 pam.avg_pass_blitzed AS avg_wkly_pass_blitzed,
		 pam.avg_pass_hurried AS avg_wkly_pass_hurried,
		 pam.avg_pass_hits AS avg_wkly_pass_hits,
		 pam.avg_pass_pressured AS avg_wkly_pass_pressured,
		 pam.avg_pass_pressured_pct AS avg_wkly_pass_pressured_pct,
		 pam.avg_pass_scrambles AS avg_wkly_pass_scrambles,
		 pam.avg_pass_yds_per_scramble AS avg_wkly_pass_yds_per_scramble,
		 pam.avg_rush_first_downs AS avg_wkly_rush_first_downs,
		 pam.avg_rush_yds_before_contact AS avg_wkly_rush_yds_before_contact,
		 pam.avg_rush_yds_before_contact_per_att AS avg_wkly_rush_yds_before_contact_per_att,
		 pam.avg_rush_yds_after_contact AS avg_wkly_rush_yds_after_contact,
		 pam.avg_rush_yds_after_contact_per_att AS avg_wkly_rush_yds_after_contact_per_att,
		 pam.avg_rush_broken_tackles AS avg_wkly_rush_broken_tackles,
		 pam.avg_rush_att_per_broken_tackle AS avg_wkly_rush_att_per_broken_tackle,
		 pam.avg_rec_first_downs AS avg_wkly_rec_first_downs,
		 pam.avg_rec_yds_before_catch AS avg_wkly_rec_yds_before_catch,
		 pam.avg_rec_yds_before_catch_per_rec AS avg_wkly_rec_yds_before_catch_per_rec,
		 pam.avg_rec_yds_after_catch AS avg_wkly_rec_yds_after_catch,
		 pam.avg_rec_yds_after_catch_per_rec AS avg_wkly_rec_yds_after_catch_per_rec,
		 pam.avg_avg_depth_of_target AS avg_wkly_avg_depth_of_target,
		 pam.avg_rec_broken_tackles AS avg_wkly_rec_broken_tackles,
		 pam.avg_rec_per_broken_tackle AS avg_wkly_rec_per_broken_tackle,
		 pam.avg_rec_dropped_passes AS avg_wkly_rec_dropped_passes,
		 pam.avg_rec_drop_pct AS avg_wkly_rec_drop_pct,
		 pam.avg_rec_int_when_targeted AS avg_wkly_rec_int_when_targeted,
		 pam.avg_rec_qbr_when_targeted AS avg_wkly_rec_qbr_when_targeted,
		 pam.avg_completions AS avg_wkly_completions,
		 pam.avg_attempts AS avg_wkly_attempts,
		 pam.avg_pass_yds AS avg_wkly_pass_yds,
		 pam.avg_pass_tds AS avg_wkly_pass_tds,
		 pam.avg_interceptions AS avg_wkly_interceptions,
		 pam.avg_rating AS avg_wkly_rating,
		 pam.avg_sacked AS avg_wkly_sacked,
		 pam.avg_rush_attempts AS avg_wkly_rush_attempts,
		 pam.avg_rush_yds AS avg_wkly_rush_yds,
		 pam.avg_rush_tds AS avg_wkly_rush_tds,
		 pam.avg_targets AS avg_wkly_targets,
		 pam.avg_receptions AS avg_wkly_receptions,
		 pam.avg_rec_yds AS avg_wkly_rec_yds,
		 pam.avg_rec_tds AS avg_wkly_rec_tds,
		 pam.avg_snap_pct AS avg_wkly_snap_pct,
		 pam.avg_offensive_snaps AS avg_wkly_offensive_snaps,
		 pam.avg_fantasy_points AS avg_wkly_fantasy_points,
        
         -- Players Team Weekly General Metrics
        tam.avg_points_for AS avg_wkly_points_for,
        tam.avg_points_allowed AS avg_wkly_points_allowed,
        tam.avg_result_margin AS avg_wkly_result_margin,

        -- Players Team Weekly Offensive Metrics
        tam.avg_tot_yds AS avg_wkly_off_tot_yds,
        tam.avg_pass_yds AS avg_wkly_off_pass_yds,
        tam.avg_rush_yds AS avg_wkly_off_rush_yds,
        tam.avg_pass_tds AS avg_wkly_off_pass_tds,
        tam.avg_pass_cmp AS avg_wkly_off_pass_cmp,
        tam.avg_pass_att AS avg_wkly_off_pass_att,
        tam.avg_pass_cmp_pct AS avg_wkly_off_pass_cmp_pct,
        tam.avg_yds_gained_per_pass_att AS avg_wkly_off_yds_gained_per_pass_att,
        tam.avg_adj_yds_gained_per_pass_att AS avg_wkly_off_adj_yds_gained_per_pass_att,
        tam.avg_pass_rate AS avg_wkly_off_pass_rate,
        tam.avg_sacked AS avg_wkly_off_sacked,
        tam.avg_sack_yds_lost AS avg_wkly_off_sack_yds_lost,
        tam.avg_rush_att AS avg_wkly_off_rush_att,
        tam.avg_rush_tds AS avg_wkly_off_rush_tds,
        tam.avg_rush_yds_per_att AS avg_wkly_off_rush_yds_per_att,
        tam.avg_total_off_plays AS avg_wkly_off_total_off_plays,
        tam.avg_yds_per_play AS avg_wkly_off_yds_per_play,

        -- Players Team Weekly Defensive Metrics
        tam.avg_opp_rush_yds AS avg_wkly_def_rush_yds,
        tam.avg_opp_tot_yds AS avg_wkly_def_tot_yds,
        tam.avg_opp_pass_yds AS avg_wkly_def_pass_yds,
        tam.avg_opp_pass_tds AS avg_wkly_def_pass_tds,
        tam.avg_opp_pass_cmp AS avg_wkly_def_pass_cmp,
        tam.avg_opp_pass_att AS avg_wkly_def_pass_att,
        tam.avg_opp_pass_cmp_pct AS avg_wkly_def_pass_cmp_pct,
        tam.avg_opp_yds_gained_per_pass_att AS avg_wkly_def_yds_gained_per_pass_att,
        tam.avg_opp_adj_yds_gained_per_pass_att AS avg_wkly_def_adj_yds_gained_per_pass_att,
        tam.avg_opp_pass_rate AS avg_wkly_def_pass_rate,
        tam.avg_opp_sacked AS avg_wkly_def_sacked,
        tam.avg_opp_sack_yds_lost AS avg_wkly_def_sack_yds_lost,
        tam.avg_opp_rush_att AS avg_wkly_def_rush_att,
        tam.avg_opp_rush_tds AS avg_wkly_def_rush_tds,
        tam.avg_opp_rush_yds_per_att AS avg_wkly_def_rush_yds_per_att,
        tam.avg_opp_tot_off_plays AS avg_wkly_def_tot_off_plays,
        tam.avg_opp_yds_per_play AS avg_wkly_def_yds_per_play,
        tam.avg_opp_fga AS avg_wkly_def_fga,
        tam.avg_opp_fgm AS avg_wkly_def_fgm,
        tam.avg_opp_xpa AS avg_wkly_def_xpa,
        tam.avg_opp_xpm AS avg_wkly_def_xpm,
        tam.avg_opp_total_punts AS avg_wkly_def_total_punts,
        tam.avg_opp_punt_yds AS avg_wkly_def_punt_yds,
        tam.avg_opp_pass_fds AS avg_wkly_def_pass_fds,
        tam.avg_opp_rsh_fds AS avg_wkly_def_rsh_fds,
        tam.avg_opp_pen_fds AS avg_wkly_def_pen_fds,
        tam.avg_opp_total_fds AS avg_wkly_def_total_fds,
        tam.avg_opp_thrd_down_conv AS avg_wkly_def_thrd_down_conv,
        tam.avg_opp_thrd_down_att AS avg_wkly_def_thrd_down_att,
        tam.avg_opp_foruth_down_conv AS avg_wkly_def_fourth_down_conv,
        tam.avg_opp_foruth_down_att AS avg_wkly_def_fourth_down_att,
        tam.avg_opp_penalties AS avg_wkly_def_penalties,
        tam.avg_opp_pentalty_yds AS avg_wkly_def_penalty_yds,
        tam.avg_opp_fmbl_lost AS avg_wkly_def_fmbl_lost,
        tam.avg_opp_int AS avg_wkly_def_int,
        tam.avg_opp_turnovers AS avg_wkly_def_turnovers,
        tam.avg_opp_time_of_possession AS avg_wkly_def_time_of_possession,



        -- Players Team Weekly Kicking Metrics
        tam.avg_fga AS avg_wkly_fga,
        tam.avg_fgm AS avg_wkly_fgm,
        tam.avg_xpa AS avg_wkly_xpa,
        tam.avg_xpm AS avg_wkly_xpm,

        -- Players Team Weekly Punting Metrics
        tam.avg_total_punts AS avg_wkly_total_punts,
        tam.avg_punt_yds AS avg_wkly_punt_yds,

        -- Players Team Weekly First Down Metrics
        tam.avg_pass_fds AS avg_wkly_pass_fds,
        tam.avg_rsh_fds AS avg_wkly_rsh_fds,
        tam.avg_pen_fds AS avg_wkly_pen_fds,
        tam.avg_total_fds AS avg_wkly_total_fds,

        -- Players Team Weekly Conversion Metrics
        tam.avg_thrd_down_conv AS avg_wkly_thrd_down_conv,
        tam.avg_thrd_down_att AS avg_wkly_thrd_down_att,
        tam.avg_fourth_down_conv AS avg_wkly_fourth_down_conv,
        tam.avg_fourth_down_att AS avg_wkly_fourth_down_att,

        -- Players Team Weekly Penalty & Turnover Metrics
        tam.avg_penalties AS avg_wkly_penalties,
        tam.avg_penalty_yds AS avg_wkly_penalty_yds,
        tam.avg_fmbl_lost AS avg_wkly_fmbl_lost,
        tam.avg_int AS avg_wkly_int,
        tam.avg_turnovers AS avg_wkly_turnovers,

        -- Players Team Weekly Time of Possession
        tam.avg_time_of_poss AS avg_wkly_time_of_poss,

        -- Opposing Teams Weekly General Metrics
        otam.avg_points_for AS avg_opp_wkly_points_for,
        otam.avg_points_allowed AS avg_opp_wkly_points_allowed,
        otam.avg_result_margin AS avg_opp_wkly_result_margin,

        -- Opposing Teams Weekly Offensive Metrics
        otam.avg_tot_yds AS avg_opp_wkly_tot_yds,
        otam.avg_pass_yds AS avg_opp_wkly_pass_yds,
        otam.avg_rush_yds AS avg_opp_wkly_rush_yds,
        otam.avg_pass_tds AS avg_opp_wkly_pass_tds,
        otam.avg_pass_cmp AS avg_opp_wkly_pass_cmp,
        otam.avg_pass_att AS avg_opp_wkly_pass_att,
        otam.avg_pass_cmp_pct AS avg_opp_wkly_pass_cmp_pct,
        otam.avg_yds_gained_per_pass_att AS avg_opp_wkly_yds_gained_per_pass_att,
        otam.avg_adj_yds_gained_per_pass_att AS avg_opp_wkly_adj_yds_gained_per_pass_att,
        otam.avg_pass_rate AS avg_opp_wkly_pass_rate,
        otam.avg_sacked AS avg_opp_wkly_sacked,
        otam.avg_sack_yds_lost AS avg_opp_wkly_sack_yds_lost,
        otam.avg_rush_att AS avg_opp_wkly_rush_att,
        otam.avg_rush_tds AS avg_opp_wkly_rush_tds,
        otam.avg_rush_yds_per_att AS avg_opp_wkly_rush_yds_per_att,
        otam.avg_total_off_plays AS avg_opp_wkly_total_off_plays,
        otam.avg_yds_per_play AS avg_opp_wkly_yds_per_play,

        -- Opposing Teams Weekly Defensive Metrics
        otam.avg_opp_tot_yds AS avg_opp_wkly_def_tot_yds,
        otam.avg_opp_pass_yds AS avg_opp_wkly_def_pass_yds,
        otam.avg_opp_rush_yds AS avg_opp_wkly_def_rush_yds,
        otam.avg_opp_pass_tds AS avg_opp_wkly_def_pass_tds,
        otam.avg_opp_pass_cmp AS avg_opp_wkly_def_pass_cmp,
        otam.avg_opp_pass_att AS avg_opp_wkly_def_pass_att,
        otam.avg_opp_pass_cmp_pct AS avg_opp_wkly_def_pass_cmp_pct,
        otam.avg_opp_yds_gained_per_pass_att AS avg_opp_wkly_def_yds_gained_per_pass_att,
        otam.avg_opp_adj_yds_gained_per_pass_att AS avg_opp_wkly_def_adj_yds_gained_per_pass_att,
        otam.avg_opp_pass_rate AS avg_opp_wkly_def_pass_rate,
        otam.avg_opp_sacked AS avg_opp_wkly_def_sacked,
        otam.avg_opp_sack_yds_lost AS avg_opp_wkly_def_sack_yds_lost,
        otam.avg_opp_rush_att AS avg_opp_wkly_def_rush_att,
        otam.avg_opp_rush_tds AS avg_opp_wkly_def_rush_tds,
        otam.avg_opp_rush_yds_per_att AS avg_opp_wkly_def_rush_yds_per_att,
        otam.avg_opp_tot_off_plays AS avg_opp_wkly_def_tot_off_plays,
        otam.avg_opp_yds_per_play AS avg_opp_wkly_def_yds_per_play,
        otam.avg_opp_fga AS avg_opp_wkly_def_fga,
        otam.avg_opp_fgm AS avg_opp_wkly_def_fgm,
        otam.avg_opp_xpa AS avg_opp_wkly_def_xpa,
        otam.avg_opp_xpm AS avg_opp_wkly_def_xpm,
        otam.avg_opp_total_punts AS avg_opp_wkly_def_total_punts,
        otam.avg_opp_punt_yds AS avg_opp_wkly_def_punt_yds,
        otam.avg_opp_pass_fds AS avg_opp_wkly_def_pass_fds,
        otam.avg_opp_rsh_fds AS avg_opp_wkly_def_rsh_fds,
        otam.avg_opp_pen_fds AS avg_opp_wkly_def_pen_fds,
        otam.avg_opp_total_fds AS avg_opp_wkly_def_total_fds,
        otam.avg_opp_thrd_down_conv AS avg_opp_wkly_def_thrd_down_conv,
        otam.avg_opp_thrd_down_att AS avg_opp_wkly_def_thrd_down_att,
        otam.avg_opp_foruth_down_conv AS avg_opp_wkly_def_fourth_down_conv,
        otam.avg_opp_foruth_down_att AS avg_opp_wkly_def_fourth_down_att,
        otam.avg_opp_penalties AS avg_opp_wkly_def_penalties,
        otam.avg_opp_pentalty_yds AS avg_opp_wkly_def_penalty_yds,
        otam.avg_opp_fmbl_lost AS avg_opp_wkly_def_fmbl_lost,
        otam.avg_opp_int AS avg_opp_wkly_def_int,
        otam.avg_opp_turnovers AS avg_opp_wkly_def_turnovers,
        otam.avg_opp_time_of_possession AS avg_opp_wkly_def_time_of_possession,

        -- Opposing Teams Weekly Kicking Metrics
        otam.avg_fga AS avg_opp_wkly_fga,
        otam.avg_fgm AS avg_opp_wkly_fgm,
        otam.avg_xpa AS avg_opp_wkly_xpa,
        otam.avg_xpm AS avg_opp_wkly_xpm,

        -- Opposing Teams Weekly Punting Metrics
        otam.avg_total_punts AS avg_opp_wkly_total_punts,
        otam.avg_punt_yds AS avg_opp_wkly_punt_yds,

        -- Opposing Teams Weekly First Down Metrics
        otam.avg_pass_fds AS avg_opp_wkly_pass_fds,
        otam.avg_rsh_fds AS avg_opp_wkly_rsh_fds,
        otam.avg_pen_fds AS avg_opp_wkly_pen_fds,
        otam.avg_total_fds AS avg_opp_wkly_total_fds,

        -- Opposing Teams Weekly Conversion Metrics
        otam.avg_thrd_down_conv AS avg_opp_wkly_thrd_down_conv,
        otam.avg_thrd_down_att AS avg_opp_wkly_thrd_down_att,
        otam.avg_fourth_down_conv AS avg_opp_wkly_fourth_down_conv,
        otam.avg_fourth_down_att AS avg_opp_wkly_fourth_down_att,

        -- Opposing Teams Weekly Penalty & Turnover Metrics
        otam.avg_penalties AS avg_opp_wkly_penalties,
        otam.avg_penalty_yds AS avg_opp_wkly_penalty_yds,
        otam.avg_fmbl_lost AS avg_opp_wkly_fmbl_lost,
        otam.avg_int AS avg_opp_wkly_int,
        otam.avg_turnovers AS avg_opp_wkly_turnovers,

        -- Opposing Teams Weekly Time of Possession
        otam.avg_time_of_poss AS avg_opp_wkly_time_of_poss,


		 -- Depth Chart Position 
		 pdc.depth_chart_pos AS depth_chart_position,

		 -- Game Date/Year
		 pgl.week,
         pgl.year as season,

		 -- Weekly Rankings Throughout Season
         t_tr.off_rush_rank,
         t_tr.off_pass_rank,
         t_td.def_rush_rank,
         t_td.def_pass_rank,
		 -- team betting odds 
         tbo.game_over_under,
         tbo.spread,

		 -- Player Demographics 
		 pd.age,
		 pd.height,
		 pd.weight,

		 -- Players Team Previous Year General Stats
		 tsgm.fumble_lost as prev_year_team_total_fumbles_lost,
		 tsgm.home_wins as prev_year_team_totals_home_wins,
		 tsgm.home_losses as prev_year_team_total_home_losses,
		 tsgm.away_wins as prev_year_team_total_away_wins,
		 tsgm.away_losses as prev_year_team_total_away_losses,
		 tsgm.wins as prev_year_team_total_wins,
		 tsgm.losses as prev_year_team_total_losses, 
		 tsgm.win_pct as prev_year_team_total_win_pct,
		 tsgm.total_games as prev_year_team_total_games,
		 tsgm.total_yards as prev_year_team_total_yards,
		 tsgm.plays_offense as prev_year_team_total_plays_offense,
		 tsgm.yds_per_play as prev_year_team_yds_per_play,
		 tsgm.turnovers as prev_year_team_total_turnovers,
		 tsgm.first_down as prev_year_team_total_first_downs, 
		 tsgm.penalties as prev_year_team_total_penalties,
         tsgm.penalties_yds as prev_year_team_total_penalties_yds,
         tsgm.pen_fd as prev_year_team_total_pen_fd,
         tsgm.drives as prev_year_team_total_drives,
         tsgm.score_pct as prev_year_team_total_score_pct,
         tsgm.turnover_pct as prev_year_team_total_turnover_pct,
         tsgm.start_avg as prev_year_team_total_start_avg,
         tsgm.time_avg as prev_year_team_total_time_avg,
         tsgm.plays_per_drive as prev_year_team_total_plays_per_drive,
         tsgm.yds_per_drive as prev_year_team_total_yds_per_drive,
         tsgm.points_avg as prev_year_team_total_points_avg,
         tsgm.third_down_att as prev_year_team_total_third_down_att,
         tsgm.third_down_success as prev_year_team_total_third_down_success,
         tsgm.third_down_pct as prev_year_team_total_third_down_pct,
         tsgm.fourth_down_att as prev_year_team_total_fourth_down_att,
         tsgm.fourth_down_success as prev_year_team_total_fourth_down_success,
         tsgm.fourth_down_pct as prev_year_team_total_fourth_down_pct,
         tsgm.red_zone_att as prev_year_team_total_red_zone_att,
         tsgm.red_zone_scores as prev_year_team_total_red_zone_scores,
         tsgm.red_zone_pct as prev_year_team_total_red_zone_pct,

         -- Opposing Team Previous Year General Stats
         opp_tsgm.fumble_lost as prev_year_opp_total_fumbles_lost,
         opp_tsgm.home_wins as prev_year_opp_totals_home_wins,
         opp_tsgm.home_losses as prev_year_opp_total_home_losses,
         opp_tsgm.away_wins as prev_year_opp_total_away_wins,
         opp_tsgm.away_losses as prev_year_opp_total_away_losses,
         opp_tsgm.wins as prev_year_opp_total_wins,
         opp_tsgm.losses as prev_year_opp_total_losses,
         opp_tsgm.win_pct as prev_year_opp_total_win_pct,
         opp_tsgm.total_games as prev_year_opp_total_games,
         opp_tsgm.total_yards as prev_year_opp_total_yards,
         opp_tsgm.plays_offense as prev_year_opp_total_plays_offense,
         opp_tsgm.yds_per_play as prev_year_opp_yds_per_play,
         opp_tsgm.turnovers as prev_year_opp_total_turnovers,
         opp_tsgm.first_down as prev_year_opp_total_first_downs,
         opp_tsgm.penalties as prev_year_opp_total_penalties,
         opp_tsgm.penalties_yds as prev_year_opp_total_penalties_yds,
         opp_tsgm.pen_fd as prev_year_opp_total_pen_fd,
         opp_tsgm.drives as prev_year_opp_total_drives,
         opp_tsgm.score_pct as prev_year_opp_total_score_pct,
         opp_tsgm.turnover_pct as prev_year_opp_total_turnover_pct,
         opp_tsgm.start_avg as prev_year_opp_total_start_avg,
         opp_tsgm.time_avg as prev_year_opp_total_time_avg,
         opp_tsgm.plays_per_drive as prev_year_opp_total_plays_per_drive,
         opp_tsgm.yds_per_drive as prev_year_opp_total_yds_per_drive,
         opp_tsgm.points_avg as prev_year_opp_total_points_avg,
         opp_tsgm.third_down_att as prev_year_opp_total_third_down_att,
         opp_tsgm.third_down_success as prev_year_opp_total_third_down_success,
         opp_tsgm.third_down_pct as prev_year_opp_total_third_down_pct,
         opp_tsgm.fourth_down_att as prev_year_opp_total_fourth_down_att,
         opp_tsgm.fourth_down_success as prev_year_opp_total_fourth_down_success,
         opp_tsgm.fourth_down_pct as prev_year_opp_total_fourth_down_pct,
         opp_tsgm.red_zone_att as prev_year_opp_total_red_zone_att,
         opp_tsgm.red_zone_scores as prev_year_opp_total_red_zone_scores,
         opp_tsgm.red_zone_pct as prev_year_opp_total_red_zone_pct,

		 -- Players Team Previous Year Passing Stats 
		 tspassingmetrics.pass_attempts as prev_year_team_total_pass_attempts,
         tspassingmetrics.complete_pass as prev_year_team_total_complete_pass,
         tspassingmetrics.incomplete_pass as prev_year_team_total_incomplete_pass,
         tspassingmetrics.passing_yards as prev_year_team_total_passing_yards,
         tspassingmetrics.pass_td as prev_year_team_total_pass_td,
         tspassingmetrics.interception as prev_year_team_total_interception,
         tspassingmetrics.net_yds_per_att as prev_year_team_total_net_yds_per_att,
         tspassingmetrics.first_downs as prev_year_team_total_passing_first_downs,
         tspassingmetrics.cmp_pct as prev_year_team_total_cmp_pct,
         tspassingmetrics.td_pct as prev_year_team_total_td_pct,
         tspassingmetrics.int_pct as prev_year_team_total_int_pct,
         tspassingmetrics.success as prev_year_team_total_success,
         tspassingmetrics.long as prev_year_team_total_long,
         tspassingmetrics.yds_per_att as prev_year_team_total_yds_per_att,
         tspassingmetrics.adj_yds_per_att as prev_year_team_total_adj_yds_per_att,
         tspassingmetrics.yds_per_cmp as prev_year_team_total_yds_per_cmp,
         tspassingmetrics.yds_per_g as prev_year_team_total_yds_per_g,
         tspassingmetrics.rating as prev_year_team_total_rating,
         tspassingmetrics.sacked as prev_year_team_total_sacked,
         tspassingmetrics.sacked_yds as prev_year_team_total_sacked_yds,
         tspassingmetrics.sacked_pct as prev_year_team_total_sacked_pct,
         tspassingmetrics.adj_net_yds_per_att as prev_year_team_total_adj_net_yds_per_att,
         tspassingmetrics.comebacks as prev_year_team_total_comebacks,
         tspassingmetrics.game_winning_drives as prev_year_team_total_game_winning_drives,

         -- Opposing Team Previous Year Passing Stats
         opp_tspassingmetrics.pass_attempts as prev_year_opp_total_pass_attempts,
         opp_tspassingmetrics.complete_pass as prev_year_opp_total_complete_pass,
         opp_tspassingmetrics.incomplete_pass as prev_year_opp_total_incomplete_pass,
         opp_tspassingmetrics.passing_yards as prev_year_opp_total_passing_yards,
         opp_tspassingmetrics.pass_td as prev_year_opp_total_pass_td,
         opp_tspassingmetrics.interception as prev_year_opp_total_interception,
         opp_tspassingmetrics.net_yds_per_att as prev_year_opp_total_net_yds_per_att,
         opp_tspassingmetrics.first_downs as prev_year_opp_total_passing_first_downs,
         opp_tspassingmetrics.cmp_pct as prev_year_opp_total_cmp_pct,
         opp_tspassingmetrics.td_pct as prev_year_opp_total_td_pct,
         opp_tspassingmetrics.int_pct as prev_year_opp_total_int_pct,
         opp_tspassingmetrics.success as prev_year_opp_total_success,
         opp_tspassingmetrics.long as prev_year_opp_total_long,
         opp_tspassingmetrics.yds_per_att as prev_year_opp_total_yds_per_att,
         opp_tspassingmetrics.adj_yds_per_att as prev_year_opp_total_adj_yds_per_att,
         opp_tspassingmetrics.yds_per_cmp as prev_year_opp_total_yds_per_cmp,
         opp_tspassingmetrics.yds_per_g as prev_year_opp_total_yds_per_g,
         opp_tspassingmetrics.rating as prev_year_opp_total_rating,
         opp_tspassingmetrics.sacked as prev_year_opp_total_sacked,
         opp_tspassingmetrics.sacked_yds as prev_year_opp_total_sacked_yds,
         opp_tspassingmetrics.sacked_pct as prev_year_opp_total_sacked_pct,
         opp_tspassingmetrics.adj_net_yds_per_att as prev_year_opp_total_adj_net_yds_per_att,
         opp_tspassingmetrics.comebacks as prev_year_opp_total_comebacks,
         opp_tspassingmetrics.game_winning_drives as prev_year_opp_total_game_winning_drives,

		 -- Players Team Previous Year Rushing/Receiving Stats
		 tsrm.rush_att as prev_year_team_total_rush_att,
         tsrm.rush_yds_per_att as prev_year_team_total_rush_yds_per_att,
         tsrm.rush_fd as prev_year_team_total_rush_fd,
         tsrm.rush_success as prev_year_team_total_rush_success,
         tsrm.rush_long as prev_year_team_total_rush_long,
         tsrm.rush_yds_per_g as prev_year_team_total_rush_yds_per_g,
         tsrm.rush_att_per_g as prev_year_team_total_rush_att_per_g,
         tsrm.rush_yds as prev_year_team_total_rush_yds,
         tsrm.rush_tds as prev_year_team_total_rush_tds,
         tsrm.targets as prev_year_team_total_targets,
         tsrm.rec as prev_year_team_total_rec,
         tsrm.rec_yds as prev_year_team_total_rec_yds,
         tsrm.rec_yds_per_rec as prev_year_team_total_rec_yds_per_rec,
         tsrm.rec_td as prev_year_team_total_rec_td,
         tsrm.rec_first_down as prev_year_team_total_rec_first_down,
         tsrm.rec_success as prev_year_team_total_rec_success,
         tsrm.rec_long as prev_year_team_total_rec_long,
         tsrm.rec_per_g as prev_year_team_total_rec_per_g,
         tsrm.rec_yds_per_g as prev_year_team_total_rec_yds_per_g,
         tsrm.catch_pct as prev_year_team_total_catch_pct,
         tsrm.rec_yds_per_tgt as prev_year_team_total_rec_yds_per_tgt,
         tsrm.touches as prev_year_team_total_touches,
         tsrm.yds_per_touch as prev_year_team_total_yds_per_touch,
         tsrm.yds_from_scrimmage as prev_year_team_total_yds_from_scrimmage,
         tsrm.rush_receive_td as prev_year_team_total_rush_receive_td,
         tsrm.fumbles as prev_year_team_total_fumbles,

         -- Opposing Team Previous Year Rushing/Receiving Stats
         opp_tsrm.rush_att as prev_year_opp_total_rush_att,
         opp_tsrm.rush_yds_per_att as prev_year_opp_total_rush_yds_per_att,
         opp_tsrm.rush_fd as prev_year_opp_total_rush_fd,
         opp_tsrm.rush_success as prev_year_opp_total_rush_success,
         opp_tsrm.rush_long as prev_year_opp_total_rush_long,
         opp_tsrm.rush_yds_per_g as prev_year_opp_total_rush_yds_per_g,
         opp_tsrm.rush_att_per_g as prev_year_opp_total_rush_att_per_g,
         opp_tsrm.rush_yds as prev_year_opp_total_rush_yds,
         opp_tsrm.rush_tds as prev_year_opp_total_rush_tds,
         opp_tsrm.targets as prev_year_opp_total_targets,
         opp_tsrm.rec as prev_year_opp_total_rec,
         opp_tsrm.rec_yds as prev_year_opp_total_rec_yds,
         opp_tsrm.rec_yds_per_rec as prev_year_opp_total_rec_yds_per_rec,
         opp_tsrm.rec_td as prev_year_opp_total_rec_td,
         opp_tsrm.rec_first_down as prev_year_opp_total_rec_first_down,
         opp_tsrm.rec_success as prev_year_opp_total_rec_success,
         opp_tsrm.rec_long as prev_year_opp_total_rec_long,
         opp_tsrm.rec_per_g as prev_year_opp_total_rec_per_g,
         opp_tsrm.rec_yds_per_g as prev_year_opp_total_rec_yds_per_g,
         opp_tsrm.catch_pct as prev_year_opp_total_catch_pct,
         opp_tsrm.rec_yds_per_tgt as prev_year_opp_total_rec_yds_per_tgt,
         opp_tsrm.touches as prev_year_opp_total_touches,
         opp_tsrm.yds_per_touch as prev_year_opp_total_yds_per_touch,
         opp_tsrm.yds_from_scrimmage as prev_year_opp_total_yds_from_scrimmage,
         opp_tsrm.rush_receive_td as prev_year_opp_total_rush_receive_td,
         opp_tsrm.fumbles as prev_year_opp_total_fumbles,

		 -- Players Team Previous Year Kicking Stats 
		 tskm.team_total_fg_long as prev_year_team_total_fg_long,
         tskm.team_total_fg_pct as prev_year_team_total_fg_pct,
         tskm.team_total_xpa as prev_year_team_total_xpa,
         tskm.team_total_xpm as prev_year_team_total_xpm,
         tskm.team_total_xp_pct as prev_year_team_total_xp_pct,
         tskm.team_total_kickoff_yds as prev_year_team_total_kickoff_yds,
         tskm.team_total_kickoff_tb_pct as prev_year_team_total_kickoff_tb_pct,

         -- Opposing Team Previous Year Kicking Stats
         opp_tskm.team_total_fg_long as prev_year_opp_total_fg_long,
         opp_tskm.team_total_fg_pct as prev_year_opp_total_fg_pct,
         opp_tskm.team_total_xpa as prev_year_opp_total_xpa,
         opp_tskm.team_total_xpm as prev_year_opp_total_xpm,
         opp_tskm.team_total_xp_pct as prev_year_opp_total_xp_pct,
         opp_tskm.team_total_kickoff_yds as prev_year_opp_total_kickoff_yds,
         opp_tskm.team_total_kickoff_tb_pct as prev_year_opp_total_kickoff_tb_pct,

		 -- Players Team Previous Year Punting Stats 
		 tspuntingmetrics.team_total_punt as prev_year_team_total_punt,
         tspuntingmetrics.team_total_punt_yds as prev_year_team_total_punt_yds,
         tspuntingmetrics.team_total_punt_yds_per_punt as prev_year_team_total_punt_yds_per_punt,
         tspuntingmetrics.team_total_punt_ret_yds_opp as prev_year_team_total_punt_ret_yds_opp,
         tspuntingmetrics.team_total_punt_net_yds as prev_year_team_total_punt_net_yds,
         tspuntingmetrics.team_total_punt_net_yds_per_punt as prev_year_team_total_punt_net_yds_per_punt,
         tspuntingmetrics.team_total_punt_long as prev_year_team_total_punt_long,
         tspuntingmetrics.team_total_punt_tb as prev_year_team_total_punt_tb,
         tspuntingmetrics.team_total_punt_tb_pct as prev_year_team_total_punt_tb_pct,
         tspuntingmetrics.team_total_punt_in_20 as prev_year_team_total_punt_in_20,
         tspuntingmetrics.team_total_punt_in_20_pct as prev_year_team_total_punt_in_20_pct,

         -- Opposing Team Previous Year Punting Stats
         opp_tspuntingmetrics.team_total_punt as prev_year_opp_total_punt,
         opp_tspuntingmetrics.team_total_punt_yds as prev_year_opp_total_punt_yds,
         opp_tspuntingmetrics.team_total_punt_yds_per_punt as prev_year_opp_total_punt_yds_per_punt,
         opp_tspuntingmetrics.team_total_punt_ret_yds_opp as prev_year_opp_total_punt_ret_yds_opp,
         opp_tspuntingmetrics.team_total_punt_net_yds as prev_year_opp_total_punt_net_yds,
         opp_tspuntingmetrics.team_total_punt_net_yds_per_punt as prev_year_opp_total_punt_net_yds_per_punt,
         opp_tspuntingmetrics.team_total_punt_long as prev_year_opp_total_punt_long,
         opp_tspuntingmetrics.team_total_punt_tb as prev_year_opp_total_punt_tb,
         opp_tspuntingmetrics.team_total_punt_tb_pct as prev_year_opp_total_punt_tb_pct,
         opp_tspuntingmetrics.team_total_punt_in_20 as prev_year_opp_total_punt_in_20,
         opp_tspuntingmetrics.team_total_punt_in_20_pct as prev_year_opp_total_punt_in_20_pct,

		 -- Players Team Previous Year Scoring Stats 
		 tssm.rush_td as prev_year_rush_td,
         tssm.rec_td as prev_year_rec_td,
         tssm.punt_ret_td as prev_year_punt_ret_td,
         tssm.kick_ret_td as prev_year_kick_ret_td,
         tssm.fumbles_rec_td as prev_year_fumbles_rec_td,
         tssm.def_int_td as prev_year_def_int_td,
         tssm.other_td as prev_year_other_td,
         tssm.total_td as prev_year_total_td,
         tssm.two_pt_md as prev_year_two_pt_md,
         tssm.def_two_pt as prev_year_def_two_pt,
         tssm.xpm as prev_year_xpm,
         tssm.xpa as prev_year_xpa,
         tssm.fgm as prev_year_fgm,
         tssm.fga as prev_year_fga,
         tssm.safety_md as prev_year_safety_md,
         tssm.scoring as prev_year_scoring,

         -- Opposing Team Previous Year Scoring Stats
         opp_tssm.rush_td as prev_year_opp_rush_td,
         opp_tssm.rec_td as prev_year_opp_rec_td,
         opp_tssm.punt_ret_td as prev_year_opp_punt_ret_td,
         opp_tssm.kick_ret_td as prev_year_opp_kick_ret_td,
         opp_tssm.fumbles_rec_td as prev_year_opp_fumbles_rec_td,
         opp_tssm.other_td as prev_year_opp_other_td,
         opp_tssm.total_td as prev_year_opp_total_td,
         opp_tssm.two_pt_md as prev_year_opp_two_pt_md,
         opp_tssm.def_two_pt as prev_year_opp_def_two_pt,
         opp_tssm.xpm as prev_year_opp_xpm,
         opp_tssm.xpa as prev_year_opp_xpa,
         opp_tssm.fgm as prev_year_opp_fgm,
         opp_tssm.fga as prev_year_opp_fga,
         opp_tssm.safety_md as prev_year_opp_safety_md,
         opp_tssm.scoring as prev_year_opp_scoring,

	     -- Players Team Previous Year Seasonal Offensive Rankings 
		 tsr.off_points as prev_year_off_points,
         tsr.off_total_yards as prev_year_off_total_yards,
         tsr.off_turnovers as prev_year_off_turnovers,
         tsr.off_fumbles_lost as prev_year_off_fumbles_lost,
         tsr.off_first_down as prev_year_off_first_down,
         tsr.off_pass_att as prev_year_off_pass_att,
         tsr.off_pass_yds as prev_year_off_pass_yds,
         tsr.off_pass_td as prev_year_off_pass_td,
         tsr.off_pass_int as prev_year_off_pass_int,
         tsr.off_pass_net_yds_per_att as prev_year_off_pass_net_yds_per_att,
         tsr.off_rush_att as prev_year_off_rush_att,
         tsr.off_rush_yds as prev_year_off_rush_yds,
         tsr.off_rush_td as prev_year_off_rush_td,
         tsr.off_rush_yds_per_att as prev_year_off_rush_yds_per_att,
         tsr.off_score_pct as prev_year_off_score_pct,
         tsr.off_turnover_pct as prev_year_off_turnover_pct,
         tsr.off_start_avg as prev_year_off_start_avg,
         tsr.off_time_avg as prev_year_off_time_avg,
         tsr.off_plays_per_drive as prev_year_off_plays_per_drive,
         tsr.off_yds_per_drive as prev_year_off_yds_per_drive,
         tsr.off_points_avg as prev_year_off_points_avg,
         tsr.off_third_down_pct as prev_year_off_third_down_pct,
         tsr.off_fourth_down_pct as prev_year_off_fourth_down_pct,
         tsr.off_red_zone_pct as prev_year_off_red_zone_pct,

         -- Opposing Team Previous Year Seasonal Offensive Rankings 
		 opp_tsr.off_points as prev_year_opp_off_points,
         opp_tsr.off_total_yards as prev_year_opp_off_total_yards,
         opp_tsr.off_turnovers as prev_year_opp_off_turnovers,
         opp_tsr.off_fumbles_lost as prev_year_opp_off_fumbles_lost,
         opp_tsr.off_first_down as prev_year_opp_off_first_down,
         opp_tsr.off_pass_att as prev_year_opp_off_pass_att,
         opp_tsr.off_pass_yds as prev_year_opp_off_pass_yds,
         opp_tsr.off_pass_td as prev_year_opp_off_pass_td,
         opp_tsr.off_pass_int as prev_year_opp_off_pass_int,
         opp_tsr.off_pass_net_yds_per_att as prev_year_opp_off_pass_net_yds_per_att,
         opp_tsr.off_rush_att as prev_year_opp_off_rush_att,
         opp_tsr.off_rush_yds as prev_year_opp_off_rush_yds,
         opp_tsr.off_rush_td as prev_year_opp_off_rush_td,
         opp_tsr.off_rush_yds_per_att as prev_year_opp_off_rush_yds_per_att,
         opp_tsr.off_score_pct as prev_year_opp_off_score_pct,
         opp_tsr.off_turnover_pct as prev_year_opp_off_turnover_pct,
         opp_tsr.off_start_avg as prev_year_opp_off_start_avg,
         opp_tsr.off_time_avg as prev_year_opp_off_time_avg,
         opp_tsr.off_plays_per_drive as prev_year_opp_off_plays_per_drive,
         opp_tsr.off_yds_per_drive as prev_year_opp_off_yds_per_drive,
         opp_tsr.off_points_avg as prev_year_opp_off_points_avg,
         opp_tsr.off_third_down_pct as prev_year_opp_off_third_down_pct,
         opp_tsr.off_fourth_down_pct as prev_year_opp_off_fourth_down_pct,
         opp_tsr.off_red_zone_pct as prev_year_opp_off_red_zone_pct,

         -- Players Team Previous Year Seasonal Defensive Rankings
         tsr.def_points as prev_year_team_def_points,
         tsr.def_total_yards as prev_year_team_def_total_yards,
         tsr.def_turnovers as prev_year_team_def_turnovers,
         tsr.def_fumbles_lost as prev_year_team_def_fumbles_lost,
         tsr.def_first_down as prev_year_team_def_first_down,
         tsr.def_pass_att as prev_year_team_def_pass_att,
         tsr.def_pass_yds as prev_year_team_def_pass_yds,
         tsr.def_pass_td as prev_year_team_def_pass_td,
         tsr.def_pass_int as prev_year_team_def_pass_int,
         tsr.def_pass_net_yds_per_att as prev_year_team_def_pass_net_yds_per_att,
         tsr.def_rush_att as prev_year_team_def_rush_att,
         tsr.def_rush_yds as prev_year_team_def_rush_yds,
         tsr.def_rush_td as prev_year_team_def_rush_td,
         tsr.def_rush_yds_per_att as prev_year_team_def_rush_yds_per_att,
         tsr.def_score_pct as prev_year_team_def_score_pct,
         tsr.def_turnover_pct as prev_year_team_def_turnover_pct,
         tsr.def_start_avg as prev_year_team_def_start_avg,
         tsr.def_time_avg as prev_year_team_def_time_avg,
         tsr.def_plays_per_drive as prev_year_team_def_plays_per_drive,
         tsr.def_yds_per_drive as prev_year_team_def_yds_per_drive,
         tsr.def_points_avg as prev_year_team_def_points_avg,
         tsr.def_third_down_pct as prev_year_team_def_third_down_pct,
         tsr.def_fourth_down_pct as prev_year_team_def_fourth_down_pct,

		 -- Players Team Previous Year Seasonal Defensive Rankings
         opp_tsr.def_points as prev_year_def_points,
         opp_tsr.def_total_yards as prev_year_def_total_yards,
         opp_tsr.def_turnovers as prev_year_def_turnovers,
         opp_tsr.def_fumbles_lost as prev_year_def_fumbles_lost,
         opp_tsr.def_first_down as prev_year_def_first_down,
         opp_tsr.def_pass_att as prev_year_def_pass_att,
         opp_tsr.def_pass_yds as prev_year_def_pass_yds,
         opp_tsr.def_pass_td as prev_year_def_pass_td,
         opp_tsr.def_pass_int as prev_year_def_pass_int,
         opp_tsr.def_pass_net_yds_per_att as prev_year_def_pass_net_yds_per_att,
         opp_tsr.def_rush_att as prev_year_def_rush_att,
         opp_tsr.def_rush_yds as prev_year_def_rush_yds,
         opp_tsr.def_rush_td as prev_year_def_rush_td,
         opp_tsr.def_rush_yds_per_att as prev_year_def_rush_yds_per_att,
         opp_tsr.def_score_pct as prev_year_def_score_pct,
         opp_tsr.def_turnover_pct as prev_year_def_turnover_pct,
         opp_tsr.def_start_avg as prev_year_def_start_avg,
         opp_tsr.def_time_avg as prev_year_def_time_avg,
         opp_tsr.def_plays_per_drive as prev_year_def_plays_per_drive,
         opp_tsr.def_yds_per_drive as prev_year_def_yds_per_drive,
         opp_tsr.def_points_avg as prev_year_def_points_avg,
         opp_tsr.def_third_down_pct as prev_year_def_third_down_pct,
         opp_tsr.def_fourth_down_pct as prev_year_def_fourth_down_pct,
         opp_tsr.def_red_zone_pct as prev_year_def_red_zone_pct,

		 -- Opposing Teams Previous Year Defensive Metrics
		 opp_tsdm.points as prev_year_opp_def_points,
         opp_tsdm.total_yards as prev_year_opp_def_total_yards,
         opp_tsdm.plays_offense as prev_year_opp_def_plays_offense,
         opp_tsdm.yds_per_play_offense as prev_year_opp_def_yds_per_play_offense,
         opp_tsdm.turnovers as prev_year_opp_def_turnovers,
         opp_tsdm.fumbles_lost as prev_year_opp_def_fumbles_lost,
         opp_tsdm.first_down as prev_year_opp_def_first_down,
         opp_tsdm.pass_cmp as prev_year_opp_def_pass_cmp,
         opp_tsdm.pass_att as prev_year_opp_def_pass_att,
         opp_tsdm.pass_yds as prev_year_opp_def_pass_yds,
         opp_tsdm.pass_td as prev_year_opp_def_pass_td,
         opp_tsdm.pass_int as prev_year_opp_def_pass_int,
         opp_tsdm.pass_net_yds_per_att as prev_year_opp_def_pass_net_yds_per_att,
         opp_tsdm.pass_fd as prev_year_opp_def_pass_fd,
         opp_tsdm.rush_att as prev_year_opp_def_rush_att,
         opp_tsdm.rush_yds as prev_year_opp_def_rush_yds,
         opp_tsdm.rush_td as prev_year_opp_def_rush_td,
         opp_tsdm.rush_yds_per_att as prev_year_opp_def_rush_yds_per_att,
         opp_tsdm.rush_fd as prev_year_opp_def_rush_fd,
         opp_tsdm.penalties as prev_year_opp_def_penalties,
         opp_tsdm.penalties_yds as prev_year_opp_def_penalties_yds,
         opp_tsdm.pen_fd as prev_year_opp_def_pen_fd,
         opp_tsdm.drives as prev_year_opp_def_drives,
         opp_tsdm.score_pct as prev_year_opp_def_score_pct,
         opp_tsdm.turnover_pct as prev_year_opp_def_turnover_pct,
         opp_tsdm.start_avg as prev_year_opp_def_start_avg,
         opp_tsdm.time_avg as prev_year_opp_def_time_avg,
         opp_tsdm.plays_per_drive as prev_year_opp_def_plays_per_drive,
         opp_tsdm.yds_per_drive as prev_year_opp_def_yds_per_drive,
         opp_tsdm.points_avg as prev_year_opp_def_points_avg,
         opp_tsdm.third_down_att as prev_year_opp_def_third_down_att,
         opp_tsdm.third_down_success as prev_year_opp_def_third_down_success,
         opp_tsdm.third_down_pct as prev_year_opp_def_third_down_pct,
         opp_tsdm.fourth_down_att as prev_year_opp_def_fourth_down_att,
         opp_tsdm.fourth_down_success as prev_year_opp_def_fourth_down_success,
         opp_tsdm.fourth_down_pct as prev_year_opp_def_fourth_down_pct,
         opp_tsdm.red_zone_att as prev_year_opp_def_red_zone_att,
         opp_tsdm.red_zone_scores as prev_year_opp_def_red_zone_scores,
         opp_tsdm.red_zone_pct as prev_year_opp_def_red_zone_pct,
         opp_tsdm.def_int as prev_year_opp_def_int,
         opp_tsdm.def_int_yds as prev_year_opp_def_int_yds,
         opp_tsdm.def_int_td as prev_year_opp_def_int_td,
         opp_tsdm.def_int_long as prev_year_opp_def_int_long,
         opp_tsdm.pass_defended as prev_year_opp_def_pass_defended,
         opp_tsdm.fumbles_forced as prev_year_opp_def_fumbles_forced,
         opp_tsdm.fumbles_rec as prev_year_opp_def_fumbles_rec,
         opp_tsdm.fumbles_rec_yds as prev_year_opp_def_fumbles_rec_yds,
         opp_tsdm.fumbles_rec_td as prev_year_opp_def_fumbles_rec_td,
         opp_tsdm.sacks as prev_year_opp_def_sacks,
         opp_tsdm.tackles_combined as prev_year_opp_def_tackles_combined,
         opp_tsdm.tackles_solo as prev_year_opp_def_tackles_solo,
         opp_tsdm.tackles_assists as prev_year_opp_def_tackles_assists,
         opp_tsdm.tackles_loss as prev_year_opp_def_tackles_loss,
         opp_tsdm.qb_hits as prev_year_opp_def_qb_hits,
         opp_tsdm.safety_md as prev_year_opp_def_safety_md,

         -- Players Team Previous Year Defensive Metrics
         tsdm.points as prev_year_team_def_points_metrics,
         tsdm.total_yards as prev_year_team_def_total_yards_metrics,
         tsdm.plays_offense as prev_year_team_def_plays_offense,
         tsdm.yds_per_play_offense as prev_year_team_def_yds_per_play_offense,
         tsdm.turnovers as prev_year_team_def_turnovers_metrics,
         tsdm.fumbles_lost as prev_year_team_def_fumbles_lost_metrics,
         tsdm.first_down as prev_year_team_def_first_down_metrics,
         tsdm.pass_cmp as prev_year_team_def_pass_cmp,
         tsdm.pass_att as prev_year_team_def_pass_att_metrics,
         tsdm.pass_yds as prev_year_team_def_pass_yds_metrics,
         tsdm.pass_td as prev_year_team_def_pass_td_metrics,
         tsdm.pass_int as prev_year_team_def_pass_int_metrics,
         tsdm.pass_net_yds_per_att as prev_year_team_def_pass_net_yds_per_att_metrics,
         tsdm.pass_fd as prev_year_team_def_pass_fd,
         tsdm.rush_att as prev_year_team_def_rush_att_metrics,
         tsdm.rush_yds as prev_year_team_def_rush_yds_metrics,
         tsdm.rush_td as prev_year_team_def_rush_td_metrics,
         tsdm.rush_yds_per_att as prev_year_team_def_rush_yds_per_att_metrics,
         tsdm.rush_fd as prev_year_team_def_rush_fd,
         tsdm.penalties as prev_year_team_def_penalties,
         tsdm.penalties_yds as prev_year_team_def_penalties_yds,
         tsdm.pen_fd as prev_year_team_def_pen_fd,
         tsdm.drives as prev_year_team_def_drives,
         tsdm.score_pct as prev_year_team_def_score_pct_metrics,
         tsdm.turnover_pct as prev_year_team_def_turnover_pct_metrics,
         tsdm.start_avg as prev_year_team_def_start_avg_metrics,
         tsdm.time_avg as prev_year_team_def_time_avg_metrics,
         tsdm.plays_per_drive as prev_year_team_def_plays_per_drive_metrics,
         tsdm.yds_per_drive as prev_year_team_def_yds_per_drive_metrics,
         tsdm.points_avg as prev_year_team_def_points_avg_metrics,
         tsdm.third_down_att as prev_year_team_def_third_down_att,
         tsdm.third_down_success as prev_year_team_def_third_down_success,
         tsdm.third_down_pct as prev_year_team_def_third_down_pct_metrics,
         tsdm.fourth_down_att as prev_year_team_def_fourth_down_att,
         tsdm.fourth_down_success as prev_year_team_def_fourth_down_success,
         tsdm.fourth_down_pct as prev_year_team_def_fourth_down_pct_metrics,
         tsdm.red_zone_att as prev_year_team_def_red_zone_att,
         tsdm.red_zone_scores as prev_year_team_def_red_zone_scores,
         tsdm.red_zone_pct as prev_year_team_def_red_zone_pct_metrics,
         tsdm.def_int as prev_year_team_def_int,
         tsdm.def_int_yds as prev_year_team_def_int_yds,
         tsdm.def_int_td as prev_year_team_def_int_td,
         tsdm.def_int_long as prev_year_team_def_int_long,
         tsdm.pass_defended as prev_year_team_def_pass_defended,
         tsdm.fumbles_forced as prev_year_team_def_fumbles_forced,
         tsdm.fumbles_rec as prev_year_team_def_fumbles_rec,
         tsdm.fumbles_rec_yds as prev_year_team_def_fumbles_rec_yds,
         tsdm.fumbles_rec_td as prev_year_team_def_fumbles_rec_td,
         tsdm.sacks as prev_year_team_def_sacks,
         tsdm.tackles_combined as prev_year_team_def_tackles_combined,
         tsdm.tackles_solo as prev_year_team_def_tackles_solo,
         tsdm.tackles_assists as prev_year_team_def_tackles_assists,
         tsdm.tackles_loss as prev_year_team_def_tackles_loss,
         tsdm.qb_hits as prev_year_team_def_qb_hits,
         tsdm.safety_md as prev_year_team_def_safety_md,

         -- Player Passing Stats From Previous Year
        pssm.games_started AS prev_year_player_passing_games_started,
        pssm.pass_att AS prev_year_player_pass_att,
        pssm.pass_cmp_pct AS prev_year_player_pass_cmp_pct,
        pssm.pass_yds AS prev_year_player_pass_yds,
        pssm.pass_td AS prev_year_player_pass_td,
        pssm.pass_td_pct AS prev_year_player_pass_td_pct,
        pssm.pass_int AS prev_year_player_pass_int,
        pssm.pass_int_pct AS prev_year_player_pass_int_pct,
        pssm.pass_first_down AS prev_year_player_pass_first_down,
        pssm.pass_success AS prev_year_player_pass_success,
        pssm.pass_long AS prev_year_player_pass_long,
        pssm.pass_yds_per_att AS prev_year_player_pass_yds_per_att,
        pssm.pass_adj_yds_per_att AS prev_year_player_pass_adj_yds_per_att,
        pssm.pass_yds_per_cmp AS prev_year_player_pass_yds_per_cmp,
        pssm.pass_yds_per_g AS prev_year_player_pass_yds_per_g,
        pssm.pass_rating AS prev_year_player_pass_rating,
        pssm.qbr AS prev_year_player_qbr,
        pssm.pass_sacked AS prev_year_player_pass_sacked,
        pssm.pass_sacked_yds AS prev_year_player_pass_sacked_yds,
        pssm.pass_sacked_pct AS prev_year_player_pass_sacked_pct,
        pssm.pass_net_yds_per_att AS prev_year_player_pass_net_yds_per_att,
        pssm.pass_adj_net_yds_per_att AS prev_year_player_pass_adj_net_yds_per_att,
        pssm.comebacks AS prev_year_player_comebacks,
        pssm.game_winning_drives AS prev_year_player_game_winning_drives,

        -- Player Rushing & Receiving Stats From Previous Year 
        psrrm.games_started AS prev_year_player_rushing_receiving_games_started,
        psrrm.rush_att AS prev_year_player_rush_att,
        psrrm.rush_yds_per_att AS prev_year_player_rush_yds_per_att,
        psrrm.rush_fd AS prev_year_player_rush_fd,
        psrrm.rush_success AS prev_year_player_rush_success,
        psrrm.rush_long AS prev_year_player_rush_long,
        psrrm.rush_yds_per_g AS prev_year_player_rush_yds_per_g,
        psrrm.rush_att_per_g AS prev_year_player_rush_att_per_g,
        psrrm.rush_yds AS prev_year_player_rush_yds,
        psrrm.rush_tds AS prev_year_player_rush_tds,
        psrrm.targets AS prev_year_player_targets,
        psrrm.rec AS prev_year_player_rec,
        psrrm.rec_yds AS prev_year_player_rec_yds,
        psrrm.rec_yds_per_rec AS prev_year_player_rec_yds_per_rec,
        psrrm.rec_td AS prev_year_player_rec_td,
        psrrm.rec_first_down AS prev_year_player_rec_first_down,
        psrrm.rec_success AS prev_year_player_rec_success,
        psrrm.rec_long AS prev_year_player_rec_long,
        psrrm.rec_per_g AS prev_year_player_rec_per_g,
        psrrm.rec_yds_per_g AS prev_year_player_rec_yds_per_g,
        psrrm.catch_pct AS prev_year_player_catch_pct,
        psrrm.rec_yds_per_tgt AS prev_year_player_rec_yds_per_tgt,
        psrrm.touches AS prev_year_player_touches,
        psrrm.yds_per_touch AS prev_year_player_yds_per_touch,
        psrrm.yds_from_scrimmage AS prev_year_player_yds_from_scrimmage,
        psrrm.rush_receive_td AS prev_year_player_rush_receive_td,
        psrrm.fumbles AS prev_year_player_fumbles,

        -- Player Scoring Metrics From Previous Year 
        player_seasonal_sm.rush_td AS prev_year_player_scoring_rush_td,
        player_seasonal_sm.rec_td AS prev_year_player_scoring_rec_td,
        player_seasonal_sm.punt_ret_td AS prev_year_player_scoring_punt_ret_td,
        player_seasonal_sm.kick_ret_td AS prev_year_player_scoring_kick_ret_td,
        player_seasonal_sm.fumbles_rec_td AS prev_year_player_scoring_fumbles_rec_td,
        player_seasonal_sm.other_td AS prev_year_player_scoring_other_td,
        player_seasonal_sm.total_td AS prev_year_player_scoring_total_td,
        player_seasonal_sm.two_pt_md AS prev_year_player_scoring_two_pt_md,
        player_seasonal_sm.scoring AS prev_year_player_total_scoring,
        CASE
           WHEN tbo.favorite_team_id = t.team_id THEN 1
		 ELSE 0
           END AS is_favorited,
		 pp.props
      FROM
         player_game_log pgl -- player game logs (week to week games)
      JOIN
         player_depth_chart pdc ON pdc.week = pgl.week AND pgl.year = pdc.season AND pgl.player_id = pdc.player_id -- player depth chart position 
      JOIN 
         player p ON p.player_id = pgl.player_id -- player information 
      JOIN 
         player_teams pt ON p.player_id = pt.player_id AND pgl.week >= pt.strt_wk AND pgl.week <= pt.end_wk AND pt.season = pgl.year -- players 
      JOIN 
         team t ON pt.team_id = t.team_id -- team the player is on
      JOIN 
         team td ON pgl.opp = td.team_id -- team the player is playing against 
      JOIN 
	  	 team_game_log tgl ON tgl.team_id = t.team_id AND tgl.week = pgl.week AND tgl.year = pgl.year
	  JOIN 
	  	 game_conditions gc ON gc.season = pgl.year AND gc.week = pgl.week AND (t.team_id = gc.home_team_id OR t.team_id = gc.visit_team_id)
      LEFT JOIN 
         player_weekly_agg_metrics pam ON pgl.week - 1 = pam.week AND pgl.year = pam.season AND pgl.player_id = pam.player_id -- player weekly aggregate metrics  
      LEFT JOIN 
         team_weekly_agg_metrics tam ON tgl.week - 1 = tam.week AND tgl.year = tam.season AND tgl.team_id = tam.team_id -- players team weekly agg metrics
      LEFT JOIN 
         team_weekly_agg_metrics otam ON tgl.week - 1 = otam.week AND tgl.year = otam.season AND tgl.opp = otam.team_id -- opposing teams weekly agg metrics
      LEFT JOIN
         player_injuries pi ON p.player_id = pi.player_id AND pi.week = pgl.week AND pi.season = pgl.year
	  LEFT JOIN 
         player_demographics pd ON p.player_id = pd.player_id AND pgl.year = pd.season -- demographic metrics for player  	 
      LEFT JOIN 
         player_seasonal_passing_metrics pssm ON p.player_id = pssm.player_id AND pssm.season = pgl.year AND pssm.team_id = t.team_id -- player seasonal passing metrics for previous year 
      LEFT JOIN 
         player_seasonal_rushing_receiving_metrics psrrm ON p.player_id = psrrm.player_id AND psrrm.season = pgl.year AND psrrm.team_id = t.team_id -- player seasonal rushing / receiving metrics for previous year 
      LEFT JOIN 
         player_seasonal_scoring_metrics player_seasonal_sm ON p.player_id = player_seasonal_sm.player_id AND player_seasonal_sm.season = pgl.year AND player_seasonal_sm.team_id = t.team_id -- player seasonal scoring metrics for previous year 
      LEFT JOIN
         team_seasonal_general_metrics tsgm ON t.team_id = tsgm.team_id AND (pgl.year - 1) = tsgm.season -- team general metrics for previous year
      LEFT JOIN
         team_seasonal_general_metrics opp_tsgm ON td.team_id = opp_tsgm.team_id AND (pgl.year - 1) = opp_tsgm.season -- opposing team general metrics for previous year
      LEFT JOIN
         team_seasonal_rushing_receiving_metrics tsrm ON t.team_id = tsrm.team_id AND (pgl.year - 1) = tsrm.season -- team rushing/receiving metrics for previous year
      LEFT JOIN
         team_seasonal_rushing_receiving_metrics opp_tsrm ON td.team_id = opp_tsrm.team_id AND (pgl.year - 1) = opp_tsrm.season -- opposing team rushing/receiving metrics for previous year
      LEFT JOIN 
         team_seasonal_passing_metrics tspassingmetrics ON t.team_id = tspassingmetrics.team_id AND (pgl.year - 1) = tspassingmetrics.season -- team passing metrics for previous year 
      LEFT JOIN 
         team_seasonal_passing_metrics opp_tspassingmetrics ON td.team_id = opp_tspassingmetrics.team_id AND (pgl.year - 1) = opp_tspassingmetrics.season -- opposing team passing metrics for previous year
      LEFT JOIN 
         team_seasonal_kicking_metrics tskm ON t.team_id = tskm.team_id AND (pgl.year - 1) = tskm.season -- team kicking metrics for previous year 
      LEFT JOIN 
         team_seasonal_kicking_metrics opp_tskm ON td.team_id = opp_tskm.team_id AND (pgl.year - 1) = opp_tskm.season -- opposing team kicking metrics for previous year
      LEFT JOIN 
         team_seasonal_punting_metrics tspuntingmetrics ON t.team_id = tspuntingmetrics.team_id AND (pgl.year - 1) = tspuntingmetrics.season -- team punting metrics for previous year 
      LEFT JOIN 
         team_seasonal_punting_metrics opp_tspuntingmetrics ON td.team_id = opp_tspuntingmetrics.team_id AND (pgl.year - 1) = opp_tspuntingmetrics.season -- opposing team punting metrics for previous year
      LEFT JOIN 
         team_seasonal_scoring_metrics tssm ON t.team_id = tssm.team_id AND (pgl.year - 1) = tssm.season -- team scoring metrics for previous years 
      LEFT JOIN 
         team_seasonal_scoring_metrics opp_tssm ON td.team_id = opp_tssm.team_id AND (pgl.year - 1) = opp_tssm.season -- opposing team scoring metrics for previous years
      LEFT JOIN 
         team_seasonal_defensive_metrics tsdm ON t.team_id = tsdm.team_id AND (pgl.year - 1) = tsdm.season -- team defensive metrics for previous year
      LEFT JOIN
         team_seasonal_defensive_metrics opp_tsdm ON td.team_id = opp_tsdm.team_id AND (pgl.year - 1) = opp_tsdm.season -- opposing team defensive metrics for previous year
      LEFT JOIN 
         team_seasonal_ranks tsr ON t.team_id = tsr.team_id AND (pgl.year - 1) = tsr.season -- team seasonal ranks for previous year 
      LEFT JOIN
         team_seasonal_ranks opp_tsr ON td.team_id = opp_tsr.team_id AND (pgl.year - 1) = opp_tsr.season -- opposing team seasonal ranks for previous year
      JOIN 
         team_ranks t_tr ON t.team_id = t_tr.team_id AND pgl.week - 1 = t_tr.week AND pgl.year = t_tr.season -- players team weekly rankings heading into matchup
      JOIN
         team_ranks t_td ON td.team_id = t_td.team_id AND pgl.week - 1 = t_td.week AND pgl.year = t_td.season -- opposing team weekly rankings heading into matchup
      JOIN
         PlayerProps pp ON p.name = pp.player_name AND pgl.week = pp.week AND pgl.year = pp.season -- player betting lines 
      JOIN 
         team_betting_odds tbo -- team betting lines 
      ON (
        (pgl.home_team = TRUE AND tbo.home_team_id = t.team_id AND tbo.away_team_id = td.team_id AND pgl.week = tbo.week AND pgl.year = tbo.season) 
            OR 
        (pgl.home_team = FALSE AND tbo.away_team_id = t.team_id AND tbo.home_team_id = td.team_id AND pgl.week = tbo.week AND pgl.year = tbo.season)
      ) 
   """
