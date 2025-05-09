CREATE TABLE team_seasonal_general_metrics (
    team_id INT,
    season INT,
    fumble_lost INT,
    home_wins INT,
    home_losses INT,
    away_wins INT,
    away_losses INT,
    wins INT,
    losses INT,
    win_pct FLOAT,
    total_games INT,
    total_yards INT,
    plays_offense INT,
    yds_per_play FLOAT,
    turnovers INT,
    first_down INT,
    penalties INT,
    penalties_yds INT,
    pen_fd INT,
    drives INT,
    score_pct FLOAT,
    turnover_pct FLOAT,
    start_avg FLOAT,
    time_avg FLOAT, -- number of seconds per drive 
    plays_per_drive FLOAT,
    yds_per_drive FLOAT,
    points_avg FLOAT,
    third_down_att INT,
    third_down_success INT,
    third_down_pct FLOAT,
    fourth_down_att INT,
    fourth_down_success INT,
    fourth_down_pct FLOAT,
    red_zone_att INT,
    red_zone_scores INT,
    red_zone_pct FLOAT,
    PRIMARY KEY (team_id, season),
	FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
);
