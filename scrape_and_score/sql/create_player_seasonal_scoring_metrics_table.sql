CREATE TABLE player_seasonal_scoring_metrics (
    player_id INT,
    season INT,
    
    team_total_rush_td INT,
    team_total_rec_td INT,
    team_total_punt_ret_td INT,
    team_total_kick_ret_td INT,
    team_total_fumbles_rec_td INT,
    team_total_def_int_td INT,
    team_total_other_td INT,
    team_total_total_td INT,
    team_total_two_pt_md INT,
    team_total_def_two_pt INT,
    team_total_xpm INT,
    team_total_xpa INT,
    team_total_fgm INT,
    team_total_fga INT,
    team_total_safety_md INT,
    team_total_scoring INT,

    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
);
