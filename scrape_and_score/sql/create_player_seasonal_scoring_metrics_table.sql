CREATE TABLE player_seasonal_scoring_metrics (
    player_id INT,
    team_id INT,
    season INT,
    
    rush_td INT,
    rec_td INT,
    punt_ret_td INT,
    kick_ret_td INT,
    fumbles_rec_td INT,
    def_int_td INT,
    other_td INT,
    total_td INT,
    two_pt_md INT,
    def_two_pt INT,
    xpm INT,
    xpa INT,
    fgm INT,
    fga INT,
    safety_md INT,
    scoring INT,

    PRIMARY KEY (player_id, season, team_id),
    FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
);
