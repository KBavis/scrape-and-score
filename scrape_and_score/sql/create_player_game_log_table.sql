CREATE TABLE player_game_log (
    player_game_log_id SERIAL PRIMARY KEY,
    player_id INT NOT NULL,
    week INT NOT NULL,
    day VARCHAR(20) NOT NULL,
	year INT NOT NULL,
    home_team BOOLEAN,
    opp INT NOT NULL, -- opponent team_id
    result VARCHAR(20),
    points_for INT,
    points_allowed INT,
    completions INT,
    attempts INT,
    pass_yd INT,
    pass_td INT,
    interceptions INT,
    rating INT,
    sacked INT,
    rush_att INT,
    rush_yds INT,
    rush_tds INT,
    tgt INT,
    rec INT,
    rec_yd INT,
    rec_td INT,
    snap_pct INT,
    FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE,
    FOREIGN KEY (opp) REFERENCES team(team_id) ON DELETE CASCADE
);