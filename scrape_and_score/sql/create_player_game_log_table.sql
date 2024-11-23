CREATE TABLE player_game_log (
    player_game_log_id SERIAL PRIMARY KEY,
    player_id INT NOT NULL,
    week INT NOT NULL,
    day VARCHAR(20) NOT NULL,
	year INT NOT NULL,
    rest_days INT,
    home_team BOOLEAN,
    distance_traveled FLOAT,
    opp INT NOT NULL, -- opponent team_id
    result VARCHAR(20),
    points_for INT,
    points_allowed INT,
    tot_yds INT,
    pass_yds INT,
    rush_yds INT,
    opp_tot_yds INT,
    opp_pass_yds INT,
    opp_rush_yds INT,
    FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE,
    FOREIGN KEY (opp) REFERENCES team(team_id) ON DELETE CASCADE
);