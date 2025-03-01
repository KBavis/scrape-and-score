CREATE TABLE player_teams (
    player_id INT NOT NULL,
    team_id INT NOT NULL,
	season INT NOT NULL, 
	week INT NOT NULL,
	PRIMARY KEY (player_id, team_id, season, week),
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE,
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
);