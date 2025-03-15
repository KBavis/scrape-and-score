-- Linkage table for a player and their corresponding team during a season
CREATE TABLE player_teams (
    player_id INT NOT NULL,
    team_id INT NOT NULL,
	season INT NOT NULL, 
	strt_wk INT NOT NULL, -- account for potential trades
	end_wk INT NOT NULL, -- account for potential trades
	PRIMARY KEY (player_id, team_id, season, strt_wk, end_wk),
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE,
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
);