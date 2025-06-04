-- Store relevant player props for a player for a particular week/season
--TODO (FFM-315): Add outcome columns and calculate whether lines are hitting
CREATE TABLE player_betting_odds (
	player_id INT NOT NULL, 
	player_name VARCHAR(100) NOT NULL,
	label VARCHAR(100) NOT NULL,
	cost INT NOT NULL, 
	line FLOAT NOT NULL,
	week INT NOT NULL, 
	season INT NOT NULL,
	PRIMARY KEY (week, season, label, player_id), 
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
)