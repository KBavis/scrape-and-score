CREATE TABLE player_betting_odds (
	label VARCHAR(100) NOT NULL,
	cost INT NOT NULL, 
	line FLOAT NOT NULL,
	week INT NOT NULL, 
	season INT NOT NULL,
	player_id INT NOT NULL, 
	player_name VARCHAR(100) NOT NULL,
	PRIMARY KEY (week, season, label), 
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
)