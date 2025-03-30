-- Store relevant player dpeth chart positions for a given week and season
CREATE TABLE player_depth_chart (
	player_id INT NOT NULL,
	week INT NOT NULL,
	season INT NOT NULL, 
	depth_chart_pos INT,
	PRIMARY KEY (week, season, player_id), 
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
)