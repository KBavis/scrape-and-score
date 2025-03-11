CREATE TABLE player_aggregate_metrics (
	player_id INT NOT NULL, 
	week INT NOT NULL, 
	season INT NOT NULL, 
	fantasy_points FLOAT, 
	PRIMARY KEY (player_id, week, season),
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE 
)