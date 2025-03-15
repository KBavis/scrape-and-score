--- Store relevant player demographic information
-- TODO: Account for things like hand size, 40 yard dash, total income, etc
CREATE TABLE player_demographics (
	player_id INT NOT NULL, 
	season INT NOT NULL, 
	age INT, 
	height FLOAT,
	weight FLOAT,
	PRIMARY KEY (player_id, season),
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
)