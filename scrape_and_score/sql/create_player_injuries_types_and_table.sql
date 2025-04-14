-- create new enum types for our player_injuries tbale 
CREATE TYPE practice_status AS ENUM('dnp', 'full', 'limited')
CREATE TYPE game_status AS ENUM('doubtful', 'questionable', 'out')

-- create new player injuries table 
CREATE TABLE player_injuries (
	player_id INT NOT NULL,
	week INT NOT NULL,
	season INT NOT NULL, 
	injury_loc VARCHAR(20),
	wed_prac_sts practice_status,
	thurs_prac_sts practice_status,
	fri_prac_sts practice_status,
	off_sts game_status,
	PRIMARY KEY (week, season, player_id), 
	FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
)