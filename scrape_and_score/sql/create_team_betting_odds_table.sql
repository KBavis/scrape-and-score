CREATE TABLE team_betting_odds (
	home_team_id INT NOT NULL,
	away_team_id INT NOT NULL, 
	home_team_score INT NOT NULL,
	away_team_score INT NOT NULL,
	week INT NOT NULL, 
	year INT NOT NULL,
	game_over_under FLOAT NOT NULL, 
	favorite_team_id INT NOT NULL,
	---outcome data---
	spread INT NOT NULL, 
	total_points INT, 
	over_hit INT,
	under_hit INT, 
	favorite_covered INT, 
	underdog_covered INT,
	PRIMARY KEY (week, year, home_team_id, away_team_id), 
	FOREIGN KEY (home_team_id) REFERENCES team(team_id) ON DELETE CASCADE,
	FOREIGN KEY (away_team_id) REFERENCES team(team_id) ON DELETE CASCADE,
	FOREIGN KEY (favorite_team_id) REFERENCES team(team_id) ON DELETE CASCADE
)