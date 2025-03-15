-- Store relevant passing metrics for a team in a given season
CREATE TABLE team_seasonal_passing_metrics(
	team_id INT NOT NULL, 
	season INT NOT NULL,
	qb_dropback INT,
	pass_snap_count INT, 
	pass_snap_pct float, 
	pass_attempts INT, 
	complete_pass INT, 
	incomplete_pass INT,
	air_yards INT,
	passing_yards INT, 
	pass_td INT, 
	interception INT, 
	targets INT, 
	receptions INT,
	yards_after_catch INT,
	receiving_td INT, 
	pass_fumble INT,
	pass_fumble_lost INT,
	PRIMARY KEY (team_id, season),
	FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
)