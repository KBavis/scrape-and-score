-- Store relevant rushing metrics for a team in a given season
CREATE TABLE team_seasonal_rushing_metrics(
	team_id INT NOT NULL, 
	season INT NOT NULL,
	rush_snaps_count INT,
	rush_snaps_pct INT,
	qb_scramble INT, 
	rushing_yards INT,
	rush_td INT,
	rush_fumble INT,
	rush_fumble_lost INT,
	PRIMARY KEY (team_id, season),
	FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
)