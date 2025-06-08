-- Store relevant rankings that are updated weekly for a specific season
CREATE TABLE team_ranks (
	team_id INT NOT NULL, 
	week INT NOT NULL, 
	season INT NOT NULL, 
	off_rush_rank INT,
    off_pass_rank INT,
    def_rush_rank INT,
    def_pass_rank INT,
	PRIMARY KEY (team_id, week, season),
	FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
)
