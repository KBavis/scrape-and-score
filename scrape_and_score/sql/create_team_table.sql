CREATE TABLE team (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    off_rush_rank INT,
    off_pass_rank INT,
    def_rush_rank INT,
    def_pass_rank INT
);