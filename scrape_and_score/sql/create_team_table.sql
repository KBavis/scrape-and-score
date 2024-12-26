CREATE TABLE team (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    offense_rank INT,
    defense_rank INT
);