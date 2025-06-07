-- Store general information about a given NFL team
CREATE TABLE team (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);