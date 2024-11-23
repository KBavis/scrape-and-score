CREATE TABLE player (
    player_id SERIAL PRIMARY KEY,
    team_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(50) NOT NULL,
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
);