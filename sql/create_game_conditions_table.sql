CREATE TABLE game_conditions (
    season INT NOT NULL,
    week INT NOT NULL,
    home_team_id INT NOT NULL,
    visit_team_id INT NOT NULL,

    game_date TIMESTAMP,
    game_time INT,
    kickoff VARCHAR(50),
    month VARCHAR(20),
    start VARCHAR(10),

    surface VARCHAR(20),
    weather_icon VARCHAR(50),
    temperature DECIMAL(5,2),
    precip_probability VARCHAR(10),
    precip_type VARCHAR(20),
    wind_speed DECIMAL(5,2),
    wind_bearing INT,

    PRIMARY KEY (season, week, home_team_id, visit_team_id),
    FOREIGN KEY (home_team_id) REFERENCES team(team_id),
    FOREIGN KEY (visit_team_id) REFERENCES team(team_id)
);
