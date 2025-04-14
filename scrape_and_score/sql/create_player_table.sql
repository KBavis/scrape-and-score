-- Store general player information
CREATE TABLE player (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(50) NOT NULL,
    normalized_name VARCHAR(100), 
    hashed_name VARCHAR(100), 
    pfr_available INT -- column used to determine if pro-football-reference HREF was found for player or not
);