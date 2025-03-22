CREATE TABLE player_seasonal_rushing_receiving_metrics (
    player_id INT,
    season INT,
    games_started INT,
    
    rush_att INT,
    rush_yds INT,
    rush_td INT,
    rush_first_down INT,
    rush_success INT,
    rush_long INT,
    rush_yds_per_att FLOAT,
    rush_yds_per_g FLOAT,
    rush_att_per_g FLOAT,

    targets INT,
    rec INT,
    rec_yds INT,
    rec_yds_per_rec FLOAT,
    rec_td INT,
    rec_first_down INT,
    rec_success INT,
    rec_long INT,
    rec_per_g FLOAT,
    rec_yds_per_g FLOAT,
    catch_pct FLOAT,
    rec_yds_per_tgt FLOAT,

    touches INT,
    yds_per_touch FLOAT,
    yds_from_scrimmage FLOAT,
    rush_receive_td INT,
    fumbles INT,
    
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
);
