CREATE TABLE player_advanced_rushing_receiving (
    player_id INT NOT NULL,
    week INT NOT NULL,
    season INT NOT NULL,
    age FLOAT NOT NULL,
    rush_first_downs INT,
    rush_yds_before_contact FLOAT,
    rush_yds_before_contact_per_att FLOAT,
    rush_yds_afer_contact FLOAT,
    rush_yds_after_contact_per_att FLOAT,
    rush_brkn_tackles INT,
    rush_att_per_brkn_tackle FLOAT,
    rec_first_downs INT,
    yds_before_catch FLOAT,
    yds_before_catch_per_rec FLOAT,
    yds_after_catch FLOAT,
    yds_after_catch_per_rec FLOAT,
    avg_depth_of_tgt FLOAT,
    rec_brkn_tackles INT,
    rec_per_brkn_tackle FLOAT,
    dropped_passes INT,
    drop_pct FLOAT,
    int_when_tgted INT,
    qbr_when_tgted FLOAT,
    PRIMARY KEY (player_id, week, season),
    FOREIGN KEY (player_id) REFERENCES player(player_id) ON DELETE CASCADE
);
