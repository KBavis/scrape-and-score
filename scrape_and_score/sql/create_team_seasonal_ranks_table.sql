CREATE TABLE team_seasonal_ranks (
    team_id INT,
    season INT,
    def_points INT,
    def_total_yards INT,
    def_turnovers INT,
    def_fumbles_lost INT,
    def_first_down INT,
    def_pass_att INT,
    def_pass_yds INT,
    def_pass_td INT,
    def_pass_int INT,
    def_pass_net_yds_per_att FLOAT,
    def_rush_att INT,
    def_rush_yds INT,
    def_rush_td INT,
    def_rush_yds_per_att FLOAT,
    def_score_pct FLOAT,
    def_turnover_pct FLOAT,
    def_start_avg FLOAT,
    def_time_avg FLOAT,
    def_plays_per_drive FLOAT,
    def_yds_per_drive FLOAT,
    def_points_avg FLOAT,
    def_third_down_pct FLOAT,
    def_fourth_down_pct FLOAT,
    def_red_zone_pct FLOAT,
    off_player INT,
    off_points INT,
    off_total_yards INT,
    off_turnovers INT,
    off_fumbles_lost INT,
    off_first_down INT,
    off_pass_att INT,
    off_pass_yds INT,
    off_pass_td INT,
    off_pass_int INT,
    off_pass_net_yds_per_att FLOAT,
    off_rush_att INT,
    off_rush_yds INT,
    off_rush_td INT,
    off_rush_yds_per_att FLOAT,
    off_score_pct FLOAT,
    off_turnover_pct FLOAT,
    off_start_avg FLOAT,
    off_time_avg FLOAT,
    off_plays_per_drive FLOAT,
    off_yds_per_drive FLOAT,
    off_points_avg FLOAT,
    off_third_down_pct FLOAT,
    off_fourth_down_pct FLOAT,
    off_red_zone_pct FLOAT,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
);
