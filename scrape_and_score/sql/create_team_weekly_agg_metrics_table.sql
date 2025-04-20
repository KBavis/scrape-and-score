-- Store weekly aggregate metrics for a team at a specific point in a season
CREATE TABLE team_weekly_agg_metrics (
    team_id INT NOT NULL,
    week INT NOT NULL,
    season INT NOT NULL,

    -- General Performance Metrics
    avg_points_for FLOAT,
    avg_points_allowed FLOAT,
    avg_result_margin FLOAT,

    -- Offensive Metrics
    avg_tot_yds FLOAT,
    avg_pass_yds FLOAT,
    avg_rush_yds FLOAT,
    avg_pass_tds FLOAT,
    avg_pass_cmp FLOAT,
    avg_pass_att FLOAT,
    avg_pass_cmp_pct FLOAT,
    avg_yds_gained_per_pass_att FLOAT,
    avg_adj_yds_gained_per_pass_att FLOAT,
    avg_pass_rate FLOAT,
    avg_sacked FLOAT,
    avg_sack_yds_lost FLOAT,
    avg_rush_att FLOAT,
    avg_rush_tds FLOAT,
    avg_rush_yds_per_att FLOAT,
    avg_total_off_plays FLOAT,
    avg_yds_per_play FLOAT,

    -- Defensive Metrics (i.e avergaes of opponents metrics corresponding to team 'team_id')
    avg_opp_tot_yds FLOAT,
    avg_opp_pass_yds FLOAT,
    avg_opp_rush_yds FLOAT,
    avg_opp_pass_tds FLOAT,
    avg_opp_pass_cmp FLOAT,
    avg_opp_pass_att FLOAT,
    avg_opp_pass_cmp_pct FLOAT, 
    avg_opp_yds_gained_per_pass_att FLOAT,
    avg_opp_adj_yds_gained_per_pass_att FLOAT,
    avg_opp_pass_rate FLOAT,
    avg_opp_sacked FLOAT, 
    avg_opp_sack_yds_lost FLOAT,
    avg_opp_rush_att FLOAT, 
    avg_opp_rush_tds FLOAT, 
    avg_opp_rush_yds_per_att FLOAT,
    avg_opp_tot_off_plays FLOAT, 
    avg_opp_yds_per_play FLOAT,
    avg_opp_fga FLOAT,
    avg_opp_fgm FLOAT, 
    avg_opp_xpa FLOAT,
    avg_opp_xpm FLOAT,
    avg_opp_total_punts FLOAT,
    avg_opp_punt_yds FLOAT,
    avg_opp_pass_fds FLOAT,
    avg_opp_rsh_fds FLOAT,
    avg_opp_pen_fds FLOAT,
    avg_opp_total_fds FLOAT,
    avg_opp_thrd_down_conv FLOAT,
    avg_opp_thrd_down_att FLOAT,
    avg_opp_foruth_down_conv FLOAT,
    avg_opp_foruth_down_att FLOAT,
    avg_opp_penalties FLOAT,
    avg_opp_pentalty_yds FLOAT,
    avg_opp_fmbl_lost FLOAT,
    avg_opp_int FLOAT,
    avg_opp_turnovers FLOAT,
    avg_opp_time_of_possession FLOAT,


    -- Kicking Metrics
    avg_fga FLOAT,
    avg_fgm FLOAT,
    avg_xpa FLOAT,
    avg_xpm FLOAT,

    -- Punting Metrics
    avg_total_punts FLOAT,
    avg_punt_yds FLOAT,

    -- First Down Metrics
    avg_pass_fds FLOAT,
    avg_rsh_fds FLOAT,
    avg_pen_fds FLOAT,
    avg_total_fds FLOAT,

    -- Conversion Metrics
    avg_thrd_down_conv FLOAT,
    avg_thrd_down_att FLOAT,
    avg_fourth_down_conv FLOAT,
    avg_fourth_down_att FLOAT,

    -- Penalty & Turnover Metrics
    avg_penalties FLOAT,
    avg_penalty_yds FLOAT,
    avg_fmbl_lost FLOAT,
    avg_int FLOAT,
    avg_turnovers FLOAT,

    -- Time of Possession
    avg_time_of_poss FLOAT,

    PRIMARY KEY (team_id, week, season),
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE
);
