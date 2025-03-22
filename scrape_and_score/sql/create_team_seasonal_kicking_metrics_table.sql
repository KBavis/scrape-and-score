CREATE TABLE team_seasonal_kicking_metrics (
    team_id INT,
    season INT, 
    team_total_fga1 INT,  -- Field goal attempts from 0-19 yards
    team_total_fgm1 INT,  -- Field goals made from 0-19 yards
    team_total_fga2 INT,  -- Field goal attempts from 20-29 yards
    team_total_fgm2 INT,  -- Field goals made from 20-29 yards
    team_total_fga3 INT,  -- Field goal attempts from 30-39 yards
    team_total_fgm3 INT,  -- Field goals made from 30-39 yards
    team_total_fga4 INT,  -- Field goal attempts from 40-49 yards
    team_total_fgm4 INT,  -- Field goals made from 40-49 yards
    team_total_fga5 INT,  -- Field goal attempts from 50+ yards
    team_total_fgm5 INT,  -- Field goals made from 50+ yards
    team_total_fga INT,  -- Total field goal attempts across all distances
    team_total_fgm INT,  -- Total field goals made across all distances
    team_total_fg_long INT,  -- Longest successful field goal (in yards)
    team_total_fg_pct FLOAT,  -- Field goal percentage (field goals made / attempts)
    team_total_xpa INT,  -- Extra point attempts
    team_total_xpm INT,  -- Extra points made
    team_total_xp_pct FLOAT,  -- Extra point percentage (extra points made / attempts)
    team_total_kickoff INT,  -- Number of kickoffs made
    team_total_kickoff_yds INT,  -- Total kickoff yards
    team_total_kickoff_tb INT,  -- Number of touchbacks on kickoffs
    team_total_kickoff_tb_pct FLOAT,  -- Touchback percentage (touchbacks / kickoffs)
    team_total_kickoff_yds_avg FLOAT,  -- Average yards per kickoff 
    PRIMARY KEY (team_id, season),  -- Composite primary key based on team_id and season
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE  -- Foreign key constraint to the team table
);