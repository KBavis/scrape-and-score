CREATE TABLE team_seasonal_punting_metrics (
    team_id INT,  -- Unique identifier for the team
    season INT,  -- Season identifier (e.g., year or season number)
    
    team_total_punt INT,  -- Total punts made
    team_total_punt_yds INT,  -- Total punt yards
    team_total_punt_yds_per_punt FLOAT,  -- Average punt yards per punt
    
    team_total_punt_ret_yds_opp INT,  -- Total punt return yards allowed by the team
    team_total_punt_net_yds INT,  -- Net punt yards (total punt yards minus return yards)
    team_total_punt_net_yds_per_punt FLOAT,  -- Average net yards per punt
    
    team_total_punt_long INT,  -- Longest punt (in yards)
    team_total_punt_tb INT,  -- Number of touchbacks on punts
    team_total_punt_tb_pct FLOAT,  -- Touchback percentage (touchbacks / total punts)
    
    team_total_punt_in_20 INT,  -- Number of punts inside the opponent's 20-yard line
    team_total_punt_in_20_pct FLOAT,  -- Percentage of punts inside the opponent's 20-yard line (in_20 / total punts)
    
    team_total_punt_blocked INT,  -- Number of punts blocked
    
    PRIMARY KEY (team_id, season),  -- Composite primary key based on team_id and season
    FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE  -- Foreign key constraint to the team table
);