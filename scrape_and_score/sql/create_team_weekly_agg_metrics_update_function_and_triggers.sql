CREATE OR REPLACE FUNCTION upsert_team_weekly_agg_metrics(t_team_id INT, t_week INT, t_season INT)
RETURNS VOID AS $$
BEGIN 
	INSERT INTO team_weekly_agg_metrics (
		team_id, 
		week, 
		season, 

		-- General Performance Metrics
    	avg_points_for,
    	avg_points_allowed,
    	avg_result_margin,

		-- Offensive Metrics 
		avg_tot_yds,
    	avg_pass_yds,
    	avg_rush_yds,
    	avg_pass_tds,
    	avg_pass_cmp,
    	avg_pass_att,
    	avg_pass_cmp_pct,
    	avg_yds_gained_per_pass_att,
    	avg_adj_yds_gained_per_pass_att,
    	avg_pass_rate,
    	avg_sacked,
    	avg_sack_yds_lost,
    	avg_rush_att,
    	avg_rush_tds,
    	avg_rush_yds_per_att,
    	avg_total_off_plays,
    	avg_yds_per_play,

		-- Defensive Opponent Metrics
    	avg_opp_tot_yds,
    	avg_opp_pass_yds,
    	avg_opp_rush_yds,

    	-- Kicking Metrics
    	avg_fga,
    	avg_fgm,
    	avg_xpa,
    	avg_xpm,

    	-- Punting Metrics
    	avg_total_punts,
    	avg_punt_yds,

    	-- First Down Metrics
    	avg_pass_fds,
    	avg_rsh_fds,
    	avg_pen_fds,
    	avg_total_fds,

    	-- Conversion Metrics
    	avg_thrd_down_conv,
    	avg_thrd_down_att,
    	avg_fourth_down_conv,
    	avg_fourth_down_att,

    	-- Penalty & Turnover Metrics
    	avg_penalties,
    	avg_penalty_yds,
    	avg_fmbl_lost,
    	avg_int,
    	avg_turnovers,

    	-- Time of Possession
    	avg_time_of_poss
	)
	SELECT 
    t_team_id, 
    t_week, 
    t_season, 

    -- General Performance Metrics 
    COALESCE(ROUND(AVG(tgl.points_for)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.points_allowed)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.points_for - tgl.points_allowed)::numeric, 2), -1),

    -- Offensive Metrics 
    COALESCE(ROUND(AVG(tgl.tot_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pass_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.rush_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pass_tds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pass_cmp)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pass_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pass_cmp_pct)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.yds_gained_per_pass_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.adj_yds_gained_per_pass_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pass_rate)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.sacked)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.sack_yds_lost)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.rush_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.rush_tds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.rush_yds_per_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.total_off_plays)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.yds_per_play)::numeric, 2), -1),

    -- Defensive Opponent Metrics
    COALESCE(ROUND(AVG(tgl.opp_tot_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.opp_pass_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.opp_rush_yds)::numeric, 2), -1),

    -- Kicking Metrics
    COALESCE(ROUND(AVG(tgl.fga)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.fgm)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.xpa)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.xpm)::numeric, 2), -1),

    -- Punting Metrics
    COALESCE(ROUND(AVG(tgl.total_punts)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.punt_yds)::numeric, 2), -1),

    -- First Down Metrics
    COALESCE(ROUND(AVG(tgl.pass_fds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.rsh_fds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.pen_fds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.total_fds)::numeric, 2), -1),

    -- Conversion Metrics
    COALESCE(ROUND(AVG(tgl.thrd_down_conv)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.thrd_down_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.fourth_down_conv)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.fourth_down_att)::numeric, 2), -1),

    -- Penalty & Turnover Metrics
    COALESCE(ROUND(AVG(tgl.penalties)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.penalty_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.fmbl_lost)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.int)::numeric, 2), -1),
    COALESCE(ROUND(AVG(tgl.turnovers)::numeric, 2), -1),

    -- Time of Possession
    COALESCE(ROUND(AVG(tgl.time_of_poss)::numeric, 2), -1)
	
	FROM 
    	team_game_log tgl 
	WHERE 
    	tgl.team_id = t_team_id AND tgl.week <= t_week AND tgl.year = t_season
 
	ON CONFLICT (team_id, week, season) DO UPDATE SET
    avg_points_for = COALESCE(EXCLUDED.avg_points_for, team_weekly_agg_metrics.avg_points_for, -1),
    avg_points_allowed = COALESCE(EXCLUDED.avg_points_allowed, team_weekly_agg_metrics.avg_points_allowed, -1),
    avg_result_margin = COALESCE(EXCLUDED.avg_result_margin, team_weekly_agg_metrics.avg_result_margin, -1),
    avg_tot_yds = COALESCE(EXCLUDED.avg_tot_yds, team_weekly_agg_metrics.avg_tot_yds, -1),
    avg_yds_per_play = COALESCE(EXCLUDED.avg_yds_per_play, team_weekly_agg_metrics.avg_yds_per_play, -1),
    avg_opp_tot_yds = COALESCE(EXCLUDED.avg_opp_tot_yds, team_weekly_agg_metrics.avg_opp_tot_yds, -1),
    avg_turnovers = COALESCE(EXCLUDED.avg_turnovers, team_weekly_agg_metrics.avg_turnovers, -1),
    avg_thrd_down_conv = COALESCE(EXCLUDED.avg_thrd_down_conv, team_weekly_agg_metrics.avg_thrd_down_conv, -1),
    avg_thrd_down_att = COALESCE(EXCLUDED.avg_thrd_down_att, team_weekly_agg_metrics.avg_thrd_down_att, -1),
    avg_penalties = COALESCE(EXCLUDED.avg_penalties, team_weekly_agg_metrics.avg_penalties, -1),
    avg_penalty_yds = COALESCE(EXCLUDED.avg_penalty_yds, team_weekly_agg_metrics.avg_penalty_yds, -1),
    avg_time_of_poss = COALESCE(EXCLUDED.avg_time_of_poss, team_weekly_agg_metrics.avg_time_of_poss, -1),
    avg_sacked = COALESCE(EXCLUDED.avg_sacked, team_weekly_agg_metrics.avg_sacked, -1),
    avg_total_off_plays = COALESCE(EXCLUDED.avg_total_off_plays, team_weekly_agg_metrics.avg_total_off_plays, -1),
    avg_rush_yds_per_att = COALESCE(EXCLUDED.avg_rush_yds_per_att, team_weekly_agg_metrics.avg_rush_yds_per_att, -1),
    avg_yds_gained_per_pass_att = COALESCE(EXCLUDED.avg_yds_gained_per_pass_att, team_weekly_agg_metrics.avg_yds_gained_per_pass_att, -1),
    avg_pass_cmp_pct = COALESCE(EXCLUDED.avg_pass_cmp_pct, team_weekly_agg_metrics.avg_pass_cmp_pct, -1),
    avg_rush_att = COALESCE(EXCLUDED.avg_rush_att, team_weekly_agg_metrics.avg_rush_att, -1),
    avg_pass_rate = COALESCE(EXCLUDED.avg_pass_rate, team_weekly_agg_metrics.avg_pass_rate, -1),
    avg_total_fds = COALESCE(EXCLUDED.avg_total_fds, team_weekly_agg_metrics.avg_total_fds, -1),
    avg_pass_fds = COALESCE(EXCLUDED.avg_pass_fds, team_weekly_agg_metrics.avg_pass_fds, -1),
    avg_rsh_fds = COALESCE(EXCLUDED.avg_rsh_fds, team_weekly_agg_metrics.avg_rsh_fds, -1),
    avg_pen_fds = COALESCE(EXCLUDED.avg_pen_fds, team_weekly_agg_metrics.avg_pen_fds, -1),
    avg_fmbl_lost = COALESCE(EXCLUDED.avg_fmbl_lost, team_weekly_agg_metrics.avg_fmbl_lost, -1),
    avg_int = COALESCE(EXCLUDED.avg_int, team_weekly_agg_metrics.avg_int, -1),
    avg_punt_yds = COALESCE(EXCLUDED.avg_punt_yds, team_weekly_agg_metrics.avg_punt_yds, -1),
    avg_total_punts = COALESCE(EXCLUDED.avg_total_punts, team_weekly_agg_metrics.avg_total_punts, -1),
    avg_fga = COALESCE(EXCLUDED.avg_fga, team_weekly_agg_metrics.avg_fga, -1),
    avg_fgm = COALESCE(EXCLUDED.avg_fgm, team_weekly_agg_metrics.avg_fgm, -1),
    avg_xpa = COALESCE(EXCLUDED.avg_xpa, team_weekly_agg_metrics.avg_xpa, -1),
    avg_xpm = COALESCE(EXCLUDED.avg_xpm, team_weekly_agg_metrics.avg_xpm, -1),
    avg_pass_cmp = COALESCE(EXCLUDED.avg_pass_cmp, team_weekly_agg_metrics.avg_pass_cmp, -1),
    avg_pass_att = COALESCE(EXCLUDED.avg_pass_att, team_weekly_agg_metrics.avg_pass_att, -1),
    avg_pass_tds = COALESCE(EXCLUDED.avg_pass_tds, team_weekly_agg_metrics.avg_pass_tds, -1),
    avg_rush_tds = COALESCE(EXCLUDED.avg_rush_tds, team_weekly_agg_metrics.avg_rush_tds, -1),
    avg_adj_yds_gained_per_pass_att = COALESCE(EXCLUDED.avg_adj_yds_gained_per_pass_att, team_weekly_agg_metrics.avg_adj_yds_gained_per_pass_att, -1),
    avg_sack_yds_lost = COALESCE(EXCLUDED.avg_sack_yds_lost, team_weekly_agg_metrics.avg_sack_yds_lost, -1),
    avg_opp_pass_yds = COALESCE(EXCLUDED.avg_opp_pass_yds, team_weekly_agg_metrics.avg_opp_pass_yds, -1),
    avg_opp_rush_yds = COALESCE(EXCLUDED.avg_opp_rush_yds, team_weekly_agg_metrics.avg_opp_rush_yds, -1),
    avg_fourth_down_conv = COALESCE(EXCLUDED.avg_fourth_down_conv, team_weekly_agg_metrics.avg_fourth_down_conv, -1),
    avg_fourth_down_att = COALESCE(EXCLUDED.avg_fourth_down_att, team_weekly_agg_metrics.avg_fourth_down_att, -1);

END;
$$ LANGUAGE plpgsql;


-- Create Triggers  & Trigger Functions 
CREATE OR REPLACE FUNCTION trigger_upsert_for_team_game_log() 
RETURNS TRIGGER AS $$
BEGIN 
	PERFORM upsert_team_weekly_agg_metrics(NEW.team_id, NEW.week, NEW.year);
	RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE TRIGGER team_game_log_trigger
AFTER INSERT OR UPDATE ON team_game_log 
FOR EACH ROW 
EXECUTE FUNCTION trigger_upsert_for_team_game_log(); 