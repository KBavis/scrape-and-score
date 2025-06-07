CREATE OR REPLACE FUNCTION upsert_team_weekly_def_agg_metrics(t_team_id INT, t_wsek INT, t_season INT)
RETURNS VOID AS $$
BEGIN 
	INSERT INTO team_weekly_agg_metrics (
		team_id,
		week, 
		season,
		avg_opp_pass_tds,
    	avg_opp_pass_cmp,
    	avg_opp_pass_att,
    	avg_opp_pass_cmp_pct,
    	avg_opp_yds_gained_per_pass_att,
    	avg_opp_adj_yds_gained_per_pass_att,
    	avg_opp_pass_rate,
    	avg_opp_sacked,
    	avg_opp_sack_yds_lost,
    	avg_opp_rush_att,
    	avg_opp_rush_tds,
    	avg_opp_rush_yds_per_att,
    	avg_opp_tot_off_plays,
    	avg_opp_yds_per_play,
    	avg_opp_fga,
    	avg_opp_fgm,
    	avg_opp_xpa,
    	avg_opp_xpm,
    	avg_opp_total_punts,
    	avg_opp_punt_yds,
    	avg_opp_pass_fds,
    	avg_opp_rsh_fds,
    	avg_opp_pen_fds,
    	avg_opp_total_fds,
    	avg_opp_thrd_down_conv,
    	avg_opp_thrd_down_att,
    	avg_opp_foruth_down_conv,
    	avg_opp_foruth_down_att,
    	avg_opp_penalties,
    	avg_opp_pentalty_yds,
    	avg_opp_fmbl_lost,
    	avg_opp_int,
    	avg_opp_turnovers,
    	avg_opp_time_of_possession
	)
	SELECT 
    t_team_id, 
    t_week, 
    t_year, 

    -- Offensive Metrics 
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
		tgl.opp = t_team_id AND tgl.week <= t_week AND tgl.year = t_season

	ON CONFLICT (team_id, week, season) DO UPDATE SET
	avg_opp_pass_tds = COALESCE(EXCLUDED.avg_opp_pass_tds, team_weekly_agg_metrics.avg_opp_pass_tds, -1),
    avg_opp_pass_cmp = COALESCE(EXCLUDED.avg_opp_pass_cmp, team_weekly_agg_metrics.avg_opp_pass_cmp, -1),
    avg_opp_pass_att = COALESCE(EXCLUDED.avg_opp_pass_att, team_weekly_agg_metrics.avg_opp_pass_att, -1),
    avg_opp_pass_cmp_pct = COALESCE(EXCLUDED.avg_opp_pass_cmp_pct, team_weekly_agg_metrics.avg_opp_pass_cmp_pct, -1),
    avg_opp_yds_gained_per_pass_att = COALESCE(EXCLUDED.avg_opp_yds_gained_per_pass_att, team_weekly_agg_metrics.avg_opp_yds_gained_per_pass_att, -1),
    avg_opp_adj_yds_gained_per_pass_att = COALESCE(EXCLUDED.avg_opp_adj_yds_gained_per_pass_att, team_weekly_agg_metrics.avg_opp_adj_yds_gained_per_pass_att, -1),
    avg_opp_pass_rate = COALESCE(EXCLUDED.avg_opp_pass_rate, team_weekly_agg_metrics.avg_opp_pass_rate, -1),
    avg_opp_sacked = COALESCE(EXCLUDED.avg_opp_sacked, team_weekly_agg_metrics.avg_opp_sacked, -1),
    avg_opp_sack_yds_lost = COALESCE(EXCLUDED.avg_opp_sack_yds_lost, team_weekly_agg_metrics.avg_opp_sack_yds_lost, -1),
    avg_opp_rush_att = COALESCE(EXCLUDED.avg_opp_rush_att, team_weekly_agg_metrics.avg_opp_rush_att, -1),
    avg_opp_rush_tds = COALESCE(EXCLUDED.avg_opp_rush_tds, team_weekly_agg_metrics.avg_opp_rush_tds, -1),
    avg_opp_rush_yds_per_att = COALESCE(EXCLUDED.avg_opp_rush_yds_per_att, team_weekly_agg_metrics.avg_opp_rush_yds_per_att, -1),
    avg_opp_tot_off_plays = COALESCE(EXCLUDED.avg_opp_tot_off_plays, team_weekly_agg_metrics.avg_opp_tot_off_plays, -1),
    avg_opp_yds_per_play = COALESCE(EXCLUDED.avg_opp_yds_per_play, team_weekly_agg_metrics.avg_opp_yds_per_play, -1),
    avg_opp_fga = COALESCE(EXCLUDED.avg_opp_fga, team_weekly_agg_metrics.avg_opp_fga, -1),
    avg_opp_fgm = COALESCE(EXCLUDED.avg_opp_fgm, team_weekly_agg_metrics.avg_opp_fgm, -1),
    avg_opp_xpa = COALESCE(EXCLUDED.avg_opp_xpa, team_weekly_agg_metrics.avg_opp_xpa, -1),
    avg_opp_xpm = COALESCE(EXCLUDED.avg_opp_xpm, team_weekly_agg_metrics.avg_opp_xpm, -1),
    avg_opp_total_punts = COALESCE(EXCLUDED.avg_opp_total_punts, team_weekly_agg_metrics.avg_opp_total_punts, -1),
    avg_opp_punt_yds = COALESCE(EXCLUDED.avg_opp_punt_yds, team_weekly_agg_metrics.avg_opp_punt_yds, -1),
    avg_opp_pass_fds = COALESCE(EXCLUDED.avg_opp_pass_fds, team_weekly_agg_metrics.avg_opp_pass_fds, -1),
    avg_opp_rsh_fds = COALESCE(EXCLUDED.avg_opp_rsh_fds, team_weekly_agg_metrics.avg_opp_rsh_fds, -1),
    avg_opp_pen_fds = COALESCE(EXCLUDED.avg_opp_pen_fds, team_weekly_agg_metrics.avg_opp_pen_fds, -1),
    avg_opp_total_fds = COALESCE(EXCLUDED.avg_opp_total_fds, team_weekly_agg_metrics.avg_opp_total_fds, -1),
    avg_opp_thrd_down_conv = COALESCE(EXCLUDED.avg_opp_thrd_down_conv, team_weekly_agg_metrics.avg_opp_thrd_down_conv, -1),
    avg_opp_thrd_down_att = COALESCE(EXCLUDED.avg_opp_thrd_down_att, team_weekly_agg_metrics.avg_opp_thrd_down_att, -1),
    avg_opp_foruth_down_conv = COALESCE(EXCLUDED.avg_opp_foruth_down_conv, team_weekly_agg_metrics.avg_opp_foruth_down_conv, -1),
    avg_opp_foruth_down_att = COALESCE(EXCLUDED.avg_opp_foruth_down_att, team_weekly_agg_metrics.avg_opp_foruth_down_att, -1),
    avg_opp_penalties = COALESCE(EXCLUDED.avg_opp_penalties, team_weekly_agg_metrics.avg_opp_penalties, -1),
    avg_opp_pentalty_yds = COALESCE(EXCLUDED.avg_opp_pentalty_yds, team_weekly_agg_metrics.avg_opp_pentalty_yds, -1),
    avg_opp_fmbl_lost = COALESCE(EXCLUDED.avg_opp_fmbl_lost, team_weekly_agg_metrics.avg_opp_fmbl_lost, -1),
    avg_opp_int = COALESCE(EXCLUDED.avg_opp_int, team_weekly_agg_metrics.avg_opp_int, -1),
    avg_opp_turnovers = COALESCE(EXCLUDED.avg_opp_turnovers, team_weekly_agg_metrics.avg_opp_turnovers, -1),
    avg_opp_time_of_possession = COALESCE(EXCLUDED.avg_opp_time_of_possession, team_weekly_agg_metrics.avg_opp_time_of_possession, -1);
END;
$$ LANGUAGE plpgsql;


-- Create Triggers  & Trigger Functions 
CREATE OR REPLACE FUNCTION trigger_upsert_for_def_team_game_log() 
RETURNS TRIGGER AS $$
BEGIN 
	PERFORM upsert_team_weekly_def_agg_metrics(NEW.team_id, NEW.week, NEW.year);
	RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE TRIGGER def_team_game_log_trigger
AFTER INSERT OR UPDATE ON team_game_log 
FOR EACH ROW 
EXECUTE FUNCTION trigger_upsert_for_def_team_game_log(); 
