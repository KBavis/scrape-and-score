-- function to insert/update player_weekly_agg_metrics table with relevant metrics 
CREATE OR REPLACE FUNCTION upsert_player_weekly_agg_metrics(p_player_id INT, p_week INT, p_season INT)
RETURNS VOID AS $$
BEGIN
    INSERT INTO player_weekly_agg_metrics (
        player_id,
        week,
        season,

        -- Advanced Passing
        avg_pass_first_downs,
        avg_pass_first_downs_per_pass_play,
        avg_intended_air_yards,
        avg_intended_air_yards_per_pass_attempt,
        avg_completed_air_yards,
        avg_completed_air_yards_per_cmp,
        avg_completed_air_yards_per_att,
        avg_pass_yds_after_catch,
        avg_pass_yds_after_catch_per_cmp,
        avg_pass_drops,
        avg_pass_drop_pct,
        avg_pass_poor_throws,
        avg_pass_poor_throws_pct,
        avg_pass_blitzed,
        avg_pass_hurried,
        avg_pass_hits,
        avg_pass_pressured,
        avg_pass_pressured_pct,
        avg_pass_scrambles,
        avg_pass_yds_per_scramble,

        -- Advanced Rushing/Receiving
        avg_rush_first_downs,
        avg_rush_yds_before_contact,
        avg_rush_yds_before_contact_per_att,
        avg_rush_yds_after_contact,
        avg_rush_yds_after_contact_per_att,
        avg_rush_broken_tackles,
        avg_rush_att_per_broken_tackle,
        avg_rec_first_downs,
        avg_rec_yds_before_catch,
        avg_rec_yds_before_catch_per_rec,
        avg_rec_yds_after_catch,
        avg_rec_yds_after_catch_per_rec,
        avg_avg_depth_of_target,
        avg_rec_broken_tackles,
        avg_rec_per_broken_tackle,
        avg_rec_dropped_passes,
        avg_rec_drop_pct,
        avg_rec_int_when_targeted,
        avg_rec_qbr_when_targeted,

        -- Game Logs
		avg_fantasy_points,
        avg_completions,
        avg_attempts,
        avg_pass_yds,
        avg_pass_tds,
        avg_interceptions,
        avg_rating,
        avg_sacked,
        avg_rush_attempts,
        avg_rush_yds,
        avg_rush_tds,
        avg_targets,
        avg_receptions,
        avg_rec_yds,
        avg_rec_tds,
        avg_snap_pct,
        avg_offensive_snaps
    )
    SELECT
        p_player_id,
        p_week,
        p_season,

	-- Advanced Passing
    COALESCE(ROUND(AVG(pap.first_downs)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.first_down_passing_per_pass_play)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.intended_air_yards)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.intended_air_yards_per_pass_attempt)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.completed_air_yards)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.completed_air_yards_per_cmp)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.completed_air_yards_per_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.yds_after_catch)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.yds_after_catch_per_cmp)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.drops)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.drop_pct)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.poor_throws)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.poor_throws_pct)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.blitzed)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.hurried)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.hits)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.pressured)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.pressured_pct)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.scrmbl)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pap.yds_per_scrmbl)::numeric, 2), -1),

	-- Advanced Rushing/Receiving
    COALESCE(ROUND(AVG(par.rush_first_downs)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rush_yds_before_contact)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rush_yds_before_contact_per_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rush_yds_afer_contact)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rush_yds_after_contact_per_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rush_brkn_tackles)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rush_att_per_brkn_tackle)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rec_first_downs)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.yds_before_catch)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.yds_before_catch_per_rec)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.yds_after_catch)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.yds_after_catch_per_rec)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.avg_depth_of_tgt)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rec_brkn_tackles)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.rec_per_brkn_tackle)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.dropped_passes)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.drop_pct)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.int_when_tgted)::numeric, 2), -1),
    COALESCE(ROUND(AVG(par.qbr_when_tgted)::numeric, 2), -1),

	-- Player Game Logs
    COALESCE(ROUND(AVG(pgl.fantasy_points)::numeric, 2), 0), -- indicate 0 fantasy points if all null instead of -1
    COALESCE(ROUND(AVG(pgl.completions)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.attempts)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.pass_yd)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.pass_td)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.interceptions)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rating)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.sacked)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rush_att)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rush_yds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rush_tds)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.tgt)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rec)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rec_yd)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.rec_td)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.snap_pct)::numeric, 2), -1),
    COALESCE(ROUND(AVG(pgl.off_snps)::numeric, 2), -1)


    FROM player_game_log pgl
    LEFT JOIN player_advanced_passing pap
        ON pgl.player_id = pap.player_id AND pgl.week = pap.week AND pgl.year = pap.season
    LEFT JOIN player_advanced_rushing_receiving par
        ON pgl.player_id = par.player_id AND pgl.week = par.week AND pgl.year = par.season
    WHERE pgl.player_id = p_player_id AND pgl.week <= p_week AND pgl.year = p_season

    ON CONFLICT (player_id, week, season) DO UPDATE SET
	-- Player Advanced Passing
    avg_pass_first_downs = COALESCE(EXCLUDED.avg_pass_first_downs, player_weekly_agg_metrics.avg_pass_first_downs, -1),
    avg_pass_first_downs_per_pass_play = COALESCE(EXCLUDED.avg_pass_first_downs_per_pass_play, player_weekly_agg_metrics.avg_pass_first_downs_per_pass_play, -1),
    avg_intended_air_yards = COALESCE(EXCLUDED.avg_intended_air_yards, player_weekly_agg_metrics.avg_intended_air_yards, -1),
    avg_intended_air_yards_per_pass_attempt = COALESCE(EXCLUDED.avg_intended_air_yards_per_pass_attempt, player_weekly_agg_metrics.avg_intended_air_yards_per_pass_attempt, -1),
    avg_completed_air_yards = COALESCE(EXCLUDED.avg_completed_air_yards, player_weekly_agg_metrics.avg_completed_air_yards, -1),
    avg_completed_air_yards_per_cmp = COALESCE(EXCLUDED.avg_completed_air_yards_per_cmp, player_weekly_agg_metrics.avg_completed_air_yards_per_cmp, -1),
    avg_completed_air_yards_per_att = COALESCE(EXCLUDED.avg_completed_air_yards_per_att, player_weekly_agg_metrics.avg_completed_air_yards_per_att, -1),
    avg_pass_yds_after_catch = COALESCE(EXCLUDED.avg_pass_yds_after_catch, player_weekly_agg_metrics.avg_pass_yds_after_catch, -1),
    avg_pass_yds_after_catch_per_cmp = COALESCE(EXCLUDED.avg_pass_yds_after_catch_per_cmp, player_weekly_agg_metrics.avg_pass_yds_after_catch_per_cmp, -1),
    avg_pass_drops = COALESCE(EXCLUDED.avg_pass_drops, player_weekly_agg_metrics.avg_pass_drops, -1),
    avg_pass_drop_pct = COALESCE(EXCLUDED.avg_pass_drop_pct, player_weekly_agg_metrics.avg_pass_drop_pct, -1),
    avg_pass_poor_throws = COALESCE(EXCLUDED.avg_pass_poor_throws, player_weekly_agg_metrics.avg_pass_poor_throws, -1),
    avg_pass_poor_throws_pct = COALESCE(EXCLUDED.avg_pass_poor_throws_pct, player_weekly_agg_metrics.avg_pass_poor_throws_pct, -1),
    avg_pass_blitzed = COALESCE(EXCLUDED.avg_pass_blitzed, player_weekly_agg_metrics.avg_pass_blitzed, -1),
    avg_pass_hurried = COALESCE(EXCLUDED.avg_pass_hurried, player_weekly_agg_metrics.avg_pass_hurried, -1),
    avg_pass_hits = COALESCE(EXCLUDED.avg_pass_hits, player_weekly_agg_metrics.avg_pass_hits, -1),
    avg_pass_pressured = COALESCE(EXCLUDED.avg_pass_pressured, player_weekly_agg_metrics.avg_pass_pressured, -1),
    avg_pass_pressured_pct = COALESCE(EXCLUDED.avg_pass_pressured_pct, player_weekly_agg_metrics.avg_pass_pressured_pct, -1),
    avg_pass_scrambles = COALESCE(EXCLUDED.avg_pass_scrambles, player_weekly_agg_metrics.avg_pass_scrambles, -1),
    avg_pass_yds_per_scramble = COALESCE(EXCLUDED.avg_pass_yds_per_scramble, player_weekly_agg_metrics.avg_pass_yds_per_scramble, -1),
	-- Player Advanced Rushing / Receiving
    avg_rush_first_downs = COALESCE(EXCLUDED.avg_rush_first_downs, player_weekly_agg_metrics.avg_rush_first_downs, -1),
    avg_rush_yds_before_contact = COALESCE(EXCLUDED.avg_rush_yds_before_contact, player_weekly_agg_metrics.avg_rush_yds_before_contact, -1),
    avg_rush_yds_before_contact_per_att = COALESCE(EXCLUDED.avg_rush_yds_before_contact_per_att, player_weekly_agg_metrics.avg_rush_yds_before_contact_per_att, -1),
    avg_rush_yds_after_contact = COALESCE(EXCLUDED.avg_rush_yds_after_contact, player_weekly_agg_metrics.avg_rush_yds_after_contact, -1),
    avg_rush_yds_after_contact_per_att = COALESCE(EXCLUDED.avg_rush_yds_after_contact_per_att, player_weekly_agg_metrics.avg_rush_yds_after_contact_per_att, -1),
    avg_rush_broken_tackles = COALESCE(EXCLUDED.avg_rush_broken_tackles, player_weekly_agg_metrics.avg_rush_broken_tackles, -1),
    avg_rush_att_per_broken_tackle = COALESCE(EXCLUDED.avg_rush_att_per_broken_tackle, player_weekly_agg_metrics.avg_rush_att_per_broken_tackle, -1),
    avg_rec_first_downs = COALESCE(EXCLUDED.avg_rec_first_downs, player_weekly_agg_metrics.avg_rec_first_downs, -1),
    avg_rec_yds_before_catch = COALESCE(EXCLUDED.avg_rec_yds_before_catch, player_weekly_agg_metrics.avg_rec_yds_before_catch, -1),
    avg_rec_yds_before_catch_per_rec = COALESCE(EXCLUDED.avg_rec_yds_before_catch_per_rec, player_weekly_agg_metrics.avg_rec_yds_before_catch_per_rec, -1),
    avg_rec_yds_after_catch = COALESCE(EXCLUDED.avg_rec_yds_after_catch, player_weekly_agg_metrics.avg_rec_yds_after_catch, -1),
    avg_rec_yds_after_catch_per_rec = COALESCE(EXCLUDED.avg_rec_yds_after_catch_per_rec, player_weekly_agg_metrics.avg_rec_yds_after_catch_per_rec, -1),
    avg_avg_depth_of_target = COALESCE(EXCLUDED.avg_avg_depth_of_target, player_weekly_agg_metrics.avg_avg_depth_of_target, -1),
    avg_rec_broken_tackles = COALESCE(EXCLUDED.avg_rec_broken_tackles, player_weekly_agg_metrics.avg_rec_broken_tackles, -1),
    avg_rec_per_broken_tackle = COALESCE(EXCLUDED.avg_rec_per_broken_tackle, player_weekly_agg_metrics.avg_rec_per_broken_tackle, -1),
    avg_rec_dropped_passes = COALESCE(EXCLUDED.avg_rec_dropped_passes, player_weekly_agg_metrics.avg_rec_dropped_passes, -1),
    avg_rec_drop_pct = COALESCE(EXCLUDED.avg_rec_drop_pct, player_weekly_agg_metrics.avg_rec_drop_pct, -1),
    avg_rec_int_when_targeted = COALESCE(EXCLUDED.avg_rec_int_when_targeted, player_weekly_agg_metrics.avg_rec_int_when_targeted, -1),
    avg_rec_qbr_when_targeted = COALESCE(EXCLUDED.avg_rec_qbr_when_targeted, player_weekly_agg_metrics.avg_rec_qbr_when_targeted, -1),
	-- Player Game Log
	avg_fantasy_points =  COALESCE(EXCLUDED.avg_fantasy_points, player_weekly_agg_metrics.avg_fantasy_points, 0), -- indicate 0 fantasy points if all null instead of -1
    avg_completions = COALESCE(EXCLUDED.avg_completions, player_weekly_agg_metrics.avg_completions, -1),
    avg_attempts = COALESCE(EXCLUDED.avg_attempts, player_weekly_agg_metrics.avg_attempts, -1),
    avg_pass_yds = COALESCE(EXCLUDED.avg_pass_yds, player_weekly_agg_metrics.avg_pass_yds, -1),
    avg_pass_tds = COALESCE(EXCLUDED.avg_pass_tds, player_weekly_agg_metrics.avg_pass_tds, -1),
    avg_interceptions = COALESCE(EXCLUDED.avg_interceptions, player_weekly_agg_metrics.avg_interceptions, -1),
    avg_rating = COALESCE(EXCLUDED.avg_rating, player_weekly_agg_metrics.avg_rating, -1),
    avg_sacked = COALESCE(EXCLUDED.avg_sacked, player_weekly_agg_metrics.avg_sacked, -1),
    avg_rush_attempts = COALESCE(EXCLUDED.avg_rush_attempts, player_weekly_agg_metrics.avg_rush_attempts, -1),
    avg_rush_yds = COALESCE(EXCLUDED.avg_rush_yds, player_weekly_agg_metrics.avg_rush_yds, -1),
    avg_rush_tds = COALESCE(EXCLUDED.avg_rush_tds, player_weekly_agg_metrics.avg_rush_tds, -1),
    avg_targets = COALESCE(EXCLUDED.avg_targets, player_weekly_agg_metrics.avg_targets, -1),
    avg_receptions = COALESCE(EXCLUDED.avg_receptions, player_weekly_agg_metrics.avg_receptions, -1),
    avg_rec_yds = COALESCE(EXCLUDED.avg_rec_yds, player_weekly_agg_metrics.avg_rec_yds, -1),
    avg_rec_tds = COALESCE(EXCLUDED.avg_rec_tds, player_weekly_agg_metrics.avg_rec_tds, -1),
    avg_snap_pct = COALESCE(EXCLUDED.avg_snap_pct, player_weekly_agg_metrics.avg_snap_pct, -1),
    avg_offensive_snaps = COALESCE(EXCLUDED.avg_offensive_snaps, player_weekly_agg_metrics.avg_offensive_snaps, -1);
END;
$$ LANGUAGE plpgsql;



---------------------------------------------------------------------------------------------------
------------TRIGGERS / TRIGGER FUNCTIONS THAT WILL INVOKE ABOVE FUNCTION----------------------------
----------------------------------------------------------------------------------------------------

-- Create Trigger Function For Player_Game_log 
CREATE OR REPLACE FUNCTION trigger_upsert_for_player_game_log() 
RETURNS TRIGGER AS $$
BEGIN 
	PERFORM upsert_player_weekly_agg_metrics(NEW.player_id, NEW.week, NEW.year);
	RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- Create Trigger Function For Player Advanced Metrics 
CREATE OR REPLACE FUNCTION trigger_upsert_for_player_advanced_metrics()
RETURNS TRIGGER AS $$
BEGIN 
	PERFORM upsert_player_weekly_agg_metrics(NEW.player_id, NEW.week, NEW.season);
	RETURN NEW;
END;
$$ LANGUAGE plpgsql;



-- Create Individual Triggers 
CREATE OR REPLACE TRIGGER player_game_log_trigger
AFTER INSERT OR UPDATE ON player_game_log 
FOR EACH ROW 
EXECUTE FUNCTION trigger_upsert_for_player_game_log(); 

CREATE OR REPLACE TRIGGER player_advanced_passing_trigger 
AFTER INSERT OR UPDATE ON player_advanced_passing
FOR EACH ROW 
EXECUTE FUNCTION trigger_upsert_for_player_advanced_metrics()


CREATE OR REPLACE TRIGGER player_advanced_rushing_receiving_trigger 
AFTER INSERT OR UPDATE ON player_advanced_rushing_receiving
FOR EACH ROW 
EXECUTE FUNCTION trigger_upsert_for_player_advanced_metrics()

