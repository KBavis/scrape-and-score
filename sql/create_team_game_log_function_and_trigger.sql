
CREATE OR REPLACE FUNCTION update_opp_stats_after_insert()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if the opponent's game log already exists
    IF EXISTS (
        SELECT 1 FROM team_game_log
        WHERE team_id = NEW.opp AND week = NEW.week AND year = NEW.year
    ) THEN
        -- Update the newly inserted row with the opponent's stats
        UPDATE team_game_log
        SET
            opp_tot_yds = (
                SELECT tot_yds FROM team_game_log
                WHERE team_id = NEW.opp AND week = NEW.week AND year = NEW.year
            ),
            opp_pass_yds = (
                SELECT pass_yds FROM team_game_log
                WHERE team_id = NEW.opp AND week = NEW.week AND year = NEW.year
            ),
            opp_rush_yds = (
                SELECT rush_yds FROM team_game_log
                WHERE team_id = NEW.opp AND week = NEW.week AND year = NEW.year
            )
        WHERE team_id = NEW.team_id AND week = NEW.week AND year = NEW.year;
    END IF;

    RETURN NULL; -- Since this is an AFTER trigger, we don't modify NEW
END;
$$ LANGUAGE plpgsql;



-- Create Trigger
CREATE TRIGGER trg_update_opp_stats
AFTER INSERT ON team_game_log
FOR EACH ROW
EXECUTE FUNCTION update_opp_stats_after_insert();
