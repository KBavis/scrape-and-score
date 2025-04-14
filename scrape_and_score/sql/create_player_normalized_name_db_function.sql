-- create function to normalize a players name 
CREATE OR REPLACE FUNCTION normalize_player_name()
RETURNS TRIGGER AS $$
BEGIN
    NEW.normalized_name := LOWER(REGEXP_REPLACE(NEW.name, '[^a-zA-Z ]', '', 'g'));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- create trigger that will apply name normalization to each newly inserted player 
CREATE TRIGGER trigger_normalize_player_name
BEFORE INSERT ON player
FOR EACH ROW
EXECUTE FUNCTION normalize_player_name();
