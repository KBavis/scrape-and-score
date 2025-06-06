def convert_time_to_seconds(time_str):
    """
    Convert time string to seconds

    Args:
        time_str (str): the string to convert to seconds

    Returns:
        float: number of seconds
    """
    minutes, seconds = map(int, time_str.split(":"))
    return float(minutes * 60 + seconds)
