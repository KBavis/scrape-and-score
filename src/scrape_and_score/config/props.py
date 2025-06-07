import yaml
import logging


_config = None


def load_configs(file_path="./resources/application.yaml"):
    """
    Functionality to load our configurations

    Args:
        file_path(str): path containing configurations

    Returns:
        loaded configurations
    """

    global _config

    if _config is None:  # load once
        with open(file_path, "r") as file:
            _config = yaml.safe_load(file)
    return _config


def get_config(key, default=None):
    """
    Functionality to retrieve a specific configuration value using a dot-seperated key

    Args:
        key (str): key to fetch config for
    """

    keys = key.split(".")
    value = _config

    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, default)
        else:
            logging.warning(f"Configuration '{key}' was not found.")
            return default

    return value
