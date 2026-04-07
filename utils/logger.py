# logging/logger.py

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    # import here to avoid circular imports since settings also uses paths
    from config.settings import LOGS_PATH

    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # without this check every import adds another handler and you get duplicate lines
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # file gets DEBUG so we can trace issues without changing log levels in prod
    log_file = LOGS_PATH / "agent.log"
    file = logging.FileHandler(log_file, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)
    logger.propagate = False

    return logger