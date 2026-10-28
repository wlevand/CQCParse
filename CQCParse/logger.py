"""
Logger is like a singleton, it's one instance of a logger by the name of it.
Plain logging only: the application (e.g. wilson) chooses how the log lines look.

https://docs.python.org/3/howto/logging.html
"""

import logging
import sys


def setup_logger(name="CQCParse", level=logging.INFO, log_to_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    plain_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Terminal handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(plain_fmt)
    logger.addHandler(handler)

    # Optional file logging (plain format)
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file, mode='w')
        file_handler.setFormatter(plain_fmt)
        logger.addHandler(file_handler)

    return logger
