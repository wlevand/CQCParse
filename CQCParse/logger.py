"""
Logger is like a singleton, it's one instance of a logger by the name of it.

https://docs.python.org/3/howto/logging.html
"""

import logging
import sys
from rich.logging import RichHandler

def setup_logger(name="CQCParse", level=logging.INFO, use_rich=True, log_to_file=None, no_color=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # Formatter for plain output (no color)
    plain_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Terminal handler (colored or plain)
    if use_rich and not no_color:
        handler = RichHandler(markup=True, rich_tracebacks=True)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(plain_fmt)

    logger.addHandler(handler)

    # Optional file logging (plain format)
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file, mode='w')
        file_handler.setFormatter(plain_fmt)
        logger.addHandler(file_handler)

    return logger
