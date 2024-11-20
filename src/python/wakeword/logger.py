import os
import sys
import logging
from config import LOG_FILE_PATH

def setup_logging():
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    datefmt='%Y-%m-%d %H:%M:%S'
    path = os.path.join(LOG_FILE_PATH)

    file_handler = logging.FileHandler(path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    handlers = [file_handler, console_handler]

    logging.basicConfig(handlers=handlers, level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.debug("logging setup complete")
    return logger