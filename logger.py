import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[38;5;39m"
    format = "[%(levelname)s][%(asctime)s][%(name)s] : %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%d-%m-%y %H:%M:%S")
        return formatter.format(record)


def get_logger(logger_name: str) -> logging.Logger:
    """Get a Logger.

    Parameters
    ----------
    logger_name : logger name (or service name)

    Returns
    ----------
    logger: custom logger object
    """
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    return logger
