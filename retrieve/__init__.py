
# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())


def add_handler_level(level=logging.DEBUG):
    """
    Inspired by urllib3 handling of logging
    """
    fmt = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    datefmt = '%Y-%m-%d:%H:%M:%S'
    name = "retrieve-handler"

    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        if handler.get_name() == name:
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.set_name(name)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.setLevel(level)


def enable_log_level(level=logging.DEBUG):
    add_handler_level(level=level)


from . import utils
from .pipeline import pipeline
