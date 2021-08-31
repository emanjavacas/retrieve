
# Set default logging handler to avoid "No handler found" warnings.
import logging
import warnings
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


# warn about available packages
try:
    import _align
except ModuleNotFoundError:
    warnings.warn("Couldn't find cython 'align' extension, defaulting to numba implementation")
    _align = None

try:
    import pyemd
except ModuleNotFoundError:
    pyemd = None
    warnings.warn("Couldn't import `pyemd` module, deactivating `wmd` method")


from . import utils
from .pipeline import pipeline, Results
from .embeddings import Embeddings


# package version
import sys

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version('text-reuse-retrieve')
