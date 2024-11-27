import logging
import os
import sys

logger = logging.getLogger(__name__)


def setupLogger(name, out_dir=None):
    """
    Create or get a Python logger for THOR. If out_dir is passed,
    then a file handler will be added to the logger.

    Returns
    -------
    logger :
        Logger with a stream handler to stdout, and if out_dir is defined
        an additional file handler to {out_dir}/thor.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    add_stream_handler = False
    add_file_handler = False

    if len(logger.handlers) == 0:
        add_stream_handler = True
        if out_dir is not None:
            add_file_handler = True
    elif len(logger.handlers) == 1 and out_dir is not None:
        add_file_handler = True
    else:
        pass

    if add_stream_handler:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_format = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        stream_handler.setFormatter(stream_format)
        logger.addHandler(stream_handler)

    if add_file_handler:
        file_handler = logging.FileHandler(os.path.join(out_dir, "thor.log"), encoding="utf-8", delay=False)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(filename)s, %(funcName)s, %(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
