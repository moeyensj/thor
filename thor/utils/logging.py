import logging
import os
import sys
import time

__all__ = ["setupLogger", "Timer"]

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
        file_handler = logging.FileHandler(
            os.path.join(out_dir, "thor.log"), encoding="utf-8", delay=False
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(filename)s, %(funcName)s, %(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class Timer:
    def __init__(
        self,
        file_name=None,
        file_dir="/tmp/thor/",
        prepend_data=[],
        sep=",",
        open_kwargs={
            "mode": "a",
            "buffering": -1,
            "encoding": "utf-8",
        },
    ):
        """
        Timing context manager that stores timing results and given user
        data to a file if desired.


        Parameters
        ----------
        file_name : {str, None}
            Name of file including extension but excluding file directory.
        file_dir : str
            Directory where to save file. Defaults to /tmp/thor/.
        prepend_data : list
            Additional data that should be prepended to outputs.
        sep : str
            If prepend_data is not an empty list, this separator will be used to concatenate
            data and the time elapsed into a single line.
        open_kwargs : dict
            Parameters with which to open the file.
        """
        self.time_start = 0
        self.time_end = 0
        self.file_name = file_name
        self.file_dir = file_dir
        self.file_path = None
        self.file = None
        self.prepend_data = prepend_data
        self.sep = sep

        if isinstance(file_name, str):
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir, exist_ok=False)
            self.file_path = os.path.join(file_dir, file_name)
            self.file = open(self.file_path, **open_kwargs)
        return

    def __enter__(self):
        self.time_start = time.time()
        return

    def __exit__(self, type, value, traceback):
        self.time_end = time.time()
        duration = self.time_end - self.time_start
        data = self.prepend_data + [duration]
        string = self.sep.join([str(d) for d in data])

        if self.file is not None:
            self.file.writelines([string + "\n"])
            self.file.close()
        else:
            if len(self.prepend_data) == 0:
                string = f"Time elapsed: {duration:.8f}s"
            logger.info(string)

        return
