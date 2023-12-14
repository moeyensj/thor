"""
Utilities for memory profiling
"""
import logging
import sys
from functools import wraps

__all__ = ["profile_ray_task", ]


logger = logging.getLogger(__name__)

def profile_ray_task(func):
    """
    Decorator to profile the memory usage of a Ray task.

    Parameters
    ----------
    func
        The function to profile.

    Returns
    -------
    The decorated function.
    """
    # Only attempt to profile memory if --memray is in the command line arguments
    # and if memray is installed.
    try:
        import memray
        import ray
    except ImportError:
        logger.debug("memray not installed, not profiling memory.")
        return func
    
    # # Detect command line arguments
    # if "--memray" not in sys.argv:
    #     logger.debug("--memray not in command line arguments, not profiling memory.")
    #     return func

    # Detect --memray-bin-path in arguments and make it the root folder for the
    # memory profile files.
    bin_path = "/tmp/ray/session_latest/logs/"
    for arg in sys.argv:
        if arg.startswith("--memray-bin-path="):
            bin_path = arg.split("=")[1]
            break

    @wraps(func)
    def wrapper(*args, **kwargs):
        with memray.Tracker(
            f"{bin_path}{func.__name__}_{ray.get_runtime_context().get_task_id()}_mem_profile.bin",
            native_traces=True,

        ):
            return func(*args, **kwargs)
    return wrapper



