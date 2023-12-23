# """
# Utilities for memory profiling
# """


def instrument_ray_worker():
    """
    Runs memray on the ray worker process.

    This has to be run when a new python interpreter is started or we will patch it too late
    and the worker process will already be running.
    Used in conjunction with ./thor_profiling.ph which has to be plated in dist-packages.
    """
    # Only enable ray profiling if ray and memray are installed.

    # Set default logger to debug
    import logging

    logging.basicConfig(level=logging.DEBUG)

    try:
        import ray
    except ImportError:
        print("ray not installed, not profiling memory.")
        return

    try:
        import memray
    except ImportError:
        print("memray not installed, not profiling memory.")
        return

    import ray._private.worker

    orig_main_loop = getattr(ray._private.worker.Worker, "main_loop")

    def patched_main_loop(self):
        """
        Patched version of ray._private.worker.Worker.main_loop that profiles the
        memory usage of the worker process with memray.
        """
        # Only initialize a tracker if there isn't already one
        with memray.Tracker(
            f"/tmp/ray/session_latest/logs/worker_{str(self.core_worker.get_worker_id())}_mem_profile.bin",
            native_traces=True,
            follow_fork=True,
            file_format=memray.FileFormat.ALL_ALLOCATIONS,
        ):
            return orig_main_loop(self)

    setattr(ray._private.worker.Worker, "main_loop", patched_main_loop)


instrument_ray_worker()
