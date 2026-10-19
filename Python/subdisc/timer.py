import time


class Timer:
    """Timer for all event timings (nanosecond based, like the java one)."""

    def __init__(self):
        self._start_time = time.perf_counter_ns()

    def get_elapsed_time(self):
        return time.perf_counter_ns() - self._start_time

    def get_elapsed_time_string(self):
        nanos = time.perf_counter_ns() - self._start_time
        minutes = nanos // 60000000000
        seconds = (nanos % 60000000000) / 1000000000.0
        return "%d minute%s and %.3f seconds.\n" % (minutes, "" if minutes == 1 else "s", seconds)
