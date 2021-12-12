import cProfile
import functools
import pstats
import signal
import time
from typing import Any, Callable


def profile(
    threshold: float = 0.0, names: list[str] = None, dump_file: str = None
) -> Callable:
    """Measure where cumulative time is spent.

    Args:
        threshold: minimum total cumulative time to display a function
        names: only display functions containing one of these substrings
    """

    def inner(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> tuple[Any, list[tuple[str, float]]]:
            # Begin profiling by enabling the profiler
            pr = cProfile.Profile()
            pr.enable()

            # The call we are profiling
            result = func(*args, **kwargs)

            # Stop profiling and analyze the results
            pr.disable()
            ps = pstats.Stats(pr)
            ps.strip_dirs()
            if dump_file:
                ps.dump_stats(dump_file)
            # NOTE: PStats doesn't appear to use typing, thus the ignores
            tot_time = max([row[3] for row in ps.stats.values()])  # type: ignore
            stats = []
            for k, v in ps.stats.items():  # type: ignore
                func_name = f"{k[0]}:{k[2]}"
                cum_frac = round(v[3] / tot_time, 6)
                if cum_frac > threshold and (
                    names is None or any([name in k for name in names])
                ):
                    stats.append((func_name, cum_frac))

            return result, stats

        return wrapper

    return inner


class TimeoutException(Exception):
    pass


def time_it(limit=5):
    """Limit the runtime of the function to the specified number of seconds"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    def inner(func):
        """Wraps a function and throws a Timeout exception if it runs too long"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Begin a signal alarm (
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(limit)
            start = time.time()

            # The call we are limiting
            try:
                result = func(*args, **kwargs)
            except TimeoutException as exc:
                print(f"{exc}: Timed out after {limit}s")
                return None, limit

            return result, time.time() - start

        return wrapper

    return inner
