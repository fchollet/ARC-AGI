import cProfile
import functools
import pstats
import signal
import time


def profile(threshold=None, names=None):
    """Limit output to a threshold fraction of total cumulative time"""

    def inner(func):
        """Wraps a function to provide easy profiling results"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Begin profiling by enabling the profiler
            pr = cProfile.Profile()
            pr.enable()

            # The call we are profiling
            result = func(*args, **kwargs)

            # Stop profiling and analyze the results
            pr.disable()
            ps = pstats.Stats(pr)
            ps.strip_dirs()
            tot_time = max([row[3] for row in ps.stats.values()])
            stats = []
            for k, v in ps.stats.items():
                cum_frac = round(v[3] / tot_time, 6)
                if names and any([name in k for name in names]):
                    name = f"{k[0]}:{k[2]}"
                    stats.append((name, cum_frac))
                elif threshold and cum_frac > threshold:
                    name = f"{k[0]}:{k[2]}"
                    stats.append((name, cum_frac))

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
