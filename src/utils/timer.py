import time
import functools

class Timer:
    """
    Context manager to measure execution time.
    Usage:
    with Timer() as t:
        # do something
    print(f"Time: {t.interval}")
    """
    def __enter__(self):
        self.start = time.time()
        self.interval = 0
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def time_execution(func):
    """
    Decorator to print the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
