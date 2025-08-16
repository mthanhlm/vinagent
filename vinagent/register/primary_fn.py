import functools


# Decorator to mark and validate the primary function
def primary_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
