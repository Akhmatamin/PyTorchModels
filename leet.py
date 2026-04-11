from typing import Callable
import time
from functools import wraps


def limit_call_deco1(limit : int):
    def wrapper(func : Callable):
        @wraps(func)
        def inner(*args, **kwargs):
            """Non-necessary docu"""
            nonlocal limit
            if limit == 0:
                print('Cannot call function!')
                return
            start = time.time()
            res = func(*args, **kwargs)
            print(f'{func.__name__} took {time.time() - start} seconds')
            limit -= 1
            return res
        return inner
    return wrapper

@limit_call_deco1(2)
def my_func(sleep_time: int):
    """Some documentation very important"""
    time.sleep(sleep_time)
    return "Hello My Decorator!"
print(my_func(1))
print(my_func.__doc__)
print(my_func.__name__)