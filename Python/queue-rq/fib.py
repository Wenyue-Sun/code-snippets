# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time

def slow_fib(n, sleep=True):
    if sleep:
        time.sleep(1)
        sleep = False
    if n <= 1:
        return 1
    else:
        return slow_fib(n-1, sleep=False) + slow_fib(n-2, sleep=False)