from collections import defaultdict, deque
import datetime
import os
import time

import torch


# Copied from: https://github.com/1zb/VecSetX/blob/master/vecset/utils/misc.py
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.data_time = SmoothedValue(fmt="{avg:.4f}")
        self.iter_time = SmoothedValue(fmt="{avg:.4f}")
        self.end = time.time()

    def begin(self):
        self.end = time.time()

    def end_data(self):
        self.data_time.update(time.time() - self.end)
        self.end = time.time()

    def end_iter(self):
        self.iter_time.update(time.time() - self.end)
        self.end = time.time()

    def total_time(self):
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        return total_time, total_time_str
