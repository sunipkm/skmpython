from __future__ import annotations
import datetime as dt
from math import inf
import re


def format_tdiff(tdiff: float | dt.timedelta, fmt: str) -> str:
    """Format time difference to string.

    Args:
        tdiff (float | dt.timedelta): Time difference.
        fmt (str): Format string. Use %p to get the sign, %d to get the number of days, %H to get hours, %M to get minutes, %S to get seconds, %f to get milliseconds.

    Raises:
        TypeError: tdiff is not of valid type.

    Returns:
        str: Formatted string.
    """
    if isinstance(tdiff, dt.timedelta):
        tdiff = tdiff.total_seconds()
    elif isinstance(tdiff, float):
        pass
    else:
        raise TypeError('%s is not a valid type for tdiff.' % (type(tdiff)))
    sgn = False
    if tdiff < 0:
        sgn = True
        tdiff = -tdiff
    rem = tdiff * 1000
    out = fmt
    prefix = '+'
    if tdiff == 0:
        prefix = ' '
    elif sgn:
        prefix = '-'
    prefix_prompts = re.findall('%p', fmt)
    if len(prefix_prompts):
        out = out.replace('%p', prefix)
    day_prompts = re.findall('%[0-9]*d', fmt)
    if len(day_prompts):
        days, rem = divmod(rem, 86400000)
        for dp in day_prompts:
            out = out.replace(dp, dp % (days))
    hour_prompts = re.findall('%H', fmt)
    if len(hour_prompts):
        hours, rem = divmod(rem, 3600000)
        out = out.replace('%H', '%02.0f' % (hours))
    minute_prompts = re.findall('%M', fmt)
    if len(minute_prompts):
        minutes, rem = divmod(rem, 60000)
        out = out.replace('%M', '%02.0f' % (minutes))
    second_prompts = re.findall('%S', fmt)
    if len(second_prompts):
        seconds, rem = divmod(rem, 1000)
        out = out.replace('%S', '%02.0f' % (seconds))
    msec_prompts = re.findall('%f', fmt)
    if len(msec_prompts):
        seconds, msec = divmod(rem, 1000)
        out = out.replace('%f', '%03.0f' % (msec))
    return out


class PerfTimer:
    """Performance Timer class.
    """

    def __init__(self, total: int, backlog: int = 2):
        """Initialize a performance timer instance.

        Args:
            total (int): Total operations to be performed. Must be > 1.
            backlog (int, optional):Count updates. Defaults to 2, minimum is 2.

        Raises:
            ValueError: Total operations < 1.
            ValueError: Backlog < 2.
        """
        if total < 1:
            raise ValueError('Items can not be less than 1.')
        if backlog < 1:
            raise ValueError('Backlog can not be less than 2.')
        self._backlog = backlog
        self._last = dt.datetime.now()
        self._init = self._last
        self._last_iter = {}
        self._total = total
        self._elapsed = 0

    def start(self):
        """Start the performance timer. This call is mostly
            unnecessary, the counter is started when created.
            This method is provided for use in rare cases.
        """
        self._last = dt.datetime.now()
        self._init = self._last

    def update(self, done: int):
        """Update counter with number of items performed.

        Args:
            done (int): Number of items completed.

        Raises:
            ValueError: Number of items completed must at least be 1.
        """
        if done < 1:
            raise ValueError('Items completed can not be less than 1.')
        now = dt.datetime.now()
        diff = (now - self._last).total_seconds()
        self._last_iter[done] = diff
        self._last = now
        if len(self._last_iter) > self._backlog:
            kold = list(self._last_iter.keys())[0]
            del self._last_iter[kold]
        self._elapsed = (now - self._init).total_seconds()

    @property
    def elapsed(self) -> float:
        """Time elapsed since the beginning of creation of this timer.

        Returns:
            float: Elapsed time since the counter started (in seconds.)
        """
        now = dt.datetime.now()
        self._elapsed = (now - self._init).total_seconds()
        return self._elapsed

    @property
    def eta(self) -> float:
        """Time remaining to complete operation.

        Returns:
            float: Time remaining. Returns inf in case no updates have been made.
        """
        if len(self._last_iter) == 0:
            _eta = inf
        if len(self._last_iter) == 1:  # happens when first update executed
            k = list(self._last_iter.keys())[0]
            _eta = (self._total - k) * self._last_iter[k] / k
        else:
            k0 = list(self._last_iter.keys())[-1]
            k1 = list(self._last_iter.keys())[-2]
            _eta = (self._total - k0) * self._last_iter[k0] / (k0 - k1)
        return _eta

    @property
    def lastiter(self) -> tuple:
        """Time taken for the last iteration to complete.

        Returns:
            tuple: (number of items, time taken)
        """
        if len(self._last_iter) == 0:
            return (0, 0)
        elif len(self._last_iter) == 1:
            k = list(self._last_iter.keys())[-1]
            return (k, self._last_iter[k])
        else:
            k0 = list(self._last_iter.keys())[-1]
            k1 = list(self._last_iter.keys())[-2]
            return (k0 - k1, self._last_iter[k0])
