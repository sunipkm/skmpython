# Library of functions and classes often used
## Functions
1. `datetime_in_timezone`: Instantiate `datetime.datetime` object using a similar constructor where the first input is `pytz.timezone()` instance, followed by `year`, `month`, `day`, `hour`, etc. Will have issues when daylight saving is set/reset.
1. `format_tdiff`: DEPRECATED. Format time difference (in seconds) using a format similar to that used by `datetime.datetime.strftime()`. Use `PerfTimer.format_tdiff()` instead.

## Decorators
1. `staticvars`: Use this decorator to declare and initialize static variables to be used
inside a function. e.g.
```py
    from skmpython import staticvars

    @staticvars(count=0)
    def func():
        func.count += 1
        print('Called %d times.'%(func.count))
```
will print how many times `func()` has been called.

## Classes
1. `PerfTimer`: Provides a way to time loops.
1. `TransformImage`: Provides image manipulation on (possibly supersampled) images, such as translation, rotation, resize. Supports loading from jpg, png and FITS. Supports saving to jpg, png.
1. `GaussFit`: Fit data to 5 degree polynomial background + N gaussians.
1. `GaussFuncs`: Provides methods to calculate backgrounds and gaussians for X given a list of parameters. This class extends the abstract `BaseGaussFuncs` class.