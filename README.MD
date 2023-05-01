# Library of functions and classes often used
## Installation
Using `pip`:
```sh
    $ pip install skmpython@"git+https://github.com/sunipkm/skmpython"
```
## Scripts
1. `update_iss_tledb`: Allows update of internal ISS TLE database. Current database is reliable between 2017-02-15 and 2017-09-30. Inputs:
    - `--start YYYYMMDD`: Start date
    - `--end YYYYMMDD`: End date
    - `--db /path/to/database` (optional): Path to database file. Do not provide to update internal database used by SatPosPredict.

## Modules
1. `GenericFit`: Provides `GenericFitFunc` and `GenericFitManager`. Instances of `GenericFitFunc` supports registration of background and feature functions, that can then be provided to
`GenericFitManager`, along with X and Y data, to fit to the given function. `GenericFitFunc` extends the abstract `FitFuncBase` class. Any custom class used as a base class in
`GenericFitManager` must derive from `FitFuncBase`.
2. `SatPosPredict`: Provides `ISSLatLonFromTstamp`. This function uses a TLE database (included default contains ISS TLE from 2017-04 to 2017-06), takes a timestamp in `datetime` or `numpy.datetime64` format.

## Functions
1. `datetime_in_timezone`: Instantiate `datetime.datetime` object using a similar constructor where the first input is `pytz.timezone()` instance, followed by `year`, `month`, `day`, `hour`, etc. Will have issues when daylight saving is set/reset.
2. `format_tdiff`: DEPRECATED. Format time difference (in seconds) using a format similar to that used by `datetime.datetime.strftime()`. Use `PerfTimer.format_tdiff()` instead.
3. `vac2air`: Convert wavelengths in vacuum (in nm) to wavelengths in air.
4. `air2vac`: Convert wavelengths in air (in nm) to wavelengths in vacuum.

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
2. `TransformImage`: Provides image manipulation on (possibly supersampled) images, such as translation, rotation, resize. Supports loading from jpg, png and FITS. Supports saving to jpg, png.
3. `GaussFit`: Fit data to 5 degree polynomial background + N gaussians (deprecated).
4. `GaussFuncs`: Provides methods to calculate backgrounds and gaussians for X given a list of parameters. This class extends the abstract `BaseGaussFuncs` class (deprecated).