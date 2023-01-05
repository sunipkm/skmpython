# %% Imports
from __future__ import annotations
from typing import Tuple
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from astropy.time import Time
from sgp4.api import Satrec
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation
from astropy import units as u
from astropy.coordinates import ITRS
import pandas as pd
from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation
from astropy import units as u
import pytz
from astropy.coordinates import ITRS

# %%
def staticvars(**kwargs):
    def decorate(func):
        for key in kwargs:
            setattr(func, key, kwargs[key])
        return func
    return decorate
    
@staticvars(tledb=None, tlefile='')
def LatLonFromTstamp(ts: datetime | np.datetime64, *, database_fname: str = 'ISS_TLE_DB.nc') -> Tuple[float, float]:
    """Get latitude, longitude for a given timestamp using a TLE database.

    Args:
        ts (datetime | np.datetime64): Timestamp for evaluation, is timezone aware.
        database_fname (str, optional): TLE dataset file (loaded using xarray.load_dataset). The dataset file must contain a timestamp (coordinate) for when the TLE is valid, and data_vars line1 and line2 containing the two TLE lines. Defaults to 'ISS_TLE_DB.nc'.

    Raises:
        RuntimeError: SGP4 runtime errors.

    Returns:
        Tuple[float, float]: (latitude, longitude) in degrees.
    """
    if LatLonFromTstamp.tledb is None or LatLonFromTstamp.tlefile != database_fname:
        if database_fname == 'ISS_TLE_DB.nc':
            database_fname = os.path.join(os.path.abspath(__file__), database_fname)
        LatLonFromTstamp.tledb = xr.load_dataset(database_fname)
        LatLonFromTstamp.tlefile = database_fname
    if isinstance(ts, datetime):
        ts = datetime.fromtimestamp(ts.astimezone(tz = pytz.utc).timestamp())
    tledb: xr.Dataset = LatLonFromTstamp.tledb
    tles = tledb.sel(timestamp=ts, method='nearest')
    l1 = str(np.asarray(tles.line1))
    l2 = str(np.asarray(tles.line2))
    jdts = Time(pd.Timestamp(ts).to_julian_date(), format='jd')
    sat = Satrec.twoline2rv(l1, l2)
    err, p_vec, v_vec = sat.sgp4(jdts.jd1, jdts.jd2)
    p_vec = CartesianRepresentation(p_vec*u.km)
    v_vec = CartesianDifferential(v_vec*u.km/u.s)
    s_vec = TEME(p_vec.with_differentials(v_vec), obstime=ts)
    itrs_geo = s_vec.transform_to(ITRS(obstime=ts))
    loc = itrs_geo.earth_location.geodetic
    if err:
        raise RuntimeError('SGP4 error %d'%(err))
    return (float(loc.lat.value), float(loc.lon.value))
# %%
if __name__ == '__main__':
    ts = datetime(2017, 2, 16, tzinfo=pytz.timezone('US/Eastern'))
    print(ts, LatLonFromTstamp(ts))