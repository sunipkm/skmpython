# %% Imports
from __future__ import annotations
import argparse
from typing import Iterable, List, Tuple
from tqdm import tqdm
import xarray as xr
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse
import shutil

import json
import requests

# %%
def datetime_from_julian(t: str) -> datetime:
    year = int(t[:2]) # first two digits
    if year > 56: # first launch in 57 so 57 is 1957
        year += 1900
    else: # < 56: 56 -> 2056
        year += 2000

    yday = int(t[2:5]) # day of year
    fday = float(t[5:]) # fractional day of year
    
    start = pytz.utc.localize(datetime(year, 1, 1)) # first day of the year
    tstamp = start.timestamp()
    tstamp += (yday - 1)*86400 # add seconds spent per day
    tstamp += fday*86400 # fraction of day to seconds
    return datetime.utcfromtimestamp(tstamp)

# %%
dict_dayofweek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dict_mon = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def get_raw_tle_from_tstamp(ts: datetime | np.datetime64 | Iterable)->Tuple[datetime, str, str, str] | np.ndarray:
    if isinstance(ts, datetime):
        ts = datetime.utcfromtimestamp(ts.timestamp())
    elif isinstance(ts, np.datetime64):
        ts = datetime.utcfromtimestamp(int(ts)*1e-9)
    elif isinstance(ts, Iterable):
        out = []
        for t in tqdm(ts):
            out.append(get_raw_tle_from_tstamp(t))
        out = list(filter(lambda x: x is not None, out))
        return np.asarray(out).T
    else:
        raise RuntimeError('ts must be datetime, np.datetime64, or Iterable of datetime or np.datetime64.')

    dayofweek = dict_dayofweek[ts.weekday()]
    day = ts.day
    mon = dict_mon[ts.month]
    year = ts.year
    hh = ts.hour
    mm = ts.minute
    ss = ts.second

    url = f'http://isstracker.com/ajax/fetchTLE.php?date={dayofweek}%2C%20{day}%20{mon}%20{year}%20{hh}%3A{mm}%3A{ss}%20GMT'

    try:
        content = requests.get(url)
        if content.status_code != 200:
            print(ts, 'status code:', content.status_code)
            return None
        tledict = json.loads(content.content)
        epoch = tledict['epoch']
        epoch = int(parse(epoch + '+00:00').timestamp()*1e9)
        lines = tledict['jsTLE'].replace('\r', '').split('\n')
        return (np.datetime64(epoch, 'ns'), lines[0], lines[1], lines[2]) # epoch is in UTC
    except requests.exceptions.RequestException as e:
        print(ts, e)
        return None

# %%
def get_tle_ds(tledb: np.ndarray):
    if not isinstance(tledb, np.ndarray) or tledb.shape[0] != 4:
        raise RuntimeError
    tstamp = np.asarray(tledb[0, :], dtype=np.datetime64)
    line_0 = np.asarray(tledb[1, :], dtype=str)
    line_1 = np.asarray(tledb[2, :], dtype=str)
    line_2 = np.asarray(tledb[3, :], dtype=str)

    tle_ds = xr.Dataset(
    data_vars={'line0': ('timestamp', line_0),
        'line1': ('timestamp', line_1), 'line2': ('timestamp', line_2)},
    coords={'timestamp': tstamp}
    )
    return tle_ds.drop_duplicates(dim=..., keep='first')
# %%
def build_tledb_exec_fn():
    parser = argparse.ArgumentParser(description='(Re)build TLE database for ISS.')
    parser.add_argument('-s', '--start', type=str, help='start date in YYYYMMDD format', required=True)
    parser.add_argument('-e', '--end', type=str, help='end date in YYYYMMDD format', required=True)
    parser.add_argument('-d', '--db', type=str, help='database file name', default=None)
    parser.usage = parser.format_help()
    args = parser.parse_args()
    start = parse(args.start).astimezone(pytz.utc)
    end = parse(args.end).astimezone(pytz.utc)
    if start < parse('1998-11-20').astimezone(pytz.utc):
        raise ValueError('Start date must be after 1998-11-20 (launch of ISS)')
    if end < start:
        raise ValueError('End date must be after start date')
    if end > datetime.now().astimezone(pytz.utc):
        raise ValueError('End date must be before now')
    if args.db is None:
        db = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ISS_TLE_DB.nc') # default, use the default Zarya storage
    elif not os.path.isfile(): # use the specified database
        raise OSError(f'{args.db} is not a file.')
    elif not args.db.endswith('.nc'):
        raise TypeError(f'Database file {args.db} must be a netCDF file.')
    else:
        db = args.db
    
    tstamps = []
    cond = True
    idx = 0
    while cond:
        next = datetime.utcfromtimestamp(start.timestamp() + 15*60*idx)
        idx += 1
        tstamps.append(next)
        if next > end.replace(tzinfo=None):
            break
    out = get_raw_tle_from_tstamp(tstamps)
    ds = get_tle_ds(out)
    oname = f'ISS_TLE_DBPART_{datetime.now().strftime("%Y%d%m")}.nc'
    if os.path.exists(oname): # allow for multiple runs in a day
        count = 1
        oname = oname.replace('.nc', f'_{count}.nc')
        while os.path.exists(oname):
            oname = oname.replace(f'_{count}.nc', f'_{count+1}.nc')
            count += 1
    ds.to_netcdf(oname) # save in calling directory
    if os.path.exists(db):
        ods = xr.load_dataset(db)
        # back up current database to its location
        shutil.copy(db, db.replace('.nc', f'_{datetime.now().strftime("%Y%d%m_%H%M%S")}.nc'))
        nds = xr.concat([ds, ods], dim='timestamp')
    else:
        nds = ds
    nds = nds.drop_duplicates(dim=..., keep='first')
    nds = nds.sortby('timestamp')
    nds.to_netcdf(db)
# %%
if __name__ == '__main__':
    build_tledb_exec_fn()