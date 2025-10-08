#!/bin/env python3

import datetime as dt
import multiprocessing as mp
import subprocess as sp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ampte_fits import read_fits_file_no_cal, read_fits_rates, \
    get_hk, add_kp_dst, calculate_range_counts, read_fits_file_only_cal, get_me

# must use fork on macos
mp.set_start_method('fork')

idx = pd.IndexSlice
_ampte_path = Path('/Users/selasley/Documents/AMPTE')
_pha_nocal_file = _ampte_path / 'AMPTE_CHEM_nocal_pha.h5'
_epoch = np.datetime64('1966-01-01')
_pap = np.array([np.nan, np.nan, 14.0, 15.3, 17.0, 18.8, 21.9, 24.1])
_pap = dict(zip(range(-1, len(_pap)), _pap))

_num_processes = 10

_apl_data_files = Path("./apl_data_files")
if not _apl_data_files.exists():
    _apl_data_files.mkdir()

_hdf_data_files = Path("./awa_data")
if not _hdf_data_files.exists():
    _hdf_data_files.mkdir()


def create_ampte_chem_nocal_phas(files: list[Path]=None,
                                 h5file: Path= _hdf_data_files / 'AMPTE_CHEM_nocal_pha.h5'):
    """
    Create a blosc compressed hdf5 file containing only spins with cshk == 0,
    i.e., only non-calibration times, and without the time period
    1984:319:19:30 to 1984:325:00:18 with strange phas with a MultiIndex containing
    SpinStartTime, PAPS_lvl, DPU_Mode, TAC_Slope, PHA_Priority, L, MLTH, Voltage_Step,
    and Sector data
    """
    files = files or sorted(_apl_data_files.glob('198*fits.gz'))
    try:
        # remove days 320 - 325 that have strange PHAs
        for doy in range(320, 326):
            files.remove(_apl_data_files / f'1984_{doy}.fits.gz')
    except ValueError:
        pass
    h5file.unlink(missing_ok=True)
    pool_chunk_size = 40
    print(f'{pool_chunk_size} {dt.datetime.now():%F %T}')

    data_cols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority',
                 'L', 'MLTH', 'Voltage_Step', 'Sector', 'SSDID', 'Energy']
    with pd.HDFStore(h5file, expectedrows=1_019_010_000, mode='w', complib='blosc:zstd') as store:
        pool = mp.Pool(processes=_num_processes)
        for i in range(0, len(files), pool_chunk_size):
            mapper = pool.map(read_fits_file_no_cal, files[i:min(len(files),i+pool_chunk_size)])
            print(f'{dt.datetime.now():%F %T}: appending data from files {i}-{i+pool_chunk_size} of {len(files)}/{pool_chunk_size}')
            store.append(key='ampte_pha',
                         value=pd.concat(mapper).sort_index(),
                         complib='blosc:zstd',
                         data_columns=data_cols,
                         index=False)
            print(f'{dt.datetime.now():%F %T} appending finished, '
                  f'{files[i].name}-{files[min(len(files),i+pool_chunk_size)-1].name}, '
                  f'{h5file.stat().st_size:_}')
        pool.close()
    time.sleep(5)

    with pd.HDFStore(h5file, complib='blosc:zstd', expectedrows=1_019_010_000) as store:
        # index on SpinStartTime and Energy
        store.create_table_index(key='ampte_pha', optlevel=9, kind='full',
                                 columns=['SpinStartTime', 'Energy'])
    print(f'{dt.datetime.now():%F %T}: index created {h5file.stat().st_size:_}')


def create_ampte_chem_cal_phas(h5file: Path=_hdf_data_files / 'AMPTE_CHEM_cal_pha.h5'):
    """
    Create a blosc compressed hdf5 file containing only spins with cshk > 0,
    i.e., only calibration times, and including the time period
    1984:319:19:30 to 1984:325:00:18 with unusual phas with a MultiIndex containing
    SpinStartTime, PAPS_lvl, DPU_Mode, TAC_Slope, PHA_Priority, L, MLTH, Voltage_Step,
    and Sector data
    """
    files = sorted(_apl_data_files.glob('198*fits.gz'))
    h5file.unlink(missing_ok=True)
    pool_chunk_size = 40
    print(f'{pool_chunk_size} {dt.datetime.now():%F %T}')

    data_cols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority',
                 'L', 'MLTH', 'Voltage_Step', 'Sector', 'SSDID', 'Energy']
    with pd.HDFStore(h5file, expectedrows=1_019_010_000, mode='w', complib='blosc:zstd') as store:
        pool = mp.Pool(processes=_num_processes)
        for i in range(0, len(files), pool_chunk_size):
            mapper = pool.map(read_fits_file_only_cal, files[i:min(len(files),i+pool_chunk_size)])
            print(f'{dt.datetime.now():%F %T}: appending data from files {i}-{i+pool_chunk_size} of {len(files)}/{pool_chunk_size}')
            store.append(key='ampte_pha',
                         value=pd.concat(mapper).sort_index(),
                         complib='blosc:zstd',
                         data_columns=data_cols,
                         index=False)
            print(f'{dt.datetime.now():%F %T} appending finished, '
                  f'{files[i].name}-{files[min(len(files),i+pool_chunk_size)-1].name}, '
                  f'{h5file.stat().st_size:_}')
        pool.close()
    time.sleep(5)

    with pd.HDFStore(h5file, complib='blosc:zstd', expectedrows=1_019_010_000) as store:
        # index on SpinStartTime and Energy
        store.create_table_index(key='ampte_pha', optlevel=9, kind='full',
                                 columns=['SpinStartTime', 'Energy'])
    print(f'{dt.datetime.now():%F %T}: index created {h5file.stat().st_size:_}')


def create_ampte_chem_rates_dst_file(files: list[Path]=None,
                                     h5file: Path=_hdf_data_files / 'AMPTE_CHEM_rates.h5'):
    """
    Create a hdf5 file containing SpinStartTime, some housekeeping and space environment
    data and rate data with a RangeIndex
    """
    def set_rate_col_types(_df: pd.DataFrame) -> pd.DataFrame:
        int8cols = ['DPU_Mode', 'PAPS_lvl', 'TAC_Slope', 'PHA_Priority', 'Period_Cntr', 'Voltage_Step']
        _df[int8cols] = _df[int8cols].astype('int8')
        int32cols = ['SSD', 'FSR', 'DCR', 'TCR', 'R0', 'R1', 'R2', 'BR0', 'BR1', 'BR2', ]
        _df[int32cols] = _df[int32cols].astype('int32')
        float32cols = ['L', 'MAGLAT', 'MAGLON']
        _df[float32cols] = _df[float32cols].astype('float32')
        return _df

    h5file.unlink(missing_ok=True)
    rates_dst_file = h5file.with_name(h5file.name.replace('rates', 'rates_dst'))
    rates_dst_file.unlink(missing_ok=True)
    repack_file = _hdf_data_files / 'AMPTE_CHEM_rates_dst_repack.h5'
    repack_file.unlink(missing_ok=True)
    files = files or sorted(_apl_data_files.glob('198*fits.gz'))
    pool_chunk_size = 40  # 120  can't use E>0 filter so 120 days gets too many PHAs
    print(f'{pool_chunk_size} {dt.datetime.now():%F %T}')
    # with pd.HDFStore(h5file, complib='blosc:zstd', expectedrows=20_255_351) as store:
    with pd.HDFStore(h5file, mode='w', expectedrows=20_600_000) as store:
        pool = mp.Pool(processes=_num_processes)
        for i in range(0, len(files), pool_chunk_size):
            mapper = pool.map(read_fits_rates, files[i:min(len(files), i+pool_chunk_size)])
            print(f'{dt.datetime.now():%F %T}: appending data from files {i}-{i+pool_chunk_size} of {len(files)}/{pool_chunk_size}')
            # ratesdf = pd.concat(mapper)
            store.append(key='ampte_rates',
                         value=set_rate_col_types(pd.concat(mapper)),
                         # value=set_rate_col_types(ratesdf),
                         # value=(ratesdf
                         #          .merge(calculate_range_counts(_df=ratesdf),
                         #                 on='SpinStartTime', how='outer')
                         #          .fillna(0)
                         #          .pipe(set_rate_col_types)
                         #        ),
                         # complib='blosc:zstd',
                         data_columns=True,
                         index=False)
            print(f'{dt.datetime.now():%F %T} appending finished, '
                  f'{i} of {len(files)}/{pool_chunk_size}, '
                  f'{files[i].name}-{files[min(len(files), i + pool_chunk_size) - 1].name}, '
                  f'{h5file.stat().st_size:_}')
        pool.close()
    time.sleep(5)
    add_kp_dst(h5file=h5file, key='ampte_rates')
    print(f'{dt.datetime.now():%F %T}: kp and dst added')
    # use ptrepack to save a copy of the h5 file sorted by SpinStartTime
    cmd = ('ptrepack --sortby SpinStartTime  --checkCSI --chunkshape auto '
           f'{rates_dst_file} {repack_file}')
    sp.call(cmd.split())

    # ndxcols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'FSR',
    #            'PHA_Priority', 'L', 'MLTH', 'MAGLAT', 'MAGLON', 'Period_Cntr',
    #            'Voltage_Step', 'FSR']

    # Testing shows that indexing several columns slows down reads with a where clause
    ndxcols = ['SpinStartTime']
    print(f'{dt.datetime.now():%F %T}: creating indexes')
    with pd.HDFStore(repack_file, expectedrows=20_600_000, ) as store:
        store.create_table_index(key='ampte_rates', optlevel=9, kind='full', columns=ndxcols)
    print(f'{dt.datetime.now():%F %T}: index created {h5file.stat().st_size:_}')
    repack_file.unlink(missing_ok=True)


def create_ampte_chem_hk_dst_file(files: list[Path] = None,
                                  h5file: Path = _hdf_data_files / 'AMPTE_CHEM_hk.h5'):
    """
    Create a blosc compressed hdf5 file containing SpinStartTime, housekeeping and space
    environment data with a RangeIndex
    """
    files = files or sorted(_apl_data_files.glob('198*fits.gz'))
    h5file.unlink(missing_ok=True)
    ndxcols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope',
               'PHA_Priority', 'L', 'MLTH', 'MAGLAT', 'MAGLON']
    pool_chunk_size = 120
    key = 'ampte_hk'
    print(f'{pool_chunk_size} {dt.datetime.now():%F %T}')
    with pd.HDFStore(h5file, mode='w', complib='blosc:zstd') as store:
        pool = mp.Pool(processes=_num_processes)
        for i in range(0, len(files), pool_chunk_size):
            mapper = pool.map(get_hk, files[i:min(len(files), i + pool_chunk_size)])
            print(f'{dt.datetime.now():%F %T}: appending data from files '
                  f'{i}-{i+pool_chunk_size} of {len(files)}/{pool_chunk_size}')
            store.append(key=key,
                         value=pd.concat(mapper)
                                 .sort_values(by=['SpinStartTime', 'L'])
                                 .reset_index(drop=True),
                         data_columns=True,
                         complib='blosc:zstd',
                         index=False)
            print(f'{dt.datetime.now():%F %T} appending finished, '
                  f'{files[i].name}-{files[min(len(files),i+pool_chunk_size)-1].name}, '
                  f'{h5file.stat().st_size:_}')
        pool.close()
    time.sleep(3)
    print(f'{dt.datetime.now():%F %T}: {h5file.stat().st_size:_} creating table index')
    with pd.HDFStore(h5file, complib='blosc:zstd') as store:
        store.create_table_index(key=key, optlevel=9, columns=ndxcols, kind='full')
    print(f'{dt.datetime.now():%F %T}: index created, {h5file.stat().st_size:_}')
    add_kp_dst(h5file=h5file, key=key)
    # chunk size 60 1_585_322, 90 1_585_335, 30 1_589_752 | 120 20_612_973 int64, 120 20494236 int32


def create_ampte_chem_me_file(files: list[Path] = None,
                              h5file: Path = _hdf_data_files / 'AMPTE_CHEM_me.h5'):
    h5file.unlink(missing_ok=True)
    files = files or sorted(_apl_data_files.glob('198*fits.gz'))
    pool_chunk_size = 120
    print(f'{pool_chunk_size} {dt.datetime.now():%F %T}')
    with pd.HDFStore(h5file, complib='blosc:zstd') as store:
        pool = mp.Pool(processes=6)
        for i in range(0, len(files), pool_chunk_size):
            mapper = pool.map(get_me, files[i:min(len(files),i+pool_chunk_size)])
            print(f'{dt.datetime.now():%F %T}: appending data from files '
                  f'{i}-{i+pool_chunk_size} of {len(files)}/{pool_chunk_size}')
            store.append(key='ampte_me',
                         value=pd.concat(mapper)
                                 .sort_values(by=['SpinStartTime', 'L']),
                         data_columns=['L'],
                         complib='blosc:zstd',
                         index=False,
                         chunksize=512,
                        )
            print(f'{dt.datetime.now():%F %T} appending finished, '
                  f'{files[i].name}-{files[min(len(files),i+pool_chunk_size)-1].name}, '
                  f'{h5file.stat().st_size:_}')
        pool.close()
    time.sleep(3)
    print(f'{dt.datetime.now():%F %T}: {h5file.stat().st_size:_} creating table index')
    with pd.HDFStore(h5file, complib='blosc:zstd') as store:
        store.create_table_index(key='ampte_me', optlevel=9, columns=['L'], kind='full')
    print(f'{dt.datetime.now():%F %T}: index created, {h5file.stat().st_size:_}')


def main():
    print('Creating nocal pha file')
    create_ampte_chem_nocal_phas()
    print('Creating cal pha file')
    create_ampte_chem_cal_phas()
    print('Creating rates_dst file')
    create_ampte_chem_rates_dst_file()
    print('Creating hk_dst file')
    create_ampte_chem_hk_dst_file()
    print('Creating me file')
    create_ampte_chem_me_file()
    # create_ampte_chem_rates_dst_file(sorted(Path('data/').glob('1985_*.fits.gz')))
    # create_ampte_chem_rates_dst_file(sorted(Path('data/').glob('1984_236*.fits.gz')))


if __name__ == '__main__':
    main()
