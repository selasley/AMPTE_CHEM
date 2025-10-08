#!/usr/bin/env python3
"""
Routines for reading AMPTE CHEM FITS files from APL
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# pandas.to_hdf requires ns datetimes
_epoch = np.datetime64('1966-01-01').astype('datetime64[ns]')
_maxtt = (pd.Timestamp('1990-1-1') - _epoch).total_seconds()
# paps levels from phaflux.for
# DATA PAP/14.0,15.3,17.0,18.8,21.9,24.1/
# _pap = np.array([np.nan, np.nan, 14.0, 15.3, 17.0, 18.8, 21.9, 24.1])
_pap = np.array([-1., -1., 14.0, 15.3, 17.0, 18.8, 21.9, 24.1, -1.0, -1.0])


def read_fits_file_no_cal(
    fits_file: Path, with_filters: bool = False, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing PHA data.  If with_filters is True filter the PHAs similar
    Trim the 1984_319 fits data to 19:20 to avoid odd PHAs later that day
    to how the phaflux program does.
    :param fits_file: Path to the file to read
    :param with_filters: filter PHAs
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            cshk = hdul[2].data['CSHK'].astype('uint8').repeat(32).ravel()
            not_cal_ndx = cshk == 0  # cshk != 0 is calibration data
            # Set times below 0 to 0 and after 1989 to 1990
            tt = hdul[4].data['TT'][not_cal_ndx].astype('float64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'][not_cal_ndx].astype('float').ravel()
            vltg_step = hdul[4].data['VS'][not_cal_ndx].astype('int8')[:, 0].ravel()
            dpu_mode = hdul[2].data['DPUMHK'].astype('int8').repeat(32)[not_cal_ndx].ravel()
            tac_slope = hdul[2].data['TAKHK'].astype('int8').repeat(32)[not_cal_ndx].ravel()
            pha_priority = hdul[2].data['PHAPHK'].astype('int8').repeat(32)[not_cal_ndx].ravel()
            paps = hdul[2].data['PAPS'].astype('int8').repeat(32)[not_cal_ndx].ravel()
            mlth = 12 / np.pi * hdul[1].data['MAGLON'].astype('float32').repeat(32)[not_cal_ndx].ravel()
            # print(fits_file, np.unique(hdul[2].data['PAPS']), np.unique(spin_start_time.shape))
            pha = hdul[4].data['PHA'][not_cal_ndx].astype('int16')

            ndxcols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority', 'L', 'MLTH', 'Voltage_Step']
            not_cal_ndx = pd.MultiIndex.from_arrays(
                [spin_start_time, paps, dpu_mode, tac_slope, pha_priority, l_value, mlth, vltg_step], names=ndxcols,
            )
            phadf = pd.DataFrame(
                index=not_cal_ndx.repeat(86),
                data=pha.reshape(-1, 5),
                columns=['Energy', 'TOF', 'Sector', 'SSDID', 'Range'],
            )
            # remove rows that are all 0 values
            phadf = phadf[phadf.sum(axis=1) > 0]
            int8_cols = ['Sector', 'SSDID', 'Range']
            phadf[int8_cols] = phadf[int8_cols].astype('int8')
            if with_filters:
                # phaflux rejects PHAs with Range = 3 or TOF = 1019
                # also remove PHAs with detector id > 3
                filt_ssdid = phadf['SSDID'] < 4  # .isin([1, 2, 3])
                filt_range = phadf['Range'] != 3
                filt_tof = phadf['TOF'] != 1019
                phadf = phadf[
                    filt_ssdid & filt_range & filt_tof
                ]
        phadf = phadf.set_index(['Sector', 'SSDID'], append=True)
        if fits_file.name == '1984_319.fits.gz':
            print('Trimming 1984_319.fits')
            phadf = phadf.sort_index().loc[:"1984-11-14 19:20"]
        if sort:
            return phadf.sort_index()
        else:
            return phadf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def read_fits_file_only_cal(
    fits_file: Path, with_filters: bool = False, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing PHA data.  If with_filters is True filter the PHAs similar
    to how the phaflux program does.
    :param fits_file: Path to the file to read
    :param with_filters: filter PHAs
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            cshk = hdul[2].data['CSHK'].astype('uint8').repeat(32).ravel()
            cal_ndx = cshk > 0  # cshk != 0 is calibration data
            if not np.any(cal_ndx):
                return pd.DataFrame()
            # Set times below 0 to 0 and after 1989 to 1990
            tt = hdul[4].data['TT'][cal_ndx].astype('int64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'][cal_ndx].astype('float').ravel()
            vltg_step = hdul[4].data['VS'][cal_ndx].astype('int8')[:, 0].ravel()
            dpu_mode = hdul[2].data['DPUMHK'].astype('int8').repeat(32)[cal_ndx].ravel()
            tac_slope = hdul[2].data['TAKHK'].astype('int8').repeat(32)[cal_ndx].ravel()
            pha_priority = hdul[2].data['PHAPHK'].astype('int8').repeat(32)[cal_ndx].ravel()
            paps = hdul[2].data['PAPS'].astype('int8').repeat(32)[cal_ndx].ravel()
            mlth = 12 / np.pi * hdul[1].data['MAGLON'].astype('float32').repeat(32)[cal_ndx].ravel()
            pha = hdul[4].data['PHA'][cal_ndx].astype('int16')

            ndxcols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority', 'L', 'MLTH', 'Voltage_Step']
            cal_ndx = pd.MultiIndex.from_arrays(
                [spin_start_time, paps, dpu_mode, tac_slope, pha_priority, l_value, mlth, vltg_step], names=ndxcols,
            )
            phadf = pd.DataFrame(
                index=cal_ndx.repeat(86),
                data=pha.reshape(-1, 5),
                columns=['Energy', 'TOF', 'Sector', 'SSDID', 'Range'],
            )
            # remove rows that are all 0 values
            phadf = phadf[phadf.sum(axis=1) > 0]
            int8_cols = ['Sector', 'SSDID', 'Range']
            phadf[int8_cols] = phadf[int8_cols].astype('int8')
            if with_filters:
                # phaflux rejects PHAs with Range = 3 or TOF = 1019
                # also remove PHAs with detector id > 3
                filt_ssdid = phadf['SSDID'] < 4  # .isin([1, 2, 3])
                filt_range = phadf['Range'] != 3
                filt_tof = phadf['TOF'] != 1019
                phadf = phadf[
                    filt_ssdid & filt_range & filt_tof
                ]
        phadf = phadf.set_index(['Sector', 'SSDID'], append=True)
        if sort:
            return phadf.sort_index()
        else:
            return phadf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def read_fits_rates(
    fits_file: Path, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing Direct Rates and Basic Rates data.
    :param fits_file: Path to the file to read
    :param sort: sort DataFrame index before returning
    :return: DataFrame with Direct Rates SSD, FSR, DCR, TCR data and an index
             with SpinStartTime, DV Step, PAPS level
    """
    try:
        with fits.open(fits_file) as hdul:
            # maxtt = (pd.Timestamp('1990-1-1') - _epoch).total_seconds()
            # Set times below 0 to 0 and above 1990 to 1990
            tt = hdul[4].data['TT'].astype('float64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'].astype('float32').ravel()
            dpu_mode = hdul[2].data['DPUMHK'].astype('int8').ravel().repeat(32)
            tac_slope = hdul[2].data['TAKHK'].astype('int8').ravel().repeat(32)
            pha_priority = hdul[2].data['PHAPHK'].astype('int8').ravel().repeat(32)
            paps = hdul[2].data['PAPS'].astype('int8').ravel().repeat(32)
            maglat = hdul[1].data['MAGLAT'].astype('float32').ravel().repeat(32)
            maglon = hdul[1].data['MAGLON'].astype('float32').ravel().repeat(32)
            vltg_step = hdul[4].data['VS'].astype('int8')[:, 0].ravel()
            period_cntr = hdul[4].data['VS'].astype('int8')[:, 1].ravel()
            pha = hdul[4].data['PHA'].astype('int16')
            dr = hdul[4].data['DR'].astype('int32')
            # sum the basic rates over all sectors to get BR0, BR1, BR2 per spin
            br = (hdul[4]
                    .data['SBR']
                    .astype('int32')
                    # reshape(-1, 32, 3) causes BR0, BR1 & BR2 to be rougly equal instead of BR0>BR1>BR2
                    # .reshape(-1, 32, 3)
                    # .sum(axis=1)
                    .reshape(-1, 3, 32)   # see DPU doc data block summary on page 33
                    .sum(axis=2)
                  )
        cols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority', 'L',
                'MAGLAT', 'MAGLON', 'Period_Cntr', 'Voltage_Step',
                'FSR', 'DCR', 'TCR', 'SSD', 'BR0', 'BR1', 'BR2']
                # the order of DRs in the FITS header documentation is wrong, see page 33 of the DPU doc
                # 'SSD', 'FSR', 'DCR', 'TCR', 'BR0', 'BR1', 'BR2']
        data = [spin_start_time, paps, dpu_mode, tac_slope, pha_priority, l_value, maglat, maglon,
                period_cntr, vltg_step,
                dr[:, 0], dr[:, 1], dr[:, 2], dr[:, 3], br[:, 0], br[:, 1], br[:, 2]]
        ratedf = pd.DataFrame(data=dict(zip(cols, data)))
        ratedf[['R0', 'R1', 'R2']] = calculate_range_counts(pha_arr=pha).to_numpy()
        if sort:
            return ratedf.sort_values(by='SpinStartTime')
        else:
            return ratedf
    except (IndexError, ValueError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def calculate_range_counts(pha_arr: np.array) -> pd.DataFrame:
    """
    Calculate Range counts in each spin for the PHAs in the same
    time range as _df.  Only use PHAs with Energy > 0 since
    R0, R1, R2 boundaries are defined by mass and mass/charge
    """
    df = pd.DataFrame(index=np.arange(len(pha_arr)).repeat(86),
                      data=pha_arr.reshape(-1, 5),
                      columns=['Energy', 'TOF', 'Sector', 'SSDID', 'Range'], )
    df.loc[df[['Energy', 'TOF', 'Sector', 'SSDID', 'Range']].sum(axis=1) == 0, 'Range'] = -1
    return (df.groupby([df.index, 'Range'])
              ['Energy']
              .count()
              .unstack(level='Range')
              .reindex(columns=[-1, 0, 1, 2, 3])
              [[0, 1, 2]]
              .pipe(lambda rdf: rdf.set_axis([f'R{n}' for n in rdf.columns], axis=1))
              .fillna(0)
            )


def add_kp_dst(
        h5file: Path, key: str
) -> None:
    """
    Append kp and dst values to the h5file and save a h5file with
    _dst added to the name. Create an index for the table.  Reads
    the entire h5file into memory
    """
    df: pd.DataFrame = pd.read_hdf(h5file, key=key)
    nrows = len(df)
    df['SpinStartHour'] = pd.Series(df['SpinStartTime']).dt.floor('h')
    dstdf = (pd.read_hdf('dst_kp/dst_Kp_1984-89.h5')
               .rename(columns={'Date': 'SpinStartHour'})
               .set_index('SpinStartHour')
             )

    newdf = df.merge(dstdf[['Dst']], on='SpinStartHour', how='left')
    # need to upsample Kp to hourly values for the merge to fill in values properly
    newdf = newdf.merge(dstdf[['Kp']]
                            .dropna()
                            .resample('h')
                            .ffill(),
                            on='SpinStartHour', how='left'
                        )
    del dstdf
    outfile = h5file.with_name(h5file.stem + '_dst.h5')
    outfile.unlink(missing_ok=True)
    # with pd.HDFStore(outfile, complib='blosc:zstd', expectedrows=nrows) as store:
    with pd.HDFStore(outfile, expectedrows=nrows) as store:
        store.append(key=key,
                     value=newdf[newdf.columns.drop(['SpinStartHour'])],
                     data_columns=True,
                     # complib='blosc:zstd',
                     index=False)
    ndxcols = ['SpinStartTime']
    # with pd.HDFStore(outfile, complib='blosc:zstd') as store:
    with pd.HDFStore(outfile, expectedrows=nrows) as store:
        store.create_table_index(key=key, optlevel=9, kind='full', columns=ndxcols)


def get_hk(
    fits_file: Path, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing housekeeping data.  Use Jon's values where possible.
    :param fits_file: Path to the file to read
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            # Use spin start times to match pha times
            tt = hdul[4].data['TT'].astype('float64').ravel()[::32]
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'].astype('float32').ravel()[::32]
            dpu_mode = hdul[2].data['DPUMHK'].astype('int8').ravel()
            tac_slope = hdul[2].data['TAKHK'].astype('int8').ravel()
            pha_priority = hdul[2].data['PHAPHK'].astype('int8').ravel()
            paps = hdul[2].data['PAPS'].astype('int32').ravel()
            maglat = hdul[1].data['MAGLAT'].astype('float32').ravel()
            maglon = hdul[1].data['MAGLON'].astype('float32').ravel()
            # print(fits_file, np.unique(hdul[2].data['PAPS']), np.unique(spin_start_time.shape))

            cols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority', 'L', 'MAGLAT', 'MAGLON']
            data = [spin_start_time, paps, dpu_mode, tac_slope, pha_priority, l_value, maglat, maglon]
            phadf = pd.DataFrame(data=dict(zip(cols, data)))
        if sort:
            return phadf.sort_values(by='SpinStartTime')
        else:
            return phadf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def get_me(
    fits_file: Path, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing matrix element data.  Since there are repeated times in
    the data and the times are not monotonic include L so we can match
    housekeeping and me spinstarttimes & L.
    :param fits_file: Path to the file to read
    :param sort: sort DataFrame index before returning
    :return: DataFrame with uncompressed matrix element data and an index with
             CycleStartTime,DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            tt = hdul[4].data['TT'].astype('float64').ravel()[::32]
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'].astype('float32').ravel()[::32]
            me = hdul[3].data['ME'].astype('int32')

            medf = (
                pd.DataFrame(index=spin_start_time, data=me)
                .rename_axis(index='SpinStartTime')
            )
            medf.insert(0, 'L', l_value)
        if sort:
            return medf.sort_values(by=['SpinStartTime', 'L'])
        else:
            return medf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def find_duplicate_times(
    fits_file: Path, sort: bool = False
) -> tuple[bool, pd.Series]:
    """
    Read an AMPTE CHEM FITS file from APL and test for
    duplicate SpinStartTimes
    """
    try:
        with fits.open(fits_file) as hdul:
            # Set times below 0 to 0 and above 1990 to 1990
            tt = hdul[4].data['TT'].astype('float64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = pd.Series(_epoch + tt.astype('timedelta64[s]'))
            ndx = spin_start_time.duplicated()
            if ndx.sum() > 0:
                return True, spin_start_time[ndx]
            else:
                return False, pd.Series(dtype=int)
    except (IndexError, ValueError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return False, pd.Series(dtype=int)




#################################
# Older functions no longer used when making mission data files
#################################
def read_fits_file_no_cal_84319(
    fits_file: Path, with_filters: bool = False, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing PHA data.  If with_filters is True filter the PHAs similar
    to how the phaflux program does. This version skips data after 1984-319-19:20
    :param fits_file: Path to the file to read
    :param with_filters: filter PHAs
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            cshk = hdul[2].data['CSHK'].astype('uint8').repeat(32).ravel()
            ndx = cshk == 0
            # maxtt = (pd.Timestamp('1990-1-1') - _epoch).total_seconds()
            # Set times below 0 to 0 and above 1990 to 1990
            tt = hdul[4].data['TT'][ndx].astype('int64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'][ndx].astype('float').ravel()
            vltg_step = hdul[4].data['VS'][ndx].astype('int8')[:, 0].ravel()
            dpu_mode = hdul[2].data['DPUMHK'].astype('int8').repeat(32)[ndx].ravel()
            tac_slope = hdul[2].data['TAKHK'].astype('int8').repeat(32)[ndx].ravel()
            pha_priority = hdul[2].data['PHAPHK'].astype('int8').repeat(32)[ndx].ravel()
            paps = hdul[2].data['PAPS'].repeat(32)[ndx].ravel()
            mlth = 12 / np.pi * hdul[1].data['MAGLON'].astype('float32').repeat(32)[ndx].ravel()
            # print(fits_file, np.unique(hdul[2].data['PAPS']), np.unique(spin_start_time.shape))
            pha = hdul[4].data['PHA'][ndx].astype('int16')

            ndxcols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority', 'L', 'MLTH', 'Voltage_Step']
            ndx = pd.MultiIndex.from_arrays(
                [spin_start_time, paps, dpu_mode, tac_slope, pha_priority, l_value, mlth, vltg_step], names=ndxcols,
            )
            phadf = pd.DataFrame(
                index=ndx.repeat(86),
                data=pha.reshape(-1, 5),
                columns=['Energy', 'TOF', 'Sector', 'SSDID', 'Range'],
            )
            # remove rows that are all 0 values
            phadf = phadf[phadf.sum(axis=1) > 0]
            int8_cols = ['Sector', 'SSDID', 'Range']
            phadf[int8_cols] = phadf[int8_cols].astype('int8')
            if with_filters:
                # phaflux rejects PHAs with Range = 3 or TOF = 1019
                # also remove PHAs with detector id > 3
                filt_ssdid = phadf['SSDID'] < 4  # .isin([1, 2, 3])
                filt_range = phadf['Range'] != 3
                filt_tof = phadf['TOF'] != 1019
                phadf = phadf[
                    filt_ssdid & filt_range & filt_tof
                ]
        phadf = phadf.set_index(['Sector', 'SSDID'], append=True)
        if fits_file.name == '1984_319.fits.gz':
            print('Trimming 1984_319.fits')
            phadf = phadf.sort_index().loc[:"1984-11-14 19:20"]
        if sort:
            return phadf.sort_index()
        else:
            return phadf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def read_fits_file(
    fits_file: Path, with_filters: bool = False, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing PHA data.  If with_filters is True filter the PHAs similar
    to how the phaflux program does.
    :param fits_file: Path to the file to read
    :param with_filters: filter PHAs
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            # maxtt = (pd.Timestamp('1990-1-1') - _epoch).total_seconds()
            # Set times below 0 to 0 and above 1990 to 1990
            tt = hdul[4].data['TT'].astype('float64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')
            l_value = hdul[4].data['L'].astype('float').ravel()
            vltg_step = hdul[4].data['VS'].astype('int8')[:, 0].ravel()
            dpu_mode = hdul[2].data['DPUMHK'].astype('int8').repeat(32).ravel()
            tac_slope = hdul[2].data['TAKHK'].astype('int8').repeat(32).ravel()
            pha_priority = hdul[2].data['PHAPHK'].astype('int8').repeat(32).ravel()
            paps = hdul[2].data['PAPS'].repeat(32).ravel()
            mlth = 12 / np.pi * hdul[1].data['MAGLON'].astype('float32').repeat(32).ravel()
            # print(fits_file, np.unique(hdul[2].data['PAPS']), np.unique(spin_start_time.shape))
            pha = hdul[4].data['PHA'].astype('int16')

            ndxcols = ['SpinStartTime', 'PAPS_lvl', 'DPU_Mode', 'TAC_Slope', 'PHA_Priority', 'L', 'MLTH', 'Voltage_Step']
            ndx = pd.MultiIndex.from_arrays(
                [spin_start_time, paps, dpu_mode, tac_slope, pha_priority, l_value, mlth, vltg_step], names=ndxcols,
            )
            phadf = pd.DataFrame(
                index=ndx.repeat(86),
                data=pha.reshape(-1, 5),
                columns=['Energy', 'TOF', 'Sector', 'SSDID', 'Range'],
            )
            # remove rows that are all 0 values
            phadf = phadf[phadf.sum(axis=1) > 0]
            int8_cols = ['Sector', 'SSDID', 'Range']
            phadf[int8_cols] = phadf[int8_cols].astype('int8')
            if with_filters:
                # phaflux rejects PHAs with Range = 3 or TOF = 1019
                # also remove PHAs with detector id > 3
                filt_ssdid = phadf['SSDID'] < 4  # .isin([1, 2, 3])
                filt_range = phadf['Range'] != 3
                filt_tof = phadf['TOF'] != 1019
                phadf = phadf[
                    filt_ssdid & filt_range & filt_tof
                ]
        phadf = phadf.set_index(['Sector', 'SSDID'], append=True)
        if sort:
            return phadf.sort_index()
        else:
            return phadf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def read_fits_file_schk(
        fits_file: Path, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing HK data from hdul[2].data['SCHK']
    SCHK3.0 Error word (normally zero, and that seems to be the case)
    SCHK3.1 Top 4 bits Tac Scope (top bit always 1, next 3 bits 000=nominal, 010-Barium)
    SCHK3.2 E-calibration (normally zero)
    SCHK3.3 Top 4 bits T-calibration, Bottom 4 bits MCP level: top bit 1=0n,0=off, bottom 3 bits are level.
    SCHK3.4 Top 4 bits Trigger Mode  Bottom 4 bits Subframe sequence
    SCHK4.0 Calibration and emergency status (normally 0)
    SCHK4.1 Top 3 bits PAPS level, bit 4 PAPS on/off, Bit 3:Negative DPPS on=1. Bit 2: Pos DPPS on=1, Bit 1: MCP on. Bit 0: automatic HV on
    SCHK4.2 Top 3 bits DV mode (seems to be normally 4) bit 4:0=Automatic stepping, 1 = DV controlled by command
    SCHK4.3 top 2 bits:  DPU mode, bits 4-5: Automatic calibration (0x=periodic, 10=off, 11=on), bits 1-2: Range 01-normal, 10=Li, 11=Ba), bit 0 not used
    :param fits_file: Path to the file to read
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, PAPS level in kv, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            schk = hdul[2].data['SCHK'].astype('uint8')
            schkdf = pd.DataFrame(
                index=pd.to_datetime(
                    hdul[1].data['ST'].ravel(), unit='s', origin='1966-01-01'
                ),
                data={
                    'Error': schk[:, 3, 0].ravel(),
                    'TAC_Slope': (schk[:, 3, 1] & 0x70) >> 4,
                    'E-cal': schk[:, 3, 2],
                    'T-cal': (schk[:, 3, 3] & 0xF0) >> 4,
                    'TCAL_On?': (schk[:, 3, 3] & 0x08) >> 3,
                    'MCP_Lvl': schk[:, 3, 3] & 0x07,
                    'Trigr': (schk[:, 3, 4] & 0xF0) >> 4,
                    'SubFrame': schk[:, 3, 4] & 0x0F,
                    'CalEmrgncy': schk[:, 4, 0],
                    'PAPS_Lvl': (8 - ((schk[:, 4, 1] & 0xE0) >> 5)).astype('int8'),
                    'PAPS_On': (schk[:, 4, 1] & 0x10) >> 4,
                    'DPPS_Neg_On': (schk[:, 4, 1] & 0x8) >> 3,
                    'DPPS_Pos_On': (schk[:, 4, 1] & 0x4) >> 2,
                    'MCP_ON?': (schk[:, 4, 1] & 0x2) >> 1,
                    'Auto_HV': schk[:, 4, 1] & 0x1,
                    'DV_Mode': (schk[:, 4, 2] & 0xE0) >> 5,
                    'Stepping': (schk[:, 4, 2] & 0x10) >> 4,
                    'DPU_Mode': (schk[:, 4, 3] & 0xC0) >> 6,
                    'AutoCal': (schk[:, 4, 3] & 0x30) >> 4,
                    'Range': (schk[:, 4, 3] & 0x06) >> 1,
                },
            )
            # set PAPS Off values to -1 and Level 7 values to 8
            schkdf.loc[schkdf['PAPS_On'] != 1, 'PAPS_Lvl'] = -1
        if sort:
            return schkdf.sort_index()
        else:
            return schkdf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


def get_phas(
    fits_file: Path, with_filters: bool = False, sort: bool = False
) -> pd.DataFrame:
    """
    Read an AMPTE CHEM FITS file from APL and return a pandas DataFrame
    containing PHA data.  If with_filters is True filter the PHAs similar
    to how the phaflux program does.
    :param fits_file: Path to the file to read
    :param with_filters: filter PHAs
    :param sort: sort DataFrame index before returning
    :return: DataFrame with PHA data and an index with SpinStartTime,
             DPPS Step, Sector and SSDID
    """
    try:
        with fits.open(fits_file) as hdul:
            # maxtt = (pd.Timestamp('1990-1-1') - _epoch).total_seconds()
            tt = hdul[4].data['TT'].astype('float64').ravel()
            tt = np.where(tt >= 0, tt, 0)
            tt = np.where(tt < _maxtt, tt, _maxtt)
            spin_start_time = _epoch + tt.astype('timedelta64[s]')

            l_value = hdul[4].data['L'].astype('float32').ravel()
            vltg_step = hdul[4].data['VS'].astype('int8')[:, 0].ravel()
            # print(fits_file, np.unique(spin_start_time.shape))
            pha = hdul[4].data['PHA'].astype('int16')

            ndxcols = ['SpinStartTime', 'L', 'Voltage_Step']
            ndx = pd.MultiIndex.from_arrays(
                [spin_start_time, l_value, vltg_step], names=ndxcols,
            )
            phadf = pd.DataFrame(
                index=ndx.repeat(86),
                data=pha.reshape(-1, 5),
                columns=['Energy', 'TOF', 'Sector', 'SSDID', 'Range'],
            )
            # remove rows that are all 0 values
            phadf = phadf[phadf.sum(axis=1) > 0]
            int8_cols = ['Sector', 'SSDID', 'Range']
            phadf[int8_cols] = phadf[int8_cols].astype('int8')
            if with_filters:
                # phaflux rejects PHAs with Range = 3 or TOF = 1019
                # also remove PHAs with detector id > 3
                filt_ssdid = phadf['SSDID'] < 4  # .isin([1, 2, 3])
                filt_range = phadf['Range'] != 3
                filt_tof = phadf['TOF'] != 1019
                phadf = phadf[
                    filt_ssdid & filt_range & filt_tof
                ]
        phadf = phadf.set_index(['Sector', 'SSDID'], append=True)
        if sort:
            return phadf.sort_index()
        else:
            return phadf
    except (IndexError, NotImplementedError, pd.errors.OutOfBoundsDatetime) as e:
        print(f'Error processing {fits_file}, error: {e}')
        return pd.DataFrame()


#                    REN = FLOAT(EN)
#                    EKEV = REN*1500/1023
# c       check that this event is the correct detector(s)
# C       first check for all detectors option, if so, take this event
#                         IF (IDET .NE. 0) THEN
# C       are selecting by detector, get rid of all DCR counts
#                          if (EKEV .LE. 20.) GO TO 150
# c       now check for one detector option, if equal, take this event
#                           IF (DET .NE. IDET) THEN
# C       check for except one detector option, if not, skip this one
#                             if (idet .gt. 0) go to 150
# c         if they match, we don't want this event
#                             IF (ABS(IDET) .EQ. DET) GO TO 150
#                           END IF   !OVER ONE DETECTOR
#                         END IF   !OVER ALL DETECTORS
#                    RCNT(ISPIN,RNG,SECT) = RCNT(ISPIN,RNG,SECT) + 1      !RANGE COUNTER
#                    IF (RNG.EQ.3) GO TO 150                      !ILLEGAL EVENT
# C   CONVERT TO KEV AND NANOSECONDS.  USE DPU ENERGY-CH # CONVERSION IN
# C   ALL CASES. Exclude events in TOF ch# 1019 (5-dec-85).
#                    IF (TOF.EQ.1019) GO TO 150
#                    RTOF = FLOAT(TOF)
#                    CALL CHANNEL_TO_NS(RTOF,TNS)
# Cd                 WRITE(6,125) EN,TOF,EKEV,TNS
# Cd125              FORMAT(X,2I8,2F8.2)
# C  GET MASS AND MASS PER CHARGE
#                    IF (EKEV.LE.20.) THEN
#                            M = 0.0
#                            GO TO 130
#                    END IF
#                    IF (TNS.EQ.0) THEN
#                            MPQ = 0.
#                            M = 0.
#                            GO TO 150
#                    END IF
#                    CALL MASS(EKEV,TNS,M,'S','N')
# 130                CALL MASSPERQ(TNS,M,VS(1),PAPS,MPQ,'S','Y')
# ...
# 150    CONTINUE
