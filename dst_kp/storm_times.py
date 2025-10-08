"""
Methods for getting dst and Kp data and creating dataFrames with the
dst and kp values  The values are used to create time ranges for the
ampte_phas web app
"""
import datetime as dt
import sys
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

_data_path = Path('/Volumes/Dagahra/AMPTE/dst_kp/data')


def get_dst_data() -> pd.DataFrame:
    """
    Script to get dst data from http://wdc.kugi.kyoto-u.ac.jp/dst_final and save the
    hourly data in a pandas parquet file and an excel file.
    """
    days = [f'{n:2d}'.encode() for n in range(1, 32)]
    dstdf = pd.DataFrame()
    for year in range(1984, 1990):
        for month in range(1, 13):
            req = requests.get(
                f'http://wdc.kugi.kyoto-u.ac.jp/dst_final/{year}{month:02d}/index.html'
            )
            if not req.ok:
                continue
            data = BytesIO(
                b'\n'.join(
                    [
                        line
                        for line in req.content.replace(b'-', b' -').split(b'\n')
                        if len(line) > 1 and line[:2] in days
                    ]
                )
            )
            df = pd.read_csv(data, delim_whitespace=True, header=None).melt(
                id_vars=[0], var_name="hour", value_name="dst"
            )
            df = df.set_index(
                pd.to_datetime(
                    f"{year}-{month:02d}-"
                    + df[0].astype(str)
                    + "-"
                    + (df["hour"] - 1).astype(str),
                    format="%Y-%m-%d-%H",
                )
            )
            df[['dst']].to_parquet(_data_path / f'df_{year}_{month}.parquet')
            dstdf = pd.concat([dstdf, df[['dst']]])
            print(f'{year}-{month} {len(df)} days')
            time.sleep(2)
    return dstdf.sort_index()


def get_kp_data() -> pd.DataFrame:
    """
    Script to read Kp_ap_Ap_SN_F107 files in the data directory downloaded
    from ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107 and create
    a parquet file with the Kp data every three hours.  Kp1-Kp8 are the Kp
    values 'for the eight eighths of the UT day', i.e. every 3 hours
    """
    kpdf = pd.DataFrame()
    for year in range(1984, 1990):
        with open(_data_path / f'Kp_ap_Ap_SN_F107_{year}.txt', 'r') as txtf:
            while (line := txtf.readline())[:4] != '#YYY':
                pass
            cols = line[1:].split()[:15]
            df = pd.read_csv(txtf, usecols=cols, names=cols, delim_whitespace=True)
        df.index = pd.to_datetime(
            df['YYY']
            .astype(str)
            .str.cat(df['MM'].astype(str), sep='-')
            .str.cat(df['DD'].astype(str), sep='-')
        )
        df = (
            df.loc[:, 'Kp1':'Kp8']
              .rename(columns={f'Kp{n}': (n - 1) * 3 for n in range(1, 9)})
              .melt(ignore_index=False, value_name='Kp', var_name='hour')
        )
        df = df.set_index(df.index + pd.to_timedelta(df['hour'], unit='hour'))
        kpdf = pd.concat([kpdf, df[['Kp']]])
    return kpdf.sort_index()


def get_start_end_times(storm_times_df: pd.DataFrame,
                        chunk_size: dt.timedelta) -> pd.DataFrame:
    """
    Create a list of start and end times, consolidating times chunk_size apart
    :param storm_times_df: dataFrame with a DatetimeIndex containing dst values
    :param chunk_size: timedelta used to consolidate times
    :return: list of start and end datetimes
    """
    #
    date1 = date2 = storm_times_df.iloc[0, 0]
    in_run = False
    run_count = 0
    storm_ranges = []
    for _, row in storm_times_df.iloc[1:].reset_index().iterrows():
        num_chunks = row['delt']
        if in_run:
            in_run = num_chunks == chunk_size
            if not in_run:
                storm_ranges.append([date1, date2])
                date1 = date2 = row['Date']
                run_count = 0
            else:
                run_count += 1
                date2 = row['Date']
            continue
        if num_chunks == chunk_size:
            in_run = True
            date2 = row['Date']
            run_count = 1
        else:
            storm_ranges.append([date1, date2])
            date1 = date2 = row['Date']
    return pd.DataFrame(data=storm_ranges, columns=['Start', 'End'])


def create_time_ranges(dstdf: pd.DataFrame,
                       chunk_size: dt.timedelta,
                       endfmt='%Y-%j-%H59') -> tuple[str, str]:
    storm_times_df = pd.DataFrame(data=dstdf.index, columns=['Date'])
    storm_times_df['delt'] = storm_times_df['Date'].diff()
    time_ranges_df = get_start_end_times(storm_times_df, chunk_size=chunk_size)
    begin = (time_ranges_df.iloc[0, 0] - chunk_size).date()
    storm_str = ''
    nonstorm_str = ''
    for _, row in time_ranges_df.iterrows():
        start = row['Start'] - chunk_size
        end = row['End'] + chunk_size
        storm_str += f'{row["Start"]:%Y-%j-%H%M} : {dt.datetime.strftime(row["End"], endfmt)}\n'
        nonstorm_str += f'{begin:%Y-%j-%H%M} : {dt.datetime.strftime(start, endfmt)}\n'
        begin = end
    return storm_str, nonstorm_str


def main():
    # dstdf = get_dst_data()
    # dstdf.to_parquet(_data_path / 'dst_1984-89.parquet')
    # dstdf.to_excel(_data_path / 'dst_1984-89.xlsx')
    # kpdf = get_kp_data()
    # kpdf.to_parquet(_data_path / 'Kp_1984-89.parquet')
    # dstdf.join(kpdf, how='left').to_excel(_data_path / 'dst_kp_1984-89.xlsx')

    dstdf = pd.read_parquet(_data_path / 'dst_1984-89.parquet')
    # dstdf = dstdf.groupby(dstdf.index.date).min()
    # dstdf.index = pd.to_datetime(dstdf.index)
    # chunk_size = dt.timedelta(days=1)
    chunk_size = dt.timedelta(hours=1)
    # storm_str, nonstorm_str = create_time_ranges(dstdf=dstdf.loc["1985-1-1":"1989-1-14", :].query('dst < -50'),
    #                                              chunk_size=chunk_size)
    # print(f'Time ranges with dst ≥ -50\n{nonstorm_str}')
    # print(f'\nTime ranges with dst < -50\n{storm_str}')
    storm_str, nonstorm_str = create_time_ranges(dstdf=dstdf.loc["1985-1-1":"1989-1-14", :].query('dst <= 0'),
                                                 chunk_size=chunk_size)
    print(f'Time ranges with dst > 0\n{nonstorm_str}')
    print(f'\nTime ranges with dst ≤ 0\n{storm_str}')


if __name__ == '__main__':
    sys.exit(main())
