import struct
import zipfile
from enum import IntEnum
from pathlib import Path

import pandas as pd


def read_data_file(filename) -> pd.DataFrame:
    # read the text data in the file filename

    if filename.endswith('.xlsb'):
        phadf = read_excel_workbook(filename)
    elif zipfile.is_zipfile(filename):
        cols = 'MPHA.GSTICS.IEEETimeStr EDB DV SE ID SD ECH TCH ST SP R MPQ MASS2'.split()
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            phafile = [filename for filename in zip_ref.namelist() if filename[0] != '_']
            with zip_ref.open(phafile[0]) as geotail_data:
                cols = geotail_data.readline()[1:].decode().split()
                phadf = pd.read_csv(geotail_data, header=None, index_col=[0], names=cols, sep=r'\s+', comment='#',)
        phadf = phadf.set_index(pd.to_datetime(phadf.index, format='%Y-%jT%H:%M:%S.%f'))
    elif '1993012033' in filename:
        phadf = pd.read_csv(filename, index_col=[0], sep='\t')
        phadf = phadf.set_index(pd.to_datetime(phadf.index))
    elif filename.endswith('.gz'):
        phadf = pd.read_csv(filename, index_col=[0, 1, 2, 3, 4], sep=r'\s+')
        phadf = phadf.set_index(pd.to_datetime([f'{y}:{d}:{h}:{m}:{s}' for y, d, h, m, s in phadf.index],
                                               format='%Y:%j:%H:%M:%S.%f'))
    else:
        phadf = pd.read_parquet(filename)

    return (
        phadf
        .sort_index()
        .dropna()
    )


def read_excel_workbook(filename) -> pd.DataFrame:
    # read the binary excel workbook .xlsb file at filename and return a
    # dataframe with a DateTimeIndex from the #MPHA.GSTICS.IEEETimeStr column
    # and float columns of MPQ and MASS
    return (
        pd.read_excel(filename, header=0, usecols=[0, 1, 2], )
        .dropna()
        .rename(columns={'#MPHA.GSTICS.IEEETimeStr': 'MPHA_GSTICS_IEEETimeStr'})
        .query('MPHA_GSTICS_IEEETimeStr.str[0] != "#"')
        .set_index('MPHA_GSTICS_IEEETimeStr')
        .pipe(lambda df_: df_.set_index(pd.to_datetime(df_.index, format='%Y-%jT%H:%M:%S.%f')))
    )
