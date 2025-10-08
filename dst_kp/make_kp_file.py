#!/bin/env python3

"""
Script to read Kp_ap_Ap_SN_F107 files in the data directory downloaded
from ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107 and create 
a parquet file with the Kp data every three hours.  Kp1-Kp8 are the Kp 
values 'for the eight eighths of the UT day', i.e. every 3 hours
"""

import pandas as pd


kpdf = pd.DataFrame()
for year in range(1984, 1990):
    with open(f'data/Kp_ap_Ap_SN_F107_{year}.txt', 'r') as txtf:
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

kpdf.sort_index().to_parquet('Kp_1984-89.parquet')
