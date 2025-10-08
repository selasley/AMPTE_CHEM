#!/bin/env python3
"""
Script to get dst data from http://wdc.kugi.kyoto-u.ac.jp/dst_final and save the
hourly data in a pandas parquet file and an excel file.
"""

import requests
import time
import pandas as pd
from io import BytesIO

days = [f'{n:2d}'.encode() for n in range(1, 32)]
dstdf = pd.DataFrame()
for year in range(1985, 1990):
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
        df[['dst']].to_parquet(f'data/df_{year}_{month}.parquet')
        dstdf = pd.concat([dstdf, df[['dst']]])
        print(f'{year}-{month} {len(df)} days')
        time.sleep(2)
dstdf.to_parquet('dst.parquet')
dstdf.to_excel('dst.xlsx')
