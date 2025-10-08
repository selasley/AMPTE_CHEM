"""
Class for plotting the various types of AMPTE CHEM plots
"""
import datetime as dt
import logging
import os
import re
import textwrap
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Union, Optional

import chem_pha_converters as cpc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from more_itertools import ichunked
from openpyxl import load_workbook

from ampte_plot_params import PlotParams
from vmidl_cmap import vmidl_cmap

# chunksize param for pd.read_hdf  6e6 is ok for 32GB of RAM
_chunk_size = os.environ.get('HDF_CHUNKSIZE', 6_000_000)

# hdf data file Paths
_me_file = Path('data/AMPTE_CHEM_me.h5')      # key='ampte_me'
_hk_file = Path('data/AMPTE_CHEM_hk_dst.h5')  # key='ampte_hk'
_rates_file = Path('data/AMPTE_CHEM_rates_dst.h5')  # key='ampte_rates'
_pha_file = Path('data/AMPTE_CHEM_nocal_pha.h5')    # key='ampte_pha'
_epoch = np.datetime64('1966-01-01')
# from phaflux.for
# DATA PAP/14.0,15.3,17.0,18.8,21.9,24.1/
_pap = np.array([-1., -1., 14.0, 15.3, 17.0, 18.8, 21.9, 24.1, -1.0, -1.0])

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, style='{', datefmt='%Y-%m-%d %H:%M:%S',
                    format='{asctime} {levelname} {module}:{funcName}:{lineno}: {message}')


class AmpteChemPlotter:
    def __init__(self, form_data: dict[str, str], plot_params: PlotParams):
        self.title_dict = dict()
        self.high_rate_ranges_str = ''
        self.plot_params = plot_params
        self.form_data = form_data
        if self.form_data.get('action') in ['plot_et', 'plot_et_csv']:
            self.xval = 'TOF'
            self.yval = 'Energy'
            self.logxy = False
        else:
            self.xval = 'MPQ'
            self.yval = 'Mass_amu'
            self.logxy = plot_params.mmpq_log
        self.xbins = self.xedges = self.ybins = self.yedges = None
        self.num_phas_plotted = 0
        self.total_phas = 0
        self.filters = self.make_filters()
        self.rate_filter = self.form_data['rate_filt'][0] != 'None'
        self.df, self.hist2d, self.querystr = self.get_data()
        self.title = self.form_data['plot_title'] or self.make_title()

    # @st.experimental_memo(ttl=60)
    def get_data(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
        """
        Read data from the appropriate hdf file based on form_data['action']
        and the filters in form_data or from the Dst_Kp file if plotting a
        Dst or Kp trend
        :return: DataFrame with requested data or None, 2d histogram np array or None, dict of filters and the filter string
        """
        action = self.form_data['action']
        logging.info(f'getting data {action}')
        df = hist2d = None
        if action in ['get_phas', 'plot_et', 'plot_mmpq']:
            if self.total_days() < 3:
                df, hist2d, querystr = self.get_pha_data()
            else:
                df, hist2d, querystr = self.get_pha_data_chunked()
        elif action in ['plot_et_csv', 'plot_mmpq_csv']:
            df, hist2d, querystr = self.read_2dhist_files(action=action)
        elif action in ['get_mes', 'plot_mes', 'plot_me_trend']:
            df, querystr = self.get_me_data()
        elif action == 'plot_trend':
            df, querystr = self.get_trend_data()
        else:
            df = pd.DataFrame()
            querystr = 'Invalid plot type'
        logging.info(f'Data gotten')
        return df, hist2d, querystr

    def read_2dhist_xlsx(self, xfile: BytesIO) -> tuple[str, str, pd.DataFrame]:
        wb = load_workbook(xfile, read_only=True, data_only=True)
        sheet = wb.active
        querystr = sheet['A1'].value[2:]
        title = sheet['A2'].value
        df = pd.read_excel(xfile, header=3, index_col=0)
        return querystr, title, df

    def read_2dhist_csv(self, csv_file: BytesIO) -> tuple[str, str, pd.DataFrame]:
        querystr = csv_file.readline()[2:-1].decode('utf-8')
        title = csv_file.readline().decode('utf-8').strip()
        # csv files from older AWA version only have 2 comment lines at the top
        loc = csv_file.tell()
        line = csv_file.readline().decode()
        csv_file.seek(loc)
        if line[0] == '#':
            rows_to_skip = 1
        else:
            rows_to_skip = 0
            if title == '#':
                title = '# PLOTTITLE: ' + querystr
                title = textwrap.fill(title, width=132)
        df = pd.read_csv(csv_file, skiprows=rows_to_skip, index_col=0, )
        df.columns = df.columns.astype(float)
        return querystr, title, df

    def sum_2dhist_files(self):
        sumdf = pd.DataFrame()
        self.total_phas = 0
        for hist_file in self.plot_params.csv_files:
            try:
                if hist_file.name.endswith('csv'):
                    querystr, title, df = self.read_2dhist_csv(hist_file)
                else:
                    querystr, title, df = self.read_2dhist_xlsx(hist_file)
                totals = re.findall(r'([\d,]+) PHAs of ([\d,]+) total', title)
                if len(totals) == 1:
                    plotted, total = [int(s.replace(',', ''))
                                      for s in totals[0]]
                    self.total_phas += total
            except ValueError:
                return '', '', pd.DataFrame()
            sumdf = sumdf.add(df, fill_value=0)
        # sumdf.columns = sumdf.columns.astype(float)
        self.num_phas_plotted = int(sumdf.loc[self.plot_params.ylo:self.plot_params.yhi,
                                              self.plot_params.xlo:self.plot_params.xhi]
                                          .sum()
                                          .sum()
                                    )
        normalized = ' Normalized' if 'Normalized' in title else ''
        title = re.sub(rf'[0-9,]+{normalized} PHAs of [0-9,]+ total',
                       f'{self.num_phas_plotted:,}{normalized} PHAs of {self.total_phas:,} total',
                       title)
        return querystr, title, sumdf

    def read_2dhist_files(self, action: str):
        querystr, title, df = self.sum_2dhist_files()
        if df.empty:
            return pd.DataFrame(), np.array([]), ''
        if 'PLOTTITLE' in title[:15]:
            self.form_data['plot_title'] = title[13:].replace('<lf>', '\n')
        xbins = df.columns.to_numpy(dtype=float)
        ybins = df.index.to_numpy(dtype=float)
        if action == 'plot_mmpq_csv':
            xinc = (xbins[1:] / xbins[:-1]).mean()
            self.xbins = np.hstack((xbins, xbins[-1] * xinc))
            yinc = (ybins[:-1] / ybins[1:]).mean()
            self.ybins = np.hstack((ybins[0] * yinc, ybins))
        else:
            xinc = (xbins[1:] - xbins[:-1]).mean()
            self.xbins = np.hstack((xbins, xbins[-1] * xinc))
            yinc = (ybins[:-1] - ybins[1:]).mean()
            self.ybins = np.hstack((ybins[0] + yinc, ybins))
        return df,  df.to_numpy(), querystr

    def pha_calculations(self, df: pd.DataFrame, inplace: bool = False):
        """
        Use DPU routines to calculate TOF ns and E keV from the TOF and energy
        channels.  Use those values to calculate Mass and Mass per charge using
        Lynn's MASS and MASSPERQ subroutines
        :param df: pandas DataFrame containing PHA data
        :param inplace: modify df if True else modify a copy of df
        :return: pandas DataFrame containg the coloums from df along with
                 columns for TOF in ns, Energy in keV, Mass in amu and Mass per charge
        """
        if inplace:
            phas = df
        else:
            phas = df.copy()
        phas['TOF_ns'] = phas['TOF'].array * 320.0 / 1023.0
        phas['EKEV'] = phas['Energy'].array * 1500.0 / 1023.0
        phas['Mass_amu'] = np.vectorize(cpc.mass)(
            phas['EKEV'].values, phas['TOF_ns'].values
        )
        phas['Mass_amu'] = phas['Mass_amu'].where(phas['EKEV'] > 20, other=0)
        phas['MPQ'] = np.vectorize(cpc.massperq)(
            phas['TOF_ns'].values,
            phas['Mass_amu'].values,
            phas.index.get_level_values('Voltage_Step').values,
            phas.index.get_level_values('PAPS_kv').values,
        )
        cols = ['Mass_amu', 'MPQ']
        phas[cols] = phas[cols].where(phas['TOF_ns'] > 0, other=0)

        #     this uses CHANNEL_TO_KEV from etconf.for.  phaflux uses the DPU algorithm instead
        #     for ssd in range(1, 4):
        #         phas.loc[idx[:, :, :, :, :, ssd], 'Energy_kev'] = \
        #             np.vectorize(partial(cpc.channel_to_kev, ssd))(phas.loc[idx[:, :, :, :, :, ssd], 'Energy'].values)
        #     phas['Energy_kev'] = phas['Energy_kev'].where(phas['Energy_kev']>= 0, other=0)
        #     phas['Mass_amu'] = np.vectorize(cpc.mass)(phas['Energy_kev'].values, phas['TOF_ns'].values)
        #     phas['Mass_amu'] = phas['Mass_amu'].where(phas['Energy_kev'] > 20, other=0)
        #     phas['MPQ'] = np.vectorize(cpc.massperq)(phas['TOF_ns'].values,
        #                                              phas['Mass_amu'].values,
        #                                              phas.index.get_level_values('Voltage_Step').values,
        #                                              phas.index.get_level_values('PAPS_kv').values)
        return phas

    def total_days(self) -> int:
        return sum([(stop - start).days
                    for start, stop in zip(*[iter(self.extract_times())] * 2)])

    def get_pha_data_chunked(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
        filters = self.filters
        skipped_filters = ['dst_filter', 'kp_filter', 'sumbr_filter', 'br2tcr_filter', 'rltbr_filter',
                           'br0max_filter', 'br1max_filter', 'br2max_filter']
        filter_rbr = any([filters['sumbr_filter'], filters['br2tcr_filter'], filters['rltbr_filter'],
                          filters['br0max_filter'], filters['br1max_filter'], filters['br2max_filter'],])
        querystr = ' and '.join(filt for param, filt in filters.items()
                                if filt and param not in skipped_filters)
        hist2d = None
        if (calc_2d_hist := self.form_data['action'] != 'get_phas'):
            raw_phas = None
        else:
            raw_phas = pd.DataFrame()
        total_phas = 0
        if (et_plot := self.form_data['action'] == 'plot_et'):
            etquery = (f'({self.plot_params.ylo} <= Energy <= {self.plot_params.yhi}) and '
                       f'({self.plot_params.xlo} <= TOF <= {self.plot_params.xhi})')
        if (mmpq_plot := self.form_data['action'] == 'plot_mmpq'):
            mmquery = (f'({self.plot_params.ylo} <= Mass_amu <= {self.plot_params.yhi}) and '
                       f'({self.plot_params.xlo} <= MPQ <= {self.plot_params.xhi})')
        for df in pd.read_hdf(_pha_file, where=querystr, iterator=True, chunksize=_chunk_size):
            # print(f'{dt.datetime.now()}df size is {df.shape}  {df.index.get_level_values("SpinStartTime").min()} {df.index.get_level_values("SpinStartTime").max()} ')
            if df.empty:
                print('df empty')
                continue
            logging.info(f': read chunk ending {df.index.get_level_values("SpinStartTime").max()}')
            if self.form_data['range']:
                valid_ranges = [int(pharange) for pharange in self.form_data['range'].split(',')]
                df = df.query('Range in @valid_ranges')
                if df.empty:
                    continue
            if self.form_data['ssdid']:
                valid_ids = [int(id) for id in self.form_data['ssdid'].split(',')]
                df = df.query('SSDID in @valid_ids')
                if df.empty:
                    continue
            df, hkdf = self.filter_dst_kp(df)
            if df.empty:
                continue
            if self.rate_filter:
                df = self.filter_on_rate(df=df)
                if df.empty:
                    continue
            if filter_rbr or self.plot_params.normalize:
                df = self.merge_rates(df=df)
            if filter_rbr:
                df, brquery = self.filter_rbr(df=df)
            if df.empty:
                continue
            total_phas += len(df)
            if et_plot:
                # remove PHAs outside of E-T plot boundaries
                df = df.query(etquery)
            # convert PAPS_lvl in the index to PAPS_kv so we can run pha_calculations
            df.index = (df.index.set_levels(_pap.__getitem__(1 + df.index.levels[1]), level='PAPS_lvl')
                          .set_names({'PAPS_lvl': 'PAPS_kv'})
                        )
            if not et_plot:
                df = self.pha_calculations(df=df)
            if self.form_data['action'] == 'get_phas':
                if self.form_data['dwnld_mass']:
                    mass_range = [float(mass) for mass in self.form_data['dwnld_mass'].split('-')]
                    df = df.query('@mass_range[0] <= Mass_amu <= @mass_range[1]')
                if self.form_data['dwnld_mpq']:
                    mpq_range = [float(mpq) for mpq in self.form_data['dwnld_mpq'].split('-')]
                    df = df.query('@mpq_range[0] <= MPQ <= @mpq_range[1]')
                if df.empty:
                    continue
                pha_times = df.index.get_level_values('SpinStartTime')
                hk_ndx = hkdf['SpinStartTime'].searchsorted(pha_times)
                hk_ndx[hk_ndx >= len(hkdf)] = len(hkdf) - 1
                df = df.join((hkdf.iloc[hk_ndx, 6:]
                                  .set_index(df.index)
                              ))
            if mmpq_plot:
                df = df.query(mmquery)
                if df.empty:
                    continue
            self.update_title_dict(df=df, hkdf=hkdf)
            if calc_2d_hist:
                if hist2d is None:
                    hist2d = self.make_2d_hist(df=df, xval=self.xval, yval=self.yval,
                                               logxy=self.logxy)
                else:
                    hist2d = np.add(hist2d, self.make_2d_hist(df=df, xval=self.xval, yval=self.yval,
                                                              logxy=self.logxy))
            else:
                if len(raw_phas) < 1e6:
                    raw_phas = pd.concat([raw_phas, df]).iloc[:1_000_000, :]
                # print(f'{len(raw_phas) =}, {raw_phas.memory_usage().sum() / 1024 // 1024 =}')
        self.total_phas = total_phas
        if self.form_data['ssdid']:
            querystr += f' MSSID in {valid_ids}'
        if self.form_data['range']:
            querystr += f' Range in {valid_ranges}'
        return raw_phas, hist2d, querystr

    def get_pha_data(self) -> tuple[Optional[pd.DataFrame], Optional[np.array], str]:
        filters = self.filters
        querystr = ' and '.join(filt for param, filt in filters.items() if filt and param not in
                                ['dst_filter', 'kp_filter', 'sumbr_filter', 'br2tcr_filter', 'rltbr_filter',
                                 'br0max_filter', 'br1max_filter', 'br2max_filter'])
        df = pd.read_hdf(_pha_file, where=querystr)
        if df.empty:
            return None, None, querystr
        if self.form_data['range']:
            valid_ranges = [int(pharange) for pharange in self.form_data['range'].split(',')]
            df = df.query('Range in @valid_ranges')
            if df.empty:
                return None, None, querystr
            querystr += f' Range in {valid_ranges}'
        if self.form_data['ssdid']:
            valid_ids = [int(id) for id in self.form_data['ssdid'].split(',')]
            df = df.query('SSDID in @valid_ids')
            querystr += f' MSSID in {valid_ids}'
            if df.empty:
                return None, None, querystr
        df, hkdf = self.filter_dst_kp(df)
        if df.empty:
            return None, None, querystr
        if self.rate_filter:
            df = self.filter_on_rate(df=df)
        if df.empty:
            return None, None, querystr
        filter_rbr = any([filters['sumbr_filter'], filters['br2tcr_filter'], filters['rltbr_filter'],
                          filters['br0max_filter'], filters['br1max_filter'], filters['br2max_filter'],])
        if filter_rbr or self.plot_params.normalize:
            df = self.merge_rates(df=df)
        if filter_rbr:
            df, brquery = self.filter_rbr(df=df)
        if df.empty:
            return None, None, querystr
        self.total_phas = len(df)

        if self.form_data['action'] == 'plot_et':
            # remove PHAs outside of E-T plot limits, x is TOF, y is Energy
            df = df.query('(@self.plot_params.ylo <= Energy <= @self.plot_params.yhi) and '
                          '(@self.plot_params.xlo <= TOF <= @self.plot_params.xhi)')
        # convert PAPS_lvl in the index to PAPS_kv so we can run pha_calculations
        df.index = (df.index.set_levels(_pap.__getitem__(1 + df.index.levels[1]), level='PAPS_lvl')
                      .set_names({'PAPS_lvl': 'PAPS_kv'})
                    )
        if self.form_data['action'] != 'plot_et':
            df = self.pha_calculations(df=df)
        if self.form_data['action'] == 'get_phas':
            # filter on Download Mass_amu and MPQ ranges
            if self.form_data['dwnld_mass']:
                mass_range = [float(mass) for mass in self.form_data['dwnld_mass'].split('-')]
                df = df.query('@mass_range[0] <= Mass_amu <= @mass_range[1]')
            if self.form_data['dwnld_mpq']:
                mpq_range = [float(mpq) for mpq in self.form_data['dwnld_mpq'].split('-')]
                df = df.query('@mpq_range[0] <= MPQ <= @mpq_range[1]')
            if df.empty:
                return None, None, querystr
            pha_times = df.index.get_level_values('SpinStartTime')
            hk_ndx = hkdf['SpinStartTime'].searchsorted(pha_times)
            hk_ndx[hk_ndx >= len(hkdf)] = len(hkdf) - 1
            df = df.join((hkdf.iloc[hk_ndx, 6:]
                          .set_index(df.index)
                          ))
        if self.form_data['action'] == 'plot_mmpq':
            # remove PHAs outside of M-MPQ plot limits, x is MPQ, y is Mass_amu
            df = df.query('(@self.plot_params.ylo <= Mass_amu <= @self.plot_params.yhi) and '
                          '(@self.plot_params.xlo <= MPQ <= @self.plot_params.xhi)')
            if df.empty:
                return None, None, querystr
        self.update_title_dict(df=df, hkdf=hkdf)
        if self.form_data['action'] == 'get_phas':
            hist2d = None
        else:
            hist2d = self.make_2d_hist(df=df, xval=self.xval, yval=self.yval,
                                       logxy=self.logxy)
            df = None
        logging.info(f'Data gotten')
        return df, hist2d, querystr

    def apply_extra_filters(self, df: pd.DataFrame):
        ...

    def filter_on_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        filter based on a rate in a given DV step
        """
        filters = self.filters
        invalid_filters = ['dpps_filter', 'mlth_filter', 'triples_filter',
                           'sumbr_filter', 'br2tcr_filter', 'rltbr_filter',
                           'br0max_filter', 'br1max_filter', 'br2max_filter']
        querystr = ' and '.join(filt for param, filt in filters.items()
                                if filt and param not in invalid_filters)
        hkdf = (pd.read_hdf(_rates_file, where=querystr)
                  .sort_values(by='SpinStartTime')
                )
        if filters['mlth_filter']:
            hkdf['MLTH'] = hkdf['MAGLON'] * 12 / np.pi
            hkdf = hkdf.query(filters['mlth_filter'])
        rate, step, limit = self.form_data['rate_filt']
        high_rate_times = (hkdf
                             .query(f'Voltage_Step == {step} and {rate} > {limit}')
                             .reset_index()
                             ['SpinStartTime']
        )
        if len(high_rate_times) == 0:
            return df
        three_minutes = dt.timedelta(minutes=3)
        cycle_start_times = hkdf.query('Period_Cntr == 0')['SpinStartTime']
        if high_rate_times.iloc[-1] == cycle_start_times.iloc[-1]:
        #     high_rate_times.iloc[-1] = hkdf.iloc[-1, 0]
            cycle_start_times.iloc[-1] = hkdf.iloc[-1, 0]
        hrt_ndx = high_rate_times.diff().dt.floor("T") > three_minutes
        hrt_ndx = hrt_ndx[hrt_ndx].index
        prev_cycles = (cycle_start_times.searchsorted(
                        pd.concat([high_rate_times.iloc[[0]],
                                   high_rate_times[hrt_ndx]]))
                       - (step != '63')
                       )
        next_cycles = (cycle_start_times.searchsorted(
                        pd.concat([high_rate_times[hrt_ndx - 1],
                                   high_rate_times.iloc[[-1]]]))
                       + (step == '63')
                       )
        next_cycles[-1] = min(len(cycle_start_times) - 1, next_cycles[-1])
        # print(f'prev_cycles: {len(prev_cycles)}, {prev_cycles[-1]}\n'
        #       f'next_cycles: {len(next_cycles)}, {next_cycles[-1]}\n'
        #       f'cycle_start_times: {len(cycle_start_times)}, {cycle_start_times.iloc[-1]}\n'
        #       f'{df.index.get_level_values("SpinStartTime").max()}'
        #       )
        high_rate_ranges = list(zip(cycle_start_times.iloc[prev_cycles],
                                    cycle_start_times.iloc[next_cycles]))
        self.high_rate_ranges_str += '\n'.join(f'{start:%F %T} - {stop:%F %T}'
                                               for start, stop in high_rate_ranges) + '\n'
        for range_chunk in ichunked(high_rate_ranges, 10):
            querystr = ' and '.join(f'(SpinStartTime < "{start:%F %T}" or SpinStartTime >= "{stop:%F %T}")'
                                    for start, stop in range_chunk)
            df = df.query(querystr)
        return df

    def filter_dst_kp(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read housekeeping data with all filters except triples, mlth and dpps and use
        the hourly index of dst and kp values to filter the dataFrame df.  Return
        the filtered and housekeeping dataFrames
        :param df: dataFrame containing pha data with SpinStartTime in the index
        :return: filtered df, housekeeping dataFrame
        """
        filters = self.filters
        invalid_filters = ['triples_filter', 'dpps_filter', 'mlth_filter',
                           'sumbr_filter', 'br2tcr_filter', 'rltbr_filter',
                           'br0max_filter', 'br1max_filter', 'br2max_filter']
        querystr = ' and '.join(filt for param, filt in filters.items()
                                if filt and param not in invalid_filters)
        hkdf = pd.read_hdf(_hk_file, key='ampte_hk', where=querystr)
        if filters['mlth_filter']:
            hkdf['MLTH'] = hkdf['MAGLON'] * 12 / np.pi
            hkdf = hkdf.query(filters['mlth_filter'])
        good_hours = hkdf['SpinStartTime'].dt.floor('h').unique()
        good_ndx = (df.index
                      .get_level_values('SpinStartTime')
                      .floor('h')
                      .isin(good_hours)
                    )
        df = df[good_ndx]
        # now filter hkdf using df times
        good_hours = (df.index
                        .get_level_values('SpinStartTime')
                        .floor('h')
                        .unique()
                      )
        good_ndx = hkdf['SpinStartTime'].dt.floor('h').isin(good_hours)
        return df, hkdf[good_ndx]

    def filter_rbr(self, df: pd.DataFrame):
        # df = self.merge_rates(df=df)
        querystr = ''
        q1 = q2 = q3 = ''
        filters = self.filters
        if filters['sumbr_filter']:
            querystr = f'{filters["sumbr_filter"]}zzz'
            q1 = 'sumbr; '
        if filters['br2tcr_filter']:
            querystr += f'{filters["br2tcr_filter"]}zzz'
            q1 = 'br2tcr; '
        if filters['rltbr_filter']:
            querystr += f'{filters["rltbr_filter"]}zzz'
            q1 = 'rltbr; '
        if filters['br0max_filter'] != '':
            querystr += f'{filters["br0max_filter"]}zzz'
        if filters['br1max_filter'] != '':
            querystr += f'{filters["br1max_filter"]}zzz'
        if filters['br2max_filter'] != '':
            querystr += f'{filters["br2max_filter"]}zzz'

        return df.query(' and '.join(querystr.split('zzz')[:-1])), q1 + q2 + q3

    def get_me_data(self) -> pd.DataFrame:
        # SpinStartTime is in the dataframe index of the me hdf file so use index
        # instead of SpinStartTime in the time filter
        invalid_filters = ['dpps_filter', 'triples_filter',
                           'sumbr_filter', 'br2tcr_filter', 'rltbr_filter',
                           'br0max_filter', 'br1max_filter', 'br2max_filter']
        filters = self.filters
        df = pd.concat([pd.read_hdf(_hk_file, where=filters['time_filter'])
                          .set_index('SpinStartTime'),
                        pd.read_hdf(_me_file, where=filters['time_filter'].replace('SpinStartTime', 'index'))
                          .loc[:, 0:]  # don't include L column since it's in the HK DataFrame
                        ], axis=1
                       )
        if self.form_data['action'] == 'plot_me_trend' and not self.form_data['filter_trend']:
            querystr = filters['time_filter']
        else:
            querystr = ' and '.join(filt for param, filt in filters.items()
                                    if filt and param not in invalid_filters)
        # querystr = filters['time_filter'] + (f' and {querystr}' if querystr else '')
        if df.empty:
            return df, querystr
        me0loc = df.columns.get_loc(0)
        df.insert(me0loc, 'MLTH', df['MAGLON'] * 12 / np.pi)
        df = df.query(querystr)
        self.update_title_dict(hkdf=df)
        # return df[df['PAPS_lvl'] > 0], querystr
        return df, querystr

    def get_trend_data(self) -> tuple[pd.DataFrame, str]:
        """
        Read trend data for the given time range from _rates_file and
        optionally apply the other filters
        """
        filters = self.filters
        filter_trend = self.form_data['filter_trend']
        # rate_trend = self.plot_params.trend_item in ['Voltage_Step', 'SSD', 'FSR', 'DCR', 'TCR']
        invalid_filters = ['triples_filter', 'mlth_filter', 'sumbr_filter', 'br2tcr_filter', 'rltbr_filter',
                           'br0max_filter', 'br1max_filter', 'br2max_filter']
        if filter_trend:
            querystr = ' and '.join(filt for param, filt in filters.items()
                                    if filt and param not in invalid_filters)
        else:
            querystr = filters['time_filter']
        # if self.plot_params.trend_item in ['Dst', 'Kp']:
        #     return pd.read_hdf(_dstkp_file, where=querystr.replace('SpinStartTime', 'Date')), querystr
        df: pd.DataFrame = (pd.read_hdf(_rates_file, key='ampte_rates', where=querystr)
                              .sort_values(by='SpinStartTime')
                            )
        if df.empty:
            return df, querystr
        if filter_trend and self.rate_filter:
            df = self.filter_on_rate(df=df.set_index('SpinStartTime')).reset_index()
        # if not rate_trend and 'Voltage_Step' not in querystr:
        #     df = df.query('Period_Cntr == 0')
        if filter_trend and any([filters['sumbr_filter'], filters['br2tcr_filter'], filters['rltbr_filter'],
                                 filters['br0max_filter'], filters['br1max_filter'], filters['br2max_filter'],]):
            df, brquery = self.filter_rbr(df=df)
        if self.form_data['pc0_only']:
            df = df.query('Period_Cntr == 0')
        df['MLTH'] = df['MAGLON'] * 12 / np.pi
        if filter_trend and filters['mlth_filter']:
            df = df.query(filters['mlth_filter'])
        df['PAPS_kv'] = _pap.__getitem__(1 + df['PAPS_lvl'])
        self.update_title_dict(None, hkdf=df)
        if set(self.plot_params.trend_items) & {'R0/BR0', 'R1/BR1', 'R2/BR2'}:
            # one or more trend items is a rate ratio so calculate them
            df = self.compute_br_r_ratios(df)
        return df, querystr

    def extract_times(self) -> Union[str, list[dt.datetime]]:
        """
        Return a list of datetime strings formatted YYYY:doy:HHMM-YYYY:doy:HHMM
        extracted from a string with format YYYYxDOYxHHMMxYYYYxDOYxHHMM
        on each line where x can be any character

        :return: list of start and stop datetime strings
        """
        timesstr = self.form_data['dates']
        times = []
        for timestr in timesstr.split('\n'):
            nums = re.findall(r'(\d+)+', timestr)
            numnums = len(nums)
            if numnums == 6:
                # YDH-YDH
                times.extend(['-'.join([':'.join(nums[:3]), ':'.join(nums[-3:])])])
            elif numnums == 5:
                # assume YDH-DH
                times.extend(['-'.join([':'.join(nums[:3]), ':'.join([nums[0]] + nums[-2:])])])
            elif numnums == 4:
                # assume YDH-H
                times.extend(['-'.join([':'.join(nums[:3]), ':'.join(nums[0:2] + nums[-1:])])])
            elif numnums == 1:
                nextyear = int(nums[0]) + 1
                times.append(f'{nums[0]}:001:0000-{nextyear}:001:0000')
        return [dt.datetime.strptime(ts, '%Y:%j:%H%M')
                for trange in times
                for ts in trange.split('-')]

    def times_to_filters(self) -> str:
        """
        Convert a list of datetime strings formatted YYYY:doy:HHMM-YYYY:doy:HHMM
        to a string suitable for filtering the hdf data being read
        :param time_ranges:
        :return:
        """
        # time_ranges = self.extract_times()
        # times_list = [dt.datetime.strptime(ts, '%Y:%j:%H%M')
        #               for times in time_ranges
        #               for ts in times.split('-')]
        times_list = self.extract_times()
        time_filters = [f'(SpinStartTime >= "{start:%F %T}" and SpinStartTime <= "{stop:%F %T}")'
                        for start, stop in zip(*[iter(times_list)] * 2)]
        return f"({' or '.join(time_filters)})"

    def range_to_filter(self, range_value: list[str], param: str) -> str:
        filtstr = []
        if range_value[0]:
            filtstr.append(f'{param} >= {range_value[0]}')
        if range_value[1]:
            filtstr.append(f'{param} <= {range_value[1]}')
        return ' and '.join(filtstr)

    def value_to_filter(self, value: Union[str, int], param: str) -> str:
        filt = f'{param} == {value[0]}' if value != 'Any' else ''
        return filt

    def list_to_filter(self, value: Union[str, int], param: str, valid_values: list) -> str:
        if value.strip() == '' or 'any' in value.casefold():
            # filt = f'{param} in {valid_values}'
            filt = ''
        elif len(value.split(',')) == 1:
            filt = f'{param} == {value.strip()}'
        else:
            filt = f'{param} in {[int(val) for val in value.split(",")]}'
        return filt

    def mlth_to_filter(self, mlth: list) -> str:
        if mlth[0] == mlth[1] == '':
            return ''
        mlth_float = [float(mlt[:2]) + float(mlt[2:]) / 60. if mlt else '' for mlt in mlth]
        if mlth_float[0] == mlth_float[1]:
            filt = f'MLTH == {mlth_float[0]}'
        elif mlth_float[0] != '' and mlth_float[1] == '':
            filt = f'MLTH >= {mlth_float[0]}'
        elif mlth_float[0] == '' and mlth_float[1] != '':
            filt = f'MLTH <= {mlth_float[1]}'
        elif mlth_float[0] > mlth_float[1]:
            filt = f'(MLTH >= {mlth_float[0]} or MLTH <= {mlth_float[1]})'
        else:
            filt = f'(MLTH >= {mlth_float[0]} and MLTH <= {mlth_float[1]})'
        return filt

    def make_filters(self) -> dict[str, str]:
        filters = dict()
        filters['time_filter'] = self.times_to_filters()
        filters['l_filter'] = self.range_to_filter(self.form_data['l_range'], param='L')
        filters['dpps_filter'] = self.range_to_filter(self.form_data['dpps_range'], param='Voltage_Step')
        filters['mlth_filter'] = self.mlth_to_filter(self.form_data['mlth_range'])
        filters['dst_filter'] = self.range_to_filter(self.form_data['dst_range'], param='Dst')
        filters['kp_filter'] = self.range_to_filter(self.form_data['kp_range'], param='Kp')
        valid_paps = list(range(-1, 7))
        filters['paps_filter'] = self.list_to_filter(self.form_data['paps_levels'], param='PAPS_lvl',
                                                     valid_values=valid_paps)
        valid_dpu_modes = [0, 1, 2, 3]
        filters['dpu_mode_filter'] = self.list_to_filter(self.form_data['dpu_modes'], param='DPU_Mode',
                                                         valid_values=valid_dpu_modes)
        filters['tac_slope_filter'] = self.value_to_filter(self.form_data['tac_slope'], 'TAC_Slope')
        pha_priority = self.form_data['pha_priority']
        filters['pha_priority_filter'] = f'PHA_Priority == {ord(pha_priority[0])}' if pha_priority != 'Any' else ''
        filters['triples_filter'] = 'Energy > 0' if self.form_data['triples'] else ''
        filters['sumbr_filter'] = 'BR0 + BR1 + BR2 <= DCR' if self.form_data['sumbr'] else ''
        filters['br2tcr_filter'] = 'BR2 <= TCR' if self.form_data['br2tcr'] else ''
        filters['rltbr_filter'] = 'R0 <= BR0 and R1 <= BR1 and R2 <= BR2' if self.form_data['rltbr'] else ''
        filters['br0max_filter'] = f'BR0 <= {self.form_data["br0max"]}' if self.form_data['br0max'] != '' else ''
        filters['br1max_filter'] = f'BR0 <= {self.form_data["br1max"]}' if self.form_data['br1max'] != '' else ''
        filters['br2max_filter'] = f'BR0 <= {self.form_data["br2max"]}' if self.form_data['br2max'] != '' else ''
        return filters

    def index_levels(self, df: pd.DataFrame, level: str):
        if level in df.index.names:
            values = sorted(df.index.get_level_values(level).unique())
        else:
            values = sorted(df[level].unique())
        if level == 'PHA_Priority':
            return [chr(val) for val in values if val != 0]
        # return ', '.join(str(val) for val in values)
        return values

    def index_minmax(self, df: pd.DataFrame, level: str):
        if level in df.index.names:
            min_value = df.index.get_level_values(level).min()
            max_value = df.index.get_level_values(level).max()
        else:
            min_value = df[level].min()
            max_value = df[level].max()
        # return f'{round(min_value, 2):.3g} to {round(max_value, 2):.3g}'
        return min_value, max_value

    def mlth_to_float(self, mlth: str) -> float:
        return int(mlth[:2]) + (int(mlth[2:]) / 60.) % 1

    def get_mlt_values(self, df):
        try:
            mlth_values = df.index.get_level_values('MLTH').unique()
        except KeyError:
            mlth_values = df['MLTH'].unique()
        return mlth_values

    def mltstr(self, mlt_tuple: tuple[float]) -> str:
        minmlth, maxmlth = [mlt % 24 for mlt in mlt_tuple]
        # use % 1 in case minutes rounds to 60
        min_mm = round(60 * (minmlth % 1))
        if min_mm == 60:
            min_mm = 0
            minmlth += 1
        max_mm = round(60 * (maxmlth % 1))
        if max_mm == 60:
            max_mm = 0
            maxmlth += 1
        return f'{int(minmlth):02d}:{min_mm:02d} to {int(maxmlth):02d}:{max_mm:02d}'

    def min_max_mlth(self, df):
        mlth_values = self.get_mlt_values(df)
        mltfilt = self.form_data['mlth_range']
        mltfiltlo = self.mlth_to_float(mltfilt[0] or '0000')
        mlth_values = np.where(mlth_values <= mltfiltlo, mlth_values + 24, mlth_values)
        return (mlth_values.min(), mlth_values.max())

    def min_max_dates(self, df: pd.DataFrame) -> tuple[dt.datetime, dt.datetime]:
        try:
            data_dates = df.index.get_level_values('SpinStartTime')
        except KeyError:
            try:
                data_dates = df['SpinStartTime']
            except KeyError:
                data_dates = df['Date']
        title_dates = self.title_dict.get('data_dates',
                                          [dt.datetime(2000, 1, 1), dt.datetime(1970, 1, 1)])
        return min(data_dates.min(), title_dates[0]), max(data_dates.max(), title_dates[1])

    def update_title_dict(self, df: pd.DataFrame = None, hkdf: pd.DataFrame = None):
        if (df is None or df.empty) and (hkdf is None or hkdf.empty):
            return
        if df is None or df.empty:
            df = hkdf
        title_dict = self.title_dict
        df_dict = {'data_dates': self.min_max_dates(df=df)}
        try:
            paps = self.index_levels(df=df, level="PAPS_kv")
        except KeyError:
            paps = self.index_levels(df=df, level="PAPS_lvl")
        df_dict['paps'] = set(paps) | self.title_dict.get('paps', set())
        l_values = self.index_minmax(df=df, level="L")
        tdl = title_dict.get('l', (1e9, -1e9))
        df_dict['l'] = min(l_values[0], tdl[0]), max(l_values[1], tdl[1])
        tac = title_dict.get('tac', set())
        df_dict['tac'] = tac | set(self.index_levels(df=df, level="TAC_Slope"))
        pprior = title_dict.get('pha_priority', set())
        df_dict['pha_priority'] = pprior | set(self.index_levels(df=df, level="PHA_Priority"))
        tmlt = title_dict.get('mlt', (1e9, -1e9))
        minmlth, maxmlth = self.min_max_mlth(df)
        df_dict['mlt'] = min(minmlth, tmlt[0]), max(maxmlth, tmlt[1])
        try:
            df_dict['MSSIDs'] = set(self.title_dict.get('MSSIDs', '')) | set(self.index_levels(df=df, level='SSDID'))
        except KeyError:
            pass
        try:
            df_dict['Ranges'] = set(self.title_dict.get('Ranges', '')) | set(df['Range'].unique())
        except KeyError:
            pass
        try:
            df_dict['dvs'] = self.index_minmax(df=df, level="Voltage_Step")
        except KeyError:
            pass

        self.title_dict.update(df_dict)
        self.title_dict['mltstr'] = self.mltstr(df_dict['mlt'])

        if not (hkdf is None or hkdf.empty):
            dst_range = self.title_dict.get('dst_range', [1e12, -1e12])
            self.title_dict['dst_range'] = [min(hkdf['Dst'].min(), dst_range[0]),
                                            max(hkdf['Dst'].max(), dst_range[1])
                                            ]
            kp_range = self.title_dict.get('kp_range', [1e12, -1e12])
            self.title_dict['kp_range'] = [round(min(hkdf['Kp'].min(), kp_range[0]), 2),
                                           round(max(hkdf['Kp'].max(), kp_range[1]), 2)
                                           ]

    def make_title(self) -> str:
        # df = self.df
        # if df is None or df.empty:
        #     return title + title_fltrs
        td = self.title_dict
        if td.get('data_dates') is None:
            # no data
            return "No Data"
        filters = self.filters
        inputs = self.form_data
        me_plot = inputs['action'] in ['plot_mes', 'plot_trend', 'plot_me_trend']
        title_re = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')
        date_ranges = title_re.findall(filters['time_filter'])
        data_dates = td['data_dates']
        l_values = self.title_dict.get("l")
        paps_levels = ','.join(str(pl) for pl in sorted(list(self.title_dict.get("paps"))))
        title = f'{data_dates[0]:%Y:%j:%H:%M} to {data_dates[1]:%Y:%j:%H:%M}; ' \
                f'L {l_values[0]:.2f} to {l_values[1]:.2f}; PAPS {paps_levels}'
        dvs = self.title_dict.get("dvs")
        if dvs:
            title += f'; DVS {dvs[0]} to {dvs[1]}'
        if len(date_ranges) > 2:
            title = f'{len(date_ranges) // 2} Time Ranges from {title}'
        tac = ','.join(str(tac) for tac in sorted(list(self.title_dict.get("tac"))))
        title_fltrs = f'\nTAC {tac}; '
        pprior = ','.join(str(tac) for tac in sorted(list(self.title_dict.get("pha_priority"))))
        title_fltrs += f'Priority {pprior}; '
        # title_fltrs += f'MLT: {td.get("mlt")[0]} to {td.get("mlt")[1]}; '
        title_fltrs += f'MLT {self.title_dict["mltstr"]}; '
        if (dst_range := self.title_dict.get('dst_range')):
            title_fltrs += f'Dst {dst_range[0]} to {dst_range[1]}; '
        if (kp_range := self.title_dict.get('kp_range')):
            title_fltrs += f'Kp {kp_range[0]} to {kp_range[1]}; '
        if self.rate_filter and not me_plot:
            rate, step, limit = self.form_data['rate_filt']
            title_fltrs += f'DVS{step} {rate} < {limit}; '
        sanity_filt_str = ''
        if self.filters['sumbr_filter']:
            sanity_filt_str += '∑BR≤DCR; '
        if self.filters['br2tcr_filter']:
            sanity_filt_str += 'BR2≤TCR; '
        if self.filters['rltbr_filter']:
            sanity_filt_str += 'Rn≤BRn; '
        if not me_plot:
            mssids = ','.join(f'{id}' for id in self.title_dict['MSSIDs'])
            ranges = ','.join(f'{rng}' for rng in self.title_dict['Ranges'])
            title_fltrs = f'{title_fltrs}MSSIDs {mssids}; Ranges {ranges}; '
            if inputs['triples']:
                title_fltrs += 'Triples Only; '
            title_fltrs += sanity_filt_str
            if self.form_data.get('action') == 'plot_mmpq' and (brmaxstr := self.make_brmax_filterstr()):
                title_fltrs = f'{title_fltrs}{brmaxstr}; '
            norm = 'Normalized ' if self.plot_params.normalize else ''
            title_fltrs += (f'Bin Width {self.plot_params.bin_width}; '
                            f'{self.num_phas_plotted:,.0f} {norm}PHAs of {self.total_phas:,.0f} total')
        else:
            title_fltrs += sanity_filt_str
        if self.form_data.get('action') == 'plot_me_trend':
            cutoff = 128
        else:
            #  76 is a good choice for title font size 16
            cutoff = 96
        if len(title_fltrs) > cutoff:
            # loc = title_fltrs.find(';', int(len(title_fltrs) / 2))
            loc = title_fltrs.find(';', cutoff)
            title_fltrs = title_fltrs[:loc] + '\n' + title_fltrs[loc + 1:]
        return title + title_fltrs

    def make_brmax_filterstr(self):
        brmaxs = [self.plot_params.br0max, self.plot_params.br1max, self.plot_params.br2max]
        return ', '.join(f'BR{n}≤{brmaxs[n]:.0f}' for n in range(3) if brmaxs[n] != '')

    def normalize(self, df: pd.DataFrame) -> np.array:
        if self.plot_params.normalize:
            return (df.assign(Weights=lambda _df: _df['BR0'] / _df['R0'])
                      .assign(Weights=lambda _df: _df['Weights'].where(_df['Range'] != 1, _df['BR1'] / _df['R1']))
                      .assign(Weights=lambda _df: _df['Weights'].where(_df['Range'] != 2, _df['BR2'] / _df['R2']))
                    )
        else:
            return df.assign(Weights=1)

    def merge_rates(self, df):
        cols = ['SpinStartTime', 'L', 'Voltage_Step', 'R0', 'R1', 'R2', 'BR0', 'BR1', 'BR2', 'DCR', 'TCR', ]
        ndx_names = df.index.names
        df = (pd.read_hdf(_rates_file, columns=cols, where=self.times_to_filters())
                # .set_index('SpinStartTime')
                .merge(df.reset_index(), on=['SpinStartTime', 'L', 'Voltage_Step'], how='right')
                .set_index(ndx_names)
              )
        return df

    def set_tick_params(self, ax):
        ax.tick_params(axis="x", which='major', length=7, labelsize=11, reset=True)
        ax.tick_params(axis="y", which='major', length=7, labelsize=11, reset=True)
        ax.tick_params(axis="x", which='minor', length=5, labelsize=10, reset=True)
        ax.tick_params(axis="y", which='minor', length=4, labelsize=10, reset=True)

    def make_2d_hist(self, df: pd.DataFrame, xval: str, yval: str, logxy: bool) -> np.array:
        plot_params = self.plot_params
        xlo = plot_params.xlo
        xhi = plot_params.xhi
        ylo = plot_params.ylo
        yhi = plot_params.yhi
        bin_width = plot_params.bin_width
        if isinstance(bin_width, (tuple, list)):
            xbinw = bin_width[0]
            ybinw = bin_width[1]
        else:
            xbinw = ybinw = bin_width
        if logxy:
            num = int((np.log10(xhi) - np.log10(xlo)) / xbinw) + 1
            xbins = np.geomspace(xlo, xhi, num=num)
            ybins = np.geomspace(ylo, yhi, num=num)
        else:
            xbins = np.arange(xlo, xhi + xbinw, xbinw)
            ybins = np.arange(ylo, yhi + ybinw, ybinw)

        df = self.normalize(df=df)
        hist2d, xedges, yedges = np.histogram2d(
            x=df[xval],
            y=df[yval],
            bins=[xbins, ybins],
            range=[[xlo, xhi + xbinw], [ylo, yhi + ybinw]],
            weights=df['Weights'],
        )
        self.num_phas_plotted += hist2d.sum().sum()
        self.xbins = xbins
        self.xedges = xedges
        self.ybins = ybins
        self.yedges = yedges
        return hist2d.T

    def plot_2d_hist(
            self,
            xval: str = 'TOF',
            xlabel: str = '',
            yval: str = 'Energy',
            ylabel: str = '',
            title: str = '',
            cmap=None,
            withcb: bool = True,
            grid: bool = False,
            logxy: bool = False,
            fig: Figure = None,
    ) -> Figure:
        """
        Create 2D plots of xval, yval PHA data
        :param xval: dataFrame column to use for x values.  Default is TOF
        :param xlabel: string to use for the x axis label.  Default is '' which uses xavl for the label
        :param yval: dataFrame column to use for y values.  Default is Energy
        :param ylabel: string to use for the y axis label.  Default is '' which uses yavl for the label
        :param title: The plot title. Pass in the empty string to generate the title automatically.
        :param cmap: matplotlib colormap, defaults to the cmap in vmidl
        :param withcb: add a colorbar to the plot.  Default is True
        :param grid: draw a grid if True.  Default is False
        :param logxy: use log x and y axes if True.  Default is False
        :param fig: matplotlib figure for plotting.  Default is None to force this routine to create the figure
        :return: matplotlib.figure.Figure
        """
        mpl.rcParams['axes.titlepad'] = 12
        mpl.rcParams['axes.titlesize'] = 13
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['axes.formatter.min_exponent'] = 4
        plot_params = self.plot_params
        xlo = plot_params.xlo
        xhi = plot_params.xhi
        ylo = plot_params.ylo
        yhi = plot_params.yhi
        tic = plot_params.tic
        zmin = plot_params.zmin or 1
        zmax = plot_params.zmax
        if self.hist2d is None:
            hist2d = self.make_2d_hist(df=self.df, xval=xval, yval=yval, logxy=logxy)
        else:
            hist2d = self.hist2d
        if zmin != 1 and title == '':
            # update NumPHAs in the plot title
            numpts = int(hist2d[hist2d >= zmin].sum())
            title = re.sub(r'; [0-9,]+ PHAs', f'; {numpts:,} PHAS', self.title, )
        cmap = cmap or vmidl_cmap
        if fig is None:
            plt.close('all')
            fig = plt.figure(figsize=[13.75, 11],
                             tight_layout=True,
                             dpi=72)
            ax = fig.add_subplot(111,
                                 xlim=[xlo, xhi],
                                 ylim=[ylo, yhi],
                                 xlabel=xlabel or xval,
                                 ylabel=ylabel or yval,
                                 title=title or self.title,
                                 )
            self.set_tick_params(ax)
        else:
            ax = fig.axes[0]
        if logxy:
            ax.set_xscale("log")  # <- Activate log scale on X axis
            ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(3, 2)))
            ax.set_yscale("log")  # <- Activate log scale on Y axis
            ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(3, 2)))
        else:
            if tic > 0:
                ax.set_xticks(np.arange(xlo, xhi + tic, tic))
                ax.set_yticks(np.arange(ylo, yhi + tic, tic))
        match (zmin, zmax):
            case (1, 0):
                norm = LogNorm()
            case (_, 0):
                norm = LogNorm(vmin=zmin)
            case (1, _):
                norm = LogNorm(vmax=zmax)
            case _:
                norm = LogNorm(vmin=zmin, vmax=zmax)
        pcm = ax.pcolormesh(
            self.xbins, self.ybins, hist2d, #np.where(hist2d >= zmin, hist2d, np.nan),
            norm=norm, shading='auto', cmap=cmap, rasterized=True
        )
        if withcb:
            cb = fig.colorbar(pcm, ax=ax, aspect=30)  # , shrink=0.8)
            cb.set_label('# PHAs', fontsize=16)
            cb.ax.tick_params(axis="y", which='major', length=10, labelsize=14)
            cb.ax.tick_params(axis="y", which='minor', length=5)
        if grid:
            ax.grid(True, color='lightgray', which='both' if logxy else 'major')
            ax.set_axisbelow(True)
        # ax.figure.text(0.72, 0.065, f'Plotted {dt.date.today()}', fontsize=10)
        ax.figure.text(0.79, 0.02, f'Plotted {dt.date.today()}', fontsize=10)
        return fig

    def plot_trend_mpl(self, colors: list[str]):
        """
        Plot a trend using matplotlib.  Non-interactive but should not crash the
        app.  Only used if the plot would have > 100_000 points
        :return: mpl Figure
        """
        fig = plt.figure(figsize=[13.75, 11],
                         dpi=72)
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(fmt='%y-%j\n%T'))
        self.set_tick_params(ax=ax)
        for cndx, trend_item in enumerate(self.plot_params.trend_items):
            self.df.plot(kind='scatter', x='SpinStartTime', y=trend_item, logy=self.form_data['logy'],
                         c=colors[cndx], label=trend_item, grid=True, ylabel='', ax=ax)
        ax.set_title(self.title, y=1.02)
        ax.legend(loc='upper right', ncol=len(self.plot_params.trend_items))
        ylo, yhi = self.plot_params.ylo, self.plot_params.yhi
        match (ylo, yhi):
            case (-1000, -1000):
                pass
            case (-1000, _):
                ax.set_ylim([None, yhi])
            case (_, -1000):
                ax.set_ylim([ylo, None])
            case _:
                ax.set_ylim([ylo, yhi])
        return fig

    def plot_trend_plotly(self, colors):
        linlog = 'log' if self.form_data['logy'] else 'linear'
        ylo, yhi = self.plot_params.ylo, self.plot_params.yhi
        if ylo == -1000:
            ylo = None
        elif linlog == 'log':
            ylo = np.log10(ylo)
        if yhi == -1000:
            yhi = None
        elif linlog == 'log':
            yhi = np.log10(yhi)

        fig = go.Figure(data=[go.Scatter(x=self.df[self.df.columns[0]],
                                         y=self.df[tplt],
                                         mode='markers',
                                         marker={'color': color},
                                         name=tplt)
                              for tplt, color in zip(self.plot_params.trend_items, colors)],
                        layout=dict(height=800,
                                    margin=dict(l=60, r=4),
                                    showlegend=True,
                                    title=self.title.replace('\n', '<br>', 1),
                                    xaxis=dict(linecolor='#888',
                                               tickcolor='#888',
                                               gridcolor='#555',
                                               griddash='dot',
                                               showgrid=True,
                                               tickformat='%y:%j<br>%H:%M:%S',
                                               ticks="outside",
                                               tickwidth=2,
                                               ticklen=8,
                                               ),
                                    yaxis=dict(linecolor='#888',
                                               tickcolor='#888',
                                               gridcolor='#555',
                                               griddash='dot',
                                               showgrid=True,
                                               range=([ylo, yhi]),
                                               type=linlog,
                                               ticks="outside",
                                               tickwidth=2,
                                               ticklen=6,
                                               ),
                                    ),
                        )
        return fig

    def plot_trend(self):
        """
        interactive plotly plot.
        """
        colors = ['#D00', '#0D0', '#00F', '#DD0', '#D0D', '#0DD', '#FAA', '#AFA', '#A9F']
        if len(self.df) * len(self.plot_params.trend_items) > 100_000:
            fig = self.plot_trend_mpl(colors=colors)
        else:
            fig = self.plot_trend_plotly(colors=colors)
        # print(f'{"+" * 32}  {self.df.shape[0] * len(self.plot_params.trend_items)}  {"+" * 32}\n')
        return fig

    def plot_me_trend(self):
        # get the offset to the first ME value, 0
        offset = self.df.columns.get_loc(0)
        melow, mehigh = [int(hl) + offset for hl in self.form_data['merange']]
        medf = (self.df
                    .iloc[:, melow:mehigh + 1]
                    .sum(axis=1)
                )
        if melow == mehigh:
            medf.name = f'Count of ME {melow-offset}'
        else:
            medf.name = f'Count of MEs {melow-offset} - {mehigh-offset}'
        logy = medf.max() > 1000
        fig = px.scatter(medf, y=medf.name, log_y=logy)
        fig.update_traces(marker=dict(color='#D00', size=9))
        fig.update_layout(height=800,
                          title=self.title.replace('\n', '<br>'),
                          xaxis_tickformat='%Y:%j<br>%H:%M')
        return fig

    def nm_to_mass(self, nm, pos):
        return f'{0.6854 * 1.20 ** (nm - 1):.3g}'

    def nq_to_mpq(self, nq, pos):
        return f'{0.6837 * 1.05 ** (nq - 1):.3g}'

    def plot_me_matrix(self,):
        """
        Return a figure containing rectangle patches corresponding to the NQ/NM
        rectangles in Figure 1 (Matrix Elements in mass - mass-per-charge space)
        in the AMPTE CHEM DPU document
        """
        xlo = self.plot_params.xlo
        xhi = self.plot_params.xhi
        ylo = self.plot_params.ylo
        yhi = self.plot_params.yhi
        zmin = self.plot_params.zmin
        zmax = self.plot_params.zmax
        title = self.title
        df = self.df.loc[:, 0:490].sum()

        fig, ax = plt.subplots(figsize=(16 * (xhi - xlo) / 96, 9 * (yhi - ylo) / 28), facecolor='w', dpi=144)
        ec = 'gold'
        rect = mpl.patches.Rectangle

        # DOUBLES, MEs 0-47 are doubles only.
        rects = [rect((2 * x, 0), 2, 1, ec=ec, lw=1) for x in range(48)]

        # # Use a grayscale colormap for the doubles
        # cm = plt.cm.gray
        # doubles = PatchCollection(rects, match_original=True, cmap=cm, zorder=10, norm=LogNorm())
        # doubles.set_clim(vmin=1, vmax=mes[:48].max())
        # doubles.set_array(mes[:48])
        # ax.add_collection(doubles)

        # TRIPLES, MEs 48:490.  Use vmidl cmap
        cm = vmidl_cmap.copy()
        cm.set_bad((0, 0, 0))
        if zmin == 0:
            zmin = max(1, df[48:491].min())
        if zmax == 0:
            zmax = df[48:491].max()
        # MEs 48-62
        rects.extend([rect((4 * x, 2 * y + 1), 4, 2)
                      for y in range(3)
                      for x in range(5)]
                     )
        ax.text(1.3, 0.3, '0', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(11.3, 0.3, '5', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(10, 48, 5):
            ax.text(x * 2 + 1.8, 0.3, str(x), c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(95.8, 0.3, 47, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(2.3, 1.8, 48, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(18.8, 1.8, 52, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(2.3, 3.8, '53', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(18.8, 3.8, 57, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(2.3, 5.8, '58', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(18.8, 5.8, 62, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 63-70
        y = 7
        rects.extend([rect((14, y), 2, 2), ] +
                     [rect((4 * x + 16, y), 4, 2) for x in range(7)])
        ax.text(13.8, 7.8, '63', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(22.8, 7.8, '65', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(42.8, 7.8, '70', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 71-100
        rects.extend([rect((2 * x + 14, 9 + y), 2, 1)
                      for y in range(2)
                      for x in range(15)])
        ax.text(13.7, 9.3, '71', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(23.3, 9.3, '5', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(33.3, 9.3, '0', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(44.3, 9.3, '85', c='w', ha='left', zorder=20, fontweight='bold', fontsize=10)
        ax.text(23.3, 10.3, '0', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(33.3, 10.3, '5', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(44.3, 10.3, '100', c='w', ha='left', zorder=20, fontweight='bold', fontsize=10)

        # MEs 101-122
        y = 11
        rects.extend([rect((2 * x + 14, y), 2, 2) for x in range(15)] +
                     [rect((4 * x + 44, y), 4, 2) for x in range(7)]
                     )
        ax.text(13.7, 11.8, '101', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(23.3, 11.8, '5', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(33.3, 11.8, '0', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(43.3, 11.8, '5', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(63.3, 11.8, '120', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(71.3, 11.8, '122', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 123-137
        y = 13
        rects.extend([rect((2 * x + 14, y), 2, 2) for x in range(1)] +
                     [rect((4 * x + 16, y), 4, 2) for x in range(14)]
                     )
        ax.text(13.8, 13.8, '123', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(23.3, 13.8, '125', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(43.3, 13.8, '130', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(63.3, 13.8, '135', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(71.3, 13.8, '137', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 138-251
        for y in range(15, 18):
            rects.extend([rect((x + 20, y), 1, 1) for x in range(24)] +
                         [rect((2 * x + 44, y), 2, 1) for x in range(14)]
                         )
        for y in range(3):
            ax.text(19.8, 15.3 + y, 138 + 38 * y, c='w', ha='right', fontweight='bold', fontsize=10)
            ax.text(72.3, 15.3 + y, 175 + 38 * y, c='w', ha='left', fontweight='bold', fontsize=10)
        for x in range(5):
            ax.text(22.9 + x * 5, 15.3, (x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
            ax.text(21.9 + x * 5, 17.3, (1 - x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(2):
            ax.text(51.3 + x * 10, 15.3, (1 - x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
            ax.text(53.3 + x * 10, 18.3, (1 - x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(3):
            ax.text(45.3 + x * 10, 16.3, (x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
            ax.text(49.3 + x * 10, 17.3, (x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
            ax.text(47.3 + x * 10, 19.3, (x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(15, 31, 5):
            ax.text(9.9 + x, 16.3, x % 10, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 252-256
        rects.append(rect((0, 7), 20, 11))
        rects.append(rect((20, 0), 24, 7))
        rects.append(rect((44, 0), 28, 11))
        rects.append(rect((72, 0), 25, 18))
        rects.append(rect((0, 18), 97, 11))
        ax.text(2, 12, '252', c='w', fontweight='bold', fontsize=14)
        ax.text(13.7, 10.3, '86', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(29, 3.5, '253', c='w', fontweight='bold', fontsize=14)
        ax.text(55, 6, '254', c='w', fontweight='bold', fontsize=14)
        ax.text(84, 9, '255', c='w', fontweight='bold', fontsize=14)
        ax.text(5, 23, '256', c='w', fontweight='bold', fontsize=14)
        ax.text(89, 20.3, '256', c='w', fontweight='bold', fontsize=14)

        # MEs 257-332
        for y in range(18, 20):
            rects.extend([rect((x + 20, y), 1, 1) for x in range(24)] +
                         [rect((2 * x + 44, y), 2, 1) for x in range(14)])
        ax.text(19.8, 18.3, '257', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(72.3, 18.3, '294', c='w', ha='left', fontweight='bold', fontsize=10)
        ax.text(19.8, 19.3, '295', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(72.3, 19.3, '332', c='w', ha='left', fontweight='bold', fontsize=10)
        for x in range(10, 31, 5):
            ax.text(13.9 + x, 18.3, x % 10, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(10, 26, 5):
            ax.text(15.9 + x, 19.3, x % 10, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 333-435
        for y in range(20, 24):
            rects.extend([rect((2 * x + 20, y), 2, 1) for x in range(18)] +
                         [rect((4 * x + 56, y), 4, 1) for x in range(7)])
        rects.extend([rect((4 * x + 84, y), 4, 1) for x in range(3)])
        for y in range(4):
            ax.text(19.8, 20.3 + y, 333 + 25 * y, c='w', ha='right', fontweight='bold', fontsize=10)
            ax.text(25.3, 20.3 + y, (1 - y % 2) * 5, zorder=20, c='w', ha='right', fontweight='bold', fontsize=10)
            ax.text(35.3, 20.3 + y, (y % 2) * 5, zorder=20, c='w', ha='right', fontweight='bold', fontsize=10)
            ax.text(45.3, 20.3 + y, (1 - y % 2) * 5, zorder=20, c='w', ha='right', fontweight='bold', fontsize=10)
            ax.text(55.3, 20.3 + y, (y % 2) * 5, zorder=20, c='w', ha='right', fontweight='bold', fontsize=10)

        for y in range(4):
            ax.text(75.4, 20.3 + y, 355 + 25 * y, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for y in range(3):
            ax.text(83.4, 20.3 + y, 357 + 25 * y, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(95.3, 23.3, '435', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 436-463
        y = 24
        rects.extend([rect((2 * x + 20, y), 2, 2) for x in range(18)] +
                     [rect((4 * x + 56, y), 4, 2) for x in range(10)])
        ax.text(19.8, 24.7, '436', c='w', ha='right', fontweight='bold', fontsize=10)
        ax.text(63.3, 24.7, '455', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(83.3, 24.7, '460', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(95.3, 24.7, '463', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(3):
            ax.text(29.3 + x * 10, 24.7, (x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # MEs 464-490
        y = 26
        rects.extend([rect((20, y), 4, 2)] +
                     [rect((2 * x + 24, y), 2, 2) for x in range(16)] +
                     [rect((4 * x + 56, y), 4, 2) for x in range(10)])
        ax.text(23.3, 26.7, '464', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(75.4, 26.7, '485', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        ax.text(95.3, 26.7, '490', c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)
        for x in range(4):
            ax.text(25.3 + x * 10, 26.7, (1 - x % 2) * 5, c='w', ha='right', zorder=20, fontweight='bold', fontsize=10)

        # Add rects to the axis in two PatchCollections
        # Set Z order to 10 for MEs 48-251 and 257-490
        fg_rects = rects[:252] + rects[257:]
        fg_trp_mes = np.hstack([df[:252], df[257:491]])
        triples = PatchCollection(fg_rects, match_original=True, cmap=cm, ec=ec,
                                  lw=1, norm=LogNorm(), zorder=10)
        triples.set_clim(vmin=zmin, vmax=zmax)
        triples.set_array(fg_trp_mes)
        ax.add_collection(triples)

        # Set Z order to 1 for MEs 252-256 to place them behind the other MEs
        bkg_rects = rects[252:257]
        bkg_trp_mes = df[252:257]
        triples = PatchCollection(bkg_rects, match_original=True, cmap=cm, ec=ec,
                                  lw=1, norm=LogNorm(), zorder=1)
        triples.set_clim(vmin=zmin, vmax=zmax)
        triples.set_array(bkg_trp_mes)
        ax.add_collection(triples)

        # Set up secondary x and y axes.  ax will be mass and mass per charge
        # secondary y is NM and secondary x is NQ
        ax.tick_params(which='both', top=False, right=False, left=True, bottom=True,
                       labeltop=False, labelbottom=True,
                       labelleft=True, labelright=False)
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=5)
        ax.tick_params(axis='x', rotation=30)
        ax.set_xlabel('Mass/charge (amu/e)', fontsize=14, )
        ax.set_ylabel('Mass (amu)', fontsize=14)
        ax.xaxis.set_major_formatter(self.nq_to_mpq)
        ax.yaxis.set_major_formatter(self.nm_to_mass)

        axnq = ax.twiny()
        axnq.xaxis.set_major_formatter(lambda x, pos: f'{x - 1}')
        axnq.set_xlabel('NQ', fontsize=12, loc='left', )
        axnq.tick_params(which='both', top=True, right=False, length=5,
                         labeltop=True, labelbottom=False,
                         labelleft=False, labelright=False)
        axnq.tick_params(which='major', length=7)

        axnm = ax.twinx()
        axnm.yaxis.set_major_formatter(lambda x, pos: f'{x - 1}')
        axnm.set_ylabel('NM', fontsize=12, loc='top', labelpad=24, rotation=0)
        axnm.tick_params(which='both', top=False, bottom=False, left=False, right=True,
                         labeltop=False, labelbottom=False,
                         labelleft=False, labelright=True)

        for axs in [ax, axnq]:
            axs.set(xlim=[xlo, xhi], xticks=range(int(xlo) + 1, int(xhi) + 2, 4), )
            axs.set_xticks(range(int(xlo) + 1, int(xhi) + 2), minor=True)

        for axs in [ax, axnm]:
            axs.set(ylim=[ylo, yhi], yticks=range(int(ylo) + 1, int(yhi) + 2))

        ax.set_title(title, fontsize=16, y=1.04)
        fig.colorbar(triples, ax=ax, fraction=0.025)
        return fig

    def make_zip_file(self, filename: str, datastr: str):
        contents = BytesIO()
        zippedfile = zipfile.ZipFile(contents, 'w')
        info = zipfile.ZipInfo(filename)
        info.date_time = time.localtime(time.time())[:6]
        info.compress_type = zipfile.ZIP_DEFLATED
        info.internal_attr = 1  # text file
        info.external_attr = 2175008768  # -rw-r--r--
        zippedfile.writestr(info, datastr)
        if self.rate_filter:
            info = zipfile.ZipInfo('filtered_times.txt')
            info.date_time = time.localtime(time.time())[:6]
            info.compress_type = zipfile.ZIP_DEFLATED
            info.internal_attr = 1  # text file
            info.external_attr = 2175008768  # -rw-r--r--
            rejected_times = self.high_rate_ranges_str or 'No times rejected by this filter.'
            zippedfile.writestr(info, 'Times with {} in DV step {} above {}\n{}'.format(*self.form_data['rate_filt'], rejected_times))
        zippedfile.close()
        return contents.getbuffer()

    def make_me_zip_file(self):
        me0loc = self.df.columns.get_loc(0)
        melow, mehigh = [int(hl)+me0loc for hl in self.form_data['merange']]
        cols = list(range(me0loc)) + list(range(melow, mehigh+1))
        df = self.df.iloc[:, cols]
        datastr = f'# {self.filters}\n#\n{df.to_csv(float_format="%.3f")}'
        return self.make_zip_file(filename='AMPTE_CHEM_MEs.csv',
                                  datastr=datastr)

    def make_trend_zip_file(self):
        df = self.df
        trend_items = self.plot_params.trend_items
        if trend_items[0] == 'ME':
            me0loc = self.df.columns.get_loc(0)
            melow, mehigh = [int(hl) + me0loc for hl in self.form_data['merange']]
            colnames = f'MEs_{"-".join(str(mer) for mer in self.form_data["merange"])}_Sum'
            df.insert(me0loc, colnames, df.iloc[:, range(melow, mehigh + 1)].sum(axis=1))
            datastr = f'# Filters: {", ".join([str(val) for val in self.filters.values() if val])}\n#\n' \
                      f'{df[colnames].to_csv(float_format="%.3f")}'
        else:
            colnames = ['SpinStartTime'] + trend_items
            datastr = f'# Filters: {", ".join([str(val) for val in self.filters.values() if val])}\n#\n' \
                      f'{df[colnames].to_csv(float_format="%.3f", index=False)}'
        filename = f'AMPTE_CHEM_{"_".join(trend_items)}_trend.csv'
        return self.make_zip_file(filename=filename,
                                  datastr=datastr)

    def make_hist2d_zip_file(self):
        histstr = (pd.DataFrame(data=self.hist2d,
                                columns=self.xedges[:-1],
                                index=self.yedges[:-1])
                     .sort_index(ascending=False)
                     .to_csv(float_format="%.4f"))
        title = self.title.replace('\n', '<lf>')
        # datastr = f'# {str(self.filters.values())}\n' \
        datastr = f'# Filters: {", ".join([str(val) for val in self.filters.values() if val])}\n' \
                  f'# PLOTTITLE: {title}\n#\n{histstr}'
        plot_type = 'ET' if self.form_data['action'] == 'plot_et' else 'MMpQ'
        return self.make_zip_file(filename=f'AMPTE_CHEM_{plot_type}_Hist.csv',
                                  datastr=datastr)

    def make_pha_zip_file(self):
        filename = 'AMPTE_CHEM_PHAs.csv'
        datastr = f'# Filters: {", ".join([str(val) for val in self.filters.values() if val])}\n#\n' \
                  f'{self.df.to_csv(float_format="%.3f", date_format="%Y-%j %T")}'
        return self.make_zip_file(filename=filename, datastr=datastr)

    def compute_br_r_ratios(self, df):
        """
        Return a dataframe with Rn/BRn ratios in columns RnBRn
        """
        return (df
                  .assign(R0BR0=(lambda _df: _df['R0'] / _df['BR0']))
                  .assign(R1BR1=(lambda _df: _df['R1'] / _df['BR1']))
                  .assign(R2BR2=(lambda _df: _df['R2'] / _df['BR2']))
                  .rename(columns={'R0BR0': 'R0/BR0', 'R1BR1': 'R1/BR1', 'R2BR2': 'R2/BR2', })
                  .replace([np.inf, -np.inf], np.nan)
                )


