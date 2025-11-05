#!/usr/bin/env python3
import re
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib as mpl
import pandas as pd
from PIL import Image
from nicegui import app, ui

from local_file_picker import local_file_picker
from plot_2d_hist import plot_2d_hist
from read_data_file import read_data_file


def get_client_val(name: str, default=''):
    return app.storage.client.get(name, default)


async def pick_file() -> None:
    filename, = await local_file_picker('data', multiple=False)
    app.storage.client['filename'] = filename
    if title.value == '':
        title.set_value(Path(filename).name)


def extract_times(timesstr: str) -> Union[str, list[str]]:
    """
    Return a list of datetime strings formatted YYYY:doy:HHMM-YYYY:doy:HHMM
    extracted from a string with format YYYYxDOYxHHMMxYYYYxDOYxHHMM
    on each line where x can be any character

    :return: list of start and stop datetime strings
    """
    times = []
    if timesstr == '':
        return times
    for timestr in timesstr.split('\n'):
        nums = re.findall(r'(\d+)+', timestr)
        numnums = len(nums)
        if numnums == 8:
            # YDHM-YDHM
            times.extend(['-'.join([':'.join(nums[:4]), ':'.join(nums[-4:])])])
        elif numnums == 7:
            # assume YDHM-DHM
            times.extend(['-'.join([':'.join(nums[:4]), ':'.join([nums[0]] + nums[-3:])])])
        elif numnums == 6:
            # assume YDHM-HM
            times.extend(['-'.join([':'.join(nums[:4]), ':'.join(nums[0:2] + nums[-2:])])])
        elif numnums == 1:
            nextyear = int(nums[0]) + 1
            times.append(f'{nums[0]}:001:0000-{nextyear}:001:0000')
    return times


def get_data(filename: str) -> pd.DataFrame:
    phadf = read_data_file(filename)

    timestrs = []
    if (timerange := get_client_val('timeranges')):
        timestrs = extract_times(timesstr=timerange)
        df = pd.DataFrame()
        for timestr in timestrs:
            start, stop = pd.to_datetime(timestr.split('-'), format='%Y:%j:%H:%M').strftime('%F %T')
            df = pd.concat([df, phadf.loc[start:stop]])
        phadf = df
    dvlow = int(get_client_val('dvlow'))
    dvhigh = int(get_client_val('dvhigh'))

    phadf = phadf.query('DV >= @dvlow and DV <= @dvhigh')
    return phadf, timestrs


def plot_data():
    filename = get_client_val('filename')
    if filename == '':
        ui.notify(f'You must get data before plotting')
        return
    try:
        df, timestrs = get_data(filename)
    except:
        ui.notify(f'Unable to read data from {filename}')
        return
    if df.empty:
        ui.notify(f'No data in this time range')
        return

    xvalue = get_client_val('xvalue')
    yvalue = get_client_val('yvalue')
    binwidth = float(get_client_val('binwidth', 0.03125))
    xlo, xhi = [float(s) for s in get_client_val('xlimits').split(',')]
    ylo, yhi = [float(s) for s in get_client_val('ylimits').split(',')]
    zmax = float(get_client_val('zmax') or 0)
    if xvalue not in df.columns:
        ui.notify(f'{filename} has no column named {xvalue}')
    if yvalue not in df.columns:
        ui.notify(f'{filename} has no column named {yvalue}')
    plot_title = title.value
    with hist_plot as figui:
        figui.clf()
        figui.get_layout_engine().set(w_pad=.25, h_pad=.25,
                                      hspace=0, wspace=0)
        fig = plot_2d_hist(df,
                           fig=figui, zmax=zmax, title=plot_title,
                           xval=xvalue, xlo=xlo, xhi=xhi,
                           yval=yvalue, ylo=ylo, yhi=yhi,
                           bin_width=binwidth
                           )
    draw_info_strings(df=df, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                      timestrs=timestrs,)


def draw_info_strings(df, xlo: float, xhi: float,
                      ylo: float, yhi: float,
                      timestrs: list[str]) -> None:
    dvlo = int(get_client_val('dvlow'))
    dvhi = int(get_client_val('dvhigh'))
    xvalue = get_client_val('xvalue')
    yvalue = get_client_val('yvalue')

    querystr = (f'{xvalue} >= {xlo} and {xvalue} <= {xhi} and '
                f'{yvalue} >= {ylo} and {yvalue} <= {yhi} and '
                f'DV >= {dvlo} and DV <= {dvhi}')
    df = df.query(querystr)
    start = df.index.min().strftime('%Y-%jT%T.%f')[:-4]
    stop = df.index.max().strftime('%Y-%jT%T.%f')[:-4]
    # above colorbar
    infostr = f'''data:{datetime.now():%d.%b.%Y_%H:%M:%S_%Z}
     start: {start}
     stop:  {stop}
     no GTL_Xfw
     scale GTL MPQ'''
    draw_text_at(.71, 0.906, infostr, fontfamily='monospace')
    # under colorbar
    if 'TCH' in df.columns:
        doubles = df.query('TCH == 0')['EDB'].count()
        triples = df.query('TCH > 0')['EDB'].count()
    else:
        doubles = 0
        triples = 0
    if len(timestrs) > 0:
        time_ranges = []
        for ndx, timestr in enumerate(timestrs, start=1):
            rstart, rstop = pd.to_datetime(timestr.split('-'), format='%Y:%j:%H:%M')
            if df.loc[rstart:rstop].empty:
                continue
            start = df.loc[rstart:rstop].index.min().strftime('%Y-%jT%T.%f')[:-4]
            stop = df.loc[rstart:rstop].index.max().strftime('%Y-%jT%T.%f')[:-4]
            time_ranges.append(f'Time range {ndx}\nStart: {start}\nStop:  {stop}')
    else:
        time_ranges = [start, stop]
    infostr = (f"""     triples   doubles
N* = {triples:<10}{doubles}
Ntot = {triples}/{len(df)}
{"\n".join(time_ranges)}
DPPS:  {df['DV'].min()}:{df['DV'].max()}
binw:  {get_client_val('binwidth', 0.03125)}, log2 binning""")
    draw_text_at(.73, 0.42, infostr, fontsize=12, fontfamily='monospace')


def draw_text() -> None:
    x, y, fontsize = [float(s) for s in app.storage.client.get('textloc').split(',')]
    text = get_client_val('text')
    halign = 'center' if ha.value else 'left'
    draw_text_at(x, y, text, halign, fontsize)


def draw_text_at(x: float, y: float, text: str,
                 halign: str = 'left',
                 fontsize=10, fontfamily='Helvetica') -> None:
    with hist_plot as figui:
        ax = figui.get_axes()
        if len(ax) == 0:
            ui.notify(f'You must plot data before drawing text')
        ax[0].figure.text(x, y, text, ha=halign, va='top',
                          fontsize=fontsize, fontfamily=fontfamily)


def save_plot() -> None:
    filename = Path(get_client_val('filename')).with_suffix('').with_suffix('').name
    timestr = datetime.now().strftime('%Y%m%d%H%M%S')
    plot_file = (Path('plots') / f'{filename}_{timestr}')

    # rename 2dhist.xlsx to the same name as the saved plot
    xlfile = Path('2dhist.xlsx')
    if xlfile.exists():
        xlfile.rename(plot_file.with_suffix('.xlsx'))

    with hist_plot as figui:
        for suffix in ('.png', '.pdf'):
            outfile = plot_file.with_suffix(suffix)
            with mpl.rc_context(rc={'ps.fonttype': 42}):
                figui.savefig(outfile, dpi=96)
            if suffix == '.png':
                (
                    Image.open(outfile)
                    .crop((0., 80., 1350., 1010.))
                    .save(outfile)
                )


def draw_box():
    limitstr = app.storage.client.get('boxlimits')
    if limitstr is None:
        return
    xlo, xhi, ylo, yhi = [float(s) for s in limitstr.split(',')]
    with hist_plot as figui:
        ax = figui.get_axes()
        if len(ax) == 0:
            ui.notify(f'You must plot data before drawing boxes')

        box = mpl.patches.Rectangle((xlo, ylo), xhi-xlo,  yhi-ylo, lw=1.5, fc='none', ec=(.75, .75, .75))
        ax[0].add_patch(box)


# layout
with ui.grid(columns=2, rows='auto auto auto') as figui:
    # row 1
    with ui.row():
        (
            ui.input(label='x column', value='MPQ')
            .bind_value(app.storage.client, 'xvalue')
            .classes('w-18')
            .tooltip('X column label in the data file')
        )
        (
            ui.input(label='x axis limits', value='0.7, 92')
            .bind_value(app.storage.client, 'xlimits')
            .classes('w-18')
        )
        (
            ui.input(label='y column', value='MASS')
            .bind_value(app.storage.client, 'yvalue')
            .classes('w-16')
            .tooltip('Y column label in the data file')
        )
        (
            ui.input(label='y axis limits', value='0.25, 92')
            .bind_value(app.storage.client, 'ylimits')
            .classes('w-18')
        )
        (
            ui.input(label='bin', value='0.03125')
            .bind_value(app.storage.client, 'binwidth')
            .classes('w-16')
        )
        (
            ui.input(label='DV low', value='1')
            .bind_value(app.storage.client, 'dvlow')
            .classes('w-12')
        )
        (
            ui.input(label='DV high', value='31')
            .bind_value(app.storage.client, 'dvhigh')
            .classes('w-12')
        )
        (
            ui.input(label='Z max', value='',)
            .bind_value(app.storage.client, 'zmax')
            .classes('w-12')
            .tooltip('Leave blank for auto zmax')
        )
    with ui.row():
        ui.button('Get Data', on_click=pick_file, )
        ui.button('Plot Data', on_click=plot_data, )
        ui.button('Save plot', on_click=save_plot, )

    # row 2
    with ui.row():
        title = (
            ui.input(label='Plot title',
                     value=app.storage.client.get('filename', ''))
            .classes('w-116')
            .tooltip('Will be filled in with the file name if blank')
        )
    with ui.row():
        ui.button('Draw box', on_click=draw_box)
        (
            ui.input(label='box limits xlo,xhi,ylo,yhi', value='15., 18., 3.0, 62.0')
            .bind_value(app.storage.client, 'boxlimits')
            .classes('w-36')
            .tooltip('Corner coordinates are in data units')
         )

    # row 3
    with ui.row():
        (
            ui.textarea(label='Time ranges (YYYY:DOY:HH:MM:SS-YYYY:DOY:HH:MM:SS', value='1993:12:18:1-1993:12:18:27')
            .bind_value(app.storage.client, 'timeranges')
            .classes('w-125')
            .tooltip('Leave blank to use all data')
            .props('rows=4')
        )
    with ui.row():
        ui.button('Draw text at', on_click=draw_text)
        (
            ui.input(label='x, y, fontsize', value='0.12, .85, 18')
            .bind_value(app.storage.client, 'textloc')
            .classes('w-30')
            .tooltip('Coordinates are in figure units, i.e. 0 to 1')
        )
        (
            ui.textarea(label='Text', value='GEOTAIL/STICS\n~87-212 keV/e\nions', )
            .bind_value(app.storage.client, 'text')
            .classes('w-33')
            .props('rows=4')
         )
        ha = ui.checkbox('Centered')

ui.separator()

with ui.row():
    hist_plot = ui.matplotlib(figsize=(15, 11), layout='constrained',).figure

ui.run(port=8081)

# ui.run(native=True, window_size=(720,960))
