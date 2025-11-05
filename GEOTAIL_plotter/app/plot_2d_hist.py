import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from geotail_cmap import geotail_cmap


def set_tick_params(ax):
    ax.tick_params(axis="x", which='major', length=7, labelsize=12, reset=True)
    ax.tick_params(axis="y", which='major', length=7, labelsize=12, reset=True)
    ax.tick_params(axis="x", which='minor', length=5, labelsize=10, reset=True)
    ax.tick_params(axis="y", which='minor', length=4, labelsize=10, reset=True)


def make_2d_hist(df: pd.DataFrame,
                 xval: str='MPQ',
                 yval: str='MASS',
                 logxy: bool=True,
                 xlo: float=0.5,
                 xhi: float=100,
                 ylo: float=0.5,
                 yhi: float=100,
                 # bin_width: float=.01,
                 bin_width: float=.03125,
                 ) -> tuple[np.array]:
    if isinstance(bin_width, (tuple, list)):
        xbinw = bin_width[0]
        ybinw = bin_width[1]
    else:
        xbinw = ybinw = bin_width
    if logxy:
        # num = int((np.log10(xhi) - np.log10(xlo)) / xbinw) + 1
        num = int((np.log2(xhi) - np.log2(xlo)) / xbinw) + 1
        xbins = np.geomspace(xlo, xhi, num=num)
        num = int((np.log2(yhi) - np.log2(ylo)) / ybinw) + 1
        ybins = np.geomspace(ylo, yhi, num=num)
    else:
        xbins = np.arange(xlo, xhi + xbinw, xbinw)
        ybins = np.arange(ylo, yhi + ybinw, ybinw)

    hist2d, xedges, yedges = np.histogram2d(
        x=df[xval],
        y=df[yval],
        bins=[xbins, ybins],
        range=[[xlo, xhi + xbinw], [ylo, yhi + ybinw]],
        # weights=df['Weights'],
    )
    (
        pd.DataFrame(index=np.round(ybins[:-1], 5),
                     columns=np.round(xbins[:-1], 5),
                     data=hist2d.T)
        .sort_index(ascending=False)
        .to_excel('2dhist.xlsx')
    )
    return xbins, ybins, hist2d.T


def plot_2d_hist(
        df: pd.DataFrame,
        xlo: float=0.5,
        xhi: float=1e2,
        ylo: float=0.5,
        yhi: float=1e2,
        zmin: float=1.,
        zmax: float=0.,
        tic: int = 0,
        bin_width: float = 0.03125,
        xval: str = 'MPQ',
        xlabel: str = '',
        yval: str = 'MASS',
        ylabel: str = '',
        title: str = '',
        cmap=None,
        withcb: bool = True,
        grid: bool = False,
        logxy: bool = True,
        fig: Figure = None,
) -> Figure:
    """
    Create 2D plots of xval, yval PHA data
    :param xval: dataFrame column to use for x values.  Default is TOF
    :param xlabel: string to use for the x axis label.  Default is '' which uses xavl for the label
    :param yval: dataFrame column to use for y values.  Default is Energy
    :param ylabel: string to use for the y axis label.  Default is '' which uses yavl for the label
    :param title: The plot title. Pass in the empty string to generate the title automatically.
    :param cmap: matplotlib colormap, defaults to the cmap in Steve's GEOTAIL plots
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
    xbins, ybins, hist2d = make_2d_hist(df=df, xval=xval, yval=yval,
                                        xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                        logxy=logxy, bin_width=bin_width,)
    cmap = cmap or geotail_cmap
    if fig is None:
        plt.close('all')
        fig = Figure(figsize=[13.75, 11],
                     tight_layout=True,
                     dpi=72)
    if len(fig.get_axes()) == 0:
        ax = fig.add_subplot(111,
                             xlim=[xlo, xhi],
                             ylim=[ylo, yhi],
                             xlabel=xlabel or xval,
                             ylabel=ylabel or yval,
                             title=title or title,
                             )
        set_tick_params(ax)
        ax.set_title(title, x=1., fontsize=10, ha='right', )
        ax.set_xlabel('Mass/Charge [amu/e]')
        ax.set_ylabel('Mass\n [amu]', rotation=0, labelpad=1)
    else:
        ax = fig.axes[0]
    # move plot to the left to accomodate the colorbar and information text
    loc = list(ax.get_position().bounds)
    loc[0] = 0.08
    ax.set_position(loc)
    if logxy:
        ax.set_xscale(LogScale(axis=ax.xaxis, base=2))
        ax.xaxis.set_major_formatter(mpl.ticker.LogFormatter(base=2, ))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=2, subs=(1.5,)))
        # ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=2, labelOnlyBase=False, minor_thresholds=(7, 2)))
        ax.set_yscale(LogScale(axis=ax.yaxis, base=2))  # <- Activate log scale on Y axis
        # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=2, subs=(8,)))
        ax.yaxis.set_major_formatter(mpl.ticker.LogFormatter(base=2, ))
        # ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=2, labelOnlyBase=False, minor_thresholds=(7, 2)))
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
        xbins, ybins, hist2d, #np.where(hist2d >= zmin, hist2d, np.nan),
        norm=norm, shading='auto', cmap=cmap, rasterized=False
    )
    # ax.set_aspect('equal')
    if withcb:
        # axins = ax.inset_axes([1.12, 0.44, .015, .5], )
        # cb = fig.colorbar(pcm, cax=axins, orientation="vertical",
        #                   extend='both', extendfrac=(.01, .01), extendrect=True,)
        cb = fig.colorbar(pcm, ax=ax, aspect=30,
                          extend='both', extendfrac=(.02, .02), extendrect=True,
                          anchor=(0.4, 0.86), shrink=0.5)
        # cb.set_label('# PHAs', fontsize=16)
        if zmax == 0:
            _, zmax = cb.ax.get_ybound()
        zmax = int(zmax)
        yticks = cb.ax.get_yticks()
        yticks = yticks[(yticks >= 1) & (yticks <= zmax)]
        cb.ax.set_yticks(np.hstack([yticks, [zmax]]))
        cb.ax.set_yticklabels(cb.ax.get_yticklabels()[:len(yticks)] + [str(zmax)])
        cb.ax.tick_params(axis="y", which='major', length=10,
                          labelsize=10, labelleft=True, left=True,
                          labelright=False, right=False)
        cb.ax.tick_params(axis="y", which='minor', length=5, left=True, right=False)
    if grid:
        ax.grid(True, color='lightgray', which='both' if logxy else 'major')
        ax.set_axisbelow(True)
    # # draw boxes
    # box = mpl.patches.Rectangle((0.5, 0.5), 10.5, 96.5, lw=1.5, fc='none', ec=(.75, .75, .75))
    # ax.add_patch(box)
    # box = mpl.patches.Rectangle((12, 4.), 2, 50, lw=1.5, fc='none', ec=(.75, .75, .75))
    # ax.add_patch(box)
    # box = mpl.patches.Rectangle((15, 3.5), 3, 60, lw=1.5, fc='none', ec=(.75, .75, .75))
    # ax.add_patch(box)
    # ax.figure.text(0.72, 0.065, f'Plotted {dt.date.today()}', fontsize=10)
    ax.figure.text(0.08, 0.895, f'Geotail/EPIC/STICS PHA Matrix', fontsize=12)
    # ax.figure.text(0.83, 0.92, f'data:01.Oct.2021_22:39:22_EDT\n        start:1993-012T18:01:00.5\n        stop:1993-012T18:26:49.2\n        no GTL_Xfw(???)\n        scale GTL MPQ',
    #                va='top', fontsize=10, fontweight=550)
    return fig
