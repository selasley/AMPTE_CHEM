import base64
import logging
from typing import Any

import matplotlib as mpl
import streamlit as st

from ampte_chem_plotter import AmpteChemPlotter
from ampte_plot_params import PlotParams
from openpyxl import load_workbook

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, style='{', datefmt='%Y-%m-%d %H:%M:%S',
                    format='{asctime} {levelname} {module}:{funcName}:{lineno}: {message}')


def ampte_layout() -> tuple[
    dict[str | Any, str | list | Any], PlotParams]:
    st.set_page_config(layout='wide', page_title='AMPTE CHEM WebApp')
    with open('static/css/ampte.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    with st.sidebar.form(key='filters'):
        # dates = st.text_area('Time ranges (can include many non-contiguous intervals)',
        #                      placeholder='1984-248-000:1984-249-1200      \n1984-248-000:249-1200')
        dates = st.text_area('Time ranges',
                             placeholder='1984-248-000:1984-249-1200      \n'
                                         '1985-066-000:1985-166-0030      \n'
                                         '1986-148-00:149-1200')
        st.write('Filters (leave box empty to skip filter)')
        col1, col2 = st.columns(2)
        llow = col1.text_input(label='L Low', value=2.0)
        lhi = col2.text_input(label='L High', value=10.0)
        col1, col2 = st.columns(2)
        dvlow = col1.text_input(label='DPPS Low')
        dvhi = col2.text_input(label='DPPS High')
        col1, col2 = st.columns(2)
        mlthlow = col1.text_input(label='MLT Low (HHMM)')
        mlthhi = col2.text_input(label='MLT High (HHMM)')
        col1, col2 = st.columns(2)
        dstlow = col1.text_input(label='Dst Low')
        dsthi = col2.text_input(label='Dst High')
        col1, col2 = st.columns(2)
        kplow = col1.text_input(label='Kp Low')
        kphi = col2.text_input(label='Kp High')
        col1, col2 = st.columns(2)
        ssdid = col1.text_input(label='MSS IDs', value='',
                                help='Valid IDs 1,2,3 though there are occasional 4\'s and 5\'s')
        range = col2.text_input(label='PHA Ranges', value='',
                                help='Valid Ranges 0,1,2 though there are occasional 3\'s')
        col1, col2 = st.columns(2)
        paps = col1.text_input(label='PAPS Levels',
                               help='Comma separated list of PAPS levels or blank for any level. '
                                    '1 (14.0kV) 2 (15.3kV) 3 (17.0kV) 4 (18.8kV) 5 (21.9kV) 6 (24.1kV)',
                               value='0,1,2,3,4,5,6')
        dpumodes = col2.text_input(label='DPU Modes',
                                   help='Comma separated list of DPU Modes or blank for any mode.  '
                                        'DPU modes 0 & 2 are normal operational modes. In mode 2 all ml commands are '
                                        'executed at beginning of a DV cycle.  In mode 0 some ml commands also executed '
                                        'at start of period 8,16,24.  Modes 1 & 3 are test modes - DV can be set by ml '
                                        'command, bubble commands can be executed every period, classification unit '
                                        'can be disabled.',
                                   value='0,2')
        col1, col2 = st.columns(2)
        tac_slope = col1.selectbox(label='Tac Slope',
                                   options=['Any',
                                            '0 Nominal',
                                            '1 +5%',
                                            '2 Ba Mode',
                                            '3 -5%',
                                            '6 +10%',
                                            ],
                                   index=1)
        pha_priority = col2.selectbox(label='PHA Priority',
                                      options=['Any',
                                               'Normal',
                                               'Li mode',
                                               'Ba mode',
                                               ],
                                      index=1,
                                      )
        sumbr = col1.checkbox(label='&Sigma;BR &le; DCR', value=True)
        br2tcr = col2.checkbox(label='BR2 &le; TCR', value=True)
        rltbr = col1.checkbox(label='Rn &le; BRn', value=True)
        triples = col2.checkbox(label='Triples Only', value=True)
        # fields for BR max filtering
        col0, col1, col2 = st.columns([1.25, 1, 1])
        br0max = col0.text_input(label='Max BR 0',
                                 help='Ignore spins with BRn > max value. Leave empty for no filtering.')
        br1max = col1.text_input(label='BR1', )
        br2max = col2.text_input(label='BR2', )
        # fields for direct rate filter
        col1, col2, col3 = st.columns([1.25, 1, 1])
        ratefilt = col1.selectbox(label='Filter by Rate',
                                  options=['None',
                                           'MSS',
                                           'FSR',
                                           'DCR',
                                           'TCR',
                                           ],
                                  index=0
                                  )
        ratestep = col2.text_input(label='in DV Step', value='63')
        ratelimit = col3.text_input(label='Limit', value='10000')
        #####################
        # PHAs Download
        #####################
        col1, col2 = st.columns(2)
        get_phas = col1.form_submit_button(label='Download PHAs',)
                                           # help='Download up to one million PHAs passing the filters')
        download_expander = st.expander(label='Download PHA Ranges')
        with download_expander:
            col1, col2 = st.columns(2)
            dwnld_mass = col1.text_input(label='Mass Range', value='', help='e.g. 50.0-60.5')
            dwnld_mpq = col2.text_input(label='MPQ Range', value='', help='e.g. 1.5-5.5')

        plot_title = st.text_area(label='Plot title', value='',
                                  placeholder=('Leave empty to create plot titles '
                                               'automatically from the data and filters'),
                                  )
        #####################
        # M/MPq plot layout
        #####################
        col1, col2 = st.columns([3, 4])
        mmpq_plot = col1.form_submit_button('Plot M-MPQ')
        mmpq_csv_plot = col2.form_submit_button('Plot MMPQ csv file')
        mmq_expander = st.expander(label='M-MPQ Plot Parameters')
        with mmq_expander:
            col1, col2 = st.columns(2)
            mmqxrange = col1.text_input(label='X Low, X High', value='0.5, 100')
            mmqyrange = col2.text_input(label='Y Low, Y High', value='0.5, 100')
            col1, col2, col3 = st.columns(3)
            mmqzmin = col1.text_input(label='Z Min', value=1)
            mmqzmax = col2.text_input(label='Z Max', value=0,
                                      help="Use 0 to get zmax from the data")
            mmqbinw = col3.text_input(label='Bin width', value=0.01)
            col1, col2 = st.columns((4, 6))
            mmpq_log = col1.checkbox(label='log', value=True,
                                    help='Deselect to use linear mass and mpq axes')
            normalize = col2.checkbox(label='Normalize',
                                    help='Normalize PHAs with BR/R for each spin')
            mmpq_csv_files = st.file_uploader("Choose one or more M-MPQ csv files to add together and plot.",
                                              type=['csv', 'xlsx'],
                                              accept_multiple_files=True)

        #####################
        # E/T plot layout
        #####################
        col1, col2 = st.columns([3, 4])
        et_plot = col1.form_submit_button('Plot E-T')
        et_plot_csv = col2.form_submit_button('Plot ET csv file')
        et_expander = st.expander(label='E-T Plot Parameters')
        with et_expander:
            col1, col2 = st.columns(2)
            etxrange = col1.text_input(key='etx', label='X Low, X High', value='0, 500')
            etyrange = col2.text_input(key='ety', label='Y Low, Y High', value='0, 500')
            col1, col2, col3 = st.columns(3)
            with col1:
                etzmin = st.text_input(key='etzl', label='Z Min', value=1)
            etzmax = col2.text_input(key='etzh', label='Z Max', value=0,
                                     help="Use 0 to get zmax from the data")
            etbinw = col3.text_input(key='etbw', label='Bin width', value=1.0)
            col1, col2 = st.columns(2)
            ettic = col1.text_input(label='Tic spacing', value=50,
                                    help='Set to 0 for automatic tic spacing')
            et_csv_files = st.file_uploader("Choose one or more E-T csv files to add together and plot",
                                            type=['csv', 'xlsx'],
                                            accept_multiple_files=True)

        #####################
        # Trend plot layout
        #####################
        col1, col2 = st.columns([.9, 1.1])
        # spacer = col1.markdown('&nbsp;', unsafe_allow_html=True)
        trend_plot = col1.form_submit_button('Plot Trend')
        filtr_trend = col1.checkbox(label='With filters', value=False)
        help = 'To reduce the number of points being plotted, only plot values with period counter equal 0'
        logy = col1.checkbox(label='Log Y', value=False)
        trend_yrange = col1.text_input(key='trndy', label='Y Low, High', value='-1000, -1000',
                                       help='Use -1000 to set that limit automatically')
        trend_labels = ['SSD', 'FSR', 'DCR', 'TCR', 'BR0', 'BR1', 'BR2', 'R0', 'R1', 'R2',
                        'R0/BR0', 'R1/BR1', 'R2/BR2',
                        'Dst', 'Kp', 'PAPS_lvl', 'PAPS_kv', 'L', 'DPU_Mode', 'TAC_Slope',
                        'PHA_Priority', 'Voltage_Step', 'MLTH', 'ME', 'MAGLAT', 'MAGLON',]
        trend_items = col2.multiselect(label='Trend Value',
                                       label_visibility='collapsed',
                                       options=trend_labels,
                                       max_selections=9,
                                       default=['L'],
                                       # index=0,
                                       help='Select the range of ME values for ME trends '
                                            'in the box next to the Downld MEs button below.')
        pc0_only = st.checkbox(label='Period Cntr 0 Only', value=False, help=help)
        #####################
        # ME matrix plot layout
        #####################
        mes_plot = st.form_submit_button('Plot ME Matrix')
        plot_mes_expander = st.expander(label='ME Matrix Plot Params')
        with plot_mes_expander:
            col1, col2 = st.columns(2)
            nqrange = col1.text_input(key='nq', label='NQ Low, NQ High', value='0, 96')
            nmrange = col2.text_input(key='nm', label='NM Low, NQ High', value='0, 28')
            col1, col2 = st.columns(2)
            mezmin= col1.text_input(key='mezmin', label='Z Min (0 for auto)', value=1.)
            mezmax= col2.text_input(key='mezmax', label='Z Max (0 for auto)', value=0.)

        # Get matrix elements layout
        col1, col2 = st.columns(2)
        get_mes = col1.form_submit_button('Dwnld MEs')
        spacer = col1.markdown('(uses appropriate filters)', unsafe_allow_html=True)
        merange = col2.text_input(label='ME Low, ME High', value='0, 490')

        if get_phas:
            action = 'get_phas'
            xlo, xhi = etxrange.split(',')
            ylo, yhi = etyrange.split(',')
            plot_params = PlotParams(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                     bin_width=etbinw, tic=ettic, zmax=etzmax,)
        elif get_mes:
            action = 'get_mes'
            xlo, xhi = nqrange.split(',')
            ylo, yhi = nmrange.split(',')
            plot_params = PlotParams(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                     zmin=mezmin, zmax=mezmax, bin_width=1,)
        elif et_plot or et_plot_csv:
            action = 'plot_et_csv' if et_plot_csv else 'plot_et'
            etbinw = max(1., float(etbinw))
            xlo, xhi = etxrange.split(',')
            ylo, yhi = etyrange.split(',')
            plot_params = PlotParams(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                     bin_width=etbinw, zmin=etzmin, zmax=etzmax, tic=ettic,
                                     csv_files=et_csv_files)
        elif mmpq_plot or mmpq_csv_plot:
            action = 'plot_mmpq_csv' if mmpq_csv_plot else 'plot_mmpq'
            xlo, xhi = mmqxrange.split(',')
            ylo, yhi = mmqyrange.split(',')
            plot_params = PlotParams(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                     bin_width=mmqbinw, zmin=mmqzmin, zmax=mmqzmax,
                                     mmpq_log=mmpq_log, normalize=normalize,
                                     csv_files=mmpq_csv_files)
        elif mes_plot:
            action = 'plot_mes'
            xlo, xhi = nqrange.split(',')
            ylo, yhi = nmrange.split(',')
            plot_params = PlotParams(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                                     zmin=mezmin, zmax=mezmax, bin_width=1,)
        else:
            # trend plot
            try:
                ylo, yhi = trend_yrange.split(',')
            except ValueError:
                ylo = yhi = -1000
            plot_params = PlotParams(xlo=0, xhi=1, ylo=ylo, yhi=yhi,
                                     bin_width=1, trend_items=trend_items,)
            if trend_items == ['ME']:
                action = 'plot_me_trend'
            else:
                action = 'plot_trend'

        return ({'action': action,
                 'dates': dates,
                 'l_range': [llow, lhi],
                 'dpps_range': [dvlow, dvhi],
                 'mlth_range': [mlthlow, mlthhi],
                 'dst_range': [dstlow, dsthi],
                 'kp_range': [kplow, kphi],
                 'ssdid': ssdid,
                 'range': range,
                 'paps_levels': paps,
                 'dpu_modes': dpumodes,
                 'tac_slope': tac_slope,
                 'pha_priority': pha_priority,
                 'sumbr': sumbr,
                 'br2tcr': br2tcr,
                 'rltbr': rltbr,
                 'br0max': br0max,
                 'br1max': br1max,
                 'br2max': br2max,
                 'triples': triples,
                 'rate_filt': [ratefilt, ratestep, ratelimit],
                 'dwnld_mass': dwnld_mass,
                 'dwnld_mpq': dwnld_mpq,
                 'plot_title': plot_title,
                 'filter_trend': filtr_trend,
                 'logy': logy,
                 'pc0_only': pc0_only,
                 'merange': [int(val) for val in merange.split(',')],
                 },
                plot_params
                )


def main():
    form_data, plot_params = ampte_layout()
    action = form_data['action']
    if action in ['plot_trend', 'plot_me_trend'] and plot_params.trend_items == []:
        st.text("You must select at least one trend item to make a trend plot.")
        return
    if action in ['plot_mmpq_csv', 'plot_et_csv'] and len(plot_params.csv_files) == 0:
        plot_type = action.split('_')[1].upper()
        st.text(f'Please upload a csv file to plot in the {plot_type} Plot Parameters dropdown.')
        return

    if form_data['dates'].strip() == '' and action not in ['plot_mmpq_csv', 'plot_et_csv']:
        st.markdown(open('README.md').read(), unsafe_allow_html=True)
        return

    ampte_plotter = AmpteChemPlotter(form_data=form_data, plot_params=plot_params)
    if (ampte_plotter.df is None and ampte_plotter.hist2d is None) or \
            (ampte_plotter.df is not None and ampte_plotter.df.empty):
        st.text(f'No data with these filters\n{ampte_plotter.filters}')
        return

    logging.info(f'{action}, {ampte_plotter.filters}\n\n')
    if action in ['plot_mmpq_csv', 'plot_et_csv']:
        csv_file = plot_params.csv_files[0]
        csv_file.seek(0)
        if csv_file.name.endswith('csv'):
            filts = csv_file.readline()[2:-1].decode('utf-8').replace('Filters: ', '')
        else:
            filts = load_workbook(csv_file).active['A1'].value[2:]

        csv_file.seek(0)
    else:
        filts = [val for val in ampte_plotter.filters.values() if val]
    st.text(f'Filters: {filts}')

    # get the name of the column containing date data, default to SpinStartTime
    # date_column = ampte_plotter.df.select_dtypes('datetime64[ns]')
    # if date_column.empty:
    #     date_column = 'SpinStartTime'
    # else:
    #     date_column = date_column.columns[0]
    #
    # try:
    #     st.text(f'Filters: {ampte_plotter.filters}\n'
    #             f'Data: Date range: {ampte_plotter.df[date_column].min():%Y:%j:%H%M} to {ampte_plotter.df[date_column].max():%Y:%j:%H%M}')
    # except KeyError:
    #     st.text(f'Filters: {ampte_plotter.filters}\n'
    #             f'Data: Date range: {ampte_plotter.df.index.get_level_values(date_column).min():%Y:%j:%H%M} to {ampte_plotter.df.index.get_level_values(date_column).max():%Y:%j:%H%M}')
    # if 'L' in ampte_plotter.df.columns or 'L' in ampte_plotter.df.index.names:
    #     try:
    #         st.text(f'      PAPS: {list(ampte_plotter.df.index.get_level_values("PAPS_kv").unique())},  L: {ampte_plotter.df.index.get_level_values("L").min():.3} - {ampte_plotter.df.index.get_level_values("L").max():.3}\n'
    #                 f'      TAC Slope: {list(ampte_plotter.df.index.get_level_values("TAC_Slope").unique())},  PHA Priority: {[chr(p) for p in ampte_plotter.df.index.get_level_values("PHA_Priority").unique()]}',
    #                 )
    #     except KeyError:
    #         st.text(f'      PAPS level: {list(ampte_plotter.df["PAPS_lvl"].unique())},  L: {ampte_plotter.df["L"].min():.3} - {ampte_plotter.df["L"].max():.3}\n'
    #                 f'      TAC Slope: {list(ampte_plotter.df["TAC_Slope"].unique())},  PHA Priority: {[chr(int(p)) for p in ampte_plotter.df["PHA_Priority"].unique()]}',
    #                 )

    if action == 'get_phas':
        zipfile = ampte_plotter.make_pha_zip_file()
        zipfile_b64 = base64.b64encode(zipfile)
        dl_link = f'<a href="data:application/octet-stream;base64,{zipfile_b64.decode()}" download="AMPTE_CHEM_PHAs.zip">Download PHA data</a>'
        st.markdown(dl_link, unsafe_allow_html=True)
    elif action in ['plot_et', 'plot_et_csv']:
        if action == 'plot_et':
            zipfile = ampte_plotter.make_hist2d_zip_file()
            zipfile_b64 = base64.b64encode(zipfile)
            dl_link = (f'<a href="data:application/octet-stream;base64,{zipfile_b64.decode()}" '
                       'download="AMPTE_CHEM_ET_Hist.zip">Download 2D Histogram data</a>')
            st.markdown(dl_link, unsafe_allow_html=True)
        fig = ampte_plotter.plot_2d_hist(xval='TOF', yval='Energy', logxy=False, )
        st.pyplot(fig)
    elif action in ['plot_mmpq', 'plot_mmpq_csv']:
        if action == 'plot_mmpq':
            zipfile = ampte_plotter.make_hist2d_zip_file()
            zipfile_b64 = base64.b64encode(zipfile)
            dl_link = (f'<a href="data:application/octet-stream;base64,{zipfile_b64.decode()}" '
                       'download="AMPTE_CHEM_MMpQ_Hist.zip">Download 2D Histogram data</a>')
            st.markdown(dl_link, unsafe_allow_html=True)
        logxy = plot_params.mmpq_log
        fig = ampte_plotter.plot_2d_hist(xval='MPQ', xlabel='Mass/Charge (amu/e)',
                                         yval='Mass_amu', ylabel='Mass (amu)',
                                         logxy=logxy, )
        st.pyplot(fig)
    elif action == 'plot_trend':
        zipfile = ampte_plotter.make_trend_zip_file()
        zipfile_b64 = base64.b64encode(zipfile)
        dl_link = (f'<a href="data:application/octet-stream;base64,{zipfile_b64.decode()}" '
                   f'download="AMPTE_CHEM_{plot_params.trend_items}_trend.zip">Download Trend data</a>')
        st.markdown(dl_link, unsafe_allow_html=True)
        fig = ampte_plotter.plot_trend()
        if isinstance(fig, mpl.figure.Figure):
            st.pyplot(fig)
        else:
            st.plotly_chart(fig, use_container_width=True,
                            config={'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d']})

    elif action == 'plot_me_trend':
        zipfile = ampte_plotter.make_trend_zip_file()
        zipfile_b64 = base64.b64encode(zipfile)
        dl_link = (f'<a href="data:application/octet-stream;base64,{zipfile_b64.decode()}" '
                   f'download="AMPTE_CHEM_ME_trend.zip">Download Trend data</a>')
        st.markdown(dl_link, unsafe_allow_html=True)
        fig = ampte_plotter.plot_me_trend()
        # st.pyplot(fig)
        st.plotly_chart(fig, use_container_width=True,
                        config={'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d']})
    elif action == 'plot_mes':
        st.pyplot(ampte_plotter.plot_me_matrix())
        # st.image('MEs.png')
    elif action in ['get_mes', ]:
        zipfile = ampte_plotter.make_me_zip_file()
        zipfile_b64 = base64.b64encode(zipfile)
        dl_link = f'<a href="data:application/octet-stream;base64,{zipfile_b64.decode()}" download="AMPTE_CHEM_MEs.zip">Download MEs zip file</a>'
        st.markdown(dl_link, unsafe_allow_html=True)
        # melow, mehigh = [int(hl)+10 for hl in form_data['merange']]
        # cols = list(range(10)) + list(range(melow, mehigh+1))
        # df = ampte_plotter.df.iloc[:, cols]
        # st.text(df.to_string())


if __name__ == '__main__':
    main()