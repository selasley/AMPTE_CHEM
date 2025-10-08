<h1 style="text-align: center">AMPTE WebApp</h1>

AMPTE WebApp (AWA) is a [streamlit](https://streamlit.io)-based python application for plotting data from [AMPTE CHEM](https://space.umd.edu/projects/ampte/ampte.html). Trend, 2D Histogram and Matrix Element plots are available.  Data from the plots can be saved to csv files.  AWA can plot data from csv or excel files in the same format as the saved files.

<hr style="width:50%;margin:auto"><br />

<h3 style="text-align: center">Getting Started</h3>

First fill in the **Time ranges** text box.  Time ranges can include multiple non-contiguous, non-overlapping intervals, each in the form YYYY1-DOY1-HHMM1\:YYYY2-DOY2-HHMM2. If DOY1 and DOY2 are in the same year, YYYY1-DOY1-HHMM1\:DOY2-HHMM2.

Select any filters you want to apply then use the XXX Plot Parameters menu to customize the plotting parameters before clicking the Plot XXX button.  The Download PHAs button can be used to download filtered raw PHA data instead of plotting it.

<hr style="width:50%;margin:auto"><br />

<h3 style="text-align: center">Filtering</h3>

Filters can be used to reject times not matching the filter.  For example, setting the DPPS filter to 57 and 59 and the PAPS Levels filter to 5 will only collect data for times when the DV Step is 57, 58 or 59 and the PAPS level is 5.  Filtering by detector rate operates slightly differently than the other filters.  An entire cycle is rejected if the rate filter condition is matched.  For example, if the filter is FSR in DV step 63 < 1000 and a cycle starts with FSR in DV Step 63 > 1000, the times until the start of the next cycle (DV Step 63) are all rejected.  The MLT filter can handle time ranges with MLT low > high, such as 1400 to 0200.  Filters can be used with M-MPQ, E-T, and Trend plots.  The Triples, Ranges and MSS ID filters do not apply to Trend plots.  The Triples, Ranges, MSS ID and Rate filters do not apply to ME plots.  Leaving a filter box empty will skip that filter.  For example, leaving Kp Dst Low empty and setting Dst High to 0 will cause the Dst filter to be Dst <= 0.  Some filters such as Tac Slope and PHA Priority can be set to Any to skip filtering on those values.  Filter by Rate can be set to None to skip filtering on detector rates.  If a rate filter is used, a file with the times rejected by the filter is included in the zip archive that is linked to above the plots.

<hr style="width:50%;margin:auto"><br />

<span style="font-size:115%; color:darkorange">Download PHAs</span>: This button will display a link to a zip archive of a csv file containing PHA data in the time ranges that pass the filters.  The csv file also contains housekeeping data.  If a rate filter is used, a file with the times rejected by the filter is included in the zip archive.  The Download PHA Ranges box allows filtering by mass and mass per charge ranges.  Note that only the first 1 million PHAs are saved in the csv file in the zip archive.


<span style="font-size:115%; color:darkorange">Plot M-MPQ</span>: This button will display a download link to a csv file containing a 2D Mass - Mass Per Charge histogram and a static plot of the histogram.  Plot parameters can be customized by clicking the M-MPQ Plot Parameters disclosure triangle.


<span style="font-size:115%; color:darkorange">Plot E-T</span>: This button will display a download link to a csv file containing a 2D Energy - Time of Flight histogram and a static plot of the histogram.  Plot parameters can be customized by clicking the E-T Plot Parameters disclosure triangle.

<hr>

Update Oct 1, 2025:
* Y limits now work in trend plots

Update Mar 25, 2023:</span>
* You can now plot M-MPq plots with linear axes
* You can set tick spacing to 0 in E-T plots to automatically place tics.

Update Feb 21, 2023:
* Fixed a bug updating the total and number plotted values in the title of mmpq csv plots if the original histograms were normalized

Update Feb 6, 2023:
* Trend plots with more than 100,000 points are plotted with a non-interactive backend to avoid bogging down or crashing the app
* You can enter y limits for non-interactive trend plots.  Use -1000 to set the limit automatically.  For example 50, -1000 will set the low y limit to 50 and set the high y limit based on the data.
* The ∑BR≤DCR, BR2≤TCR and Rn≤BRn filters are now included in plot titles
* Bug fix: BR filters are only applied to trend plots if the "With filters" checkbox is checked
* Enhancement: Total and number of PHAs plotted is updated in the title of summed csv file plots

Update Jan 30, 2023:
* Added sanity check filters Sum BR < DCR, BR2 < TCR, RateCountn < BRn
* Fixed a bug computing y bins when plotting from csv files
* You can select multiple csv files for plotting.  The values will be summed.

Update Jan 5, 2023:
* New rates data file with better Range Counts.  The problem counting PHAs in duplicated times in a single Range Count value has been fixed.  Mostly.
* M/MPQ plots can now be "normalized" by plotting histograms of the sum of BR/R instead of the some of the number of PHAs.  BRs are summed over all sectors.  Not tested over a wide range of inputs and BR/R histograms have not been verified by hand yet.  Note that BRs have spikes, possibly from data hits, which can create "hot pixels" in the histograms.
* BR filters are available for M/MPQ plots.  The filters can be used for non-normalized and normalized plots.
* Ratio trend labels now include the /
* Increased maxMessageSize to 300 to allow up to 300MB files to be downloaded.


Update Dec 27, 2022:

* New rates data file with the detector rates labeled correctly (FSR;DCR;TCR;MSS). The rates were previously labeled following the FITS files' internal documentation (SSD;FSR;DCR;TCR)
* Log Y checkbox for trend plots
* You can now plot up to 9 values in trend plots.  ME trends have not been tested yet with this version of AWA


Update Dec 15, 2022:

* Trend plots now include BR/R ratios
* BRs and Range Counts are summed over spins in trend plots.  Most of the APL data files contain repeated times which currently affect summing over cycles.  For one of the worst cases, 1985-247, look at a trend plot of BRs without filters from 1985-246-0400-248-0400.  Data in the duplicated times seems to be OK but the timestamps that are bad.  I am working on either correcting the timestamps or a new method for summing over cycles that ignores time.

As of September 16, 2022 the PHA data file does not contain times with calibration data (when CSHK .ne. 0), and does not contain the time range 1984-319 19:30 to 1984-325 00:20 with an unusual PHA distribution.  See the [AMPTE web page](https://voyager-mac.umd.edu/ampte/) for information.  The rates+housekeeping and matrix element data files do contain calibration times and the unusual time range.

