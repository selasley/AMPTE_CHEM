<span style="font-size: 0.8rem">October 8, 2025</span>


The AMPTE CHEM WebApp (AWA) uses hdf files containing pulse height data, housekeeping, rates, and matrix element data.  In retrospect hdf may not be the best choice for storing this much data.  HDF was chosen in order to have compressed local files with no need for a database server.

AMPTE CHEM [data files](http://sd-www.jhuapl.edu/AMPTE/chem/data/) are available at the [AMPTE CCE Science Data Center](http://sd-www.jhuapl.edu/AMPTE/) at the Johns Hopkins University Applied Physics Lab.  There are 1480 compressed [FITS](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/) files totaling roughly 11.4GB.  All files should be placed in the apl\_data\_files directory.  Do not place the files in subdirectories for each year.

The python script create\_mission\_hdf\_files.py  creates the hdf files AWA uses from the JHUAPL FITS files.


1. Install [uv](https://docs.astral.sh/uv/) from Astral if it is not installed on your system
		
	For macos and linux use the command 
	<pre>curl -LsSf https://astral.sh/uv/install.sh | sh</pre>
	
	or, if your system does not have curl,
	
	<pre>wget -qO- https://astral.sh/uv/install.sh | sh</pre>
	
	For powershell in windows
	<pre>powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"</pre>
	
2. Use uv to create a python 3.13 virtual environment and install the needed packages.  Setup\_AMPTE\_WebApp.md has instructions for installing uv in the Manual Setup section.
	<pre>uv venv -p 3.13 --no-project
	source .venv/bin/activate
	uv pip install -r requirements.txt
	</pre>

3. Run the create\_mission\_hdf\_files.py and have a nice cup of tea.  The script takes roughly an hour to create all of the hdf files.

	<pre>python3 create\_mission\_hdf\_files.py</pre>

This will create 7 hdf files in the awa\_data directory.  The AMPTE\_CHEM\_cal\_pha.h5 and AMPTE\_CHEM\_rates.h5 files are not used by AWA.

Once the hdf files are created you can run the web app as described in Setup\_AMPTE\_WebApp.md

The bottom of the create\_mission\_hdf\_files.ipynb notebook contains information and the first few rows of the hdf files.  View the notebook with [jupyter](https://jupyter.org)
	<pre>
	jupyter lab</pre>
