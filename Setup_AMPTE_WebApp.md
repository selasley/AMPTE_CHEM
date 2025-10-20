<span style="font-size: 0.8rem">October 18, 2025</span>

<h1 style="text-align: center">AMPTE WebApp Setup</h1>

First create the hdf files AWA uses as described in Create\_AWA\_Datafiles.md.  Then clone the AMPTE CHEM repository on Github, or download a zipped archive by clicking the green "<> Code" button.  

Change to the ampte\_chem\_webapp directory nd follow the instructions for running AWA with Docker or manually.

<h2 style="text-align: center">Using Docker or Podman</h2>

The easiest way to run the AMPTE WEbApp is using Docker or Podman.  If Docker is not already on your system you can download and install [Docker Desktop](https://www.docker.com) for Macos, Windows, or x86-64 based linux systems, or [install Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) on linux systems.  [Podman Desktop](https://podman-desktop.io) is another popular container system.  Replace docker with podman in the commands below if you are using it.

Start Docker Desktop and use the command below from within the ampte\_chem\_webapp directory.  This will create a docker image named ampte\_chem\_webapp and run the streamlit app in a container named awa based on the image 

<pre>docker compose up -d</pre>

Or you can manually create the image and run the container with these commands.  (note the . at the end of the image build command)-

<pre>
docker image build -t ampte_chem_webapp .

docker container run -d \
  -p 8501:8501 \
  --name awa \
  --cap-drop=ALL \
  -v $(pwd)/../awa_data/:/ampte_chem_webapp/data/:ro \
  ampte_chem_webapp
</pre>

AWA assumes the AMPTE CHEM hdf files are located in a directory named awa\_data in the parent directory of the ampte\_chem\_webapp directory.

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser to use the webapp.  

<div style="text-align:center; margin: 0 auto; ">
  <hr style="width: 75%">
  <h1>Manual Setup</h1>
</div>

**NOTE: A Fortran compiler is needed to set up the AMPTE CHEM WebApp**   
See scipy's [System-level dependencies](https://scipy.github.io/devdocs/building/index.html#system-level-dependencies) website for compatibility information.

These instructions for macOS or linux systems may need changes to work in Windows systems.

1. Install [uv](https://docs.astral.sh/uv/) from Astral if it is not installed on your system
		
	For macos and linux use the command 
	<pre>curl -LsSf https://astral.sh/uv/install.sh | sh</pre>
	
	or, if your system does not have curl,
	
	<pre>wget -qO- https://astral.sh/uv/install.sh | sh</pre>
	
	For powershell in windows
	<pre>powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"</pre>
	
2. Change to the ampte\_chem\_webapp directory.  Use uv to install a python 3.13 virtual environment with the packages required to run AWA, then activate the virtual environment.

	<pre>uv sync
	source .venv/bin/activate
	</pre>

3. Use numpy.f2py to compile chem\_pha\_converters.f90 to a module that can be imported.  f2py requires a Fortran compiler.

	<pre>python3 -m numpy.f2py -c chem_pha_converters.f90 -m chem_pha_converters</pre>

4. Copy or symlink the AMPTE hdf files from Create\_AWA\_Datafiles.md into the data/ directory.


5. Start the streamlit app then visit [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser

	<pre>python3 -m streamlit run awa_streamlit.py</pre>
