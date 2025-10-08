# syntax=docker/dockerfile:1

# parts of this file are from https://hynek.me/articles/docker-uv/
# start with the uv bookworm-slim image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# streamlit apps use port 8501
EXPOSE 8501

WORKDIR /ampte_chem_webapp

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Setup a non-root user
RUN <<EOT
  groupadd --system --gid 999 nonroot
  useradd --system --gid 999 --uid 999 --create-home nonroot
EOT

COPY ["chem_pha_converters.f90", "pyproject.toml", "./"]

# create a virtual environment, install python packages, use f2py
# to compile chem_pha_converters.f90 then clean up 
RUN <<EOT
  export DEBIAN_FRONTEND=noninteractive 
  apt-get update -qy
  apt-get -qy upgrade
  apt-get -qy install \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    build-essential \
    gfortran
  uv sync
  uv run f2py -c chem_pha_converters.f90 -m chem_pha_converters
  uv cache clean
  apt-get purge -y build-essential gfortran
  apt-get autoremove -y
  rm -rf /var/lib/apt/lists/*
EOT

# copy the remaining files
COPY ./ ./

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Use the non-root user to run our application
USER nonroot

CMD ["uv", "run", "streamlit", "run", "--browser.gatherUsageStats", "false", "awa_streamlit.py"]