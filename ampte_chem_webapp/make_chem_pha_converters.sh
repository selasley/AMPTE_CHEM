#!/bin/sh
# export LDFLAGS="$LDFLAGS -Wl,-ld_classic"
python3 -m numpy.f2py -c chem_pha_converters.f90 -m chem_pha_converters
