# ***** NO LONGER USED *****
# https://stackoverflow.com/questions/44043990/how-to-build-a-python-wheel-with-compiled-fortran-extension-module-without-requi/49765622#49765622
# https://stackoverflow.com/questions/14453208/mixing-f2py-with-distutils
# https://stackoverflow.com/questions/21136266/typeerror-dist-must-be-a-distribution-instance

import setuptools
from numpy.distutils.core import Extension, setup


setup(
    name="chem_pha_converters",
    version='3.2',
    packages = setuptools.find_packages(),
    package_data = {'': ['*.f90']},
    include_package_data = True,
    ext_modules=[
        Extension(name="chem_pha_converters",
                  sources=["chem_pha_converters.f90"],
                  extra_link_args=["-static", "-static-libgfortran", "-static-libgcc"]
                  )
    ]
)
