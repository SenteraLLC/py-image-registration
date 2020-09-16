"""This performs feature based multiband alignment for multispectral images"""

import re
import setuptools

VERSIONFILE = "multi_spect_tools/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi_spect_tools",
    version=verstr,
    description="Python multi_spect_tools utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SenteraLLC/py-image-registration",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "opencv-python", "SimpleITK"],
    extras_require={
        "dev": [ ]
    },
)