import setuptools
from setuptools import setup
from codecs import open
from os import path
# import subprocess as sp
import warnings
import glob

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))


with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="ehrzero",
    version = str(version['__version__']),
    author='Dymtro Onishchenko, Ishanu Chattopadhyay ',
    author_email="ishanu@uchicago.edu",
    description="Zero-Knowledge Risk Oracle for predictive diagnoses of childhood neuropsychiatric disorders from sparse  electronic health records",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://zed.uchicago.edu/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    # package_data={
    #     "": ["*.dat"]
    # },
    install_requires=[
        "pandas>=0.23.4",
        "scikit-learn>=0.20.3",
        "lightgbm>=2.2.3",
        "matplotlib>=3.0.2"

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
