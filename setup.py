import setuptools
from codecs import open
from os import path
import warnings
import glob

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))


# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="ehrzero",
    version = str(version['__version__']),
    author='Ishanu Chattopadhyay, Dymtro Onishchenko',
    author_email="ishanu@uchicago.edu",
    description="Pipeline for autism prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zeroknowledgediscovery/",
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
    classifiers=[\
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps."]
)
