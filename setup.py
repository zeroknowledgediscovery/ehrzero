import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ehrzero",
    version="1.0.7",
    author='Ishanu Chattopadhyay, Dymtro Onishchenko',
    author_email="ishanu@uchicago.edu",
    description="Pipeline for autism prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://pypi.python.org/pypi/ehrzero/",
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
