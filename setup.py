import os
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bakaano-hydro",
    version="1.1.5",
    author="Confidence Duku",
    author_email="confidence.duku@gmail.com",
    description="Distributed hydrology-guided neural network for streamflow prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/confidence-duku/bakaano-hydro",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        "dask==2024.11.2",
        "earthengine-api==1.4.3",
        "fiona==1.10.1",
        "future==1.0.0",
        "geemap==0.35.1",
        "geopandas==1.0.1",
        "hydroeval==0.1.0",
        "ipython==8.12.3",
        "ipywidgets==8.1.5",
        "isimip-client==1.0.1",
        "keras==3.6.0",
        "keras-tcn==3.5.6",
        "matplotlib==3.9.2",
        "netCDF4==1.7.2",
        "numpy==1.26.4",
        "OWSLib==0.32.0",
        "pandas==2.2.3",
        "pyproj==3.7.0",
        "pysheds==0.3.3",
        "rasterio==1.4.2",
        "requests==2.32.3",
        "richdem==0.3.4",
        "rioxarray==0.18.1",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "shapely==2.0.6",
        "tensorflow==2.18.0",
        "tensorflow-addons==0.23.0",
        "tensorflow_probability==0.25.0",
        "tf_keras==2.18.0",
        "whitebox==2.3.5",
        "xarray==2024.10.0",
        "tqdm==4.67.1"
    ],
    extras_require={
        "gpu": [
            "tensorflow[and-cuda]==2.18.0"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="flood forecasting hydrology deep learning distributed runoff tcn",
    license="Apache 2.0",
    project_urls={
        "Source": "https://github.com/confidence-duku/bakaano-hydro",
        "Bug Tracker": "https://github.com/confidence-duku/bakaano-hydro/issues",
        "Documentation": "https://github.com/confidence-duku/bakaano-hydro#readme",
    },
)
