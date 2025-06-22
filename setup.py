import os
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bakaano-hydro",
    version="1.2.9",
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
        "isimip-client==1.0.1",
        "keras==3.6.0",
        "localtileserver==0.10.6",
        "leafmap==0.43.6",
        "keras-tcn==3.5.6",
        "matplotlib==3.9.2",
        "netCDF4==1.7.2",
        "numpy==1.26.4",
        "slicer==0.0.7",
        "pandas==2.2.3",
        "pysheds==0.3.3",
        "rasterio==1.4.2",
        "requests==2.32.3",
        "rioxarray",
        "scipy==1.14.1",
        "shapely==2.0.6",
        "tensorflow==2.18.0",
        "tensorflow_probability==0.25.0",
        "tf_keras==2.18.0",
        "xarray==2024.10.0",
        "tqdm==4.67.1",
        "scikit-learn==1.5.2"
    ],
    extras_require={
        "gpu": [
            "nvidia-cublas-cu12==12.5.3.2",
            "nvidia-cuda-cupti-cu12==12.5.82",
            "nvidia-cuda-nvcc-cu12==12.5.82",
            "nvidia-cuda-nvrtc-cu12==12.5.82",
            "nvidia-cuda-runtime-cu12==12.5.82",
            "nvidia-cudnn-cu12==9.3.0.75",
            "nvidia-cufft-cu12==11.2.3.61",
            "nvidia-curand-cu12==10.3.6.82",
            "nvidia-cusolver-cu12==11.6.3.83",
            "nvidia-cusparse-cu12==12.5.1.3",
            "nvidia-nvjitlink-cu12==12.5.82"
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="hydrology earth system flood forecasting deep learning streamflow prediction climate risk",
    license="Apache 2.0",
    project_urls={
        "Source": "https://github.com/confidence-duku/bakaano-hydro",
        "Bug Tracker": "https://github.com/confidence-duku/bakaano-hydro/issues",
        "Documentation": "https://github.com/confidence-duku/bakaano-hydro#readme",
    },
)
