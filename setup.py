import os
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bakaano-hydro",
    version="1.3.6",
    author="Confidence Duku",
    author_email="confidence.duku@gmail.com",
    description="Distributed hydrology-guided neural network for streamflow prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/confidence-duku/bakaano-hydro",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        "dask>=2024.11,<2026",
        "earthengine-api>=1.4,<2",
        "fiona>=1.10,<2",
        "future>=1,<2",
        "geemap>=0.35,<0.40",
        "geopandas>=1.1,<2",
        "hydroeval>=0.1,<1",
        "isimip-client>=1.0,<2",
        "keras>=3.12,<4",
        "localtileserver>=0.10,<0.11",
        "leafmap>=0.43,<0.50",
        "keras-tcn>=3.5,<4",
        "matplotlib>=3.9,<4",
        "netCDF4>=1.7,<2",
        "numpy>=1.26,<2.1",
        "slicer>=0.0.7,<0.1",
        "pandas>=2.2,<2.4",
        "pysheds>=0.3,<0.4",
        "rasterio>=1.4,<2",
        "requests>=2.32,<3",
        "rioxarray>=0.18,<0.20",
        "scipy>=1.14,<2",
        "shapely>=2.0,<2.2",
        "tensorflow>=2.18,<2.20",
        "tensorflow_probability>=0.25,<0.27",
        "tf_keras>=2.18,<2.20",
        "xarray>=2024.10,<2026",
        "tqdm>=4.67,<5",
        "scikit-learn>=1.5,<1.7",
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
