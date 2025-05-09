
[![DOI](https://zenodo.org/badge/923830097.svg)](https://doi.org/10.5194/egusphere-2025-1633) [![License](https://img.shields.io/github/license/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/blob/main/LICENSE) [![PyPI version](https://badge.fury.io/py/bakaano-hydro.svg)](https://pypi.org/project/bakaano-hydro/)
 [![GitHub release](https://img.shields.io/github/v/release/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/releases) [![Last Commit](https://img.shields.io/github/last-commit/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/commits/main) [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) 


# Bakaano-Hydro

## Description

Bakaano-Hydro is a distributed hydrology-guided neural network model for streamflow prediction. It uniquely integrates physically based hydrological principles with the generalization capacity of machine learning in a spatially explicit and physically meaningful way. This makes it particularly valuable in data-scarce regions, where traditional hydrological models often struggle due to sparse observations and calibration limitations, and where current state-of-the-art data-driven models are constrained by lumped modeling approaches that overlook spatial heterogeneity and the inability to capture hydrological connectivity. 

By learning spatially distributed, physically meaningful runoff and routing dynamics, Bakaano-Hydro is able to generalize across diverse catchments and hydro-climatic regimes. This hybrid design enables the model to simulate streamflow more accurately and reliably—even in ungauged or poorly monitored basins—while retaining interpretability grounded in hydrological processes.

The name Bakaano comes from Fante, a language spoken along the southern coast of Ghana. Loosely translated as "by the river side" or "stream-side", it reflects the  lived reality of many vulnerable riverine communities across the Global South - those most exposed to flood risk and often least equipped to adapt.

![image](https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf)

## Key Features
- **Distributed architecture**: Captures spatial heterogeneity of hydrological processes using gridded runoff and flow routing.
- **Hybrid modeling**: Combines physically based hydrology with deep learning for enhanced accuracy and realism.
- **Scalable and generalizable**: Trains a single model across basins, regions, or continents—no need for basin-specific calibration.
- **Reliable in data-scarce regions**: Designed to perform well even with sparse observational data.
- **High-performance ready**: Compatible with GPU acceleration for fast training and inference on large-scale datasets.
- **Seamless integration**: Modular components allow for easy adaptation with other runoff models, routing schemes, or neural network architectures.
- **Automated end-to-end pipeline**: From climate data ingestion and preprocessing to runoff simulation, routing, and streamflow prediction—Bakaano-Hydro automates the entire workflow with minimal user intervention.
- **Easy deployment**: Installable via pip and designed with reproducibility in mind.
- **Versatile applications**: Suitable for streamflow forecasting, climate adaptation planning, flood risk assessment, and more.

## Installation

Bakaano-Hydro is built on TensorFlow and is designed to leverage GPU acceleration for training. This requires a system with an NVIDIA GPU installed or bundled CUDA and cuDNN runtime libraries.
GPU acceleration is strongly recommended for training deep learning components and running large-scale simulations, as it significantly improves speed and scalability.

If you have a compatible NVIDIA GPU and drivers installed, install with:

```bash
pip install bakaano-hydro[gpu]
```

This will automatically install the correct version of TensorFlow along with CUDA and cuDNN runtime libraries

If you do not have access to a GPU, you can still install and use Bakaano-Hydro in CPU mode (e.g., for inference, testing or small-scale evaluation):

```bash
pip install bakaano-hydro
```

Note: Training on CPU is supported but will be significantly slower, especially on large datasets or deep learning tasks.

## Getting started / Example notebooks

Bakaano-Hydro requires three primary data or inputs

1. Shapefile of study area or river basin
2. Observed streamflow data in NetCDF format from Global Runoff Data Center (<https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Home>). Because Bakaano-Hydro aims to use only open-source data, it currently accepts observed streamflow data only from GRDC.
3. Registration at Google Earth Engine (<https://code.earthengine.google.com/register>). Bakaano-Hydro retrieves, NDVI, tree cover and meteorological variables from ERA5-land or CHIRPS from Google Earth Engine Data Catalog. This platform requires prior registration for subsequent authentication during execution of the model

Model execution then involves only a few guided steps. See the quick start notebook <https://github.com/confidence-duku/bakaano-hydro/blob/main/quick_start.ipynb> for guidance.

## Code architecture

![bakaanohydro-2025-04-16-132235](https://github.com/user-attachments/assets/4a98f643-b703-4352-bfd7-3d4d13e9fdfe)

## Support

For assistance, please contact Confidence Duku (<confidence.duku@wur.nl>)

## Contributing

No contributions are currently accepted.

## Authors and acknowledgment

See CITATION.cff file.

Bakaano-Hydro was developed as part of Wageningen University & Research Investment theme 'Data-driven discoveries in a changing climate' and also as part of the KB program 'Climate resilient land use'.

## License

Apache License
