[![DOI](https://zenodo.org/badge/923830097.svg)](https://doi.org/10.5194/egusphere-2025-1633) [![License](https://img.shields.io/github/license/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/blob/main/LICENSE) [![PyPI version](https://badge.fury.io/py/bakaano-hydro.svg)](https://pypi.org/project/bakaano-hydro/)
 [![GitHub release](https://img.shields.io/github/v/release/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/releases) [![Last Commit](https://img.shields.io/github/last-commit/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/commits/main) [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) 


# Bakaano-Hydro

## Overview

Bakaano-Hydro is a distributed hydrology-guided neural network model for streamflow prediction. It uniquely integrates physically based hydrological principles with the generalization capacity of machine learning in a spatially explicit and physically meaningful way. This makes it particularly valuable in data-scarce regions, where traditional hydrological models often struggle due to sparse observations and calibration limitations, and where current state-of-the-art data-driven models are constrained by lumped modeling approaches that overlook spatial heterogeneity and the inability to capture hydrological connectivity. 

By learning spatially distributed, physically meaningful runoff and routing dynamics, Bakaano-Hydro is able to generalize across diverse catchments and hydro-climatic regimes. This hybrid design enables the model to simulate streamflow more accurately and reliably—even in ungauged or poorly monitored basins—while retaining interpretability grounded in hydrological processes.

The name Bakaano comes from Fante, a language spoken along the southern coast of Ghana. Loosely translated as "by the river side" or "stream-side", it reflects the  lived reality of many vulnerable riverine communities across the Global South - those most exposed to flood risk and often least equipped to adapt.

![image](https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf)

## Documentation

Full documentation is available at:
https://confidence-duku.github.io/bakaano-hydro/

## Conceptual model

Bakaano-Hydro consists of three tightly coupled components:

**1. Distributed runoff generation**
Vegetation, soil, and meteorological drivers are used to compute grid-cell runoff using a VegET-based approach.

**2. Physically informed routing**
Runoff is routed through the river network using flow-direction-based routing (e.g. MFD/WFA), preserving spatial connectivity.

**3. Neural network**
A Temporal Convolutional Network (TCN), conditioned on static catchment descriptors, learns hydrological dynamics from physically routed runoff, enabling robust generalization across diverse basins.

The neural network augments hydrology—it does not replace it.

## Installation

Bakaano-Hydro is built on TensorFlow and supports both CPU and GPU execution.
Create new environment
```bash
  conda create --name bakaano_env python=3.10
  conda activate bakaano_env
  ```

**GPU (recommended)**
```bash
  pip install bakaano-hydro[gpu]
  ```
This installs TensorFlow with compatible CUDA and cuDNN runtime libraries as well as supported versions of dependent libraries 

CPU-only
```bash
  pip install bakaano-hydro
  ```
⚠️ CPU training is supported but can be slow for large basins or long time series.


## Data Requirements

1. **Shapefile**: Defines the study area or river basin.
2. **Observed Streamflow Data**: NetCDF format from the Global Runoff Data Center (https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Stations)
3. **Google Earth Engine Registration**: Required for retrieving NDVI, tree cover, and meteorological data (https://earthengine.google.com/signup/).



## Project directory structure

After running Bakaano-Hydro, the working directory follows this structure:

```text
working_dir/
├── alpha_earth/                     # AlphaEarth satellite embeddings (A00–A63)
│   ├── band_A00.tif
│   ├── ...
│   └── band_A63.tif
│
├── catchment/                       # Catchment-level static descriptors
│   └── river_grid.tif
│
├── elevation/                       # DEM and derived topographic layers
│   ├── dem_clipped.tif
│   ├── hyd_glo_dem_30s.tif
│   └── hyd_glo_dem_30s.zip
│
├── ERA5/                            # ERA5-Land meteorological forcing (processed)
│   ├── precip.nc
│   ├── tasmin.nc
│   ├── tasmax.nc
│   └── tmean.nc
│
├── era5_scratch/                    # Temporary ERA5 download & reprojection files
│   └── *.tmp
│
├── models/                          # Trained Bakaano-Hydro models & scalers
│   ├── bakaano_model.keras
│   ├── alpha_earth_scaler.pkl
│   ├── runoff_scaler.pkl
│   └── response_scaler.pkl
│
├── ndvi/                            # MODIS NDVI products
│   └── daily_ndvi_climatology.pkl
│
├── predicted_streamflow_data/       # Model simulation outputs
│   ├── predicted_streamflow_lat{lat}_lon{lon}.csv
│   └── bakaano_{station_id}.csv
│
├── runoff_output/                   # Distributed runoff & routed flow tensors
│   └── wacc_sparse_arrays.pkl
│
├── scratch/                         # Temporary working files (safe to delete)
│   └── *.tmp
│
├── soil/                            # Soil hydraulic properties
│   ├── wilting_point.tif
│   ├── saturation_point.tif
│   └── available_water_content.tif
│
└── vcf/                             # Vegetation cover fractions
    ├── mean_tree_cover.tif
    ├── mean_herb_cover.tif
    └── vegetation_metadata.json
```

## Quick start

See the documentation quick start for a full runnable walkthrough:
- docs/_build/quickstart.html
## How to cite

If you use Bakaano-Hydro in academic work, please cite:

- Duku, C.: Bakaano-Hydro (v1.1). A distributed hydrology-guided deep learning model for streamflow prediction, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-1633, 2025.

- Duku, C.: Enhancing flood forecasting reliability in data-scarce regions with a distributed hydrology-guided neural network framework, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-2294, 2025.

See CITATION.cff.

## Acknowledgment

Bakaano-Hydro was developed at Wageningen Environmental Research with funding from the Netherlands Ministry of Agriculture, Fisheries, Food Security and Nature (LVVN). This work is part of the Knowledge Base (KB) programme **Climate Resilient Water and Land Use**, within the project **Compound and Cascading Climate Risks and Social Tipping Points**, and builds directly on earlier research conducted under the programme **Data-Driven Discoveries in a Changing Climate**.

## License

Apache License
