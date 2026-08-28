[![Paper DOI](https://img.shields.io/badge/DOI-10.5194%2Fegusphere--2025--1633-blue)](https://doi.org/10.5194/egusphere-2025-1633) [![License](https://img.shields.io/github/license/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/blob/main/LICENSE) [![PyPI version](https://badge.fury.io/py/bakaano-hydro.svg)](https://pypi.org/project/bakaano-hydro/) [![GitHub release](https://img.shields.io/github/v/release/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/releases) [![Last Commit](https://img.shields.io/github/last-commit/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/commits/main) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

# Bakaano-Hydro

## What is Bakaano-Hydro?

Bakaano-Hydro is a distributed hydrology-guided neural network model for streamflow prediction. It combines physically based runoff generation and river-network routing with a Temporal Convolutional Network, so streamflow predictions remain spatially explicit and grounded in hydrological processes.

The model is designed for data-scarce and poorly monitored basins, where traditional hydrological models are difficult to calibrate and purely data-driven models often lose important spatial heterogeneity and flow connectivity.

## Who is it for?

Bakaano-Hydro is intended for hydrologists, climate-risk researchers, flood-forecasting teams, and applied machine-learning users working with basin-scale streamflow simulation, ungauged or sparsely gauged catchments, and scenario or flood-mapping workflows.

![image](https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf)

## Installation

Bakaano-Hydro is built on TensorFlow, requires Python 3.10 or newer, and supports both CPU and GPU execution.

Create a new environment:

```bash
conda create --name bakaano_env python=3.10
conda activate bakaano_env
```

**GPU (recommended)**

```bash
pip install bakaano-hydro[gpu]
```

This installs the pinned NVIDIA CUDA 12 runtime libraries used by Bakaano-Hydro. GPU execution requires a supported Linux environment, compatible NVIDIA hardware, and a sufficiently recent NVIDIA driver.

**CPU-only**

```bash
pip install bakaano-hydro
```

CPU training is supported but can be slow for large basins or long time series.

## First workflow

Minimal project setup:

```python
from bakaano.core.project import ProjectContext

working_dir = "/path/to/working_dir"
study_area = "/path/to/basin.shp"

project = ProjectContext(
    working_dir=working_dir,
    study_area=study_area,
    climate_data_source="ERA5",
)

project.validate_project(for_task="preprocess")
```

Run the preprocessing and hydrology modules, confirm that the required training artifacts exist, and then train the model:

```python
from bakaano.neuralnet.train import train_streamflow_model

project.validate_project(for_task="train")

model_path = train_streamflow_model(
    working_dir=working_dir,
    study_area=study_area,
    train_start="1981-01-01",
    train_end="2020-12-31",
    grdc_netcdf="/path/to/GRDC.nc",
    area_normalize=True,
    log_transform=True,
)
```

Recommended notebooks:

- [Beginner local workflow](quick_start_beginner.ipynb)
- [Advanced local workflow](quick_start_advanced.ipynb)
- [Scenario workflow](scenario_quickstart.ipynb)
- [Flood-mapping workflow](flood_mapper_quickstart.ipynb)

## Where are the docs?

Full documentation is available on the [Bakaano-Hydro documentation site](https://confidence-duku.github.io/bakaano-hydro/).

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Beginner%20Colab.ipynb)

- Beginner Colab: https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Beginner%20Colab.ipynb
- Advanced Colab: https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Advanced%20Colab.ipynb

## Data requirements

1. **Study-area shapefile**: Defines the river basin or watershed.
2. **Observed streamflow data**: Either a GRDC NetCDF file or per-station CSV files accompanied by a station lookup CSV. GRDC data are available through the [Global Runoff Data Centre portal](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Stations).
3. **Google Earth Engine registration**: Required for retrieving NDVI, tree cover, AlphaEarth embeddings, and ERA5/CHIRPS meteorological inputs. Register through the [Google Earth Engine website](https://earthengine.google.com/signup/). CHELSA meteorological inputs use a separate download workflow.

## Conceptual model

Bakaano-Hydro consists of three tightly coupled components:

**1. Distributed runoff generation**
Vegetation, soil, and meteorological drivers are used to compute grid-cell runoff using a VegET-based approach.

**2. Physically informed routing**
Runoff is routed through the river network using flow-direction-based routing such as MFD, D8, or D-infinity, preserving spatial connectivity. The routing step produces weighted flow-accumulation outputs used by the neural network.

**3. Neural network**
A Temporal Convolutional Network (TCN), conditioned on static catchment descriptors, learns hydrological dynamics from physically routed runoff, enabling robust generalization across diverse basins.

The neural network augments hydrology; it does not replace it.

The name Bakaano comes from Fante, a language spoken along the southern coast of Ghana. Loosely translated as "by the riverside" or "stream-side", it reflects the lived reality of many vulnerable riverine communities across the Global South—those most exposed to flood risk and often least equipped to adapt.

## Package organization

The codebase is organized into focused subpackages:

- `bakaano.core`: shared infrastructure, utilities, and project helpers
- `bakaano.data`: DEM, soil, vegetation, NDVI, AlphaEarth, and meteorological preprocessing
- `bakaano.hydrology`: VegET runoff generation, routing, PET, routed rainfall utilities, and routed-runoff visualization
- `bakaano.neuralnet`: neural-network training and simulation
- `bakaano.extensions`: optional extensions such as scenarios and flood mapping

Canonical usage is now module-first rather than runner-first:

- use `bakaano.core.project.ProjectContext` for working-directory paths and readiness checks
- use `bakaano.neuralnet.train.train_streamflow_model(...)` for training
- use `bakaano.neuralnet.simulate.simulate_grdc_csv_stations(...)` or `simulate_streamflow(...)` for inference

## Project directory structure

After running Bakaano-Hydro, the working directory follows this structure:

```text
working_dir/
├── alpha_earth/                     # AlphaEarth satellite embeddings (A00-A63)
│   ├── band_A00.tif
│   ├── ...
│   └── band_A63.tif
│
├── catchment/                       # Catchment-level static descriptors
│   └── river_grid.tif
│
├── elevation/                       # DEM and derived topographic layers
│   ├── dem_clipped.tif
│   ├── slope_clipped.tif
│   ├── hyd_glo_dem_30s.tif
│   └── hyd_glo_dem_30s.zip
│
├── ERA5/                            # ERA5-Land meteorological forcing (processed)
│   ├── prep/pr.nc
│   ├── tasmin/tasmin.nc
│   ├── tasmax/tasmax.nc
│   └── tmean/tas.nc
│
├── era5_scratch/                    # Intermediate daily ERA5 GeoTIFFs
│   └── *.tif
│
├── models/                          # Trained Bakaano-Hydro models and scalers
│   ├── bakaano_model.keras
│   ├── alpha_earth_scaler.pkl
│   └── predictor_response_data.pkl
│
├── ndvi/                            # MODIS NDVI products
│   └── daily_ndvi_climatology.pkl
│
├── predicted_streamflow_data/       # Model simulation outputs
│   ├── predicted_streamflow_lat{lat}_lon{lon}.csv
│   └── bakaano_{station_id}.csv
│
├── runoff_output/                   # Distributed runoff and routed flow tensors
│   ├── wacc_sparse_arrays.pkl
│   ├── wacc_output_metadata.pkl
│   ├── rainfall_sparse_arrays.pkl   # Present when routed rainfall is enabled
│   ├── rainfall_output_metadata.pkl # Present when routed rainfall is enabled
│   ├── wacc_resume_state.pkl        # Present only during resumable runs
│   ├── wacc_resume_chunks/          # Present only during resumable runs
│   ├── rainfall_resume_state.pkl    # Present only during resumable runs
│   └── rainfall_resume_chunks/      # Present only during resumable runs
│
├── shapes/                           # Generated vector and river-network layers
│   └── prj_study_area.shp
│
├── scratch/                         # Temporary working files (safe to delete)
│   └── runoff_scratch.tif
│
├── soil/                            # Soil hydraulic properties
│   ├── clipped_WWP_M_sl6_1km_ll.tif
│   ├── clipped_AWCtS_M_sl6_1km_ll.tif
│   └── clipped_AWCh3_M_sl6_1km_ll.tif
│
├── vcf/                             # Vegetation cover fractions
│   ├── mean_tree_cover.tif
│   └── mean_herb_cover.tif
│
├── scenarios/                       # Optional land-cover scenario workspaces
│   └── {scenario_name}/
│       ├── scenario_geometry.geojson
│       ├── scenario_metadata.json
│       ├── vcf/
│       ├── ndvi/
│       ├── runoff_output/
│       └── predicted_streamflow_data/
│
└── flood/                           # Optional flood-mapping outputs
    ├── rating_curves.pkl
    ├── inundation_depth_{period}yr.tif
    └── scratch/
```

Depending on the selected climate source, the meteorological directory is named `ERA5/`, `CHIRPS/`, or `CHELSA/`. CHIRPS processing also uses `chirps_scratch/`; CHELSA stores its downloaded NetCDF files in the same `prep/`, `tasmin/`, `tasmax/`, and `tmean/` subdirectories.

## How to cite

If you use Bakaano-Hydro in academic work, please cite:

- Duku, C.: Bakaano-Hydro (paper version v1.1). A distributed hydrology-guided deep learning model for streamflow prediction, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-1633, 2025.

- Duku, C.: Enhancing flood forecasting reliability in data-scarce regions with a distributed hydrology-guided neural network framework, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-2294, 2025.

For software citation metadata, see CITATION.cff.

## Acknowledgment

Bakaano-Hydro was developed at Wageningen Environmental Research with funding from the Netherlands Ministry of Agriculture, Fisheries, Food Security and Nature (LVVN). This work is part of the Knowledge Base (KB) programme **Climate Resilient Water and Land Use**, within the project **Compound and Cascading Climate Risks and Social Tipping Points**, and builds directly on earlier research conducted under the programme **Data-Driven Discoveries in a Changing Climate**.

## License

Apache License 2.0
