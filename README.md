[![DOI](https://zenodo.org/badge/923830097.svg)](https://doi.org/10.5194/egusphere-2025-1633) [![License](https://img.shields.io/github/license/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/blob/main/LICENSE) [![PyPI version](https://badge.fury.io/py/bakaano-hydro.svg)](https://pypi.org/project/bakaano-hydro/)
 [![GitHub release](https://img.shields.io/github/v/release/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/releases) [![Last Commit](https://img.shields.io/github/last-commit/confidence-duku/bakaano-hydro.svg)](https://github.com/confidence-duku/bakaano-hydro/commits/main) [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) 


# Bakaano-Hydro

## Overview

Bakaano-Hydro is a distributed hydrology-guided neural network model for streamflow prediction. It uniquely integrates physically based hydrological principles with the generalization capacity of machine learning in a spatially explicit and physically meaningful way. This makes it particularly valuable in data-scarce regions, where traditional hydrological models often struggle due to sparse observations and calibration limitations, and where current state-of-the-art data-driven models are constrained by lumped modeling approaches that overlook spatial heterogeneity and the inability to capture hydrological connectivity. 

By learning spatially distributed, physically meaningful runoff and routing dynamics, Bakaano-Hydro is able to generalize across diverse catchments and hydro-climatic regimes. This hybrid design enables the model to simulate streamflow more accurately and reliably—even in ungauged or poorly monitored basins—while retaining interpretability grounded in hydrological processes.

The name Bakaano comes from Fante, a language spoken along the southern coast of Ghana. Loosely translated as "by the river side" or "stream-side", it reflects the  lived reality of many vulnerable riverine communities across the Global South - those most exposed to flood risk and often least equipped to adapt.

![image](https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf)

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
│   ├── A00.tif
│   ├── ...
│   └── A63.tif
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
│   └── alpha_earth_scaler.pkl
│
├── ndvi/                            # MODIS NDVI products
│   └── daily_ndvi_climatology.pkl
│
├── predicted_streamflow_data/       # Model simulation outputs
│   ├── streamflow_lat_lon*.csv
│   └── metadata.json
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

## Quick start: runnable walkthrough (↔ workflow steps)

### Step 1 – Download & preprocess input data

```python
working_dir='/lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger'
study_area='/home/WUR/duku002/Scripts/NBAT/hydro/common_data/niger.shp'
```

**Tree cover**
```python
from bakaano.tree_cover import TreeCover
vf = TreeCover(
    working_dir=working_dir, 
    study_area=study_area, 
    start_date='2001-01-01', 
    end_date='2020-12-31'
)
vf.get_tree_cover_data()
vf.plot_tree_cover(variable='tree_cover') # options for plot are 'tree_cover' and 'herb_cover'
```
    
![png](quick_start_files/quick_start_3_1.png)
    

**NDVI**

```python
from bakaano.ndvi import NDVI
nd = NDVI(
    working_dir=working_dir, 
    study_area=study_area, 
    start_date='2001-01-01', 
    end_date='2010-12-31'
)
nd.get_ndvi_data()
nd.plot_ndvi(interval_num=10)  # because NDVI is in 16-day interval the 'interval_num' represents a 16-day period. 
                               #Hence 0 is the first 16 day period
```
    
![png](quick_start_files/quick_start_4_2.png)
    


**DEM**

```python
from bakaano.dem import DEM
dd = DEM(
    working_dir=working_dir, 
    study_area=study_area, 
    local_data=False, 
    local_data_path=None
)
dd.get_dem_data()
dd.plot_dem()
```

    
![png](quick_start_files/quick_start_5_2.png)
    

**Soil**

```python
from bakaano.soil import Soil
sgd = Soil(
    working_dir=working_dir, 
    study_area=study_area
)
sgd.get_soil_data()
sgd.plot_soil(variable='wilting_point')  #options are 'wilting_point', 'saturation_point' and 'available_water_content'
```
    
![png](quick_start_files/quick_start_6_2.png)
    

**Alpha Earth Embeddings**

```python
from bakaano.alpha_earth import AlphaEarth
dd = AlphaEarth(
    working_dir=working_dir, 
    study_area=study_area,
    start_date='2013-01-01', 
    end_date = '2024-01-01',
)
dd.get_alpha_earth()
dd.plot_alpha_earth('A35') #Band options are A00 to A63
```

    
![png](quick_start_files/quick_start_7_2.png)
    


**Meteorological forcings**

```python
from bakaano.meteo import Meteo
cd = Meteo(
    working_dir=working_dir, 
    study_area=study_area, 
    start_date='2001-01-01', 
    end_date='2010-12-31',
    local_data=False, 
    data_source='ERA5'
)
cd.plot_meteo(variable='tasmin', date='2006-12-01') # variable options are 'tmean', 'precip', 'tasmax', 'tasmin'
```

    
![png](quick_start_files/quick_start_8_2.png)
    


### Step 2 – Compute runoff, route to river network & visualize output

```python
from bakaano.veget import VegET
vg = VegET(
    working_dir=working_dir, 
    study_area=study_area,
    start_date='2001-01-01', 
    end_date='2010-12-31',
    climate_data_source='ERA5',
    routing_method='mfd'
)
vg.compute_veget_runoff_route_flow()
```


```python
from bakaano.plot_runoff import RoutedRunoff
rr = RoutedRunoff(
    working_dir=working_dir, 
    study_area=study_area
)
rr.map_routed_runoff(date='2020-09-03', vmax=6) #output values have been log transformed for better visualization
```

    
![png](quick_start_files/quick_start_11_1.png)
    
This step computes grid-cell runoff and routes it through the river network using a multi-flow direction routing scheme.


### Step 3 – Interactive exploration of GRDC data (optional but recommended)


```python
from bakaano.runner import BakaanoHydro
bk = BakaanoHydro(
    working_dir=working_dir, 
    study_area=study_area,
    climate_data_source='ERA5'
)
bk.explore_data_interactively('1981-01-01', '2016-12-31', '/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-africa-south-america.nc')
```
Use this to inspect station coverage before training.


### Step 4 – Train an instance of Bakaano-Hydro model

(Workflow node 4)


```python
from bakaano.runner import BakaanoHydro
bk = BakaanoHydro(  
    working_dir=working_dir, 
    study_area=study_area,
    climate_data_source='ERA5'
)
```


```python
bk.train_streamflow_model(
    train_start='1991-01-01', 
    train_end='2020-12-31', 
    grdc_netcdf='/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-africa-south-america.nc', 
    batch_size=32, 
    num_epochs=300,
    learning_rate=0.001
)
```
### Step 5 – Evaluate and apply the model

```python
model_path = f'{working_dir}/models/bakaano_model.keras' 

bk.evaluate_streamflow_model_interactively(
    model_path=model_path, 
    val_start='1981-01-01', 
    val_end='1988-12-31', 
    grdc_netcdf='/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-africa-south-america.nc', 
)
```

    
![png](quick_start_files/quick_start_17_5.png)
    



```python
model_path = f'{working_dir}/models/bakaano_model.keras'

bk.simulate_streamflow(
    model_path=model_path, 
    sim_start='1981-01-01', 
    sim_end='1988-12-31', 
    latlist=[13.8, 13.9, 9.15, 8.75, 10.66, 9.32, 7.8, 8.76, 6.17],
    lonlist=[3.0, 4.0, 4.77, 5.91, 4.69, 4.63, 8.91, 10.82, 6.77],
)

```
## How to cite

If you use Bakaano-Hydro in academic work, please cite:

- Duku, C.: Bakaano-Hydro (v1.1). A distributed hydrology-guided deep learning model for streamflow prediction, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-1633, 2025.

- Duku, C.: Enhancing flood forecasting reliability in data-scarce regions with a distributed hydrology-guided neural network framework, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-2294, 2025.

See CITATION.cff.

## Acknowledgment

Bakaano-Hydro was developed at Wageningen Environmental Research with funding from the Netherlands Ministry of Agriculture, Fisheries, Food Security and Nature (LVVN). This work is part of the Knowledge Base (KB) programme **Climate Resilient Water and Land Use**, within the project **Compound and Cascading Climate Risks and Social Tipping Points**, and builds directly on earlier research conducted under the programme **Data-Driven Discoveries in a Changing Climate**.

## License

Apache License

