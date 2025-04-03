# Bakaano-Hydro

## Description
Bakaano-Hydro is a distributed hydrology-guided neural network model for streamflow prediction. Bakaano-Hydro employs a serial hybridization approach and integrates a gridded process-based rainfall-runoff method that captures spatial heterogeneity and dynamic interactions of meteorological forcings and physiographic attributes generating spatially distributed runoff estimates; a flow routing method propagating runoff through the river network based on topographic constraints to preserve hydrological connectivity.; and a sequential neural network that uses routed flow sequences extracted at hydrological stations to predict streamflow. This approach ensures that primary hydrological responses to climate, soil, topography, and vegetation interactions and changes are captured by process-based components, enhancing interpretability while leveraging deep learning for pattern recognition. 

![image](https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf)

Bakaano-Hydro leverages extensive data inputs—ranging from digital elevation models (DEMs) to meteorological time-series—and processes them through a robust sequence of automated steps. This includes the download, preprocessing, and alignment of source data, as well as regridding inputs to the desired spatial resolution, ensuring consistency and accuracy across all datasets.
It is highly adaptable, providing users with two primary options for data input: they can either let the model automatically download and preprocess all relevant input data or supply their own datasets. If users choose the latter, Bakaano-Hydro accommodates them by accepting data in the widely-used WGS84 geographic coordinate system (EPSG:4326), without the need for time-consuming clipping or regridding. The model seamlessly adjusts input data to match the DEM's spatial resolution, ensuring that all variables are consistently aligned for optimal performance.

## Installation

- Create and activate a conda environment 

```
conda create --name envname python=3.10.4
conda activate envname
```

- Install the Python libraries to that conda environment

```
sudo apt-get update
sudo apt-get install g++
pip install -r requirements.txt
```


## Usage

Bakaano-Hydro requires three primary data or inputs
1. Shapefile of study area or river basin
2. Observed streamflow data in NetCDF format from Global Runoff Data Center (https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Home). Because Bakaano-Hydro aims to use only open-source data, it currently accepts observed streamflow data only from GRDC. 
3. Registration at Google Earth Engine (https://code.earthengine.google.com/register). Bakaano-Hydro retrieves, NDVI, tree cover and meteorological variables from ERA5-land or CHIRPS from Google Earth Engine Data Catalog. This platform requires prior registration for subsequent authentication during execution of the model

Model execution then involves only five steps. See the quick start notebook https://github.com/confidence-duku/bakaano-hydro/blob/main/quick_start.ipynb for guidance.


## Code architecture

```mermaid
classDiagram
    class NDVI {
	    +download_ndvi()
        +interpolate_daily_ndvi()
	    +preprocess_ndvi()
    }
    class Soil {
	    +get_soil_data()
	    +preprocess()
        +plot_soil()
    }
    class DEM {
	    +get_dem_data()
	    +preprocess()
        +plot_dem()
    }
    class TreeCover {
	    +download_tree_cover()
	    +preprocess_tree_cover()
        +plot_tree_cover()
    }
    class Meteo {
	    +get_meteo_data()
    }
    class Utils {
	    +reproject_raster()
        +align_rasters()
        +get_bbox()
        +concat_nc()
        +clip()
    }
    class VegET {
	    +compute_veget_runoff_route_flow()
    }
    class PET {
	    +compute_pet()
    }
    class Router {
	    +compute_flow_dir()
        +compute_weighted_flow_accumulation()
    }
    class BakaanoHydro {
	    +train_streamflow_model()
	    +evaluate_streamflow_model()
        +compute_metrics()
        +simulate_streamflow()
    }
    class DataPreprocessor {
	    +get_data()
        +encode_lat_lon()
        +load_observed_streamflow()
    }
    class StreamflowModel {
        +prepare_data()
        +quantile_transform()
	    +train_model()
    }
    class PredictDataPreprocessor {
	    +get_data()
        +encode_lat_lon()
    }
    class PredictStreamflow {
	    +prepare_data()
        +quantile_transform()
        +load_model()
    }
	note for NDVI "Defined in ndvi.py"
	note for Soil "Defined in soil.py"
	note for DEM "Defined in dem.py"
	note for TreeCover "Defined in tree_cover.py"
	note for Meteo "Defined in meteo.py"
	note for Utils "Defined in utils.py"
	note for VegET "Defined in veget.py"
	note for PET "Defined in pet.py"
	note for Router "Defined in router.py"
	note for BakaanoHydro "Defined in runner.py"
	note for DataPreprocessor "Defined in streamflow_trainer.py"
	note for StreamflowModel "Defined in streamflow_trainer.py"
    note for PredictDataPreprocessor "Defined in streamflow_simulator.py"
	note for PredictStreamflow "Defined in streamflow_simulator.py"
    NDVI --|> Utils
    Soil --|> Utils
    DEM --|> Utils
    TreeCover --|> Utils
    Meteo --|> Utils
    VegET --> NDVI
    VegET --> Soil
    VegET --> DEM
    VegET --> TreeCover
    VegET --> Meteo
    VegET --> PET
    VegET --> Router
    Router --> DEM
    PET --> Meteo
    DataPreprocessor --> VegET
    StreamflowModel --> VegET
    PredictDataPreprocessor --> VegET
    PredictStreamflow --> VegET
    BakaanoHydro --> DataPreprocessor
    BakaanoHydro --> StreamflowModel
    BakaanoHydro --> PredictDataPreprocessor
    BakaanoHydro --> PredictStreamflow
```

## Support
For assistance, please contact Confidence Duku (confidence.duku@wur.nl)

## Contributing
No contributions are currently accepted.

## Authors and acknowledgment
See CITATION.cff file.

## License
Apache License
