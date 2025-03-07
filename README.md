# Bakaano-Hydro

## Name
Bakaano-Hydro

## Description
Bakaano-Hydro  is a distributed hydrology-guided neural network modelling framework for simulating streamflows. It integrates the distributed hydrological representations of physical-based models with the capacity of deep learning techniques to learn and generalize across basins.Bakaano-Hydro provides a complete, integrated solution for simulating land surface hydrological processes, river flow routing, and streamflow, from raw data processing to model deployment.

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

- See https://github.com/confidence-duku/bakaano-hydro/blob/main/Bakaano%20Hydro%20Tutorials.ipynb

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
