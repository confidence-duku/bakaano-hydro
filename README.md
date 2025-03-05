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

- See Bakaano-Hydro Tutorials

```mermaid
classDiagram
    %% Data Processing Modules
    class NDVI {
        +download_ndvi()
        +preprocess_ndvi()
    }
    class Soil {
        +download_soil()
        +preprocess_soil()
    }
    class DEM {
        +download_dem()
        +preprocess_dem()
    }
    class TreeCover {
        +download_tree_cover()
        +preprocess_tree_cover()
    }
    class Meteo {
        +download_meteo()
        +preprocess_meteo()
    }
    class Utils {
        +helper_functions()
    }

    %% NDVI, Soil, DEM, TreeCover, and Meteo use Utils
    NDVI --> Utils
    Soil --> Utils
    DEM --> Utils
    TreeCover --> Utils
    Meteo --> Utils

    %% VegET Model
    class VegET {
        +compute_veget_runoff()
    }
    class PET {
        +compute_pet()
    }
    class Router {
        +route_flow()
    }

    %% Connections for VegET
    VegET --> NDVI
    VegET --> Soil
    VegET --> DEM
    VegET --> TreeCover
    VegET --> Meteo
    VegET --> PET
    VegET --> Router

    %% Router uses DEM outputs
    Router --> DEM

    %% PET uses Meteo outputs
    PET --> Meteo

    %% Main Script (Now with BakaanoHydro)
    class BakaanoHydro {
        +run_simulation()
        +train_model()
        +simulate_streamflow()
    }

    %% Streamflow Modules
    class StreamflowTrainer {
        +train_model()
    }
    class DataPreprocessor {
        +prepare_training_data()
    }
    class StreamflowModel {
        +train_streamflow_model()
    }

    %% Streamflow Trainer relationships
    StreamflowTrainer --> DataPreprocessor
    StreamflowTrainer --> StreamflowModel

    class StreamflowSimulator {
        +simulate_streamflow()
    }
    class PredictDataPreprocessor {
        +prepare_prediction_data()
    }
    class PredictStreamflow {
        +run_prediction()
    }

    %% Streamflow Simulator relationships
    StreamflowSimulator --> PredictDataPreprocessor
    StreamflowSimulator --> PredictStreamflow

    %% Streamflow modules depend on VegET outputs
    StreamflowTrainer --> VegET
    StreamflowSimulator --> VegET
    StreamflowSimulator --> StreamflowTrainer

    %% BakaanoHydro in main.py calls StreamflowTrainer and StreamflowSimulator
    BakaanoHydro --> StreamflowTrainer
    BakaanoHydro --> StreamflowSimulator

    %% File Names (Added as Notes)
    note for NDVI "Defined in ndvi.py"
    note for Soil "Defined in soil.py"
    note for DEM "Defined in dem.py"
    note for TreeCover "Defined in tree_cover.py"
    note for Meteo "Defined in meteo.py"
    note for Utils "Defined in utils.py"
    note for VegET "Defined in veget.py"
    note for PET "Defined in pet.py"
    note for Router "Defined in router.py"
    note for BakaanoHydro "Main script in main.py"
    note for StreamflowTrainer "Part of main.py (Uses DataPreprocessor, StreamflowModel)"
    note for StreamflowSimulator "Part of main.py (Uses PredictDataPreprocessor, PredictStreamflow)"
```

## Support
For assistance, please contact Confidence Duku (confidence.duku@wur.nl)

## Contributing
No contributions are currently accepted.

## Authors and acknowledgment
See CITATION.cff file.

## License
Apache License
