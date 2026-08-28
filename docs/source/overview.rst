Overview
========

Bakaano-Hydro combines three layers:

1. Data preparation
   Static and dynamic raster inputs are downloaded, clipped, and aligned to a
   common DEM grid.
2. Hydrology
   VegET generates runoff and routes it through the river network.
3. Neural network
   The TCN-based model learns streamflow dynamics from routed hydrologic
   predictors and static catchment descriptors.

Pipeline diagram
----------------

.. code-block:: text

   Study area + working_dir
            |
            v
   +-----------------------+
   |  bakaano.data.*       |
   |  DEM / Soil / NDVI    |
   |  Tree cover / Meteo   |
   |  AlphaEarth           |
   +-----------------------+
            |
            v
   +-----------------------+
   |  bakaano.hydrology    |
   |  VegET runoff/routing |
   |  Rainfall features    |
   +-----------------------+
            |
            v
   +-----------------------+
   |  bakaano.neuralnet    |
   |  train / evaluate /   |
   |  simulate             |
   +-----------------------+
            |
            v
   +-----------------------+
   |  Outputs              |
   |  model + CSVs         |
   +-----------------------+

Package diagram
---------------

.. code-block:: text

   bakaano/
     core/         shared project helpers and utilities
     data/         preprocessing and downloads
     hydrology/    runoff, routing, rainfall products, plotting
     neuralnet/    training, evaluation, simulation
     extensions/   scenario and flood-mapping add-ons

Artifact flow
-------------

.. code-block:: text

   elevation/dem_clipped.tif
   soil/*.tif
   vcf/*.tif
   ndvi/daily_ndvi_climatology.pkl
   ERA5|CHIRPS|CHELSA/{prep,tasmax,tasmin,tmean}/*.nc
   alpha_earth/band_A*.tif
        |
        v
   runoff_output/wacc_sparse_arrays.pkl
   runoff_output/rainfall_sparse_arrays.pkl   [optional]
        |
        v
   models/bakaano_model.keras
   models/alpha_earth_scaler.pkl
        |
        v
   predicted_streamflow_data/*.csv

Which function should I use?
----------------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Canonical entry point
   * - Inspect project layout and readiness
     - ``bakaano.core.project.ProjectContext``
   * - Train with GRDC or CSV observations
     - ``bakaano.neuralnet.train.train_streamflow_model``
   * - Validate a trained model interactively
     - ``bakaano.neuralnet.simulate.evaluate_streamflow_model_interactively``
   * - Simulate official station sets
     - ``bakaano.neuralnet.simulate.simulate_grdc_csv_stations``
   * - Simulate arbitrary lat/lon points
     - ``bakaano.neuralnet.simulate.simulate_streamflow``
   * - Land-cover-change scenarios
     - ``bakaano.extensions.scenario.ScenarioManager``
   * - Flood mapping
     - ``bakaano.extensions.flood_mapper.FloodMapper``
