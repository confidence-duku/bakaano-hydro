API Structure
=============

This page summarizes the high-level structure of the package and where to start.

Canonical package layout
------------------------

Bakaano is organized into focused subpackages:

- ``bakaano.core``: shared infrastructure and utilities
- ``bakaano.data``: input-data acquisition and preprocessing
- ``bakaano.hydrology``: runoff, routing, PET, and routed-runoff visualization
- ``bakaano.neuralnet``: neural-network training and simulation
- ``bakaano.extensions``: optional extensions such as scenarios and flood mapping

High-level entry points
-----------------------

- ``bakaano.core.project.ProjectContext``: shared working-directory paths and readiness checks
- ``bakaano.neuralnet.train.train_streamflow_model``: direct training entry point
- ``bakaano.neuralnet.simulate.evaluate_streamflow_model_interactively``: direct interactive evaluation entry point
- ``bakaano.neuralnet.simulate.simulate_grdc_csv_stations``: direct station-set simulation entry point
- ``bakaano.neuralnet.simulate.simulate_streamflow``: direct arbitrary-point simulation entry point
- ``bakaano.neuralnet.train.DataPreprocessor``: builds training datasets, optionally adding routed rainfall if available
- ``bakaano.neuralnet.train.StreamflowModel``: model definition and training
- ``bakaano.neuralnet.simulate.PredictDataPreprocessor``: builds simulation inputs, optionally adding routed rainfall if available
- ``bakaano.neuralnet.simulate.PredictStreamflow``: loads model and runs inference

Data preparation modules
------------------------

- ``bakaano.data.dem.DEM``: DEM download/clip
- ``bakaano.data.soil.Soil``: soil properties
- ``bakaano.data.ndvi.NDVI``: NDVI climatology
- ``bakaano.data.tree_cover.TreeCover``: vegetation cover fractions
- ``bakaano.data.alpha_earth.AlphaEarth``: AlphaEarth embeddings
- ``bakaano.data.meteo.Meteo``: meteorological forcing

Hydrology modules
-----------------

- ``bakaano.hydrology.veget.VegET``: runoff generation + routing
- ``bakaano.hydrology.rainfall_features.RainfallFeatures``: optional routed rainfall products
- ``bakaano.hydrology.router.RunoffRouter``: flow direction and routing utilities
- ``bakaano.hydrology.pet.PotentialEvapotranspiration``: PET calculations

Plotting helpers
----------------

- ``bakaano.hydrology.plot_runoff.RoutedRunoff``: routed runoff maps and time series

Advanced workflows
------------------

- ``bakaano.extensions.scenario.ScenarioManager``: land-cover-change scenarios
- ``bakaano.extensions.flood_mapper.FloodMapper``: rating curves and inundation mapping
