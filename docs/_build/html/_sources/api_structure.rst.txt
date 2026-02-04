API Structure
=============

This page summarizes the high-level structure of the package and where to start.

High-level entry points
-----------------------

- ``bakaano.runner.BakaanoHydro``: main orchestration class for training/evaluation
- ``bakaano.streamflow_trainer.DataPreprocessor``: builds training datasets
- ``bakaano.streamflow_trainer.StreamflowModel``: model definition and training
- ``bakaano.streamflow_simulator.PredictDataPreprocessor``: builds simulation inputs
- ``bakaano.streamflow_simulator.PredictStreamflow``: loads model and runs inference

Data preparation modules
------------------------

- ``bakaano.dem.DEM``: DEM download/clip
- ``bakaano.soil.Soil``: soil properties
- ``bakaano.ndvi.NDVI``: NDVI climatology
- ``bakaano.tree_cover.TreeCover``: vegetation cover fractions
- ``bakaano.alpha_earth.AlphaEarth``: AlphaEarth embeddings
- ``bakaano.meteo.Meteo``: meteorological forcing

Hydrology modules
-----------------

- ``bakaano.veget.VegET``: runoff generation + routing
- ``bakaano.router.RunoffRouter``: flow direction and routing utilities

Plotting helpers
----------------

- ``bakaano.plot_runoff.RoutedRunoff``: routed runoff maps and time series
