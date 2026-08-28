Quick Start
===========

This page shows one canonical minimal path for new users.

Colab notebooks
---------------

For the fastest start, use one of the maintained notebooks:

- `Open Beginner Colab <https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Beginner%20Colab.ipynb>`_
- `Open Advanced Colab <https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Advanced%20Colab.ipynb>`_
- `Scenario notebook <https://github.com/confidence-duku/bakaano-hydro/blob/main/scenario_quickstart.ipynb>`_
- `Flood-mapping notebook <https://github.com/confidence-duku/bakaano-hydro/blob/main/flood_mapper_quickstart.ipynb>`_

Minimal working example
-----------------------

1. Set up project context

.. code-block:: python

   from bakaano.core.project import ProjectContext

   working_dir = "/path/to/working_dir"
   study_area = "/path/to/basin.shp"

   project = ProjectContext(
       working_dir=working_dir,
       study_area=study_area,
       climate_data_source="ERA5",
   )

2. Run preprocessing and hydrology modules

Run these modules in order:

- ``bakaano.data.dem.DEM``
- ``bakaano.data.tree_cover.TreeCover``
- ``bakaano.data.ndvi.NDVI``
- ``bakaano.data.soil.Soil``
- ``bakaano.data.alpha_earth.AlphaEarth``
- ``bakaano.data.meteo.Meteo``
- ``bakaano.hydrology.veget.VegET``

Use :doc:`inputs_outputs` for the exact files each module reads and writes.

3. Train a model

.. code-block:: python

   from bakaano.neuralnet.train import train_streamflow_model

   project.validate_project(for_task="train")

   model_path = train_streamflow_model(
       working_dir=working_dir,
       study_area=study_area,
       train_start="1981-01-01",
       train_end="2020-12-31",
       grdc_netcdf="/path/to/GRDC.nc",
       batch_size=32,
       num_epochs=300,
       learning_rate=1e-3,
       routing_method="mfd",
       area_normalize=True,
       log_transform=True,
   )

4. Evaluate or simulate

.. code-block:: python

   from bakaano.neuralnet.simulate import (
       evaluate_streamflow_model_interactively,
       simulate_grdc_csv_stations,
       simulate_streamflow,
   )

   evaluate_streamflow_model_interactively(
       working_dir=working_dir,
       study_area=study_area,
       model_path=model_path,
       val_start="2013-01-01",
       val_end="2020-12-31",
       grdc_netcdf="/path/to/GRDC.nc",
   )

   simulate_grdc_csv_stations(
       working_dir=working_dir,
       study_area=study_area,
       model_path=model_path,
       sim_start="1981-01-01",
       sim_end="2020-12-31",
       grdc_netcdf="/path/to/GRDC.nc",
   )

   simulate_streamflow(
       working_dir=working_dir,
       study_area=study_area,
       model_path=model_path,
       sim_start="1981-01-01",
       sim_end="1990-12-31",
       latlist=[13.8, 13.9],
       lonlist=[3.0, 4.0],
   )

Before you run expensive steps
------------------------------

- ``project.validate_project(for_task="train")`` before training
- ``project.validate_project(for_task="evaluate")`` before interactive evaluation
- ``project.validate_project(for_task="simulate")`` before simulation

Important notes
---------------

- Simulation outputs begin after a one-year warmup period.
- If ``rainfall_sparse_arrays.pkl`` exists for the requested dates, Bakaano uses
  routed rainfall automatically as an extra temporal predictor.
- New models save ``area_normalize`` and ``log_transform`` beside the Keras
  checkpoint, and inference loads those settings automatically.

Where to go next
----------------

- :doc:`project_setup`
- :doc:`training`
- :doc:`evaluation`
- :doc:`simulation`
- :doc:`extensions`
- :doc:`migration`
