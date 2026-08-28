Evaluation
==========

Use ``bakaano.neuralnet.simulate.evaluate_streamflow_model_interactively`` to
compare a trained model against one station at a time.

Minimal working example
-----------------------

.. code-block:: python

   from bakaano.neuralnet.simulate import evaluate_streamflow_model_interactively

   evaluate_streamflow_model_interactively(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       model_path="/path/to/working_dir/models/bakaano_model.keras",
       val_start="2013-01-01",
       val_end="2020-12-31",
       grdc_netcdf="/path/to/GRDC.nc",
       routing_method="mfd",
   )

Evaluate with station CSVs
--------------------------

.. code-block:: python

   evaluate_streamflow_model_interactively(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       model_path="/path/to/working_dir/models/bakaano_model.keras",
       val_start="2013-01-01",
       val_end="2020-12-31",
       grdc_netcdf=None,
       csv_dir="/path/to/observed_csvs",
       lookup_csv="/path/to/station_lookup.csv",
   )

How it works
------------

- Loads one trained model.
- Builds predictors for the requested evaluation period.
- Prompts you to select a station.
- Plots observed versus predicted streamflow.

Common failure modes
--------------------

- The model file does not exist.
- The requested validation period is not covered by routed runoff.
- CSV observations are incomplete or use different station IDs than the lookup table.
- The trained model does not have a sidecar config file and fallback scaling
  options were not set to match training.
