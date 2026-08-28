Simulation
==========

Bakaano has two canonical simulation entry points:

- ``simulate_grdc_csv_stations`` for station sets
- ``simulate_streamflow`` for arbitrary lat/lon points

Simulate station sets
---------------------

.. code-block:: python

   from bakaano.neuralnet.simulate import simulate_grdc_csv_stations

   simulate_grdc_csv_stations(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       model_path="/path/to/working_dir/models/bakaano_model.keras",
       sim_start="1981-01-01",
       sim_end="2020-12-31",
       grdc_netcdf="/path/to/GRDC.nc",
       routing_method="mfd",
   )

Simulate station CSV sets
-------------------------

.. code-block:: python

   simulate_grdc_csv_stations(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       model_path="/path/to/working_dir/models/bakaano_model.keras",
       sim_start="1981-01-01",
       sim_end="2020-12-31",
       grdc_netcdf=None,
       csv_dir="/path/to/observed_csvs",
       lookup_csv="/path/to/station_lookup.csv",
   )

Simulate arbitrary lat/lon points
---------------------------------

.. code-block:: python

   from bakaano.neuralnet.simulate import simulate_streamflow

   simulate_streamflow(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       model_path="/path/to/working_dir/models/bakaano_model.keras",
       sim_start="1981-01-01",
       sim_end="1990-12-31",
       latlist=[13.8, 13.9],
       lonlist=[3.0, 4.0],
       routing_method="mfd",
   )

Outputs
-------

- Station simulations:
  ``{working_dir}/predicted_streamflow_data/bakaano_<station_id>.csv``
- Lat/lon simulations:
  ``{working_dir}/predicted_streamflow_data/predicted_streamflow_lat<lat>_lon<lon>.csv``

Important timing rule
---------------------

Simulation outputs begin after a one-year warmup period. The first 365 days are
used as model context and are not written to the output CSVs.

Common failure modes
--------------------

- Requested simulation period is not covered by routed runoff.
- Lat/lon lists have different lengths.
- Points fall outside the study area.
- A model trained with routed rainfall is run without ``rainfall_sparse_arrays.pkl``.
