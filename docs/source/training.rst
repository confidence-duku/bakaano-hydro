Training
========

Use ``bakaano.neuralnet.train.train_streamflow_model`` for the canonical
training workflow.

Minimal working example
-----------------------

.. code-block:: python

   from bakaano.neuralnet.train import train_streamflow_model

   model_path = train_streamflow_model(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       train_start="1981-01-01",
       train_end="2020-12-31",
       grdc_netcdf="/path/to/GRDC.nc",
       batch_size=32,
       num_epochs=300,
       learning_rate=1e-3,
       routing_method="mfd",
       area_normalize=True,
       log_transform=True,
       model_overwrite=True,
   )

Train with station CSVs
-----------------------

.. code-block:: python

   model_path = train_streamflow_model(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       train_start="1981-01-01",
       train_end="2020-12-31",
       grdc_netcdf=None,
       csv_dir="/path/to/observed_csvs",
       lookup_csv="/path/to/station_lookup.csv",
       id_col="id",
       lat_col="latitude",
       lon_col="longitude",
       date_col="date",
       discharge_col="discharge",
       file_pattern="{id}.csv",
       batch_size=32,
       num_epochs=300,
       routing_method="mfd",
       area_normalize=True,
   )

Required inputs
---------------

.. list-table::
   :header-rows: 1

   * - Input
     - Requirement
   * - ``working_dir``
     - Must contain aligned preprocessing outputs
   * - ``study_area``
     - Basin shapefile in EPSG:4326
   * - ``train_start`` / ``train_end``
     - Must be fully covered by routed runoff and observations
   * - Observations
     - Provide exactly one of ``grdc_netcdf`` or ``csv_dir`` + ``lookup_csv``

Training outputs
----------------

- ``{working_dir}/models/bakaano_model.keras``
- ``{working_dir}/models/alpha_earth_scaler.pkl``

Parameter guide
---------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Guidance
   * - ``batch_size``
     - Start with 32-64 on GPU, 8-16 on CPU
   * - ``num_epochs``
     - Typical range: 150-300
   * - ``learning_rate``
     - Lower if training is unstable
   * - ``loss_function``
     - ``asym_laplace_nll`` is the default and emits uncertainty parameters;
       standard Keras losses such as ``huber`` train against the point estimate
   * - ``area_normalize``
     - Keep consistent between training and inference
   * - ``log_transform``
     - If ``True`` (default), train on ``log1p`` predictors/targets; keep the
       same value for evaluation and simulation. New model checkpoints save
       this value in a JSON sidecar used automatically by inference.
   * - ``routing_method``
     - Must match the routed runoff used to build predictors
   * - ``model_overwrite``
     - ``False`` continues from an existing saved model if present

Optional routed rainfall
------------------------

If ``runoff_output/rainfall_sparse_arrays.pkl`` exists and fully covers the
requested training period, Bakaano automatically appends routed rainfall as an
extra temporal feature channel. The TCN input channel count is inferred from
the prepared predictors, so runoff-only and runoff-plus-rainfall models are
both supported. A checkpoint trained with rainfall still requires rainfall at
inference time.

Common failure modes
--------------------

- Requested training dates are not covered by routed runoff.
- Lookup CSV or station CSVs are missing required columns.
- Observation files and routed runoff cover different date ranges.
- A model trained with routed rainfall later receives only runoff during inference.
