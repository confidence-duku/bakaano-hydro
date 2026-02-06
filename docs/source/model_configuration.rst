Model Configuration
===================

This page summarizes recommended settings and common tradeoffs.

User-defined inputs
-------------------

This section defines the user-facing inputs used across training, evaluation,
and simulation, with guidance on when to adjust them.

Data scope and timing
^^^^^^^^^^^^^^^^^^^^^
- ``train_start`` / ``train_end``: Training window (YYYY-MM-DD). Ensure routed runoff
  and observations fully cover this period. If you see "date not found" errors,
  your routed runoff window is shorter than the requested training range.
- ``val_start`` / ``val_end``: Evaluation window (YYYY-MM-DD). Use a period outside
  the training range for a realistic check. Keep it long enough to capture
  wet/dry variability (multiple seasons if possible).
- ``sim_start`` / ``sim_end``: Simulation window (YYYY-MM-DD). Must be covered by
  the routed runoff inputs. Simulation outputs begin after a one-year warmup
  (the first 365 days are used as model context).

Paths and sources
^^^^^^^^^^^^^^^^^
- ``working_dir``: Project folder where inputs/outputs are stored. Keep it on
  fast local storage; large basins can generate many GB of routed runoff.
- ``study_area``: Basin boundary shapefile in EPSG:4326.
- ``grdc_netcdf``: GRDC NetCDF path for observed streamflow. Required unless you
  provide CSV observations. The file must include station name, coordinates,
  and discharge time series.
- ``model_path``: Path to a trained model (e.g., ``{working_dir}/models/bakaano_model.keras``).

Training controls
^^^^^^^^^^^^^^^^^
- ``batch_size``: Number of samples per step. Larger is faster but can smooth peaks.
  Start with 32–64 (GPU) or 8–16 (CPU). If peaks are muted, reduce it.
- ``num_epochs``: Number of training epochs. 150–300 is a common range.
- ``learning_rate``: Base optimizer step size. Lower it (e.g., 5e-5) if training
  is unstable or the loss diverges early.
- ``lr_schedule``: ``cosine`` or ``exp_decay`` for decayed learning rates. Set to
  ``None`` for a fixed rate. ``cosine`` is a robust default.
- ``warmup_epochs``: Number of warmup epochs before scheduling kicks in.
  Use 3 to stabilize early training; increase if the loss spikes in the
  first few epochs.
- ``min_learning_rate``: Floor learning rate when using a schedule (e.g., 1e-5).
- ``loss_function``: Training objective. ``huber`` is a stable default; ``mse`` is
  smoother but can suppress peaks; ``msle`` emphasizes relative errors.
- ``seed``: Random seed for reproducibility. Fix this to compare experiments.

When to choose ``asym_laplace_nll``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``loss_function="asym_laplace_nll"`` when you care about cross-basin
generalization or want the model to express asymmetric uncertainty. It can
improve transfer to ungauged basins by letting the model adapt its error
tolerance across regimes, and by treating under- vs over-prediction differently.

Hydrology and scaling
^^^^^^^^^^^^^^^^^^^^^
- ``routing_method``: Routing method used to generate runoff (``mfd``, ``d8``,
  or ``dinf``). Must match the method used in VegET routing; mismatches will
  cause inconsistent predictors.
- ``catchment_size_threshold``: Minimum catchment size filter for stations.
  Increase it to exclude very small basins or to reduce noise from tiny catchments.
- ``area_normalize``: If ``True`` (recommended), the model trains on area-normalized
  depth (mm/day) and converts outputs back to m³/s. Set to ``False`` if you need
  raw discharge units during training, evaluation, and simulation. Make sure you
  use the same value for training and inference.

CSV observation options (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``csv_dir``: Folder containing one CSV per station. Useful when you have custom
  observations not in GRDC.
- ``lookup_csv``: Station metadata CSV with id/coordinates. Must align with the
  station IDs used in the per-station CSV filenames.
- ``id_col``, ``lat_col``, ``lon_col``: Column names in the lookup CSV.
- ``date_col``, ``discharge_col``: Column names in station CSVs. Dates must be
  parseable (e.g., ``YYYY-MM-DD``), and discharge must be in m³/s.
- ``file_pattern``: Filename pattern for station CSVs (default ``{id}.csv``).
  For example, use ``station_{id}.csv`` if your files are prefixed.

Simulation coordinates (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``latlist`` / ``lonlist``: Lists of coordinates for point simulations. Points
  should fall inside the study area and on or near the river network for
  meaningful results.

Recommended settings
--------------------

These defaults are a good starting point for most basins. Adjust only if you
see instability or poor peak performance.

General defaults
^^^^^^^^^^^^^^^^
- ``learning_rate``: 1e-4
- ``min_learning_rate``: 1e-5
- ``warmup_epochs``: 3
- ``lr_schedule``: ``cosine``
- ``loss_function``: ``huber``
- ``batch_size``: 32–64 (GPU), 8–16 (CPU)
- ``num_epochs``: 150–300
- ``routing_method``: match the method used in runoff routing (``mfd`` recommended)
- ``catchment_size_threshold``: 1–10 (raise if you want to exclude tiny basins)
- ``area_normalize``: ``True`` (recommended for multi-basin/generalization)
