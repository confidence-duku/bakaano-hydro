Inputs and Outputs
==================

This page summarizes required inputs, expected units/CRS, and outputs by module.

Global assumptions
------------------

- CRS: EPSG:4326 for all rasters and vector inputs unless noted.
- Area units: number of DEM grid cells; the effective cell area depends on the
  DEM resolution.
- Discharge units: m³/s (raw), area-normalized to mm/day for model inputs.

Observed streamflow CSV schema
------------------------------

Lookup CSV (station metadata)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required columns (default names in parentheses):
- station id (``id``)
- latitude (``latitude``)
- longitude (``longitude``)

Notes:
- Coordinates must be in EPSG:4326.
- Station IDs are treated as strings.

Per-station CSV (time series)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required columns (default names in parentheses):
- date (``date``)
- discharge (``discharge``)

Notes:
- Dates must be parseable by pandas (e.g., ``YYYY-MM-DD``).
- Discharge is expected in m³/s.
- One CSV per station; filenames follow ``{id}.csv`` by default.

Predicted streamflow units
--------------------------

By default, model training is performed on ``log1p``-transformed targets
(``log_transform=True``). When ``area_normalize=True``, the target is
area-normalized discharge depth (mm/day) before the optional ``log1p``
transform. During inference, predictions are converted back with ``expm1`` only
when ``log_transform=True`` and then converted to volumetric discharge (m³/s)
by reversing the area normalization. The CSV outputs written to
``{working_dir}/predicted_streamflow_data`` are in m³/s.

New training runs save ``area_normalize`` and ``log_transform`` in a JSON
sidecar next to the Keras model. Evaluation and simulation load those settings
before preparing inference tensors, so users normally do not pass them manually.

When ``loss_function="asym_laplace_nll"``, the model predicts 3 values per
sample (location + asymmetric scales). The runner/simulator uses the first
value (location term) as discharge prediction for plots and CSV outputs.

If ``area_normalize=False`` is used, the model target is raw m³/s before the
optional ``log1p`` transform, and no area-based conversion is applied at
inference time.

Note: Prediction time series start after a one-year warmup period. The first 365
days of the simulation window are used as model context and are not written to
the output CSVs.

Module reference
----------------

DEM (bakaano.data.dem.DEM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- study_area: basin shapefile (EPSG:4326)
- local_data_path (optional): local DEM GeoTIFF (EPSG:4326)

Outputs:
- ``{working_dir}/elevation/dem_clipped.tif``
- ``{working_dir}/elevation/slope_clipped.tif``

Soil (bakaano.data.soil.Soil)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif``
- ``{working_dir}/soil/clipped_WWP_M_sl6_1km_ll.tif``
- ``{working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif``

NDVI (bakaano.data.ndvi.NDVI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/ndvi/daily_ndvi_climatology.pkl``
- Intermediate NDVI GeoTIFFs in ``{working_dir}/ndvi/``

Tree cover (bakaano.data.tree_cover.TreeCover)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/vcf/mean_tree_cover.tif``
- ``{working_dir}/vcf/mean_herb_cover.tif``

AlphaEarth (bakaano.data.alpha_earth.AlphaEarth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/alpha_earth/band_A00.tif`` ... ``band_A63.tif``

Meteo (bakaano.data.meteo.Meteo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- data_source: CHELSA, ERA5, or CHIRPS

Outputs:

- ERA5/CHIRPS consolidated NetCDFs in
  ``{working_dir}/{data_source}/prep/pr.nc``,
  ``tasmax/tasmax.nc``, ``tasmin/tasmin.nc``, and ``tmean/tas.nc``.
- CHELSA source NetCDFs in the corresponding ``prep/``, ``tasmax/``,
  ``tasmin/``, and ``tmean/`` subdirectories.
- For Earth Engine downloads (ERA5/CHIRPS), intermediate GeoTIFFs are stored
  in ``era5_scratch/`` and, for precipitation from CHIRPS,
  ``chirps_scratch/`` before conversion to NetCDF.

VegET + routing (bakaano.hydrology.veget.VegET)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:

- DEM, soil, NDVI, tree cover, meteo
- routing_method: mfd, d8, dinf
- climate_data_source: CHELSA, ERA5, or CHIRPS
- resume / checkpoint_days (optional): resume interrupted routing runs and control checkpoint frequency

Outputs:

- Routed runoff in ``{working_dir}/runoff_output/wacc_sparse_arrays.pkl``
- Resume state during interrupted runs in ``{working_dir}/runoff_output/wacc_resume_state.pkl``
- Temporary checkpoint chunks during interrupted runs in ``{working_dir}/runoff_output/wacc_resume_chunks/*.pkl``
- Completed-run metadata in ``{working_dir}/runoff_output/wacc_output_metadata.pkl``
- River grid in ``{working_dir}/catchment/river_grid.tif`` (if generated)

Notes:

- Resume files are written only when a checkpoint flush occurs. With the default ``checkpoint_days=30``, runs interrupted before day 30 restart from day 1.
- ``resume=False`` deletes any existing VegET resume state before starting.

Optional routed rainfall products (bakaano.hydrology.rainfall_features.RainfallFeatures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outputs:

- Routed rainfall in ``{working_dir}/runoff_output/rainfall_sparse_arrays.pkl``
- Completed-run metadata in ``{working_dir}/runoff_output/rainfall_output_metadata.pkl``
- Resume state during interrupted runs in ``{working_dir}/runoff_output/rainfall_resume_state.pkl``
- Temporary checkpoint chunks during interrupted runs in ``{working_dir}/runoff_output/rainfall_resume_chunks/*.pkl``

Notes:

- This routed-rainfall product is optional. Bakaano training and simulation use VegET routed runoff as the primary predictor.
- If ``rainfall_sparse_arrays.pkl`` is available for the requested date range, training and simulation automatically append routed rainfall as an extra temporal predictor channel. The model input channel count is inferred from the prepared predictors.

Streamflow training (bakaano.neuralnet.train)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- GRDC NetCDF (or CSV lookup + per-station CSVs)
- Routed runoff in {working_dir}/runoff_output
- AlphaEarth bands in {working_dir}/alpha_earth

Outputs:
- Trained model: ``{working_dir}/models/bakaano_model.keras``
- AlphaEarth scaler: ``{working_dir}/models/alpha_earth_scaler.pkl``

Streamflow simulation (bakaano.neuralnet.simulate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- Trained model
- Routed runoff and AlphaEarth bands
- GRDC NetCDF or station CSVs (optional)

Outputs:
- Predicted streamflow CSVs in ``{working_dir}/predicted_streamflow_data``
