Inputs and Outputs
==================

This page summarizes required inputs, expected units/CRS, and outputs by module.

Global assumptions
------------------

- CRS: EPSG:4326 for all rasters and vector inputs unless noted.
- Area units: number of 1 km x 1 km grid cells (km²).
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

Model training is performed on area-normalized targets (mm/day) and uses a
``log1p`` transform during training. Predictions are inverse-transformed with
``expm1`` and converted back to volumetric discharge (m³/s) by reversing the
area normalization. The CSV outputs written to
``{working_dir}/predicted_streamflow_data`` are in m³/s.

Module reference
----------------

DEM (bakaano.dem.DEM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- study_area: basin shapefile (EPSG:4326)
- local_data_path (optional): local DEM GeoTIFF (EPSG:4326)

Outputs:
- ``{working_dir}/elevation/dem_clipped.tif``
- ``{working_dir}/elevation/slope_clipped.tif``

Soil (bakaano.soil.Soil)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/soil/clipped_AWCh3_M_sl6_1km_ll.tif``
- ``{working_dir}/soil/clipped_WWP_M_sl6_1km_ll.tif``
- ``{working_dir}/soil/clipped_AWCtS_M_sl6_1km_ll.tif``

NDVI (bakaano.ndvi.NDVI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/ndvi/daily_ndvi_climatology.pkl``
- Intermediate NDVI GeoTIFFs in ``{working_dir}/ndvi/``

Tree cover (bakaano.tree_cover.TreeCover)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/vcf/mean_tree_cover.tif``
- ``{working_dir}/vcf/mean_herb_cover.tif``

AlphaEarth (bakaano.alpha_earth.AlphaEarth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- study_area: basin shapefile (EPSG:4326)

Outputs:
- ``{working_dir}/alpha_earth/band_A00.tif`` ... ``band_A63.tif``

Meteo (bakaano.meteo.Meteo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- start_date / end_date: YYYY-MM-DD
- data_source: CHELSA, ERA5, or CHIRPS

Outputs:

- NetCDFs in ``{working_dir}/{data_source}/`` (pr, tasmax, tasmin, tas)
- For Earth Engine downloads (ERA5/CHIRPS), intermediate GeoTIFFs are stored
  in scratch folders before conversion to NetCDF.

VegET + routing (bakaano.veget.VegET)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- DEM, soil, NDVI, tree cover, meteo
- routing_method: mfd, d8, dinf

Outputs:
- Routed runoff in ``{working_dir}/runoff_output/*.pkl``
- River grid in ``{working_dir}/catchment/river_grid.tif`` (if generated)

Streamflow training (bakaano.streamflow_trainer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- GRDC NetCDF (or CSV lookup + per-station CSVs)
- Routed runoff in {working_dir}/runoff_output
- AlphaEarth bands in {working_dir}/alpha_earth

Outputs:
- Trained model: ``{working_dir}/models/bakaano_model.keras``
- AlphaEarth scaler: ``{working_dir}/models/alpha_earth_scaler.pkl``

Streamflow simulation (bakaano.streamflow_simulator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:
- Trained model
- Routed runoff and AlphaEarth bands
- GRDC NetCDF or station CSVs (optional)

Outputs:
- Predicted streamflow CSVs in ``{working_dir}/predicted_streamflow_data``
