Troubleshooting
===============

Common issues and fixes.

Google Earth Engine authentication
-----------------------------------

Symptoms:

- Errors when downloading NDVI/Tree cover/AlphaEarth/meteorology
- Authentication prompt keeps appearing

Fix:

- Ensure you have registered for GEE and completed the browser login flow.
- On headless servers, run ``earthengine authenticate`` on a machine with a browser
  and copy ``~/.config/earthengine/credentials`` to the server.

No stations found inside the basin
----------------------------------

Symptoms:

- ``No stations from the lookup table intersect the study area.``

Fix:

- Confirm station coordinates are in EPSG:4326.
- Verify the study area polygon and station coordinates overlap.

Missing columns in CSVs
-----------------------

Symptoms:

- ``Missing columns in station CSV for id=...``

Fix:

- Ensure the per-station CSVs include the required ``date`` and ``discharge`` columns,
  or pass the correct ``date_col`` / ``discharge_col`` arguments.

Predictions are all zeros or flatlines
--------------------------------------

Symptoms:

- Predicted streamflow is near-zero or flat across time.

Fix:

- Check that training targets are correctly scaled and not dominated by zeros.
- Verify loss settings and learning rate are reasonable for the data range.
- Inspect a few stations to confirm the observed discharge has non-zero values.
- Re-train the model if you recently changed preprocessing or scaling options
  such as ``area_normalize`` or ``log_transform``.

Missing or empty raster inputs
------------------------------

Symptoms:

- Runoff computation fails or returns empty arrays.
- Interactive map shows blank layers.

Fix:

- Confirm the DEM, soil, NDVI, tree cover, and meteorology folders exist under
  ``working_dir``.
- Re-run preprocessing if the study area or DEM resolution changed.
- Verify raster CRS is EPSG:4326 and extents overlap the basin.

Observed streamflow dates out of range
--------------------------------------

Symptoms:

- No data returned for training or evaluation periods.
- Empty observed time series after filtering.

Fix:

- Ensure the GRDC NetCDF or CSVs cover the selected date range.
- Check that dates are parsed correctly (``YYYY-MM-DD``) and are daily.

Mismatched routing method
-------------------------

Symptoms:

- Training or simulation runs but outputs look inconsistent with runoff maps.

Fix:

- Use the same ``routing_method`` (e.g., ``mfd``/``d8``/``dinf``) in VegET and
  in training/simulation calls.

Unsupported climate data source
-------------------------------

Symptoms:

- VegET fails immediately with ``Unsupported climate_data_source ...``.

Fix:

- Use one of the supported values exactly as implemented:
  ``CHELSA``, ``ERA5``, or ``CHIRPS``.
- Pass the same source consistently when preprocessing forcing data and when
  launching runoff generation from the high-level runner.

VegET rerun starts from day 1 after interruption
------------------------------------------------

Symptoms:

- An interrupted VegET run is launched again, but the log starts from day 1.
- The output does not show ``Resuming VegET from day ...``.

Fix:

- Keep ``resume=True`` when calling ``compute_veget_runoff_route_flow``.
- Confirm the earlier run had reached a checkpoint flush. With the default
  ``checkpoint_days=30``, interruptions before day 30 cannot resume.
- Check that ``{working_dir}/runoff_output/wacc_resume_state.pkl`` and
  ``{working_dir}/runoff_output/wacc_resume_chunks/`` still exist before rerunning.
- Reuse the same ``start_date``, ``end_date``, ``routing_method``,
  ``climate_data_source``, ``study_area``, NDVI path, and vegetation cover paths;
  changing these invalidates the saved resume signature.
- Do not pass ``resume=False`` unless you intend to discard existing checkpoints.

Model expects routed rainfall but simulation only has runoff
------------------------------------------------------------

Symptoms:

- Simulation fails with a feature-count mismatch.
- The error says the trained model expects more temporal features than the
  prepared inputs.

Fix:

- If the model was trained while ``runoff_output/rainfall_sparse_arrays.pkl``
  was available, provide the same routed-rainfall file for the requested
  simulation period.
- If you want to run without routed rainfall, retrain the model without
  ``rainfall_sparse_arrays.pkl`` present.
- Older one-channel models remain supported; if rainfall is present at
  inference time, Bakaano automatically drops the extra rainfall feature for
  those models.

AlphaEarth scaling or missing scaler
------------------------------------

Symptoms:

- Model fails to load or predictions are unstable.

Fix:

- Ensure ``models/alpha_earth_scaler.pkl`` exists in the training workspace.
- Re-train if the AlphaEarth inputs changed.

Model outputs in unexpected units
---------------------------------

Symptoms:

- Predicted streamflow magnitudes are implausible.

Fix:

- Outputs are written in m³/s.
- If ``area_normalize=True``, verify that training and inference both use the
  same setting so the area-based unit conversion is reversed consistently.
- New models save ``area_normalize`` and ``log_transform`` beside the Keras
  checkpoint. If you are using an older checkpoint without that JSON sidecar,
  pass the same settings used during training.
- If ``area_normalize=False``, outputs stay in raw discharge units throughout
  training and inference.
