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

AlphaEarth scaling or missing scaler
------------------------------------

Symptoms:

- Model fails to load or predictions are unstable.

Fix:

- Ensure ``models/alpha_earth_scaler.pkl`` exists in the training workspace.
- Re-train if the AlphaEarth inputs changed.

Runoff/response scaler missing
------------------------------

Symptoms:

- Predictions look shifted or unstable after scaling changes.
- Inference fails to find scaler files.

Fix:

- Ensure ``models/alpha_earth_scaler.pkl`` exists and matches the model inputs.
- Re-train the model to regenerate scalers if missing.

Model outputs in unexpected units
---------------------------------

Symptoms:

- Predicted streamflow magnitudes are implausible.

Fix:

- Outputs are written in m³/s. Verify that the response scaler is being
  inverse-transformed before reversing area normalization, and that basin area
  and input runoff units are consistent.
