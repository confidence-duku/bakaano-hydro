Model Configuration
===================

This page summarizes recommended settings and common tradeoffs.

Loss functions
--------------

Point losses:

- ``mae``: stable baseline, good overall fit.
- ``huber``: robust to outliers; often better than MSE in linear space.
- ``mse``: emphasizes large errors; can suppress peaks in linear space.
- ``msle``: emphasizes relative error; often better than MSE for wide ranges.

Distributional loss:

- ``asym_laplace_nll``: predicts (mu, b_plus, b_minus) per time step.
  Useful when you want asymmetric penalties and heteroscedastic uncertainty.

Transforms and scaling
----------------------

- Area normalization is handled automatically in preprocessing.
- You can disable area normalization via ``area_normalize=False`` in the
  training, evaluation, and simulation APIs.
- Area normalization is recommended for large-scale modeling and for
  generalizing to unseen basins.
- Accumulated runoff and discharge are converted to area-invariant depth
  (mm/day) using basin area (km²) so basins of different size are comparable.
- Predictors and responses are log-transformed (``log1p``) after area
  normalization.
- AlphaEarth embeddings are standardized with a ``StandardScaler``.
- The AlphaEarth scaler is saved to ``{working_dir}/models`` and reused at
  inference (``alpha_earth_scaler.pkl``).

Model inputs
------------

- Multi-scale temporal inputs: 45, 90, 180, and 365-day windows.
- Static conditioning inputs: AlphaEarth (64-D) and basin area (scalar).

Spatial resolution
------------------

The analysis resolution follows the DEM grid. By default the model downloads
a 1 km DEM, but you can supply a finer or coarser DEM via ``local_data_path``.

Learning rate schedule
----------------------

Typical starting point:

- base LR: 1e-4
- min LR: 1e-5
- warmup epochs: 5

If training is unstable:

- lower base LR to 5e-5
- reduce batch size

Batch size and generalization
-----------------------------

- Large batches improve throughput but can hurt peaks.
- Recommended ranges: 256–512 for 2 GPUs.
- If you need stronger peak learning, reduce batch size or use peak-weighted
  losses.

Bootstrapping
-------------

Streaming bootstrap sampling improves generalization but adds loss noise.
Recommended ``epoch_data_fraction``: 0.5–0.8.

Asymmetric Laplace outputs
--------------------------

If using ``asym_laplace_nll``:

- Output layer is 3 units: ``mu``, ``log_b_plus``, ``log_b_minus``.
- Prediction uses ``mu`` as the point estimate.
