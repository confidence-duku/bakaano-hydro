Installation
============

Requirements
------------

- Python 3.10+
- Conda (recommended)

Create a new environment
------------------------

.. code-block:: bash

   conda create --name bakaano_env python=3.10
   conda activate bakaano_env

Install (GPU)
-------------

.. code-block:: bash

   pip install bakaano-hydro[gpu]

This installs TensorFlow with compatible CUDA and cuDNN runtime libraries.

Install (CPU)
-------------

.. code-block:: bash

   pip install bakaano-hydro

CPU training is supported but can be slow for large basins.

Data requirements
-----------------

- Study-area shapefile (river basin boundary)
- Observed streamflow (GRDC NetCDF or station CSVs with a lookup table)
- Google Earth Engine access (for NDVI, tree cover, meteorology)
