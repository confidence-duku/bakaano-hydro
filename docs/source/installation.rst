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

Preview docs locally
--------------------

To build and live-preview the documentation while editing:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-autobuild docs/source docs/_build/html

Then open the URL shown in the terminal, usually:

.. code-block:: text

   http://127.0.0.1:8000

For a one-off static build instead of live preview:

.. code-block:: bash

   python -m sphinx -b html docs/source docs/_build/html

Data requirements
-----------------

- Study-area shapefile (river basin boundary)
- Observed streamflow (GRDC NetCDF or station CSVs with a lookup table)
- Google Earth Engine access (for NDVI, tree cover, meteorology)
