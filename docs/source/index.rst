Bakaano-Hydro Documentation
===========================

Bakaano-Hydro is a distributed hydrology-guided neural network model for
streamflow prediction. The documentation below is organized by user task first,
then by module reference.

.. figure:: https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf
   :alt: Bakaano-Hydro conceptual model diagram
   :align: center
   :width: 85%

   Conceptual overview of the Bakaano-Hydro pipeline.

Start Here
----------

Common tasks
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - I want to...
     - Go here
   * - Set up a project and check required files
     - :doc:`project_setup`
   * - Run one end-to-end minimal example
     - :doc:`quickstart`
   * - Train a model
     - :doc:`training`
   * - Validate a trained model
     - :doc:`evaluation`
   * - Simulate stations or lat/lon points
     - :doc:`simulation`
   * - Understand package layout and data flow
     - :doc:`overview`
   * - Use scenario or flood extensions
     - :doc:`extensions`
   * - Migrate from older package paths
     - :doc:`migration`

Start with Colab
~~~~~~~~~~~~~~~~

Recommended entry point for new users:

- `Open Beginner Colab <https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Beginner%20Colab.ipynb>`_
- `Open Advanced Colab <https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Advanced%20Colab.ipynb>`_
- `Notebook sources on GitHub <https://github.com/confidence-duku/bakaano-hydro/blob/main/>`_

Advanced workflow notebooks:

- `Scenario workflow notebook <https://github.com/confidence-duku/bakaano-hydro/blob/main/scenario_quickstart.ipynb>`_
- `Flood-mapping workflow notebook <https://github.com/confidence-duku/bakaano-hydro/blob/main/flood_mapper_quickstart.ipynb>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Open In Colab
   :target: https://colab.research.google.com/github/confidence-duku/bakaano-hydro/blob/main/Bakaano-Hydro%20Beginner%20Colab.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   overview
   project_setup
   quickstart
   training
   evaluation
   simulation
   extensions
   inputs_outputs
   model_configuration
   troubleshooting
   migration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_structure
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
