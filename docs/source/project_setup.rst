Project Setup
=============

Use ``ProjectContext`` to establish a shared project configuration and check
whether the required artifacts exist before expensive tasks.

Minimal setup
-------------

.. code-block:: python

   from bakaano.core.project import ProjectContext

   project = ProjectContext(
       working_dir="/path/to/working_dir",
       study_area="/path/to/basin.shp",
       climate_data_source="ERA5",
   )

Project helper methods
----------------------

.. list-table::
   :header-rows: 1

   * - Method
     - Use
   * - ``project.project_paths()``
     - Return canonical paths under ``working_dir``
   * - ``project.project_status()``
     - Inspect which key files already exist
   * - ``project.validate_project(for_task=...)``
     - Fail early when required artifacts are missing
   * - ``project.workflow_overview()``
     - Show the recommended module-level flow

Preflight checks
----------------

.. code-block:: python

   project.project_paths()
   project.project_status()
   project.validate_project(for_task="train")

Available tasks for validation:

- ``preprocess``
- ``train``
- ``evaluate``
- ``simulate``
- ``flood``
- ``scenario``

Common failure modes
--------------------

- Study area is missing or not readable as vector data.
- DEM/reference rasters have not been created yet.
- VegET runoff output does not exist for training or simulation.
- Model file is missing for evaluation or simulation.
