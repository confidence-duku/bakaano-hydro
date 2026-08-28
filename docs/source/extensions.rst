Extensions
==========

Bakaano keeps scenario analysis and flood mapping outside the core train/sim
path. They are optional extensions built on the main preprocessing, hydrology,
and neural-network modules.

Scenario extension
------------------

Canonical entry point:

- ``bakaano.extensions.scenario.ScenarioManager``

Use it for:

- vegetation-change experiments
- scenario-specific runoff recomputation
- scenario-specific streamflow simulations

Flood-mapping extension
-----------------------

Canonical entry point:

- ``bakaano.extensions.flood_mapper.FloodMapper``

Use it for:

- rating-curve generation
- outlet hydrograph-driven inundation mapping
- screening-level flood-depth products

When to use extensions
----------------------

Use the extensions only after the core project is already working:

1. preprocessing finished
2. VegET runoff/routing finished
3. model training or simulation path verified

Notebook entry points
---------------------

- ``scenario_quickstart.ipynb``
- ``flood_mapper_quickstart.ipynb``
