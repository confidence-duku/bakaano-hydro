Migration
=========

This page summarizes the current canonical API and the most important path
changes from earlier Bakaano layouts.

Canonical package names
-----------------------

.. list-table::
   :header-rows: 1

   * - Old
     - New canonical path
   * - ``bakaano.modeling``
     - ``bakaano.neuralnet``
   * - ``bakaano.workflows``
     - ``bakaano.extensions``
   * - legacy runner facade
     - module-first APIs under ``bakaano.core.project`` and ``bakaano.neuralnet``
   * - ``bakaano.hydrology.rainroute``
     - ``bakaano.hydrology.rainfall_features``

Canonical entry points
----------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Canonical path
   * - Project setup and readiness checks
     - ``bakaano.core.project.ProjectContext``
   * - Interactive map inspection
     - ``bakaano.hydrology.plot_runoff.RoutedRunoff.interactive_station_map``
   * - Training
     - ``bakaano.neuralnet.train.train_streamflow_model``
   * - Interactive evaluation
     - ``bakaano.neuralnet.simulate.evaluate_streamflow_model_interactively``
   * - Station-set simulation
     - ``bakaano.neuralnet.simulate.simulate_grdc_csv_stations``
   * - Lat/lon simulation
     - ``bakaano.neuralnet.simulate.simulate_streamflow``
   * - Scenario extension
     - ``bakaano.extensions.scenario.ScenarioManager``
   * - Flood-mapping extension
     - ``bakaano.extensions.flood_mapper.FloodMapper``

Legacy notes
------------

- The legacy runner facade has been removed.
- New code should use the canonical paths listed above.
