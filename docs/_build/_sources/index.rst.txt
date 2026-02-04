Bakaano-Hydro Documentation
===========================

Introduction
------------

Bakaano-Hydro is a distributed hydrology-guided neural network model for streamflow prediction.
It uniquely integrates physically based hydrological principles with the generalization capacity
of machine learning in a spatially explicit and physically meaningful way. This makes it
particularly valuable in data-scarce regions, where traditional hydrological models often
struggle due to sparse observations and calibration limitations, and where current
state-of-the-art data-driven models are constrained by lumped modeling approaches that overlook
spatial heterogeneity and the inability to capture hydrological connectivity.

By learning spatially distributed, physically meaningful runoff and routing dynamics,
Bakaano-Hydro is able to generalize across diverse catchments and hydro-climatic regimes. This
hybrid design enables the model to simulate streamflow more accurately and reliably—even in
ungauged or poorly monitored basins—while retaining interpretability grounded in hydrological
processes.

The name Bakaano comes from Fante, a language spoken along the southern coast of Ghana. Loosely
translated as "by the river side" or "stream-side", it reflects the lived reality of many
vulnerable riverine communities across the Global South - those most exposed to flood risk and
often least equipped to adapt.

Bakaano-Hydro consists of three tightly coupled components:

1. Distributed runoff generation
   Vegetation, soil, and meteorological drivers are used to compute grid-cell runoff using a
   VegET-based approach.
2. Physically informed routing
   Runoff is routed through the river network using flow-direction-based routing (e.g. MFD/WFA),
   preserving spatial connectivity.
3. Neural network
   A Temporal Convolutional Network (TCN), conditioned on static catchment descriptors, learns
   hydrological dynamics from physically routed runoff, enabling robust generalization across
   diverse basins.

The neural network augments hydrology, it does not replace it.

.. figure:: https://github.com/user-attachments/assets/8cc1a447-c625-4278-924c-1697e6d10fbf
   :alt: Bakaano-Hydro conceptual model diagram
   :align: center
   :width: 85%

   Conceptual overview of the Bakaano-Hydro pipeline.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   inputs_outputs
   model_configuration
   troubleshooting

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
