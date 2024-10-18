.. _secprecice:

Coupling multiple simulations with preCICE
==========================================

Pyoomph already comes with a adapter for the library `preCICE <https://precice.org/>`__.

This allows to couple pyoomph simulations with other pyoomph simulations or with any other simulation software that is compatible to use preCICE.

To use the preCICE coupling, you require the preCICE library (see `installation instructions <https://precice.org/installation-overview.html>`__) and the `preCICE python bindings <https://precice.org/installation-bindings-python.html>`__.

.. toctree::
   :maxdepth: 5
   :hidden:

   precice/coupled_heat.rst
   precice/coupled_heat_circle.rst   

