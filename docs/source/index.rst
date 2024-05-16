.. pyoomph documentation master file, created by
   sphinx-quickstart on Tue Apr 16 16:36:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. include:: <isonum.txt>



Welcome to pyoomph's documentation!
===================================

pyoomph is a python multi-physics finite element framework based on `oomph-lib <http://www.oomph-lib.org>`__ and `GiNaC <http://www.ginac.de>`__.

pyoomph lets you assemble quite arbitrary multi-physics and multi-domain problems directly in python, including custom equations, constraints and moving meshes.
By generating symbolically derived C code, the performance of pyoomph is on par with hand-coded finite element implementations, but with a considerably lower workload to set up complicated problems.

You can either browse the tutorial online or download a `PDF version <https://pyoomph.readthedocs.io/_/downloads/en/latest/pdf/>`__.

The source code of pyoomph is hosted on `Github <https://github.com/cdiddens/pyoomph>`__.


Tutorial
========

.. toctree::
   :maxdepth: 1      

   tutorial.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
