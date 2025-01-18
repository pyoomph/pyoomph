.. pyoomph documentation master file, created by
   sphinx-quickstart on Tue Apr 16 16:36:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. include:: <isonum.txt>



Welcome to pyoomph's documentation!
===================================

.. only:: html
	
   .. image:: _static/pyoomph_logo_full.png
      :width: 300
      :alt: Pyoomph Logo
      :align: center


pyoomph is a python multi-physics finite element framework based on `oomph-lib <http://www.oomph-lib.org>`__ and `GiNaC <http://www.ginac.de>`__.

pyoomph lets you assemble quite arbitrary multi-physics and multi-domain problems directly in python, including custom equations, constraints and moving meshes.
By generating symbolically derived C code, the performance of pyoomph is on par with hand-coded finite element implementations, but with a considerably lower workload to set up complicated problems.

You can either browse the tutorial online or download a `PDF version <https://pyoomph.readthedocs.io/_/downloads/en/latest/pdf/>`__.

The source code of pyoomph is hosted on `Github <https://github.com/pyoomph/pyoomph>`__ and you can find its homepage at `pyoomph.github.io <https://pyoomph.github.io>`__.


Tutorial
========

.. toctree::
   :maxdepth: 1      

   tutorial.rst


.. |tuto_lorenz| image:: tutorial/temporal/timestepping/plot_lorenz.*
  :height: 12em
  :align: top  
  :alt: Adaptive timestepping for the Lorenz attractor
  
.. |tuto_odebif| image:: tutorial/temporal/stability/bifurcs.*
  :height: 12em
  :align: top  
  :alt: Stability analysis and bifurcation tracking of ODEs
  
.. |tuto_adaptpoisson| image:: tutorial/spatial/poisson/poisson2d_adapt.*
  :height: 12em
  :align: top
  :alt: Spatial adaptivity for a 2d Poisson equation
  
.. |tuto_stokeslaw| image:: tutorial/spatial/stokes/stokes_law.*
  :height: 12em
  :align: top  
  :alt: Flow around a sphere (Stokes law)
  
.. |tuto_stokesdrop| image:: tutorial/spatial/stokes/stokes_noflux.*
  :height: 12em
  :align: top  
  :alt: No normal flow through a curved interface
  
.. |tuto_fishmesh| image:: tutorial/spatial/mesh/fishgmsh_eye.*
  :height: 12em
  :align: top  
  :alt: Generation of custom meshes
  
.. |tuto_waveeq| image:: tutorial/pde/wave/waveeqdoubleslit.*
  :height: 12em
  :align: top  
  :alt: Wave equation
  
.. |tuto_rtinstab| image:: tutorial/pde/navier/rayleigh_taylor.*
  :height: 12em
  :align: top  
  :alt: Rayleigh-Taylor instability
  
.. |tuto_lubric| image:: tutorial/pde/lubric/lubric_coalescence.*
  :height: 12em
  :align: top  
  :alt: Coalescence of droplets via lubrication theory
  
.. |tuto_pattern| image:: tutorial/pde/patterns/kse_ics.*
  :height: 12em
  :align: top  
  :alt: Pattern formation and stability analysis

.. |tuto_remeshing| image:: tutorial/ale/remeshing.*
  :width: 100%
  :align: top  
  :alt: Moving meshes and mesh reconstruction

.. |tuto_nsfreesurf| image:: tutorial/ale/free_surface_ns.*
  :height: 12em
  :align: top  
  :alt: Transient simulation of a free surface
  
.. |tuto_drop3d| image:: tutorial/ale/spread/threedim_spread_single.*
  :height: 12em
  :align: top  
  :alt: Three-dimensional droplet with varying wettability
  
.. |tuto_dropmara| image:: tutorial/ale/spread/droplet_mara_grav.*
  :width: 100%
  :align: top  
  :alt: Droplet with Marangoni flow and gravity
  
.. |tuto_icecyl| image:: tutorial/multidom/icecylinder.*
  :height: 12em
  :align: top  
  :alt: Melting of an ice cylinder
  
.. |tuto_stokessurf| image:: tutorial/multidom/falling_droplet.*
  :height: 12em
  :align: top  
  :alt: Flow around a sphere with insoluble surfactants
  
.. |tuto_evap| image:: tutorial/multidom/dropevap.*
  :width: 100%
  :align: top  
  :alt: Evaporation of a water droplet
  
.. |tuto_heleshaw| image:: tutorial/mcflow/heleshaw_half.*
  :height: 10em
  :align: top  
  :alt: Mixture evaporation from a Hele-Shaw cell
  
.. |tuto_dg| image:: tutorial/dg/dg_fvm_d2.*
  :width: 100%
  :align: top  
  :alt: Discontinuous Galerkin methods
  
.. |tuto_linresp| image:: tutorial/advstab/response/pdrdrum.*
  :width: 100%
  :align: top  
  :alt: Linear response to periodic driving
  
.. |tuto_dropdetach| image:: tutorial/advstab/movmesh/dropstab.*
  :width: 100%
  :align: top  
  :alt: Bifurcation tracking of a detaching droplet
  
.. |tuto_azimuthal| image:: tutorial/advstab/azimuthal/rb_cyl.*
  :width: 100%
  :align: top  
  :alt: Azimuthal stability analysis

.. |tuto_cartnormal| image:: tutorial/advstab/cartesiannormal/rivuletplots.*
  :width: 100%
  :align: top  
  :alt: Cartesian normal mode stability analysis

.. |tuto_precice| image:: tutorial/precice/heat_circle.*
  :height: 12em
  :align: top  
  :alt: Coupling multiple simulations with preCICE


.. list-table:: Some representative tutorial cases
	:widths: 50 50
	:align: center
	
	* - 	|tuto_lorenz|  
		
		:ref:`Adaptive timestepping for the Lorenz attractor<secODEtemporaladapt>`
	  - 	|tuto_odebif|
	  
		:ref:`Stability analysis and bifurcation tracking of ODEs<secODEarclength>`	  
		
	* -	|tuto_adaptpoisson|
	
		:ref:`Spatial adaptivity for a 2d Poisson equation<secspatialadapt>`
		
	  - 	|tuto_stokeslaw|
	  
		:ref:`Flow around a sphere (Stokes law)<secspatialstokes_law>`
		
	* -	|tuto_stokesdrop|
	
		:ref:`No normal flow through a curved interface<secspatialzeroflowenforcing>`
		
	  - 	|tuto_fishmesh|
	  
		:ref:`Generation of custom meshes<secspatialfishmesheye>`
		
	* -	|tuto_waveeq|
	
		:ref:`Wave equation<secpdedoubleslit>`
		
	  - 	|tuto_rtinstab|
	  
		:ref:`Rayleigh-Taylor instability<secpdertinstab>`
		
	* -	|tuto_lubric|
	
		:ref:`Coalescence of droplets via lubrication theory<secpdelubric_coalescence>`
		
	  - 	|tuto_pattern|
	  
		:ref:`Pattern formation and stability analysis<secpdepattformeigs>`

		
	* - 	|tuto_nsfreesurf|
	  
		:ref:`Transient simulation of a free surface<secALEfreesurfNS>`

	  -	|tuto_drop3d|
	
		:ref:`Three-dimensional droplet with varying wettability<threedimdroplet>`

	* -	|tuto_remeshing|
	
		:ref:`Moving meshes and mesh reconstruction<secaleremeshing>`		
		
	  - 	|tuto_dropmara|
	  
		:ref:`Droplet with Marangoni flow and gravity<secALEstatdroplet>`
				
	* -	|tuto_icecyl|
	
		:ref:`Melting of an ice cylinder<secmultidomicecyl>`		
		
	  - 	|tuto_stokessurf|
	  
		:ref:`Flow around a sphere with insoluble surfactants<secmultidomstokessurfact>`						

	* -	|tuto_evap|
	
		:ref:`Evaporation of a water droplet<secmultidomdropevap>`		
		
	  - 	|tuto_heleshaw|
	  
		:ref:`Mixture evaporation from a Hele-Shaw cell<secRicardo>`
		
	* -	|tuto_dg|
	
		:ref:`Discontinuous Galerkin methods<secdgweakdbc>`		
		
	  - 	|tuto_linresp|
	  
		:ref:`Linear response to periodic driving<secadvstabdrumresponse>`
		
	* -	|tuto_dropdetach|
	
		:ref:`Bifurcation tracking of a detaching droplet<secdropletdetach>`		
		
	  - 	|tuto_azimuthal|
	  
		:ref:`Azimuthal stability analysis<secadvstabrbconv>`														
		
	* -	|tuto_cartnormal|
	
		:ref:`Cartesian normal mode stability analysis<secadvstabrivulets>`		
		
	  - 	|tuto_precice|
	  
		:ref:`Coupling multiple simulations with preCICE<secprecicenonmatch>`																
	
	
	

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
