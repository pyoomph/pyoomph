.. _secmcflow:

Multi-Component Flow
====================

The last example of the previous section, i.e. the evaporating binary droplet has shown that one has to set a lot of properties, e.g. the mass density and surface tension as function of the composition. The resulting equation system can easily become complicated, e.g. when considering ternary droplets with all components being volatile, etc. pyoomph has a set of predefined equations, that allows to easily add flow of multi-component mixtures, potentially with mass transfer between different phases, surfactants and thermal effects. All these equations are just :py:class:`~pyoomph.generic.codegen.Equations` or :py:class:`~pyoomph.generic.codegen.InterfaceEquations` and all required knowledge to develop these by hand yourself was already introduced throughout this manual. Therefore, we will not go into the details of the implementation of these predefined equations, but feel free to have a look at the code :py:mod:`pyoomph.equations.multi_component`.

.. toctree::
   :maxdepth: 5
   :hidden:

   mcflow/general.rst       
   mcflow/matlib.rst    
   mcflow/rtinstab.rst
   mcflow/freesurf.rst
   mcflow/marainstab.rst
   mcflow/contact.rst
   mcflow/surfact.rst
   mcflow/unifac.rst

