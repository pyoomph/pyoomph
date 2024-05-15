Stability analysis involving the shape, i.e. on a moving mesh
-------------------------------------------------------------

Performing linear stability analysis on a moving mesh is a cumbersome task, since the mass and Jacobian matrices of the generalized eigenproblem will contain plenty of terms stemming from the mesh motion, i.e. the feedback of a change in a nodal position on the discretized equations. Pyoomph can handle this automatically by its full symbolical differentiation. This is discussed in detail in our article :cite:`Diddens2024`.

.. toctree::
   :maxdepth: 5
   :hidden:
   
   movmesh/dropdetach.rst
