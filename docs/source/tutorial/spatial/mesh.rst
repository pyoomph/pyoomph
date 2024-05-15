.. _secspatialmeshgen:

Creating custom meshes
----------------------

So far, we only used predefined mesh classes, namely the :py:class:`~pyoomph.meshes.simplemeshes.LineMesh` and :py:class:`~pyoomph.meshes.simplemeshes.RectangularQuadMesh`. In general, the problem to solve has a more intricate geometry, and the mesh has to be created by the user. This section will show how to create a custom mesh, either by specifying each element by hand or by using pyoomph's interface to the mesh generator Gmsh (`gmsh.info <https://gmsh.info/>`_).


.. toctree::
   :maxdepth: 5
   :hidden:

   mesh/byhand.rst
   mesh/curvedline.rst
   mesh/unitsandmacro.rst
   mesh/gmsh.rst
   mesh/resolution.rst

