Convection-diffusion equation
-----------------------------

A very important equation for transport phenomena is the convection diffusion equation

.. math:: \partial_t c + \nabla\cdot\left(c\vec{u}\right)=\nabla\cdot\left(D\nabla c\right) \,.

Here :math:`c` can be understood as e.g. some concentration field or a temperature field, :math:`\vec{u}` is the advecting velocity field and :math:`D` is a diffusion coefficient. After multiplication with a test function :math:`\phi`, we can have two different weak formulations, depending on whether the advection term is included in the partial integration or not, namely:

.. math:: :label: eqpdeconvdiffuweakA

   \left(\partial_t c,\phi\right) + \left(\nabla\cdot\left(c\vec{u}\right),\phi\right)+\left(D\nabla c,\nabla \phi\right) +\left\langle -D\nabla c\cdot \vec{n},\phi\right\rangle=0

and

.. math:: :label: eqpdeconvdiffuweakB

   \left(\partial_t c,\phi\right) - \left(c\vec{u},\nabla\phi\right)+\left(D\nabla c,\nabla \phi\right) +\left\langle \left(\vec{u}c-D\nabla c\right)\cdot \vec{n},\phi\right\rangle=0\,.

Note that, if the velocity :math:`\vec{u}` is divergence free (incompressible), the second term in :math:numref:`eqpdeconvdiffuweakA` reads :math:`(\vec{u}\cdot\nabla c,\phi)`. The first observation between both variants is that the Neumann term :math:`\langle \cdot,\cdot \rangle` is different. In :math:numref:`eqpdeconvdiffuweakA`, we impose pure diffusive fluxes as Neumann conditions, whereas in :math:numref:`eqpdeconvdiffuweakB` total fluxes, i.e. the sum of advection and diffusion, are imposed by Neumann conditions.

The second observation is more severe: The advection terms are not symmetric with respect to the spatial derivative order. While the time derivative has zeroth order spatial derivatives on both :math:`c` and :math:`\phi` and the diffusion term has both first order spatial derivatives, i.e. :math:`\nabla c` and :math:`\nabla \phi`, the advection term is mixed. Either the field :math:`c` or the test function :math:`\phi` is derived. This asymmetry leads to considerable complications if the equation is advection-dominated.

.. toctree::
   :maxdepth: 5
   :hidden:

   convdiffu/naive.rst
   convdiffu/supg.rst   

