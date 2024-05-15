Solids
~~~~~~

Compared to gases and liquids, solids are trivial. At the moment, only pure solids are allowed and we only can set thermal properties. Since solids are not fluids, we cannot use them with the :py:func:`~pyoomph.equations.multi_component.CompositionFlowEquations`, but we still can solve the temperature field. In the future, also elasticity and porosity might be added. Pure solids must inherit from the :py:class:`~pyoomph.materials.generic.PureSolidProperties` class and can be decorated with a ``@MaterialProperties.register()`` to register the solid to the library based on the ``name``, i.e. analogous to pure gases or pure liquids. As properties, only the :py:attr:`~pyoomph.materials.generic.MaterialProperties.molar_mass`, :py:attr:`~pyoomph.materials.generic.MaterialProperties.mass_density`, :py:attr:`~pyoomph.materials.generic.MaterialProperties.thermal_conductivity` and :py:attr:`~pyoomph.materials.generic.MaterialProperties.specific_heat_capacity` can be set.

Mixtures of solids are not implemented yet, so for e.g. an alloy, one has to define it as a pure solid with the properties of the alloy.
