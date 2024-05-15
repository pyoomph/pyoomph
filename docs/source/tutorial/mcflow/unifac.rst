.. _secmcflowunifac:

UNIFAC models
-------------

For the calculation of activity coefficients, group contribution methods provide a promising approach. Of course, these models cannot yield the exact activity coefficients for arbitrary mixtures, but for a widespread range of mixtures they give quite reasonable results.

pyoomph has three different parameter sets of UNIFAC implemented, namely the original UNIFAC (``"UNIFAC"``) :cite:`Fredenslund1975,Fredenslund1977`, Dortmund modified UNIFAC (``"Dortmund"``) :cite:`Lohmann2001` and AIOMFAC (``"AIOMFAC"``) :cite:`Zuend2008,Zuend2011`.

Group contribution methods are based on the assumption that each molecule can be split into functional groups of atoms. Water consists of just a single group, namely a ``"H2O"`` group, whereas most other molecules are separated into several subgroups. Ethanol, for example, is split in the original UNIFAC into three functional groups, namely ``"CH3"``, ``"CH2"`` and ``"OH"``, all appearing once in the ethanol molecule. Groups can also appear multiple times in each molecule, e.g. :math:`1\times`\ ``"CH3"``, :math:`4\times`\ ``"CH2"``, :math:`1\times`\ ``"CH"`` and :math:`2\times`\ ``"OH"`` is the group decomposition of 1,2-hexanediol in the original UNIFAC model. To set these for the ``"Original"`` UNIFAC model, one has to use the :py:meth:`~pyoomph.materials.generic.PureLiquidProperties.set_unifac_groups` method, i.e. add the line

.. container:: center

   ``self.set_unifac_groups({"CH3": 1, "CH2": 4, "CH": 1, "OH": 2``\ ``}``\ ``, only_for="Original")``

to the definition of the pure liquid 1,2-hexanediol. One can add more lines with different ``only_for`` arguments to set the groups for ``"Dortmund"`` UNIFAC and ``"AIOMFAC"``.


Details on UNIFAC-like models are not given here, but can be found e.g. in the references :cite:`Fredenslund1975,Fredenslund1977,Lohmann2001,Zuend2008,Zuend2011`. For just the equations, *Wikipedia* provides a brief overview at https://en.wikipedia.org/wiki/UNIFAC.

The possible groups and their parameters as well as the interaction table of different groups can be found in the python files in :py:mod:`pyoomph.materials.UNIFAC`, e.g. in :py:mod:`pyoomph.materials.UNIFAC.aiomfac`. To use these models in a liquid mixture for the activity coefficients, the groups of all components in the mixture have to be defined by the :py:meth:`~pyoomph.materials.generic.PureLiquidProperties.set_unifac_groups` for the desired model as described above. In the mixture itself, you can assemble the activity coefficients with the :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_activity_coefficients_by_unifac` method of the liquid mixture class. By default, it will also set the vapor pressures of all components according to the non-ideal Raoult's law. If the vapor pressure of a pure component is not set, this component will be considered as nonvolatile also in the mixture.


.. _secboxunifacinfo:

.. important::

   When publishing results based on any of these UNIFAC-like models, please cite the corresponding papers:

   The published parameters of the original and modified (Dortmund) UNIFAC model were taken with kind permission from the *DDBST* website https://www.ddbst.com. Please cite the papers listed at https://www.ddbst.com/published-parameters-unifac.html for original UNIFAC and https://www.ddbst.com/PublishedParametersUNIFACDO.html for modified (Dortmund) UNIFAC.

   Also note that the *UNIFAC Consortium* provides *updated and revised parameters*, which will increase the accuracy of the predicted activity coefficients. Please refer to the `https://unifac.ddbst.com/unifac_.html <https://unifac.ddbst.com/unifac_.html>`_ for more information. These updated parameters are not included in pyoomph.
   
   When using the AIOMFAC model, please cite the papers listed here https://aiomfac.lab.mcgill.ca/citation.html.
