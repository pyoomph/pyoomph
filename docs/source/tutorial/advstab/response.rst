Linear response to periodic driving
-----------------------------------

Periodic driving of a system will eventually lead to a response that attains the same frequency as the driving, since all transients are damped out (provided there is damping in the system). In general, one has to integrate the system over time and extract the response amplitude and the phase shift from the long-term solution. However, for linear equations or for small driving amplitudes, one can directly solve a linear eigenvalue problem to obtain the complex-valued response. pyoomph comes with a utility for this, which can be applied to arbitrary problems.

.. toctree::
   :maxdepth: 5
   :hidden:
   
   response/dampedho.rst
   response/drums.rst
