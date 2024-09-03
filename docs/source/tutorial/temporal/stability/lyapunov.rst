Lyapunov exponents
~~~~~~~~~~~~~~~~~~

After a stationary state becomes unstable due to bifurcations, it can either converge to another stationary solution, diverge or show nontrivial temporal dynamics. In the latter case, the dynamics can be either periodic, quasi-periodic or chaotic. A prominent example of chaotic dynamics is given by the Lorenz system, which was already discussed in :numref:`secODEtemporaladapt`. However, just from the trajectory of the system alone, it is not directly apparent that it is indeed a chaotic system. 

A necessary requirement for deterministic chaos is the sensitivity to small changes of the initial conditions. If a system is chaotic, a small perturbation in the initial condition will grow over time and the long term dynamics of the unperturbed and the perturbed system will be separated by a growing distance in phase space. However, usually, dissipative systems are considered, i.e. systems that relax on a *strange attractor*, which is bounded in phase space. Therefore, eventually both the unperturbed and perturbed trajectory, can converge to the same *strange attractor* and the distance between both trajectories in time will be bounded as well. 

Therefore, instead of measuring the divergence of perturbtions in the initial conditions (which is ultimately bounded) it is beneficial to investiage the growth of the a tiny perturbation along the trajectory. Let :math:`\vec{x}(t)` be a trajectory, then we are interested in how a perturbation :math:`\vec{x}(t)+\delta\vec{x}(t)` will develop. Here, we choose a tiny initial pertrubation vector :math:`\delta\vec{x}(0)` and make sure that it remains tiny during its evolution, so that the linear dynamics around :math:`\vec{x}(t)` alone govern the evolution of :math:`\delta\vec{x}(0)`. This can be achieved by either choosing the magnitude of the initial perturbation :math:`\delta\vec{x}(0)` sufficiently small that it does not grow within the considered simulation time to a magnitude where nonlinear contributions become relevant. Alternatively, one can renormalize :math:`\delta\vec{x}(t)`, e.g. after each :math:`n^\text{th}` time step, to a tiny magnitude, but monitoring its exponential growth.

More precisely, for a system :math:`\partial_t\vec{x}=\vec{F}(\vec{x})`, it is sufficient to calculate the evolution of 

.. math:: :label: eqlyapdynsys

   \begin{aligned}
   \partial_t\delta\vec{x}=\mathbf{J}(\vec{x}(t))\delta\vec{x} 
   \end{aligned}
  
where :math:`\mathbf{J}` is the linearization of :math:`\vec{F}`, i.e. the Jacobian along the trajectory. Due to nonlinearities, the Jacobian is not constant along the trajectory, but the long term dynamics still follow an exponential growth or decay in the long-term limit, i.e. :math:`\delta\vec{x}(t)\sim\exp(\lambda t)`. For an :math:`N`-dimensional system, we generically have :math:`N` different solutions for the exponent :math:`\lambda`, which are called Lyapunov exponents. If at least one :math:`\lambda` is positive, a random perturbation will generically grow over time. If the sum of all Lyapunov exponents is additionally negative, we converge to a strange attractor, i.e. a clear indicator for chaos. Note that one Lyapunov exponent is usually zero, corresponding to the initial perturbation :math:`\delta\vec{x}(0)\propto\partial_t\vec{x}(0)`, i.e. just a shift in the direction of the trajectory :math:`\vec{x}(0)`.

Conventional implementations to numerically calculate Ljapunov exponents rely on the explicit form :math:numref:`eqlyapdynsys`. However, not all systems can be written in this form, e.g. the pendulum constraint in :numref:`secODEpendulum`, which has no time derivative. Since pyoomph allows to formulate equations in the implicit residual formulation :math:`\vec{R}(\partial_t\vec{x},\vec{x})=0`, a general form of :math:numref:`eqlyapdynsys` reads

.. math:: :label: eqlyapimpl

   \begin{aligned}
   \mathbf{M}\partial_t\delta\vec{x}+\mathbf{J}\delta\vec{x}=0 
   \end{aligned}

with the mass matrix :math:`\mathbf{M}` and the Jacobian (without any time-derivatives) :math:`\mathbf{J}`. 

.. warning::

	This form is of course only valid it the maximum time derivative order is one, i.e. for second order time derivatives, again the usual substitution has to be done, cf. :numref:`secODEhigheroderdt`.
	
	
To calculate Lyapunov exponents, pyoomph comes the class :py:class:`~pyoomph.utils.lyapunov.LyapunovExponentCalculator` from the :py:mod:`pyoomph.utils.lyapunov` module. It starts with one or multiple initially random guesses of :math:`\delta\vec{x}(0)` and integrates :math:numref:`eqlyapimpl` along the trajectory. If more than one guess is taken, a Gram-Schmidt orthogonalization is performed after each step. Thereby, components of the faster growing perturbations (i.e. corresponding to higher Lyapunov exponents :math:`\lambda`) are removed for the slower growing or even decaying perturbation components. Automatically, the class :py:class:`~pyoomph.utils.lyapunov.LyapunovExponentCalculator` writes the desired number of Lyapunov exponents :math:`\lambda` to a text file. Due to the random initial guess, usually the largest Lyapunov exponents will be calculated, but these are normally also the most interesting ones. Moreover, the class :py:class:`~pyoomph.utils.lyapunov.LyapunovExponentCalculator` allows to select an average time :math:`T_\text{avg}` by the keyword argument ``average_time``. As mentioned before, the exponential behavior only holds in the long time limit, since the matrices :math:`\mathbf{M}` and :math:`\mathbf{J}` will change along the trajectory :math:`\vec{x}(t)`. The Lyapunov exponents will then be estimated by 

.. math:: :label: eqlyapavg

   \begin{aligned}
   \lambda(t) = \frac{1}{T_\text{avg}}\ln\frac{\|\delta\vec{x}(t)\|}{\|\delta\vec{x}(t-T_\text{avg})\|}
   \end{aligned}

Of course, for :math:`t<T_\text{avg}`, the averaging process will only go from :math:`0` to :math:`t` instead, which is also always the case if ``average_time=None`` is selected, i.e. the averaging goes over the full simulation time, corresponding to the real definition of the Lyapunov exponents.

As an example, we will check the Lorenz system (with the default parameters :math:`\sigma=10`, :math:`\rho=28` and :math:`\beta=8/3`) from :numref:`secODEtemporaladapt` for chaos in the following. When modifying the run code of section :numref:`secODEtemporaladapt` to 
	
.. code:: python

	# Import the LyapunovExponentCalculator from the utils module
	from pyoomph.utils.lyapunov import LyapunovExponentCalculator

	with LorenzProblem() as problem:
		# We want to save memory, since we have a fine temporal discretization. 
		# So we do not write state files for continue simulations
		problem.write_states=False 
		# Add the LyapunovExponentCalculator to the problem
		# Averaging over T_avg=20 and calculating N=3 Lyapunov exponents
		problem+=LyapunovExponentCalculator(average_time=20,N=3)
		# Run it with a rather fine time step and temporal error
		problem.run(endtime=200,outstep=0.0025,startstep=0.01,temporal_error=0.05,maxstep=0.01)        

we get a file called ``lyapunov.txt`` in the output directory. The average time is chosen to :math:`T_\text{avg}=20`, which averages over several typical frequencies of the Lorenz system. The resulting plot is the following, where we also added the long-time limit literature values by dotted lines. The sum of all Lyapunov exponents corresponds to the phase space divergence, i.e. the trace of the Jacobian, which can be obtained analyically by :math:`\sum_{i=1}^3 \lambda_i=-\sigma-1-\beta\approx -13.666`.

..  figure:: lorenzlyapunov.*
	:name: figodelorenzlyapunov
	:align: center
	:alt: Lyapunov spectrum of the Lorenz system
	:class: with-shadow
	:width: 70%
	
	Lyapunov spectrum of the Lorenz system with :math:`\sigma=10`, :math:`\rho=28` and :math:`\beta=8/3`. Dotted lines are the long-time limit literature values.



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lorenz_lyapunov.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		
               
