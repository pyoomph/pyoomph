Womersley flow
~~~~~~~~~~~~~~

As an example for the action of inertia in this equation, we calculate a Womersley pipe flow, i.e. a flow through a pipe driven by an oscillating pressure:

.. code:: python

   class WomersleyFlowProblem(Problem):
   	def __init__(self):
   		super(WomersleyFlowProblem, self).__init__()
   		self.rho,self.mu=10,1 # density and viscosity
   		self.omega,self.delta_p=10,10 # frequency and pressure amplitude
   		self.L,self.R=1,1 # size of the pipe
   		# Corresponding to a Womersley number of 10
   		self.max_refinement_level=3 # refine due to the velocity profile
   		
   	def define_problem(self):
   		self.set_coordinate_system("axisymmetric") # Pipe: Axisymmetric
   		Nr=4 # number of radial mesh elements
   		self.add_mesh(RectangularQuadMesh(N=[Nr,int(self.L/self.R*Nr)],size=[self.R,self.L]))
   		
   		eqs=NavierStokesEquations(self.rho,self.mu)
   		eqs+=MeshFileOutput()
   		eqs+=DirichletBC(velocity_x=0)@"left" # no r-velocity at the axis
   		eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"right" # no-slip at the wall
   		eqs+=DirichletBC(velocity_x=0)@"top" # no r-velocity at in and outflow
   		eqs+=DirichletBC(velocity_x=0)@"bottom"														
   		# impose oscillating pressure
   		eqs+=NeumannBC(velocity_y=-self.delta_p*cos(self.omega*var("time")))@"bottom"
   		eqs+=SpatialErrorEstimator(velocity=1) # Refine where necessary
   		# eqs+=DirichletBC(velocity_x=0) # We can also deactivate the entire x-velocity in this problem
   		self.add_equations(eqs@"domain") # adding the equation
   		
   if __name__=="__main__":
   	with WomersleyFlowProblem() as problem:
   		problem.run(1,outstep=True,startstep=0.01,spatial_adapt=1)

Due to the inertia, the flow reversal does not happen instantaneously, but shows a Womersley flow profile, see :numref:`figpdewomersley`.

..  figure:: womersley.*
	:name: figpdewomersley
	:align: center
	:alt: Womersley flow in a pipe
	:class: with-shadow
	:width: 100%

	Womersley flow in a pipe


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <navier_stokes.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
