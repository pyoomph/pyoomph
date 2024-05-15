.. _eqpdelubric_relax:

Relaxation of a perturbation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first problem class, let us calculate the relaxation of a thin film with a modulation:

.. code:: python

   		
   class LubricationProblem(Problem):	
   	def define_problem(self):
   		self.add_mesh(LineMesh(N=100)) # simple line mesh		
   		eqs=LubricationEquations() # equations
   		eqs+=TextFileOutput() # output	
   		eqs+=InitialCondition(h=0.05*(1+0.25*cos(2*pi*var("coordinate_x"))))  # small height with a modulation
   		self.add_equations(eqs@"domain") # adding the equation

   		
   if __name__=="__main__":
   	with LubricationProblem() as problem:
   		problem.run(50,outstep=True,startstep=0.25)

The result is depicted in :numref:`figpdelubrication`.

..  figure:: lubrication.*
	:name: figpdelubrication
	:align: center
	:alt: Relaxation of a perturbed surface $h$ in the lubrication limit
	:class: with-shadow
	:width: 70%

	Relaxation of a perturbed surface :math:`h` in the lubrication limit.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <lubrication.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
