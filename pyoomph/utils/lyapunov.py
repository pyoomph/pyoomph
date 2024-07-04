from ..generic.problem import GenericProblemHooks
import numpy
from scipy.sparse import csr_matrix
from ..expressions import ExpressionNumOrNone
from ..typings import NPFloatArray,List,Optional
from collections import deque

class LyapunovExponentCalculator(GenericProblemHooks):
    """
    A class for calculating Lyapunov exponents. Add it to the problem by ``problem+=LyapunovExponentCalculator(...)`` and it will do the rest for you.
    However, note that we only may have first order time derivatives in the equations. Second order time derivatives must be rewritten as first order time derivatives before.
    Also, the time derivatives in the system must use the fully implicit "BDF2" time scheme, which is the default (unless set otherwise stated by either using ``scheme="..."`` in :py:func:`~pyoomph.expressions.generic.partial_t` or by altering :py:attr:`~pyoomph.generic.problem.Problem.default_timestepping_scheme` of the :py:class:`~pyoomph.generic.problem.Problem`).

    Args:
        average_time: The time interval over which to average the Lyapunov exponents. If None, we average over the entire time
        N: The number of Lyapunov exponents to calculate. N>=2 will invoke Gram-Schmidt on the perturbation vectors. Defaults to 1.
        filename: The name of the output file. Defaults to "lyapunov.txt".
        relative_to_output: Whether to save the output file relative to the problem's output directory. Defaults to True.
        min_perturbation_norm: The minimum norm of the perturbation. Defaults to 1e-20.
        max_perturbation_norm: The maximum norm of the perturbation. Defaults to 1e10.
        start_perturbation_norm: The initial norm of the perturbation. Defaults to 1e-4.
    """    
    def __init__(self,average_time:ExpressionNumOrNone,N:int=1,filename="lyapunov.txt",relative_to_output:bool=True,min_perturbation_norm:float=1e-20,max_perturbation_norm:float=1e10,start_perturbation_norm=1e-4):
        super().__init__()
        self.filename=filename
        self.relative_to_output=relative_to_output
        self.min_perturbation_norm=min_perturbation_norm
        self.max_perturbation_norm=max_perturbation_norm
        self.start_perturbation_norm=start_perturbation_norm    
        
        self.perturbation:List[NPFloatArray]=[] # Storing the last perturbation
        self.old_perturbation:Optional[List[NPFloatArray]]=None # Storing the perturbation one step before
        self.outputfile=None # Output file
        self.average_time=average_time
        self.ringbuffer=deque()
        self.N=N
        if self.N<=0:
            raise ValueError("N must be a positive integer")
        
    def renormalize(self,i:int):
        if self.old_perturbation is None:
            self.old_perturbation=[None]*self.N
        if self.old_perturbation[i] is not None:
            # Scale the old perturbation. Note: We divide by the norm of self.perturbation to keep the ratio between both
            self.old_perturbation[i]=self.old_perturbation[i]/numpy.linalg.norm(self.perturbation[i])*self.start_perturbation_norm
        # And renormalize the current perturbation to start_perturbation_norm
        self.perturbation[i]=self.perturbation[i]/numpy.linalg.norm(self.perturbation[i])*self.start_perturbation_norm
    
    
    def actions_after_newton_solve(self):
        problem=self.get_problem()
        if len(self.perturbation)!=self.N:
            self.perturbation=[[] for i in range(self.N)]
        if len(self.perturbation[0])!=problem.ndof():
            for i in range(self.N):
                self.perturbation[i]=(numpy.random.rand(problem.ndof())*2-1)
                if self.old_perturbation is None:
                    self.old_perturbation=[None]*self.N
                self.old_perturbation[i]=None
                self.renormalize(i) # and scale it to the length
            
        # Open the file if necessary
        if self.outputfile is None:
            if self.relative_to_output:                
                self.outputfile=open(problem.get_output_directory(self.filename),"w")
            else:
                self.outputfile=open(self.filename,"w")
        
        t=problem.get_current_time(as_float=True)
        # History time stepping weights
        if problem.timestepper.get_num_unsteady_steps_done()==0: # The first step is degraded to BDF1 by default
            w1=problem.timestepper.weightBDF1(1,1)
            w2=0            
        else:
            w1=problem.timestepper.weightBDF2(1,1)
            w2=problem.timestepper.weightBDF2(1,2)            
        # Second history perturbation
        
        # Get the mass matrix and the Jacobian
        n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0.0) #type:ignore # Mass and zero Jacobian
        M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n)).copy()	#type:ignore        
        M.eliminate_zeros() #type:ignore
        
        growths=[]
        for i in range(self.N):
            pert1=self.perturbation[i].copy() # First history perturbation
            pert2=(self.old_perturbation[i] if (self.old_perturbation[i] is not None) else self.perturbation[i]).copy()
            # Assemble the RHS
            rhs=-M@(w1*pert1+w2*pert2)
            # And (re)solve the linear system for the new perturbation
            problem.get_la_solver().solve_serial(2,n,J_nzz,1,J_val,J_rs,J_ci,rhs,0,1)
            # Update the perturbation (rhs stores the solution after solving)
            self.perturbation[i]=rhs.copy()
            # Check whether we have to renormalize
            currnorm=numpy.linalg.norm(self.perturbation[i])
            self.old_perturbation[i]=pert1.copy()
            if currnorm>self.max_perturbation_norm or currnorm<self.min_perturbation_norm:
                self.renormalize(i)            
        
            # Calculate the growth, update the ring buffer and write the current estimate to the file
            growths.append(numpy.log(numpy.linalg.norm(self.perturbation[i])/numpy.linalg.norm(self.old_perturbation[i])))
            
        # Gram-Schmidt
        if self.N>1:
            new_basis=self.perturbation.copy()
            for i in range(self.N):            
                for j in range(i):
                    new_basis[i]-=numpy.dot(self.perturbation[j],self.perturbation[i])/numpy.dot(self.perturbation[j],self.perturbation[j])*self.perturbation[j]
            self.perturbation=new_basis
        
        self.ringbuffer.append((t,numpy.array(growths)))
        # this is essentially 1/(t2-t1)*log(norm(t2)/norm(t1)) by accumulating over the buffer and using the logarithmic addition rule
        if len(self.ringbuffer)>=2:            
            ljap_estimate=sum(r[1] for r in self.ringbuffer)/(self.ringbuffer[-1][0]-self.ringbuffer[0][0])
            if self.average_time is not None:
                while self.ringbuffer[0][0]<t-self.average_time:
                    self.ringbuffer.popleft()
            self.outputfile.write(str(t)+"\t"+"\t".join(map(str,ljap_estimate))+"\n")
            self.outputfile.flush()
    
