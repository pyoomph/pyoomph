from pyoomph import * 
from pyoomph.expressions import * 


# Same Lorenz system as before
class LorenzSystem(ODEEquations):
    def __init__(self,*,sigma=10,rho=28,beta=8/3,scheme="BDF2"): # Default parameters used by Lorenz
        super(LorenzSystem,self).__init__()
        self.sigma=sigma
        self.rho=rho
        self.beta=beta
        self.scheme=scheme        

    def define_fields(self):
        self.define_ode_variable("x","y","z") 
    
    def define_residuals(self):
        x,y,z=var(["x","y","z"])
        residual=(partial_t(x)-self.sigma*(y-x))*testfunction(x)
        residual+=(partial_t(y)-x*(self.rho-z)+y)*testfunction(y)
        residual+=(partial_t(z)-x*y+self.beta*z)*testfunction(z)
        self.add_residual(time_scheme(self.scheme,residual))        


# Problem as before
class LorenzProblem(Problem):    
    def define_problem(self):
        eqs=LorenzSystem()
        eqs+=InitialCondition(x=0.01)  # Some non-trivial initial position
        eqs+=TemporalErrorEstimator(x=1,y=1,z=1) # Weight all temporal error with unity
        eqs+=ODEFileOutput()  
        self+=eqs@"lorenz_attractor"


if __name__=="__main__":
    
    # Import the LyapunovExponentCalculator from the utils module
    from pyoomph.utils.lyapunov import LyapunovExponentCalculator
    
    with LorenzProblem() as problem:
        # We want to save memory, since we have a fine temporal discretization. 
        # So we do not write state files for continue simulations
        problem.write_states=False 
        # Add the LyapunovExponentCalculator to the problem. 
        # Averaging over T_avg=20 and calculating N=3 Lyapunov exponents
        problem+=LyapunovExponentCalculator(average_time=20,N=3)
        # Run it with a rather fine time step and temporal error
        problem.run(endtime=200,outstep=0.0025,startstep=0.01,temporal_error=0.05,maxstep=0.01)        
