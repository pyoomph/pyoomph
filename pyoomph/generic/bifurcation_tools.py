from .problem import Problem
from ..expressions import GlobalParameter
from ..typings import *
import numpy

def get_hopf_lyapunov_coefficient(problem:Problem,param:Union[GlobalParameter,str],FD_delta:float=1e-5,FD_param_delta:float=1e-5,omega:Optional[float]=None,q:Optional[NPComplexArray]=None,mu0:float=0,omega_epsilon:float=1e-6):
    # Taken from § 10.2 of Yuri A. Kuznetsov, Elements of Applied Bifurcation Theory, Fourth Edition, Springer, 2004
    # Also implemented analogously in pde2path, file hogetnf.m 
    
    if isinstance(param,str):
        param=problem.get_global_parameter(param)

    eigensolve_kwargs={}
        
    wrapped_diffs=True
    u=problem.get_current_dofs()[0]
    def nodalf(up):
        problem.set_current_dofs(up)
        res=-numpy.array(problem.get_residuals())        
        return res
    
    def solve_mat(A,rhs):
        return numpy.linalg.solve(A.toarray(),rhs) # TODO: Improve here
    
    delt=FD_delta
    
    
        
    
    ntstep=problem.ntime_stepper()
    was_steady=[False]*ntstep
    for i in range(ntstep):
        ts=problem.time_stepper_pt(i)
        was_steady[i]=ts.is_steady()
        ts.make_steady()
    
    from scipy.sparse import csr_matrix      
    n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = problem.assemble_eigenproblem_matrices(0) #type:ignore
    M=csr_matrix((M_val, M_ci, M_rs), shape=(n, n))	#TODO: Is csr or csc?
    A=csr_matrix((-J_val, J_ci, J_rs), shape=(n, n))
    AT=A.transpose()
    MT=M.transpose()
    
    if omega is None or q is None:
        eval,evect,_,_=problem.get_eigen_solver().solve(2,custom_J_and_M=(A,M),**eigensolve_kwargs)                    
        omega0=numpy.imag(eval[0])
        if omega0<0:
            omega0=-omega0
            qR=numpy.real(evect[1])
            qI=numpy.imag(evect[1])
        else:
            qR=numpy.real(evect[0])
            qI=numpy.imag(evect[0])
        mu0=numpy.real(eval[0])
    else:
        qR=numpy.real(q)
        qI=numpy.imag(q)
        omega0=omega
    qdenom=numpy.dot(qR,qR)+numpy.dot(qI,qI)
    qR/=numpy.sqrt(qdenom)
    qI/=numpy.sqrt(qdenom)
    if numpy.abs(numpy.dot(qR,qI))>1e-7:
        raise ValueError("qR and qI are not orthogonal. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
    
    evalT,evectT,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(AT,MT),**eigensolve_kwargs,shift=-(1j+omega_epsilon)*omega0,v0=numpy.conjugate(q),sort=False)   # TODO: Is MT right here?                 
    #print("GOT",evalT,evectT)
    #print("Omega0",omega0)
    if numpy.imag(evalT[0])<0 and numpy.abs(numpy.imag(evalT[0])+omega0)<1e-6:                    
        #print("Omega0'[0]",numpy.imag(evalT[0]))
        pR=numpy.real(evectT[0])
        pI=numpy.imag(evectT[0])
    else:
        raise ValueError("Could not find the correct eigenvector. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")

    p=(pR+pI*1j)
    #print("EIGENGL",-1j*omega0*(MT*p)-AT*p)
    #print("qR",qR)
    #print("qI",qI)
    #print("pR",pR)
    #print("pI",pI)
    verbose=True

    theta=numpy.angle(numpy.dot(pR,qR)+numpy.dot(pI,qI)+(numpy.dot(pR,qI)-numpy.dot(pI,qR))*1j); 
    p=(pR+pI*1j)*numpy.exp(1j*theta)
    pR=numpy.real(p) 
    pI=numpy.imag(p)
    pnorm=numpy.dot(pR,qR)+numpy.dot(pI,qI)
    if numpy.abs(pnorm)<1e-10:                
        raise ValueError("pnorm is very small. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
        #p*=numpy.exp(1j*numpy.pi/3)
        #pR=numpy.real(p)
        #pI=numpy.imag(p)
        
        #print("q",numpy.dot(qR,qR)+numpy.dot(qI,qI))
        #print("<qR,qI>",numpy.dot(qR,qI))
       
        #pR=numpy.real(p) 
        #pI=numpy.imag(p)
        #pnorm=numpy.dot(pR,qR)+numpy.dot(pI,qI)
        #print(pnorm)
        
        #exit()
        #pnorm=1
    pR=pR/pnorm
    pI=pI/pnorm
    
    if verbose:
        print("Step 1 : Checking equations")
        print("Q Matrix equations (should be zero)")
        print(A*qR+omega0*qI)
        print(-omega0*qR+A*qI)
        print("P Matrix equations (should be zero)")
        print(AT*pR-omega0*pI)
        print(omega0*pR+AT*pI)
        print("Normalisation (1,0) required")
        print(numpy.dot(qR,qR)+numpy.dot(qI,qI),numpy.dot(qR,qI))
        print(numpy.dot(pR,qR)+numpy.dot(pI,qI),numpy.dot(pR,qI)-numpy.dot(pI,qR))
        print("THIS gives:")
        print("qR",qR)
        print("qI",qI)
        print("pR",pR)
        print("pI",pI)
    
    
    
    
    f0=nodalf(u)
    def d2f(direct):            
        # TODO: Make via Hessian products instead        
        fp=nodalf(u+delt*direct)
        fm=nodalf(u-delt*direct)
        return (fm-2*f0+fp)/(delt**2)
        
    def d3f(direct):
        fmm=nodalf(u-2*delt*direct)
        fm=nodalf(u-delt*direct)
        fp=nodalf(u+delt*direct)
        fpp=nodalf(u+2*delt*direct)
        return (-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)
    
    
    
    # Step 2 
    # TODO: Make via Hessian products instead
    
    if wrapped_diffs:
        a=d2f(qR)
        b=d2f(qI)
        c=0.25*(d2f(qR+qI)-d2f(qR-qI))
    else:
        f0=nodalf(u)
        fp=nodalf(u+delt*qR)
        fm=nodalf(u-delt*qR)
        a=(fm-2*f0+fp)/(delt**2)
        fp=nodalf(u+delt*qI)
        fm=nodalf(u-delt*qI)
        b=(fm-2*f0+fp)/(delt**2)
        f1p=nodalf(u+delt*(qR+qI))
        f2p=nodalf(u+delt*(qR-qI))
        f1m=nodalf(u-delt*(qR+qI))
        f2m=nodalf(u-delt*(qR-qI))
        c=0.25*(f1m+f1p-f2m-f2p)
    
    if verbose:
        print("Step 2")
        print("a",a)
        print("b",b)
        print("c",c)

    #step 3
    r=solve_mat(A,M*(a+b))
    sv=solve_mat(-A+2j*M*omega0,M*(a-b+2j*c))
    sR=numpy.real(sv)
    sI=numpy.imag(sv)
    if verbose:
        print("Step 3")
        print("r",r)
        print("sR",sR)
        print("sI",sI)
        print("CHECKING r",A*r-M*(a+b))
        print("CHECKING sR",A*sR-2*omega0*M*sI,a-b)
        print("CHECKING sI",2*omega0*M*sR-A*sI,2*c)

    # step 4
    if wrapped_diffs:
        sig1=0.25*numpy.dot(pR,d2f(qR+r)-d2f(qR-r))
        sig2=0.25*numpy.dot(pI,d2f(qI+r)-d2f(qI-r)) # TODO: In the book, it is pI, in pde2path is is pR
    else:
        f1p=nodalf(u+delt*(qR+r))
        f2p=nodalf(u+delt*(qR-r))
        f1m=nodalf(u-delt*(qR+r))
        f2m=nodalf(u-delt*(qR-r))
        sig1=0.25*numpy.dot(pR,(f1m+f1p-f2m-f2p))
        f1p=nodalf(u+delt*(qI+r))
        f2p=nodalf(u+delt*(qI-r))
        f1m=nodalf(u-delt*(qI+r))
        f2m=nodalf(u-delt*(qI-r));
        sig2=0.25*numpy.dot(pI,(f1m+f1p-f2m-f2p)) # TODO: In the book, it is pI, in pde2path is is pR
    sig=sig1+sig2
    if verbose:
        print("Step 4")
        print("sig1",sig1)
        print("sig2",sig2)
        print("sig",sig)
    
    # step 5
    if wrapped_diffs:
        d1=0.25*numpy.dot(pR,d2f(qR+sR)-d2f(qR-sR))
        d2=0.25*numpy.dot(pR,d2f(qI+sI)-d2f(qI-sI))
        d3=0.25*numpy.dot(pI,d2f(qR+sI)-d2f(qR-sI)) # TODO: In the book, it is pI, in pde2path is is pR
        d4=0.25*numpy.dot(pI,d2f(qI+sR)-d2f(qI-sR)) # TODO: In the book, it is pI, in pde2path is is pR
    else:
        f1p=nodalf(u+delt*(qR+sR))
        f2p=nodalf(u+delt*(qR-sR))
        f1m=nodalf(u-delt*(qR+sR))
        f2m=nodalf(u-delt*(qR-sR))
        d1=0.25*numpy.dot(pR,(f1m+f1p-f2m-f2p))
        f1p=nodalf(u+delt*(qI+sI))
        f2p=nodalf(u+delt*(qI-sI))
        f1m=nodalf(u-delt*(qI+sI))
        f2m=nodalf(u-delt*(qI-sI))
        d2=0.25*numpy.dot(pR,(f1m+f1p-f2m-f2p))
        f1p=nodalf(u+delt*(qR+sI))
        f2p=nodalf(u+delt*(qR-sI))
        f1m=nodalf(u-delt*(qR+sI))
        f2m=nodalf(u-delt*(qR-sI))
        d3=0.25*numpy.dot(pI,(f1m+f1p-f2m-f2p)) # TODO: In the book, it is pI, in pde2path is is pR
        f1p=nodalf(u+delt*(qI+sR))
        f2p=nodalf(u+delt*(qI-sR))
        f1m=nodalf(u-delt*(qI+sR))
        f2m=nodalf(u-delt*(qI-sR))
        d4=0.25*numpy.dot(pI,(f1m+f1p-f2m-f2p)) # TODO: In the book, it is pI, in pde2path is is pR
    d0=d1+d2+d3-d4
    if verbose:
        print("Step 5")
        print("d1",d1)
        print("d2",d2)
        print("d3",d3)
        print("d4",d4)
        print("d0",d0)
    
    # Step 6
    
    #d=[qr;qz]; 
    if wrapped_diffs:
        g1=numpy.dot(pR,d3f(qR))
        g2=numpy.dot(pI,d3f(qI))
        g3=numpy.dot(pR+pI,d3f(qR+qI))
        g4=numpy.dot(pR-pI,d3f(qR-qI))
    else:
        fmm=nodalf(u-2*delt*qR)
        fm=nodalf(u-delt*qR)
        fp=nodalf(u+delt*qR)
        fpp=nodalf(u+2*delt*qR); 
        g1=numpy.dot(pR,(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)) 
        #d=[qi;qz]; 
        fmm=nodalf(u-2*delt*qI)
        fm=nodalf(u-delt*qI)
        fp=nodalf(u+delt*qI)
        fpp=nodalf(u+2*delt*qI)
        g2=numpy.dot(pI,(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3))
        d=qR+qI
        fmm=nodalf(u-2*delt*d)
        fm=nodalf(u-delt*d)
        fp=nodalf(u+delt*d)
        fpp=nodalf(u+2*delt*d)        
        g3=numpy.dot(pR+pI,(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3))
        d=qR-qI
        fmm=nodalf(u-2*delt*d)
        fm=nodalf(u-delt*d)
        fp=nodalf(u+delt*d)
        fpp=nodalf(u+2*delt*d)
        g4=numpy.dot(pR-pI,(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3))
    g0=2*(g1+g2)/3+(g3+g4)/6
    if verbose:
        print("Step 6")
        print("g1",g1)
        print("g2",g2)
        print("g3",g3)
        print("g4",g4)
        print("g0",g0)
    
    # step 7
    ga=(g0-2*sig+d0)/abs(2*omega0);
    c1=abs(omega0)*ga
    if verbose:
        print("Step 7")
        print("ga",ga)
        print("c1",c1)
  

    # Get dmu/dparam    
    old=param.value
    param.value+=FD_param_delta
    evalsP,_=problem.solve_eigenproblem(2,**eigensolve_kwargs,shift=(1j+omega_epsilon)*omega0)
    param.value=old
    mu=numpy.real(evalsP[0])    
    mup=-(mu-mu0)/FD_param_delta
    if ((c1>0 and mup>0) or (c1<0 and mup<0)):
        dlam=-1
    else:
        dlam=1
    al=numpy.sqrt(-dlam*mup/c1); 
    if verbose:
        print("dmu_dparam",mup,al)
        print("Would return",dlam,al)
    

    problem.set_current_dofs(u) # TODO: Do not do it every time, but only when required      
    for i in range(ntstep):
        if not was_steady[i]:
            problem.time_stepper_pt(i).undo_make_steady()

    return ga,dlam,al,qR,qI