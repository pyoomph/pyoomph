from .problem import Problem
from ..expressions import GlobalParameter
from ..typings import *

import numpy,scipy

def get_hopf_lyapunov_coefficient(problem:Problem,param:Union[GlobalParameter,str],FD_delta:float=1e-5,FD_param_delta:float=1e-5,omega:Optional[float]=None,q:Optional[NPComplexArray]=None,mu0:float=0,omega_epsilon:float=1e-3):
    # Taken from § 10.2 of Yuri A. Kuznetsov, Elements of Applied Bifurcation Theory, Fourth Edition, Springer, 2004
    # Also implemented analogously in pde2path, file hogetnf.m 
    # XXX Here is the generalization of the code with mass matrix
    # In Kuznetsov, it is assumed that the dofs x can be cast to a complex number z via z=<p,x>
    # If you have a mass matrix, it must be z=<p,Mx> where M is the mass matrix
    # The rest stays the same
    # Thereby, when having the eigenvector q corresponding to lambda=i*omega (i*omega*M*q=A*q=-J*q)
    # we must find the adjoint eigenvector p, which fulfills A^T*p=-i*omega*M^T*p
    # and it must be normalized so that <p,Mq>=1 and <p,M(q*)>=0
    #
    # Thereby, you can take the dynamics close to the bifurcations:
    #
    #   M*partial_t(x)=A*x+F(x,alpha)
    #
    # Multiply it via <p| and use the definition of z (see above), to get
    #
    #   partial_t(z)=<p,A*x> + <p,F(x,alpha)>
    #
    # write <p,A*x>=<A^T*p,x>=<lambda* *M^T*p,x>=lambda*<p,Mx>=lambda*z
    # so that we end up at
    # 
    #   partial_t(z)=lambda*z + <p,F(x,alpha)>
    #   
    # Now, we can apply the method of Kuznetsov to get the Lyapunov coefficient.
    #
    # However, when mapping the expansion coefficients h_lk of the quadratic and cubic terms of the normal form
    #
    #      partial_t(z)=lambda*z + sum_{2<=l+k<=3} h_lk z^l z*^k
    #
    # We get the following equation:
    #
    #   h20 = (2*i*omega*M-A)^(-1) B(q,q)       instead of     h20 = (2*i*omega*I-A)^(-1) B(q,q) [5.30 in the book]
    #   h11= -A^(-1) B(q,qb)                    remains the same    [5.31 in the book]
    #
    # The cubic term [see 5.32 in the book] is also augmented with A mass matrix
    #
    #   (i*omega0*M-A)*h21=C(q,q,qb)+B(qb,h20)+2*B(q,h11)-2*c1*M*q
    #
    # But the argument is the same: When applying <p| to this equation, the lhs vanishes.
    # Since <p,q>=1, we get
    #
    #   c1=1/2*<p,C(q,q,qb)+2*B(q,h11)+B(qb,h20)>
    #
    # as in the book
    
    esolver=problem.get_eigen_solver()
    
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
        #return numpy.linalg.solve(A.toarray(),rhs) # TODO: Improve here
        return scipy.sparse.linalg.spsolve(A,rhs)# TODO: Improve here to use e.g. Pardiso (however, requires complex support)
    
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
    AT=A.transpose().tocsr()
    MT=M.transpose().tocsr()
    
    if omega is None or q is None:
        print("Solving for omega and q")
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
    print("PREDOT",numpy.dot(qR,qI))
    qdenom=numpy.dot(qR,qR)+numpy.dot(qI,qI)
    qR/=numpy.sqrt(qdenom)
    qI/=numpy.sqrt(qdenom)
    print("POSTDOT",numpy.dot(qR,qI))
    
    if False or (numpy.amax(numpy.abs(A@qR+omega0*M@qI))>1e-7 or numpy.amax(-omega0*M@qR+A*qI)>1e-7):
        print("Given q does not fulfill the eigenvector equation. Resolving it.")
        esolv_kwargs=eigensolve_kwargs.copy()
        if esolver.supports_target():
            esolv_kwargs["target"]=1j*omega0
        eval,evects,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(A,M),**esolv_kwargs,shift=(1j+omega_epsilon)*omega0,v0=numpy.array(q),sort=False,quiet=False)
        #for ev,evec in zip(eval,evects):
        #    print("EVAL",ev)
        #    #evec=numpy.conjugate(evec)
        #    print(numpy.amax(numpy.abs(-A@evec+ev*M@evec)))
        #    print(numpy.amax(numpy.abs(A@numpy.real(evec)+numpy.imag(ev)*M@numpy.imag(evec))))
        #    print(numpy.amax(numpy.abs(-numpy.imag(ev)*M@numpy.real(evec)+A@numpy.imag(evec))))
        omega0=numpy.imag(eval[0])
        if omega0<0:
            omega0=-omega0
            qR=numpy.real(evects[0])
            qI=-numpy.imag(evects[0])
        else:
            qR=numpy.real(evects[0])
            qI=numpy.imag(evects[0])
        
        #print("AFTER SETTING")    
        #print(numpy.amax(numpy.abs(-A@(qR+1j*qI)+1j*omega0*M@(qR+1j*qI))))
        #print(numpy.amax(numpy.abs(A@qR+omega0*M@qI)))
        #print(numpy.amax(numpy.abs(-omega0*M@qR+A@qI)))
        
    if numpy.abs(numpy.dot(qR,qI))>1e-7:
        raise ValueError("qR and qI are not orthogonal. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
    
    esolv_kwargs=eigensolve_kwargs.copy()
    if esolver.supports_target():
        esolv_kwargs["target"]=-1j*omega0
    evalT,evectT,_,_=problem.get_eigen_solver().solve(1,custom_J_and_M=(AT,MT),**esolv_kwargs,shift=-(1j+omega_epsilon)*omega0,v0=numpy.conjugate(q),sort=False,quiet=False)   # TODO: Is MT right here?                 
    #print("GOT",evalT,evectT)
    #print("Omega0",omega0)
    if numpy.imag(evalT[0])<0 and numpy.abs(numpy.imag(evalT[0])+omega0)<1e-6:                    
        #print("Omega0'[0]",numpy.imag(evalT[0]))
        pR=numpy.real(evectT[0])
        pI=numpy.imag(evectT[0])
        #print("Omega0 for q",omega0,"and for p",numpy.imag(evalT[0]),"sum",numpy.imag(evalT[0])+omega0)
        #print("Precheck: P Matrix equations (should be zero)")
        #print(numpy.amax(AT@pR-omega0*MT@pI))
        #print(numpy.amax(omega0*MT@pR+AT@pI))
    else:
        raise ValueError("Could not find the correct eigenvector. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")

    #p=MT@(pR+pI*1j)
    #pR=numpy.real(p) 
    #pI=numpy.imag(p)
    #print("EIGENGL",-1j*omega0*(MT*p)-AT*p)
    #print("qR",qR)
    #print("qI",qI)
    #print("pR",pR)
    #print("pI",pI)
    verbose=True
    
    # XXX Here is the generalization of the code with mass matrix
    Mq=M@(qR+qI*1j)
    MqR=numpy.real(Mq)
    MqI=numpy.imag(Mq)
    #print("<p,Mq>",numpy.vdot(pR+1j*pI,M@(qR+1j*qI)))
    #print("<p,Mq*>",numpy.vdot(pR+1j*pI,M@(qR-1j*qI)))
    #exit()
    #theta=numpy.angle(numpy.dot(pR,qR)+numpy.dot(pI,qI)+(numpy.dot(pR,qI)-numpy.dot(pI,qR))*1j); 
    theta=numpy.angle(numpy.dot(pR,MqR)+numpy.dot(pI,MqI)+(numpy.dot(pR,MqI)-numpy.dot(pI,MqR))*1j); 
    p=(pR+pI*1j)*numpy.exp(1j*theta)
    pR=numpy.real(p) 
    pI=numpy.imag(p)
    pnorm=numpy.dot(pR,MqR)+numpy.dot(pI,MqI)
    if numpy.abs(pnorm)<1e-10:                
        print(numpy.dot(qR,qR)+numpy.dot(qI,qI),numpy.dot(qR,qI))
        print(numpy.dot(pR,qR)+numpy.dot(pI,qI),numpy.dot(pR,qI)-numpy.dot(pI,qR))
        print(numpy.linalg.norm(pR),numpy.linalg.norm(pI))
        print("<p,q>",numpy.vdot(pR+1j*pI,qR+1j*qI))
        print("<p,q*>",numpy.vdot(pR+1j*pI,qR-1j*qI))
        for theta in numpy.linspace(0,2*numpy.pi,20):
            p=(pR+pI*1j)*numpy.exp(1j*theta)
            pRt=numpy.real(p) 
            pIt=numpy.imag(p)
            print("Theta",theta,"<p,q>",numpy.vdot(pRt+1j*pIt,qR+1j*qI),"<p,q*>",numpy.vdot(pRt+1j*pIt,qR-1j*qI))
        #exit()
        #print("pnorm is very small. This is likely an issue with the eigenvalue solver. Please check the eigenvalue solver settings.")
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
    
    qunitscale=max(numpy.amax(numpy.absolute(qR)),numpy.amax(numpy.absolute(qI)))
    punitscale=max(numpy.amax(numpy.absolute(pR)),numpy.amax(numpy.absolute(pI)))
    print("Q unitscale",qunitscale)
    print("Q magnitude",numpy.linalg.norm(qR),numpy.linalg.norm(qI))
    print("P unitscale",punitscale)
    print("P magnitude",numpy.linalg.norm(pR),numpy.linalg.norm(pI))
    qRus=qR/qunitscale
    qIus=qI/qunitscale
    pRus=pR/punitscale
    pIus=pI/punitscale
    
    
    if verbose:
        print("Step 1 : Checking equations")
        print("Q Matrix equations (should be zero)")
        print(numpy.amax(numpy.abs(A@qR+omega0*M@qI)))
        print(numpy.amax(-omega0*M@qR+A@qI))
        print("P Matrix equations (should be zero)")
        print(numpy.amax(AT@pR-omega0*MT@pI))
        print(numpy.amax(omega0*MT@pR+AT@pI))
        print("Normalisation (1,0) required")
        print(numpy.dot(qR,qR)+numpy.dot(qI,qI),numpy.dot(qR,qI))
        print(numpy.dot(pR,MqR)+numpy.dot(pI,MqI),numpy.dot(pR,MqI)-numpy.dot(pI,MqR))
        exit()
        #print("THIS gives:")
        #print("qR",qR)
        #print("qI",qI)
        #print("pR",pR)
        #print("pI",pI)
    
    
    
    
    f0=nodalf(u)
    def d2f(direct):                    
        res_hess=-numpy.array(problem.get_second_order_directional_derivative(direct))        
        if True:
            fp=nodalf(u+delt*direct)
            fm=nodalf(u-delt*direct)
            problem.set_current_dofs(u)
            res_fd=(fm-2*f0+fp)/(delt**2)        
            print("DIFFERENCE",numpy.amax(numpy.absolute(res_hess-res_fd)),"FD",numpy.amax(numpy.absolute(res_fd)),"HESS",numpy.amax(numpy.absolute(res_hess)),"DIRECT",numpy.amax(numpy.absolute(res_hess)))
            return res_hess
        return res_hess
        
    def d3f(direct):
        direct_scale=numpy.amax(numpy.abs(direct))                
        if direct_scale<1e-10:
            direct_scale=1
        else:
            direct_scale=1/direct_scale        
        direct_scale=1
        problem.set_current_dofs(u+delt*direct*direct_scale)
        res_hessp=-numpy.array(problem.get_second_order_directional_derivative(direct))
        problem.set_current_dofs(u-delt*direct*direct_scale)
        res_hessm=-numpy.array(problem.get_second_order_directional_derivative(direct))
        problem.set_current_dofs(u)        
        res_hess=0.5*(res_hessp-res_hessm)/(delt*direct_scale)
        
        if True:        
            fmm=nodalf(u-2*delt*direct)
            fm=nodalf(u-delt*direct)
            fp=nodalf(u+delt*direct)
            fpp=nodalf(u+2*delt*direct)
            res_fd=(-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)
            print("DIFFERENCE 3d order",numpy.amax(numpy.absolute(res_hess-res_fd)),"FD",numpy.amax(numpy.absolute(res_fd)),"HESS",numpy.amax(numpy.absolute(res_hess)),"DIRECT",numpy.amax(numpy.absolute(direct)))
            #print("RESP",res_hessp,res_hessm,"DIRECT",direct)
            return res_hess # (-0.5*fmm+fm-fp+0.5*fpp)/(delt**3)
        return res_hess
    
    
    # Step 2 
    # TODO: Make via Hessian products instead
    
    use_unitscales=False
    
    if wrapped_diffs:
        if use_unitscales:
            a=d2f(qRus)
            b=d2f(qIus)
            c=0.25*(d2f(qRus+qIus)-d2f(qRus-qIus))
        else:
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
        print("A magnitude",numpy.linalg.norm(a))
        print("B magnitude",numpy.linalg.norm(b))
        print("C magnitude",numpy.linalg.norm(c))
        #a/=numpy.linalg.norm(a)
        #b/=numpy.linalg.norm(b)
        #c/=numpy.linalg.norm(c)
        #print("a",a)
        #print("b",b)
        #print("c",c)

    #step 3
    r=solve_mat(A,M*(a+b))
    sv=solve_mat(-A+2j*M*omega0,M*(a-b+2j*c))
    sR=numpy.real(sv)
    sI=numpy.imag(sv)
    if verbose:
        print("Step 3")
        #print("r",r)
        #print("sR",sR)
        #print("sI",sI)
        print("CHECKING r",numpy.amax(numpy.absolute(A@r-M@(a+b))),"Rmagnitude",numpy.linalg.norm(r))
        print("CHECKING sR",numpy.amax(numpy.absolute(-A@sR-2*omega0*M@sI - M@(a-b))),"Sr magnitude",numpy.linalg.norm(sR))
        print("CHECKING sI",numpy.amax(numpy.absolute(2*omega0*M@sR-A@sI - 2*M@c)), "SI magnitude",numpy.linalg.norm(sI))

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
    if False:
        param.value+=FD_param_delta    
        evalsP,_=problem.solve_eigenproblem(2,**eigensolve_kwargs,shift=(1j+omega_epsilon)*omega0)
        mu=numpy.real(evalsP[0])    
        mup=-(mu-mu0)/FD_param_delta
    else:
        problem.activate_eigenbranch_tracking("complex",eigenvector=q,eigenvalue=1j*omega0)        
        problem.solve()
        mu1=numpy.real(problem.get_last_eigenvalues()[0])
        problem.go_to_param(**{param.get_name():param.value+FD_param_delta})
        mu2=numpy.real(problem.get_last_eigenvalues()[0])
        mup=-(mu2-mu1)/FD_param_delta
        problem.deactivate_bifurcation_tracking()
        problem.set_current_dofs(u)                            

    param.value=old
    
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