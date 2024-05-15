#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2024  Christian Diddens & Duarte Rocha
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
from .generic import GenericLinearSystemSolver, GenericEigenSolver, EigenSolverWhich
from collections import OrderedDict
import petsc4py #type:ignore
import sys

petsc4py.init(sys.argv) #type:ignore

import slepc4py #type:ignore

slepc4py.init(sys.argv) #type:ignore

from petsc4py import PETSc #type:ignore
from slepc4py import SLEPc #type:ignore
from ..generic.mpi import *
from ..typings import *
import numpy

if TYPE_CHECKING:
    from ..generic.problem import Problem


@GenericLinearSystemSolver.register_solver()
class PETSCSolver(GenericLinearSystemSolver):
    idname = "petsc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self._do_not_set_any_args:bool=False

    #		opts=PETSc.Options().getAll()
    #		if "add_zero_diagonal" in opts.keys():
    #			problem.set_diagonal_zero_entries(True)

    def set_default_petsc_option(self,name:str,val:Any=None,force:bool=False)->None:
        _SetDefaultPetscOption(name,val, force) #type:ignore

    def setup_solver(self):        
        opts = PETSc.Options().getAll() #type:ignore
        if "add_zero_diagonal" in opts.keys(): #type:ignore
            #			print(dir(self.petsc_mat))
            self.petsc_mat.setOption(19, 0) #type:ignore
            self.petsc_mat.shift(0) #type:ignore

        self.ksp = PETSc.KSP().create() #type:ignore
        self.ksp.setOperators(self.petsc_mat) #type:ignore
        if not self._do_not_set_any_args:
            self.ksp.setType('preonly') #type:ignore
        pc = self.ksp.getPC() #type:ignore
        if not self._do_not_set_any_args:
            pc.setType('lu') #type:ignore
        opts = PETSc.Options().getAll() #type:ignore
        if "pc_factor_mat_solver_type" in opts.keys(): #type:ignore
            if hasattr(pc, "setFactorSolverPackage"): #type:ignore
                if not self._do_not_set_any_args: #type:ignore
                    pc.setFactorSolverPackage(opts["pc_factor_mat_solver_type"]) #type:ignore
        pc.setFromOptions() #type:ignore
        self.ksp.setFromOptions() #type:ignore
        #print('Solving with:', self.ksp.getType())  # ,dir(pc)

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag == 1:
            self.petsc_mat = PETSc.Mat().createAIJ(size=(n, n), csr=(colptr, rowind, values)) #type:ignore
            self.x = PETSc.Vec().createSeq(n) #type:ignore
        elif op_flag == 2:
            self.setup_solver()
            bv = PETSc.Vec().createWithArray(b) #type:ignore
            self.ksp.solve(bv, self.x) #type:ignore
            xv = self.x.getArray() #type:ignore
            b[:] = xv[:] #type:ignore

            print('Converged in', self.ksp.getIterationNumber(), 'iterations.') #type:ignore

            self.petsc_mat.destroy() #type:ignore
            self.ksp.destroy() #type:ignore
            self.x.destroy() #type:ignore
            bv.destroy() #type:ignore
        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")
        return 0  # TODO: Return sign of Jacobian

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:

        if op_flag == 1:
            #print("PETSCINF",nrow_local,n)
            self.petsc_mat = PETSc.Mat().createAIJ(size=((nrow_local, n), (nrow_local, n),),csr=(row_start, col_index, values)) #type:ignore
        #			print("PROCESSOR Ns",get_mpi_rank(),nrow_local,n)
        #			print("PROCESSOR RS",get_mpi_rank(),row_start)
        #			print("PROCESSOR CI",get_mpi_rank(),col_index)
        #			print("FIRST ROW",get_mpi_rank(),first_row)
        # self.petsc_mat
        elif op_flag == 2:
            self.setup_solver()
            bv = PETSc.Vec().createWithArray(b, (len(b), n)) #type:ignore
            xv = bv.duplicate() #type:ignore
            self.ksp.solve(bv, xv) #type:ignore
            res = xv.getArray() #type:ignore
            b[:] = res[:] #type:ignore

            self.petsc_mat.destroy() #type:ignore
            self.ksp.destroy() #type:ignore
            xv.destroy() #type:ignore
            bv.destroy() #type:ignore
        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")

    def set_options(self,**kwargs:Any):
        for a,b in kwargs.items():
            PETSc.Options().setValue(a,b) #type:ignore

def _SetDefaultPetscOption(key:str, val:Any,force:bool=False):
    if force or (not PETSc.Options().hasName(key)): #type:ignore
        PETSc.Options().setValue(key, val) #type:ignore


@GenericEigenSolver.register_solver()
class SlepcEigenSolver(GenericEigenSolver):
    idname = "slepc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)

    def further_setup(self,E): #type:ignore
        pass
    
    def use_mumps(self):
        _SetDefaultPetscOption("mat_mumps_icntl_6",5)
        _SetDefaultPetscOption("st_ksp_type","preonly")
        _SetDefaultPetscOption("st_pc_type","lu")
        _SetDefaultPetscOption("st_pc_factor_mat_solver_type","mumps")

    def solve(self, neval:int, shift:Union[float,None,complex]=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Optional[Literal["r","i"]]=None,v0:Optional[Union[NPComplexArray,NPFloatArray]]=None,target:Optional[complex]=None)->Tuple[NPComplexArray,NPComplexArray]:
        if which!="LM":
            raise RuntimeError("Implement which="+str(which))
        if OPpart is not None:
            raise RuntimeError("Implement OPpart="+str(OPpart))
#        if v0 is not None:
#            raise RuntimeError("Implement v0="+str(v0))
        
        if False:
            n, M_nzz, M_nr, M_val, M_ci, M_rs, J_nzz, J_nr, J_val, J_ci, J_rs = self.problem.assemble_eigenproblem_matrices(0) #type:ignore

            #print("M value ranges",numpy.amin(M_val),numpy.amax(M_val)) #type:ignore
            M = PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(M_rs, M_ci, M_val)) #type:ignore
            J = PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(J_rs, J_ci, -J_val)) #type:ignore

        else:
            Jin,Min,n,complex_mat=self.get_J_M_n_and_type()
            upscale_to_complex=complex_mat and (PETSc.ScalarType in {numpy.float64,numpy.float128,numpy.float32})
            if upscale_to_complex:
                raise RuntimeError("SLEPc cannot handle a complex matrix here...")
            M=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Min.indptr, Min.indices, Min.data))
            J=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Jin.indptr, Jin.indices, Jin.data))
            
        if self.imag_contribution is not None:
            raise RuntimeError("Cannot have imaginary matrix contributions yet here")
#        for manip in self.matrix_manipulators:
#            raise RuntimeError("Cannot have MatrixManipulators yet here: "+str(manip))
            #J, M = manip.apply_on_J_and_M(self, J, M)

        # TODO: Working example
        ##--petsc -st_pc_type lu -st_pc_factor_mat_solver_type umfpack
        # print(dir(PETSc.Options.hasName))
        # exit()
        _SetDefaultPetscOption("st_ksp_type", "preonly")
        _SetDefaultPetscOption("st_type", "sinvert")
        _SetDefaultPetscOption("eps_type", "arnoldi")

        #_SetDefaultPetscOption("eps_target_magnitude", 1)
        if target:
            _SetDefaultPetscOption("eps_target_magnitude",0)
        else:
            _SetDefaultPetscOption("eps_target_real", 0)
        if shift is not None:            
            _SetDefaultPetscOption("st_shift", shift)
        E = SLEPc.EPS()  #type:ignore
        E.create() #type:ignore
        if target is not None:
            if abs(numpy.real(target))<1e-20:
                if abs(numpy.imag(target))<1e-20:
                    trg="0.0"
                elif numpy.imag(target)>0:
                    trg="0.0+"+str(numpy.imag(target))+"i"
                else:
                    trg="0.0-"+str(numpy.imag(target))+"i"
            elif abs(numpy.imag(target))<1e-20:
                trg=str(numpy.real(target))
            else:
                if numpy.imag(target)>0:
                    trg=str(numpy.real(target))+"+"+str(numpy.imag(target))+"i"
                else:
                    trg=str(numpy.imag(target))+"-"+str(numpy.imag(target))+"i"
            print(trg)
            _SetDefaultPetscOption("eps_target",trg)
            
            #trgt=PETSc.toScalar(target)
            #print(trgt)
            #E.setTarget(trgt)
        E.setOperators(J, M) #type:ignore
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #type:ignore
        if v0 is not None:
            _v0=PETSc.Vec().createWithArray(v0)
            E.setInitialSpace(_v0)
        #E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        ncv=max(2 * neval + 1, 5 + neval)
        mdp=ncv #TODO: Can be smaller for higher
        E.setDimensions(neval,ncv,mdp) #type:ignore
        #print(dir(E))
        #exit()
        # E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        E.setFromOptions() #type:ignore

        self.further_setup(E) #type:ignore
        E.solve() #type:ignore

        Print = PETSc.Sys.Print #type:ignore
        Print()
        Print("******************************")
        Print("*** SLEPc Solution Results ***")
        Print("******************************")
        Print()
        its = E.getIterationNumber() #type:ignore
        Print("Number of iterations of the method: %d" % its) #type:ignore
        eps_type = E.getType() #type:ignore
        Print("Solution method: %s" % eps_type) #type:ignore
        nev, ncv, mpd = E.getDimensions()  #type:ignore
        Print("Number of requested eigenvalues: %d" % nev)
        tol, maxit = E.getTolerances() #type:ignore
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit)) #type:ignore
        nconv = E.getConverged() #type:ignore
        Print("Number of converged eigenpairs %d" % nconv) #type:ignore

        Print(M) #type:ignore

        evals = []
        evects = []
        if nconv > 0:
            # Create the results vectors

            vr, wr = J.getVecs() #type:ignore
            vi, wi = J.getVecs() #type:ignore
            #
            Print()
            Print(" k ||Ax-kx||/||kx|| ")
            Print("----------------- ------------------")
            #lastev = None
            for i in range(nconv): #type:ignore
                k = E.getEigenpair(i, vr, vi) #type:ignore
                error = E.computeError(i) #type:ignore
                evals.append(k) #type:ignore
                _vr = vr.getArray() #type:ignore
                #Print("IN K %9f%+9f j"%(k.real,k.imag))
                if k.imag != 0.0: #type:ignore
                    #Print("LASTEV "+("None" if lastev is None else "NOTNONE"))

                    if False:
                        if lastev is not None:
                            Print("DIFF  "+str(numpy.abs(lastev - numpy.conjugate(k))))
                            if numpy.abs(lastev - numpy.conjugate(k)) == 0.0:
                                Print("ADDING VI  "+str(vi.getArray()))
                                evects.append(0+vi.getArray())
                            #lastev = None
                        else:
                            evects.append(0+_vr)
                            #lastev = k
                        Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                    else:
                        evects.append(_vr+vi.getArray()*1j) #type:ignore
                else:
                    #lastev = None
                    evects.append(0+_vr) #type:ignore
                    Print(" %12f %12g" % (k.real, error)) #type:ignore

        evals = numpy.array(evals) #type:ignore
        if sort:
            if sort==True:
                srt = numpy.argsort(-evals)[0:min(neval, len(evals))] #type:ignore
            else:
                srt = numpy.argsort(numpy.array([sort(x) for x in evals]))[0:min(neval, len(evals))] #type:ignore
            #print("SORTING",evals,srt)
            evals = evals[srt] #type:ignore
            evects = numpy.array(evects)[srt] #type:ignore
        else:
            evects = numpy.array(evects) #type:ignore
        E.destroy() #type:ignore
        return numpy.array(evals), numpy.array(evects) #type:ignore



class FieldSplitPETSCSolver(PETSCSolver):
    def __init__(self,problem:"Problem"):
        super(FieldSplitPETSCSolver, self).__init__(problem)
        self._fieldsplit_map:Optional[NPIntArray]=None
        self._fieldsplit:Optional[List[Tuple[str,Any]]]=None
        self.default_field_split:Optional[int]=None

    def define_options(self):
        pass

    def define_field_split(self):
        #Call split_fields here (with e.g. domain/velocity_x=0, ...)
        pass

    def split_fields(self,**kwargs:int):
        meshblocks:OrderedDict[str,Dict[str,int]]=OrderedDict()
        wholemesh:OrderedDict[str,int]=OrderedDict()
        allkeys:OrderedDict[str,bool]=OrderedDict()
        for n,b in kwargs.items():
            tmesh=self.problem.get_mesh(n,return_None_if_not_found=True)
            if tmesh is None:
                #Simple field only
                sp=n.split("/")
                sn="/".join(sp[0:-1])
                tmesh = self.problem.get_mesh(sn, return_None_if_not_found=True)
                if tmesh is None:
                    raise RuntimeError("Cannot perform a field split for the unknown field "+n)
                if not (sn in meshblocks.keys()):
                    meshblocks[sn]={}
                meshblocks[sn][sp[-1]]=b
                allkeys[sn]=True
            else: #Whole mesh
                if n in wholemesh.keys():
                    raise RuntimeError("Duplicated argument "+n)
                wholemesh[n]=b
                allkeys[n]=True


        for k in allkeys.keys():
            mesh=self.problem.get_mesh(k)
            typesI, names = mesh.describe_global_dofs()
            name_look_up={v:i for i,v in enumerate(names)}
            types:NPIntArray = numpy.array(typesI,dtype=numpy.int32) #type:ignore
            if self._fieldsplit_map is None:
                self._fieldsplit_map=0*types-1
            if k in wholemesh.keys():
                dest=wholemesh[k]
                where=numpy.where(types>=0)[0] #type:ignore
                self._fieldsplit_map[where]=dest
            if k in meshblocks.keys():
                for vn,dest in meshblocks[k].items():
                    if not (vn in name_look_up.keys()):
                        raise RuntimeError("Cannot find the field "+vn+" on mesh "+k+" to split")
                    where = numpy.where(types == name_look_up[vn])[0] #type:ignore
                    self._fieldsplit_map[where] = dest


    def _perform_field_split(self):
        if self._fieldsplit_map is None:
            return
        where = numpy.where(self._fieldsplit_map == -1)[0] #type:ignore
        if len(where)>0:
            if self.default_field_split is None:
                raise RuntimeError("Found a defined field split. Use either default_field_split or specify all fields in the define_field_split")
            else:
                self._fieldsplit_map[where] = self.default_field_split

        numfields:int=numpy.amax(self._fieldsplit_map)+1 #type:ignore
        fields = []
        for i in range(numfields):
            IS = PETSc.IS() #type:ignore
            subdofs = numpy.where(self._fieldsplit_map == i)[0] #type:ignore
            subdofs:NPIntArray  = numpy.array(subdofs, dtype="int32") #type:ignore
            IS.createGeneral(subdofs) #type:ignore
            name = str(i)
            fields.append((name, IS)) #type:ignore
        pc = self.ksp.getPC() #type:ignore
        pc.setFieldSplitIS(*fields) #type:ignore
        self._fieldsplit=fields

    def assemble_preconditioner(self,name:str,restrict_on_field_split:Optional[int]=None)->Any:        
        _, n, _, _, _, J_ci, J_rs = self.problem._assemble_residual_jacobian(name)
        P = PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(J_rs, J_ci, J_val)) #type:ignore
        if restrict_on_field_split is not None:
            assert self._fieldsplit is not None
            ps = self._fieldsplit[restrict_on_field_split][1] #type:ignore
            P = P.createSubMatrix(ps, ps) #type:ignore
        return P #type:ignore

    def define_preconditioner(self):
        pass

    def setup_solver(self):
        self.define_options()
        super().setup_solver()
        self._fieldsplit_map=None
        self.define_field_split()
        self._perform_field_split()
        self.define_preconditioner()

    def get_PC(self)->Any:
        pc = self.ksp.getPC() #type:ignore
        return pc #type:ignore





