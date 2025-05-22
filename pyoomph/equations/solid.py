from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import BaseMovingMeshEquations

class BaseSolidConstitutiveLaw:
    def __init__(self,use_subexpressions=True):
        self.use_subexpressions=use_subexpressions
        
    def _subexpression(self,expr):
        if self.use_subexpressions:
            return subexpression(expr)
        else:
            return expr
        
    def is_incompressible(self):
        return False
    
    def get_Gij(self):        
        # Deformed position
        x=var("mesh")
        return self._subexpression(matproduct(transpose(grad(x,lagrangian=True)),grad(x,lagrangian=True)))
    
    def get_G_up_ij(self):
        # G^ij=(G^(-1))_ij
        Gij=self.get_Gij()        
        return self._subexpression(inverse_matrix(Gij,fill_to_vector_dim_3=True))
        
    
    def get_gij(self):
        # Undeformed position
        X=var("lagrangian")
        return self._subexpression(matproduct(transpose(grad(X,lagrangian=True)),grad(X,lagrangian=True)))
        #return identity_matrix() # Only true for Cartesian coordinates
    
    def get_gammaij(self):
        # Deformed position
        return self._subexpression((self.get_Gij()-self.get_gij())/2)
    
    def get_sigma(self,dim:Optional[int]=None,pressure_var:Optional[Expression]=None):
        raise RuntimeError("get_sigma must be implemented in the derived class")
    
class IncompressibleSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    def is_incompressible(self):
        return True
    
class GeneralizedHookeanSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    def __init__(self, E:ExpressionOrNum=1, nu:ExpressionOrNum=0.4,use_subexpressions=True):
        super().__init__()
        self.E=E
        self.nu=nu
        self.use_subexpressions=use_subexpressions
        

    def get_sigma(self,dim:Optional[int]=None,pressure_var:Optional[Expression]=None):
        se=lambda x : self._subexpression(x)
        Gup=self.get_G_up_ij()
        gammakl=self.get_gammaij()
        if dim is None:
            raise RuntimeError("dim must be specified")
        
        E=lambda i,j,k,l : se(self.E/(1+self.nu))*(se(self.nu/(1-2*self.nu))*Gup[i,j]*Gup[k,l]+(Gup[i,k]*Gup[j,l]+Gup[i,l]*Gup[j,k])/2)
        sigma_up_ij=lambda i,j : sum(E(i,j,k,l)*gammakl[k,l] for k in range(dim) for l in range(dim))
        sigma=matrix([[sigma_up_ij(i,j) for j in range(dim)] for i in range(dim)])
        return se(sigma)
        
class IncompressibleHookeanSolidConstitutiveLaw(IncompressibleSolidConstitutiveLaw):
    def __init__(self,E:ExpressionOrNum=1,use_subexpressions=True):
        super().__init__(use_subexpressions=use_subexpressions)
        self.E=E

    def get_sigma(self,dim:Optional[int]=None,pressure_var:Optional[Expression]=None):
        se=lambda x : self._subexpression(x)
        Gup=self.get_G_up_ij()
        gammakl=self.get_gammaij()
        if dim is None:
            raise RuntimeError("dim must be specified")
        bar_sigma_up_ij=lambda i,j : se(sum((Gup[i,k]*Gup[j,l]+Gup[i,l]*Gup[j,k])*gammakl[k,l] for k in range(dim) for l in range(dim)))
        if pressure_var is None:
            raise RuntimeError("pressure_var must be specified")
        return -pressure_var*Gup+self.E/3*matrix([[bar_sigma_up_ij(i,j) for j in range(dim)] for i in range(dim)])
    

class DeformableSolidEquations(BaseMovingMeshEquations):
    # TODO: Bulk force in Lagrangian or Eulerian coordinates? Make two
    def __init__(self, constitutive_law:BaseSolidConstitutiveLaw, mass_density:ExpressionOrNum=1,bulkforce:ExpressionOrNum=0,coordinate_space = None,first_order_time_derivative=False):
        super().__init__(coordinate_space, False, None)
        self.constitutive_law=constitutive_law
        self.mass_density=mass_density
        self.bulkforce=bulkforce
        self.pressure_space="C1"
        self.pressure_name="pressure"
        self.first_order_time_derivative=first_order_time_derivative
        
    def define_fields(self):
        super().define_fields()
        if self.constitutive_law.is_incompressible():
            self.define_scalar_field(self.pressure_name,self.pressure_space)
        if self.first_order_time_derivative:
            if self.coordinate_space is None:
                raise RuntimeError("coordinate_space must be specified for first order time derivative")
            self.define_vector_field("dt_mesh",self.coordinate_space,scale=scale_factor("spatial")/scale_factor("temporal"),testscale=scale_factor("temporal")/scale_factor("spatial"))
        
    def define_residuals(self):
        x,xtest=var_and_test("mesh")
        X=var("lagrangian")
        if self.first_order_time_derivative:
            self.add_weak(var("dt_mesh")-mesh_velocity(),testfunction("dt_mesh"),lagrangian=False)            
            accel=partial_t(var("dt_mesh"),ALE=False)            
        else:
            accel=partial_t(x,2,ALE=False)
        #Gij=self.constitutive_law.get_Gij()
        dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
        pvar=var(self.pressure_name) if self.constitutive_law.is_incompressible() else None
        
        sigma=self.constitutive_law.get_sigma(dim,pvar)            
        self.add_weak( self.mass_density*accel-self.bulkforce,xtest,lagrangian=True ).add_weak(matproduct((sigma),grad(x,lagrangian=True)),grad(xtest,lagrangian=True),lagrangian=True)
        if self.constitutive_law.is_incompressible():
            detG=determinant(self.constitutive_law.get_Gij())
            detg=determinant(self.constitutive_law.get_gij())
            self.add_weak(detG - detg,testfunction(self.pressure_name),lagrangian=True)

