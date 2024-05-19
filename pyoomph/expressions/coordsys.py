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
 

import _pyoomph
from .generic import *

from ..typings import *
from ..expressions import *

if TYPE_CHECKING:
    from ..generic.codegen import Equations,FiniteElementCodeGenerator


pi = 2 * _pyoomph.GiNaC_asin(1)


class BaseCoordinateSystem(_pyoomph.CustomCoordinateSystem):
    def __init__(self):
        super(BaseCoordinateSystem, self).__init__()
        self.x_rel_scale=1
        self.y_rel_scale=1
        self.z_rel_scale=1

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        raise RuntimeError("Implement the define_vector_field function for this coordinate system")

    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations",symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        raise RuntimeError("Implement the define_tensor_field function for this coordinate system")


    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the vector_gradient for this coordinate system. Occured upon taking the grad of " + str(arg))

    # Dimension of the vector gradient matrix for dimension dim
    def vector_gradient_dimension(self, basedim:int, lagrangian:bool=False)->int:
        return basedim  # Normally identity

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the vector_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))


    def geometric_jacobian(self)->Expression:
        raise RuntimeError("Implement the geometric_jacobian for this coordinate system")

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the scalar_gradient for this coordinate system. Occured upon taking the grad of " + str(arg))

    def surface_scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        return self.scalar_gradient(arg, ndim, edim, with_scales, lagrangian)

    def surface_vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        return self.vector_gradient(arg, ndim, edim, with_scales, lagrangian)

    def surface_vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        return self.vector_divergence(arg, ndim, edim, with_scales, lagrangian)

    def grad(self, arg:Expression, ndim:int, edim:int, flags:int) -> Expression:
        with_scales = (flags & 1 != 0)
        lagrangian = (flags & 8 != 0)
        is_vector=0
        if flags & 6 == 0:
            if _pyoomph.GiNaC_is_a_matrix(arg):
                is_vector = 2
            else:
                is_vector = 1
        if flags & 6 == 2 or is_vector == 1:
            if ndim != edim:
                return self.surface_scalar_gradient(arg, ndim, edim, with_scales, lagrangian)
            else:
                return self.scalar_gradient(arg, ndim, edim, with_scales, lagrangian)
        elif flags & 6 == 4 or is_vector == 2:
            if ndim == edim:
                return self.vector_gradient(arg.evalm(), ndim, edim, with_scales, lagrangian)
            elif ndim == edim + 1:
                return self.surface_vector_gradient(arg.evalm(), ndim, edim, with_scales, lagrangian)
            else:
                raise RuntimeError("Trying to perform a vector gradient on an element with dimension " + str(
                    edim) + " but with a nodal dimension of " + str(ndim))
        else:
            raise RuntimeError("Strange flags in grad: ", flags, flags & 6)



    def get_normal_vector_or_component(self,cg:"FiniteElementCodeGenerator",component:Optional[int]=None,only_base_mode:bool=False,only_perturbation_mode:bool=False,where:str="Residual"):
        dim = cg.get_nodal_dimension()
        if component is None:
            posscompos=[nondim("normal_"+d,domain=cg,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode) for d in ["x","y","z"]]
            mycomps=posscompos[:dim]        
            return vector(*mycomps)
        else:
            return cg._get_normal_component(component)            
        
        

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")


    def directional_derivative(self,arg:Expression,direct:Expression,ndim:int,edim:int,flags:int)->Expression:
        with_scales = (flags & 1 != 0)
        lagrangian = (flags & 8 != 0)
        vectorial=(flags & 4 !=0)
        tensorial=(flags &16 !=0)
        if tensorial:
            return self.directional_tensor_derivative(arg,direct,lagrangian,with_scales,ndim,edim,with_scales)
        elif vectorial:
            return matproduct(grad(arg,lagrangian=lagrangian,nondim=not with_scales),direct)
        else:
            return dot(grad(arg,lagrangian=lagrangian,nondim=not with_scales),direct)
    



    def div(self, arg:Expression, ndim:int, edim:int, flags:int)->Expression:
        with_scales = (flags % 2 == 1)
        lagrangian = (flags & 8 != 0)
        tensorial=(flags & 16 != 0)
        #		is_vector=(flags//2)	#TODO

        if tensorial:
            return self.tensor_divergence(arg.evalm(),ndim,edim,with_scales,lagrangian)
        elif edim == ndim:
            return self.vector_divergence(arg.evalm(), ndim, edim, with_scales, lagrangian)
        elif ndim == edim + 1:
            return self.surface_vector_divergence(arg.evalm(), ndim, edim, with_scales, lagrangian)
        else:
            raise RuntimeError("Cannot calculate this divergence yet: element dimension "+str(edim)+" and nodal dimension "+str(ndim))

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int)->ExpressionOrNum:
        return spatial_scale ** elem_dim

    def integral_dx(self, ndim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:        
        if lagrangian:
            if with_scale:
                return spatial_scale ** (ndim) * nondim("dX" )
            else:
                return nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** (ndim) * nondim("dx" )
            else:
                return nondim("dx")

    def get_coords(self, ndim:int, with_scales:bool, lagrangian:bool) -> List[Expression]:
        rel_scales = [self.x_rel_scale, self.y_rel_scale, self.z_rel_scale]
        if lagrangian:
            if with_scales:
                x, y, z = var(["lagrangian_x", "lagrangian_y", "lagrangian_z"])
                x,y,z=rel_scales[0]*x,rel_scales[1]*y,rel_scales[2]*z
            else:
                x, y, z = nondim(["lagrangian_x", "lagrangian_y", "lagrangian_z"])
        else:
            if with_scales:
                x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
                x,y,z=rel_scales[0]*x,rel_scales[1]*y,rel_scales[2]*z
            else:
                x, y, z = nondim(["coordinate_x", "coordinate_y", "coordinate_z"])
        all_coords = [x, y, z]
        if ndim == 1:
            return [x]
        else:
            return all_coords[:ndim]


    def get_id_name(self)->str:
        return "<unknown coord.sys>"

    def get_actual_dimension(self, reduced_dim:int)->int:
        return reduced_dim

    def jacobian_for_element_size(self)->Expression:
        return self.geometric_jacobian()


    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        dim=cg.get_nodal_dimension()
        if dimensional:
            vr=lambda n : var(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
        else:
            vr=lambda n : nondim(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
        if dim == 1:
            return vector([vr(name+"_x")])
        elif dim == 2:
            return vector([vr(name+"_x"), vr(name+"_y")])
        elif dim == 3:
            return vector([vr(name+"_x"), vr(name+"_y"), vr(name+"_z")])

#####################


class ODECoordinateSystem(BaseCoordinateSystem):
    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool) -> Expression:
        return Expression(0)

    def geometric_jacobian(self)->Expression:
        return Expression(1)

    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        raise RuntimeError("ODEs don't have coordinates")

# Cartesian
class CartesianCoordinateSystem(BaseCoordinateSystem):
    def __init__(self,x_rel_scale:ExpressionOrNum=1,y_rel_scale:ExpressionOrNum=1,z_rel_scale:ExpressionOrNum=1):
        super().__init__()
        self.x_rel_scale:ExpressionOrNum=x_rel_scale
        self.y_rel_scale:ExpressionOrNum=y_rel_scale
        self.z_rel_scale:ExpressionOrNum=z_rel_scale

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        inds = ["x", "y", "z"]
        namelist:List[str] = []
        for i in range(ndim):
            namelist.append(name + "_" + inds[i])
        v:List[Expression] = []
        vtest:List[Expression] = []
        s = scale_factor(name)
        S = test_scale_factor(name)
        for i, f in enumerate(namelist):
            if i >= ndim: break
            element.set_scaling(**{f: name})
            vc = element.define_scalar_field(f, space)
            vc = var(f)
            v.append(vc / s)
            element.set_test_scaling(**{f: name})
            vtest.append(testfunction(f) / S)
        return v, vtest, namelist


    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations", symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        inds = ["x", "y", "z"]
        namelist:List[List[str]] = []
        for i in range(ndim):
            nlst:List[str]=[]
            for j in range(ndim):
                if symmetric and j<i:
                    nlst.append(name + "_" + inds[j]+ inds[i])
                else:
                    nlst.append(name + "_" + inds[i]+ inds[j])
            namelist.append(nlst)
        v:List[List[Expression]] = []
        vtest:List[List[Expression]] = []
        s = scale_factor(name)
        S = test_scale_factor(name)
        for i, fl in enumerate(namelist):
            if i >= ndim: break
            vl:List[Expression]=[]
            vtl:List[Expression]=[]
            for j, f in enumerate(fl):
                if j >= ndim: break
                if symmetric and j<i:
                    vl.append(v[j][i])
                    vtl.append(vtest[j][i])
                else:
                    element.set_scaling(**{f: name})
                    vc = element.define_scalar_field(f, space)
                    vc = var(f)
                    vl.append(vc / s)
                    element.set_test_scaling(**{f: name})
                    vtl.append(testfunction(f) / S)
            v.append(vl)
            vtest.append(vtl)
        #print("NL",namelist)
        #exit()
        #if symmetric:
        #    for i, fl in enumerate(namelist):
        #        for j, f in enumerate(fl):
        #            if j<i:
        #                namelist[i].remove(f)
        return v, vtest, namelist


    def geometric_jacobian(self)->Expression:
        return rational_num(1, 1)

    def get_id_name(self)->str:
        return "Cartesian"

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        for a in self.get_coords(ndim, with_scales, lagrangian):
            res.append(diff(arg, a))
        return vector(res)

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:Expression = Expression(0)
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        for i in range(arg.nops()):
            res += diff(arg[i], coords[i])
        return res
    
    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[List[ExpressionOrNum]] = []
        for b in range(arg.nops()):
            line:List[ExpressionOrNum] = []
            entry = arg[b]
            for a in self.get_coords(ndim, with_scales, lagrangian):
                line.append(diff(entry, a))
            res.append(line)
        return matrix(res)
    
    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[Expression] = []
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        for i in range(3):
            div_line = 0 
            for j,coord in enumerate(coords):
                entry=arg[j,i]
                div_line+=diff(entry, coord)
            res.append(div_line)
        return vector(res)

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int,with_scales:bool)->Expression:
        res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
        coords=self.get_coords(ndim, with_scales, lagrangian)
        for i in range(3):
            for j in range(3):
                for k,coord in enumerate(coords):                       
                    res[i][j]+=diff(T[i,j],coord)*direct[k]
        return matrix(res)



# Axisymmetric

class AxisymmetricCoordinateSystem(BaseCoordinateSystem):
    def __init__(self,r_rel_scale:ExpressionOrNum=1,z_rel_scale:ExpressionOrNum=1,cartesian_error_estimation:bool=False,use_x_as_symmetry_axis:bool=False,):
        super().__init__()
        self.x_rel_scale=r_rel_scale
        self.y_rel_scale=z_rel_scale
        self.cartesian_error_estimation=cartesian_error_estimation
        self.use_x_as_symmetry_axis=use_x_as_symmetry_axis

    def get_actual_dimension(self, reduced_dim:int)->int:
        return reduced_dim + 1

    def get_id_name(self)->str:
        return "Axisymmetric"

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int)->ExpressionOrNum:
        return spatial_scale ** (elem_dim + 1)

    def integral_dx(self, ndim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:
        if ndim >= 3:
            raise RuntimeError("Axisymmetry does not work for dimension " + str(ndim))
        ndim_offs=ndim+1
        if lagrangian:
            if with_scale:
                return spatial_scale ** ndim_offs * 2 * pi * nondim("lagrangian_y" if self.use_x_as_symmetry_axis else "lagrangian_x") * nondim("dX")
            else:
                return 2 * pi * nondim("lagrangian_y" if self.use_x_as_symmetry_axis else "lagrangian_x") * nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** ndim_offs * 2 * pi * nondim("coordinate_y" if self.use_x_as_symmetry_axis else "coordinate_x") * nondim("dx")
            else:
                return 2 * pi * nondim("coordinate_y" if self.use_x_as_symmetry_axis else "coordinate_x") * nondim("dx")

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        for i, a in enumerate(self.get_coords(3, with_scales, lagrangian)):
            res.append(diff(arg, a) if i < ndim else 0)
        return vector(res)

    def vector_gradient_dimension(self, basedim:int, lagrangian:bool=False)->int:
        # Just alwas 3
        return 3

    def geometric_jacobian(self)->Expression:
        if self.cartesian_error_estimation:
            return Expression(1)
        else:
            return 2 * pi * nondim("coordinate_y" if self.use_x_as_symmetry_axis else "coordinate_x")

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim >= 3:
            raise RuntimeError("Vector gradient in axisymmetry does not work for dimension " + str(ndim))
        if ndim == 1:
            if self.use_x_as_symmetry_axis:
                raise RuntimeError("Cannot have use_x_as_symmetry_axis in an axisymmetric coordinate system in 1d")
            r, = self.get_coords(ndim, with_scales, lagrangian)
            res:List[List[ExpressionOrNum]] = [[diff(arg[0], r), 0, 0], [0, 0, 0], [0, 0, arg[0] / r]]
        else:
            if arg.nops() != 3:
                raise RuntimeError(
                    "Cannot take a 2d axisymmetric vector gradient from a vector with dim!=2:  " + str(arg))
            x, y = self.get_coords(ndim, with_scales, lagrangian)
            r= y if self.use_x_as_symmetry_axis else x
            ri = 1 if self.use_x_as_symmetry_axis else 0
            res:List[List[ExpressionOrNum]] = [[diff(arg[0], x), diff(arg[0], y), 0],
                   [diff(arg[1], x), diff(arg[1], y), 0], [0, 0, arg[ri] / r]]
        return matrix(res)
    

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        zero=Expression(0)
        s = scale_factor(name)
        S = test_scale_factor(name)          
        if ndim == 2:
            vx = element.define_scalar_field(name + "_x", space)
            vy = element.define_scalar_field(name + "_y", space)
            vx = var(name + "_x")
            vy = var(name + "_y")
            element.set_scaling(**{name + "_x": name, name + "_y": name})
            element.set_test_scaling(**{name + "_x": name, name + "_y": name})          
            return [vx / s, vy / s, zero], [testfunction(name + "_x") / S,testfunction(name + "_y") / S, zero], [name + "_x",name + "_y"]
        elif ndim == 1:
            vx = element.define_scalar_field(name + "_x", space)
            vx = var(name + "_x")
            element.set_scaling(**{name + "_x": name})
            element.set_test_scaling(**{name + "_x": name})
            return [vx / s, zero], [testfunction(name + "_x") / S], [name + "_x"]
        else:
            raise RuntimeError("Axisymmetric vector fields do not work for dimension " + str(ndim))

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res = Expression(0)
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        nops = arg.nops()
        if nops >= 1:
            res += diff(arg[0], coords[0]) + (arg[1] / coords[1] if self.use_x_as_symmetry_axis  else arg[0] / coords[0])
        if nops >= 2:
            res += diff(arg[1], coords[1])
        return res
    
    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations", symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        s = scale_factor(name)
        S = test_scale_factor(name)
        if ndim==1:
            txx = element.define_scalar_field(name + "_xx", space)
            txx = var(name + "_xx")
            taa = element.define_scalar_field(name + "_aa", space)
            taa = var(name + "_aa")
            element.set_scaling(**{name + "_xx": name,name+"_aa":name})
            element.set_test_scaling(**{name + "_xx": name, name + "_aa": name})
            return [[txx / s, 0, 0], [0, 0, 0], [0, 0, taa/s]], [[testfunction(name + "_xx") / S, 0, 0], [0, 0, 0], [0, 0, testfunction(name + "_aa") / S]], [[name + "_xx", 0,0], [0,0, 0], [0, 0, name + "_aa"]] 
        elif ndim==2:
            txx = element.define_scalar_field(name + "_xx", space)
            txx = var(name + "_xx")
            tyy = element.define_scalar_field(name + "_yy", space)
            tyy = var(name + "_yy")
            txy = element.define_scalar_field(name + "_xy", space)
            txy = var(name + "_xy")
            element.set_scaling(**{name + "_xx": name, name + "_yy": name, name + "_xy": name})
            element.set_test_scaling(**{name + "_xx": name, name + "_yy": name, name + "_aa": name, name + "_xy": name})
            taa = element.define_scalar_field(name + "_aa", space)
            taa = var(name + "_aa")
            element.set_scaling(**{name + "_aa": name})
            element.set_test_scaling(**{name + "_aa": name})
        if not symmetric:
            tyx = element.define_scalar_field(name + "_yx", space)
            tyx = var(name + "_yx")
            element.set_scaling(**{name + "_yx": name})
            element.set_test_scaling(**{name + "_yx": name})
            return [[txx / s, txy / s, 0], [tyx / s, tyy / s, 0], [0, 0, taa / s]], [[testfunction(name + "_xx") / S, testfunction(name + "_xy") / S, 0], [testfunction(name + "_yx") / S, testfunction(name + "_yy") / S, 0], [0, 0, testfunction(name + "_aa") / S]], [[name + "_xx", name + "_xy", 0], [name + "_yx", name + "_yy", 0], [0, 0, name + "_aa"]] 
        else:
            return [[txx / s, txy / s, 0], [txy / s, tyy / s, 0], [0, 0, taa / s]], [[testfunction(name + "_xx") / S, testfunction(name + "_xy") / S, 0], [testfunction(name + "_xy") / S, testfunction(name + "_yy") / S, 0], [0, 0, testfunction(name + "_aa") / S]], [[name + "_xx", name + "_xy", 0], [name + "_xy", name + "_yy", 0], [0, 0, name + "_aa"]] 


    def tensor_divergence(self, T:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        coords = self.get_coords(T.nops(), with_scales, lagrangian)
        if self.use_x_as_symmetry_axis:
            if ndim==1:
                raise RuntimeError("Cannot have use_x_as_symmetry_axis in an axisymmetric coordinate system in 1d")
            else:
                div_x=diff(T[0,0],coords[0])+diff(T[1,0],coords[1])+T[1,0] / coords[1] 
                div_y=diff(T[0,1],coords[0])+diff(T[1,1],coords[1])+(T[1,1]-T[2,2]) / coords[1]
                div_theta=diff(T[0,2],coords[0])+diff(T[1,2],coords[1])+(T[1,2] - T[2,1]) / coords[1]
            return vector(div_x,div_y,div_theta)
        else:
            div_x=diff(T[0,0],coords[0])+(T[0,0] - T[2,2]) / coords[0]
            div_y=diff(T[0,1],coords[0])+T[0,1] / coords[0]
            div_theta=diff(T[0,2],coords[0])+(T[0,2] - T[2,0]) / coords[0]
            if ndim==2:
                div_x+=diff(T[1,0],coords[1])
                div_y+= diff(T[1,1],coords[1])
                div_theta+=diff(T[1,2],coords[1])
            return vector(div_x, div_y, div_theta)

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int,with_scales:bool,)->Expression:        
        if ndim==1:
            if self.use_x_as_symmetry_axis:
                raise RuntimeError("Cannot have use_x_as_symmetry_axis in an axisymmetric coordinate system in 1d")
            res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
            coords = self.get_coords(1, with_scales, lagrangian)
            res[0][0]=diff(T[0,0],coords[0])*direct[0]
            res[0][2]=1/coords[0]*(T[0,0]-T[2,2])*direct[2]
            res[2][0]=res[0][2]
            res[2][2]=diff(T[2,2],coords[0])*direct[0]

            return matrix(res)
        elif ndim==2:
            if self.use_x_as_symmetry_axis:
                #raise RuntimeError("TODO")
                res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
                coords = self.get_coords(T.nops(), with_scales, lagrangian)
                #diff_tensor_theta=[[-T[2,0]-T[0,2], T[2,1], T[0,0]-T[2,2]], [-T[1,2], 0, T[1,0]], [T[0,0]-T[2,2], T[0,1], T[0,2]+T[2,0]]]
                diff_tensor_theta=[[0, -T[0,2], T[1,0]], [-T[2,0], -T[2,1]-T[1,2], T[1,1]-T[2,2]], [T[1,0], T[1,1]-T[2,2], T[1,2]+T[2,1]]]
                for i in range(3):
                    for j in range(3):
                        res[i][j]=direct[0]*diff(T[i,j],coords[0])+direct[1]*diff(T[i,j],coords[1])+direct[2]/coords[1]*diff_tensor_theta[i][j]
                return matrix(res)
            else:
                res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
                coords = self.get_coords(T.nops(), with_scales, lagrangian)
                diff_tensor_theta=[[-T[2,0]-T[0,2], T[2,1], T[0,0]-T[2,2]], [-T[1,2], 0, T[1,0]], [T[0,0]-T[2,2], T[0,1], T[0,2]+T[2,0]]]
                for i in range(3):
                    for j in range(3):
                        res[i][j]=direct[0]*diff(T[i,j],coords[0])+direct[1]*diff(T[i,j],coords[1])+direct[2]/coords[0]*diff_tensor_theta[i][j]
                return matrix(res)
        else:
            raise RuntimeError("Not possible")




class AxisymmetryBreakingCoordinateSystem(AxisymmetricCoordinateSystem):
    def __init__(self,angular_mode:Expression,map_real_part_of_mode0:bool=False):        
        self.imaginary_i=_pyoomph.GiNaC_imaginary_i()
        super(AxisymmetryBreakingCoordinateSystem, self).__init__()
        self.angular_mode = angular_mode
        # Expansion smallness parameters epsilon: U=U_base+epsilon*U_angular
        self.expansion_eps=_pyoomph.GiNaC_new_symbol("eps")
        # m_angular and angular variable phi.
        # We first take a generic symbol for m and replace it later with the parameter or 0 (base state)
        self.m_angular_symbol = _pyoomph.GiNaC_new_symbol("m_angular")
        self.phi = _pyoomph.GiNaC_new_symbol("phi")
        # Field mode exp(I*m*phi)
        self.field_mode=_pyoomph.GiNaC_FakeExponentialMode(self.imaginary_i*self.m_angular_symbol*self.phi)
        # Mode of the test functions
        self.test_mode=_pyoomph.GiNaC_FakeExponentialMode(-self.imaginary_i*self.m_angular_symbol*self.phi)
        self.map_real_part_of_mode0=map_real_part_of_mode0 #Normally not necessary
        self.with_phi_component_in_mesh_coordinates=False # TODO: Implement that some day
        self.expand_with_modes_for_python_debugging=False # If you want to use expand_expression_for_debugging, set this

        

    def get_mode_expansion_of_var_or_test(self,code:_pyoomph.FiniteElementCode,fieldname:str,is_field:bool,is_dim:bool,expr:Expression,where:str,expansion_mode:int)->Expression:
        #print("MODE EXPANSION",fieldname,is_field)
        #print("EEXXX",where)
        if where!="Residual" and (not self.expand_with_modes_for_python_debugging or where!="Python"):
            return expr # Don't do this for integral expressions, fluxes, initial and dirichlets
        
        base_flag=1
        pert_flag=1
        if expansion_mode!=0:
            if expansion_mode==-1:
                pert_flag=0
                expr=_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(0))
            elif expansion_mode==-2:
                base_flag=0
                expr=_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(0))
            
        
        ignore_fields={"time","lagrangian_x","lagrangian_y","lagrangian_z"}
        if not code._coordinates_as_dofs:
            ignore_fields=ignore_fields.union({"coordinate_x","coordinate_y","coordinate_z","mesh_x","mesh_y","mesh_z"})
        if fieldname in ignore_fields:
            return expr # Do not modify these
        
        # Each field and test function are expanded with a mode perturbation
        if is_field:
            # Expand fields as Ubase+epsilon*Upert*exp(I*m*phi)
            return base_flag*expr+pert_flag*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(1))*self.field_mode
        else:
            # Test functions are not required to be expanded. They only enter linearly
            return expr*self.test_mode

    def map_residual_on_base_mode(self,residual:Expression)->Expression:
        zero=_pyoomph.Expression(0)
        # The base equations are obtained by setting epsilon and m to zero
        res=_pyoomph.GiNaC_SymSubs(_pyoomph.GiNaC_SymSubs(residual,self.m_angular_symbol,zero),self.expansion_eps,zero)
        if self.map_real_part_of_mode0:
            res=_pyoomph.GiNaC_get_real_part(res)
        return res

    def _map_residal_on_angular_eigenproblem(self,residual:Expression,re_im_mapping:Callable[[Expression],Expression])->Expression:
        zero = _pyoomph.Expression(0)
        # First order in epsilon is just derive by epsilon and set epsilon to zero afterwards
        first_order_in_eps = _pyoomph.GiNaC_SymSubs(diff(_pyoomph.GiNaC_expand(residual), self.expansion_eps), self.expansion_eps, zero)
        replaced_m = _pyoomph.GiNaC_SymSubs(first_order_in_eps, self.m_angular_symbol, self.angular_mode)
        # Map on the real/imag part part
        real_or_imag = re_im_mapping(replaced_m)
        # Remove any contributions of the base mode from the Jacobian and mass matrix
        # The Jacobian arises by deriving with respect to all degrees of freedom.
        # Here, we must ensure that we only derive with respect to the perturbed mode, not to the base mode (mode zero)
        # For the Hessian, we just want the zero mode terms, i.e. getting the second derivatives with respect to the base mode with the correct I*m-terms
        rem_jacobian_flag = _pyoomph.Expression(1)
        azimode = _pyoomph.Expression(1)
        rem_hessian_flag = _pyoomph.Expression(2)
        no_jacobian_entries_from_base_mode = _pyoomph.GiNaC_remove_mode_from_jacobian_or_hessian(real_or_imag, zero,rem_jacobian_flag)
        #no_hessian_entries_from_azimuthal_mode = _pyoomph.GiNaC_remove_mode_from_jacobian_or_hessian(no_jacobian_entries_from_base_mode, azimode,rem_hessian_flag)
        no_hessian_entries_from_azimuthal_mode=no_jacobian_entries_from_base_mode
        return _pyoomph.GiNaC_collect_common_factors(no_hessian_entries_from_azimuthal_mode)

    def map_residual_on_angular_eigenproblem_real(self,residual:Expression)->Expression:
        real_part=_pyoomph.GiNaC_get_real_part
        return self._map_residal_on_angular_eigenproblem(residual,real_part)

    def map_residual_on_angular_eigenproblem_imag(self,residual:Expression)->Expression:
        imag_part=_pyoomph.GiNaC_get_imag_part
        return self._map_residal_on_angular_eigenproblem(residual,imag_part)

    def get_id_name(self)->str:
        raise RuntimeError("Is this required?")
        return "Axisymmetric"

    def map_to_zero_epsilon(self,input):
        if isinstance(input,list):
            return [self.map_to_zero_epsilon(i) for i in input]
        else:
            return  _pyoomph.GiNaC_SymSubs(input,self.expansion_eps,Expression(0))
        
    def map_to_first_order_epsilon(self,input,with_epsilon:bool=True):
        if isinstance(input,list):
            return [self.map_to_first_order_epsilon(i,with_epsilon) for i in input]
        else:
            return _pyoomph.GiNaC_SymSubs(diff(_pyoomph.GiNaC_expand(input), self.expansion_eps), self.expansion_eps, Expression(0))*(self.expansion_eps if with_epsilon else 1)            



    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        coords = self.get_coords(3, with_scales, lagrangian)
        dcoords=self.map_to_zero_epsilon(coords)
        pcoords=self.map_to_first_order_epsilon(coords)
        for i, a in enumerate(dcoords):
            if i < ndim:
                res.append(diff(arg, a))
            elif i == ndim:
                res.append( diff(arg,self.phi) / dcoords[0] - (0 if lagrangian else 1)*_pyoomph.GiNaC_EvalFlag("moving_mesh")*(diff(arg,self.phi)-arg)/dcoords[0]**2 * pcoords[0])
            else:
                res.append(0)
        return vector(res)

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool) -> Expression:
        # dvr/dr        dvr/(r*dt)-vt/r
        # dvt/dr        dvt/(r*dt)+vr/r

        # Or here:
        # df(v[0],r)        d_by_dt*v[0]/r-v[1]/r
        # df(v[1],r)        d_by_dt*v[1]/r+v[0]/r

        # dvr/dr        dvr/(r*dt)-vt/r      dvr/dz
        # dvt/dr        dvt/(r*dt)+vr/r      dvt/dz
        # dvz/dr        dvz/(r*dt)           dvz/dz
        # Or
        # df(v[0],r)        df(v[0],z)               d_by_dt*v[0]/r-v[2]/r
        # df(v[1],r)        df(v[1],z)               d_by_dt*v[1]/r
        # df(v[2],r)        df(v[2],z)               d_by_dt*v[2]/r+v[0]/r


        df = diff
        zero=Expression(0)
        if ndim >= 3:
            raise RuntimeError("Vector gradient in axisymmetry does not work for dimension " + str(ndim))
        if ndim == 1:
            r, = self.get_coords(ndim, with_scales, lagrangian)
            dr=self.map_to_zero_epsilon(r)
            pr=self.map_to_first_order_epsilon(r)
            #res = [[df(v[0], r), d_by_dt * v[0] / r - v[1] / r, 0], [df(v[1], r), d_by_dt * v[1] / r + v[0] / r, 0],[0, 0, 0]]
            res:List[List[ExpressionOrNum]] = [[df(arg[0], dr), df(arg[0],self.phi) / dr - arg[1] / dr,zero],
                   [df(arg[1], dr), df(arg[1],self.phi) / dr + arg[0] / dr,zero],[zero,zero,zero]]
#            res = [[df(v[0], r), df(v[1], r), 0], [d_by_dt * v[0] / r - v[1] / r, d_by_dt * v[1] / r + v[0] / r, 0],[0, 0, 0]]
            if not lagrangian:
                res[0][1]+= -_pyoomph.GiNaC_EvalFlag("moving_mesh")*(df(arg[0],self.phi)-arg[1])/dr**2 * pr
                res[1][1]+= -_pyoomph.GiNaC_EvalFlag("moving_mesh")*(df(arg[1],self.phi)+arg[0])/dr**2 * pr            
        else:
#            raise RuntimeError("TODO")
            if arg.nops() != 3:
                raise RuntimeError(
                    "Cannot take a 2d axisymmetric vector gradient from a vector with dim!=2:  " + str(arg))
            r, z = self.get_coords(ndim, with_scales, lagrangian)
            dr=self.map_to_zero_epsilon(r)
            dz=self.map_to_zero_epsilon(z)
            pr=self.map_to_first_order_epsilon(r)
            #pr=r-dr            
            res:List[List[ExpressionOrNum]]=[[ df(arg[0],dr),df(arg[0],dz),df(arg[0],self.phi)/dr-arg[2]/dr],
                 [df(arg[1],dr),df(arg[1],dz),df(arg[1],self.phi)/dr],
                 [df(arg[2],dr) ,df(arg[2],dz),df(arg[2],self.phi)/dr+arg[0]/dr]]
            if not lagrangian:
                res[0][2]+= -_pyoomph.GiNaC_EvalFlag("moving_mesh")*(df(arg[0],self.phi)-arg[2])/dr**2 * pr
                res[1][2]+= -_pyoomph.GiNaC_EvalFlag("moving_mesh")*(df(arg[1],self.phi))/dr**2 * pr
                res[2][2]+= -_pyoomph.GiNaC_EvalFlag("moving_mesh")*(df(arg[2],self.phi)+arg[0])/dr**2 * pr
        return matrix(res)



    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res = 0
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        dcoords=self.map_to_zero_epsilon(coords)
        #pcoords=[cm-c0 for cm,c0 in zip(coords,dcoords) ] # Perturbed coordinates
        pcoords=self.map_to_first_order_epsilon(coords)
#        nops = arg.nops()
        if ndim== 1:
            res = diff(arg[0], dcoords[0]) + arg[0] / dcoords[0] + diff(arg[1],self.phi)/dcoords[0]
        elif ndim == 2:
            res = diff(arg[0], dcoords[0]) + arg[0] / dcoords[0] + diff(arg[1], dcoords[1])+diff(arg[2],self.phi) / dcoords[0]
        else:
            raise RuntimeError("Cannot use this coordinate system on a 3d mesh")
        if not lagrangian:
            #print(coords)
            #if cg._coordinates_as_dofs and (where=="Residual" or (not self.expand_with_modes_for_python_debugging or where!="Python")):
            # Derive the u_r/r term by the denominator 
            # Only on a moving mesh:
            res+= -_pyoomph.GiNaC_EvalFlag("moving_mesh")*(arg[0]+ diff(arg[ndim],self.phi))/dcoords[0]**2 * pcoords[0]
            #if ndim==1:
                
                #exit()
            #else:
            #    raise RuntimeError("TODO")
        return res


    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations") -> Tuple[List[Expression], List[Expression], List[str]]:
        s = scale_factor(name)
        S = test_scale_factor(name)        
        if ndim == 2:
            vx = element.define_scalar_field(name + "_x", space)
            vy = element.define_scalar_field(name + "_y", space)
            vt = element.define_scalar_field(name + "_phi", space)
            vx = var(name + "_x")
            vy = var(name + "_y")
            vt = var(name + "_phi")
            element.set_scaling(**{name + "_x": name, name + "_y": name, name + "_phi": name})
            element.set_test_scaling(**{name + "_x": name, name + "_y": name,name + "_phi": name})            
            return [vx / s, vy / s, vt/s], [testfunction(name + "_x") / S,testfunction(name + "_y") / S, testfunction(name + "_phi") / S], [name + "_x",name + "_y",name+"_phi"]
        elif ndim == 1:
            vx = element.define_scalar_field(name + "_x", space)
            vt = element.define_scalar_field(name + "_phi", space)
            vx = var(name + "_x")
            vt = var(name + "_phi")
            element.set_scaling(**{name + "_x": name,name+"_phi":name})
            element.set_test_scaling(**{name + "_x": name,name+"_phi":name})
            return [vx / s, vt/s], [testfunction(name + "_x") / S,testfunction(name + "_phi") / S], [name + "_x",name + "_phi"]
        else:
            raise RuntimeError("Axisymmetric vector fields do not work for dimension " + str(ndim))


    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        if not cg._coordinates_as_dofs:
            return super().expand_coordinate_or_mesh_vector(cg,name,dimensional,no_jacobian,no_hessian)
        else:
            dim=cg.get_nodal_dimension()
            if dimensional:
                vr=lambda n : var(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
            else:
                vr=lambda n : nondim(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
            if self.with_phi_component_in_mesh_coordinates:
                if dim == 1:
                    return vector([vr(name+"_x"),vr(name+"_phi")])
                elif dim == 2:
                    return vector([vr(name+"_x"), vr(name+"_y"),vr(name+"_phi")])
                elif dim == 3:
                    raise RuntimeError("Cannot use symmetry-breaking coordinate system in 3d")
            else:
                if dim == 1:
                    return vector([vr(name+"_x"),0])
                elif dim == 2:
                    return vector([vr(name+"_x"), vr(name+"_y"),0])
                elif dim == 3:
                    raise RuntimeError("Cannot use symmetry-breaking coordinate system in 3d")



    def get_normal_vector_or_component(self,cg:"FiniteElementCodeGenerator",component:Optional[int]=None,only_base_mode:bool=False,only_perturbation_mode:bool=False,where:str="Residual"):
        dim = cg.get_nodal_dimension()
        if not cg._coordinates_as_dofs or (where!="Residual" and (not self.expand_with_modes_for_python_debugging or where!="Python")):
            return super().get_normal_vector_or_component(cg,component,only_base_mode,only_perturbation_mode,where=where)
        
        if dim not in [1,2]:
            RuntimeError("Cannot use this coordinate system with a nodal dimension of "+str(dim))
        if component is None:
            comp=lambda s : nondim(s,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode)
            if dim==1:
                return vector(comp("normal_x"),comp("normal_y"),0)
            else:
                return vector(comp("normal_x"),comp("normal_y"),comp("normal_z")) #TODO: Normal z here?
        else:
            # Normal expansion            
            base_factor=0 if only_perturbation_mode else 1
            pert_factor=0 if only_base_mode else 1
            
            if component<dim: # Radial normal
                return base_factor*cg._get_normal_component(component) +pert_factor*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(cg._get_normal_component_eigenexpansion(component),Expression(1))*self.field_mode
            elif component==dim:                
                nr0=cg._get_normal_component(0)
                #rm=self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(var("mesh_x"),_pyoomph.Expression(1))*self.field_mode
                rm=var("mesh_x",only_perturbation_mode=True)
                n0_dot_xeigen=nr0*rm
                if dim==2:
                    nz0=cg._get_normal_component(1)
                    zm=var("mesh_y",only_perturbation_mode=True)
                    #zm=self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(var("mesh_y"),_pyoomph.Expression(1))*self.field_mode
                    n0_dot_xeigen+=nz0*zm
                if self.with_phi_component_in_mesh_coordinates:
                    phim=var("mesh_phi",only_perturbation_mode=True)
                else:
                    phim=0
                #return pert_factor*self.expansion_eps*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen))*self.field_mode
                #print(pert_factor*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen)))                
                #exit()
                return pert_factor*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen))
                
            else:
                raise RuntimeError("Normal component "+str(component)+" not available")
        print(dim)
        exit()
        if component is None:
            posscompos=[nondim("normal_"+d,domain=cg) for d in ["x","y","z"]]
            mycomps=posscompos[:dim]        
            return vector(*mycomps)
        else:
            return cg._get_normal_component(component)            
            

    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")



# Radial symmetric (1d only)

class RadialSymmetricCoordinateSystem(BaseCoordinateSystem):

    def __init__(self, Rcenter:ExpressionOrNum=0):
        super(RadialSymmetricCoordinateSystem, self).__init__()
        self.Rcenter = Rcenter

    def get_actual_dimension(self, reduced_dim:int)->int:
        return 3

    def get_id_name(self)->str:
        return "RadialSymmetric"

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations") -> Tuple[List[Expression], List[Expression], List[str]]:
        if ndim == 1:
            vx = element.define_scalar_field(name + "_x", space)
            vx = var(name + "_x")
            element.set_scaling(**{name + "_x": name})
            element.set_test_scaling(**{name + "_x": name})
            s = scale_factor(name)
            S = test_scale_factor(name)
            return [vx / s], [testfunction(name + "_x") / S], [name + "_x"]
        else:
            raise RuntimeError("Radialsymmetric vector fields do not work for dimension " + str(ndim))

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int) -> ExpressionOrNum:
        return spatial_scale ** (elem_dim + 2)

    def integral_dx(self, ndim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:
        if ndim >= 3:
            raise RuntimeError("Does not work for dimension " + str(ndim))        
        if lagrangian:
            if with_scale:
                return spatial_scale ** (ndim + 2) * 4 * pi * (nondim(
                    "lagrangian_x") - self.Rcenter) ** 2 * nondim("dX")
            else:
                return 4 * pi * (nondim(
                    "lagrangian_x") - self.Rcenter / scale_factor(
                    "spatial")) ** 2 * nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** (ndim + 2) * 4 * pi * (nondim(
                    "coordinate_x") - self.Rcenter / scale_factor(
                    "spatial")) ** 2 * nondim("dx")
            else:
                return 4 * pi * (nondim(
                    "coordinate_x") - self.Rcenter / scale_factor(
                    "spatial")) ** 2 * nondim("dx")

    def geometric_jacobian(self) -> Expression:
        return 4 * pi * (nondim("coordinate_x") - self.Rcenter / scale_factor(
            "spatial")) ** 2

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim != 1:
            raise RuntimeError("Vector gradient in radial symmetry does not work for dimension " + str(ndim))
        else:
            r, = self.get_coords(ndim, with_scales, lagrangian)
            if with_scales:
                r = r - self.Rcenter
            else:
                r = r - self.Rcenter / scale_factor("spatial")
            res:List[List[ExpressionOrNum]] = [[diff(arg[0], r), 0, 0], [0, arg[0] / r, 0], [0, 0, arg[0] / r]]
        return matrix(res)

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim != 1:
            raise RuntimeError("Vector divergence in radial symmetry does not work for dimension " + str(ndim))
        res = 0
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        if with_scales:
            coords[0] -= self.Rcenter
        else:
            coords[0] -= self.Rcenter / scale_factor("spatial")
        res += diff(arg[0], coords[0]) + 2 * arg[0] / coords[0]
        return res

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        for i, a in enumerate(self.get_coords(3, with_scales, lagrangian)):
            res.append(diff(arg, a) if i < ndim else 0)
        return vector(res)


    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")
