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
 
 
from ..generic import Equations
from ..expressions import *  # Import grad et al


class LinearElasticity(Equations):
   """
      Represents the linear elasticity equation, which is a second order partial differential equation given by:
         lambda = 2 * E / 2 / (1 + nu) * (E * nu / (1 + nu) / (1 - 2 * nu)) / (nu / (1 + nu) / (1 - 2 * nu) + 2 * E / 2 / (1 + nu))
         sigma = lambda * tr(sym(grad(x - X))) * I + 2 * mu * sym(grad((x - X)))
         div(sigma) + f = 0
   
      where x is the unknown Eulerian coordinate, X is the is the Lagrangian coordinate, E is the Young's modulus, nu is the Poisson's ratio, lambda is the Lamé parameter, sigma is the stress tensor, tr is the trace operator, sym(grad()) is the symmetric gradient operator, div is the divergence operator, f is the bulk force, and I is the identity matrix.

      Args:
         E (ExpressionOrNum): Young's modulus. Default is 1.
         nu (ExpressionOrNum): Poisson's ratio. Default is 0.3.
         bulk_force (ExpressionNumOrNone): Bulk force. Default is None.
         spatial_error_factor (Optional[float]): Spatial error factor. Default is None.
   """
      
   def __init__(self, E:ExpressionOrNum=1, nu:ExpressionOrNum=0.3, bulk_force:ExpressionNumOrNone=None,spatial_error_factor:Optional[float]=None):
      super(LinearElasticity, self).__init__()
      self.E = E
      self.nu = nu
      self.bulk_force = bulk_force
      self.spatial_error_factor=spatial_error_factor

   def define_fields(self):
      self.activate_coordinates_as_dofs()

   def define_residuals(self):
      E = self.E
      nu = self.nu

      mu = E / 2 / (1 + nu)
      lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
      lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)
      eps:Callable[[ExpressionOrNum],Expression] = lambda v: sym(grad(v, lagrangian=True))
      vdim = self.get_coordinate_system().vector_gradient_dimension(self.get_element_dimension(), lagrangian=True)
      sigma:Callable[[ExpressionOrNum],Expression] = lambda v: lmbda * trace(eps(v)) * identity_matrix(vdim) + 2 * mu * eps(v)

      x, x_test = var_and_test("mesh")  # Symbolic variables and test functions
      X = var("lagrangian")
      dX = self.get_dx(lagrangian=True)
      displ = x - X
      self.add_residual(double_dot(sigma(displ), eps(x_test)) * dX)
      if self.bulk_force is not None:
         self.add_residual(- dot(self.bulk_force, x_test) * dX)

   def define_error_estimators(self):
      if (self.spatial_error_factor is not None) and self.spatial_error_factor!=0:
         self.add_spatial_error_estimator(self.spatial_error_factor*grad(nondim("mesh"), lagrangian=True))
