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
 
 
from ..generic import ODEEquations
from ..expressions import * 



class DynamicODEEquations(ODEEquations):
	"""
		Represents a dynamic ordinary differential equation.
	"""
	def __init__(self):
		super().__init__() #Really important, otherwise it will crash
		self._eqs:List[Tuple[str,ExpressionOrNum]]=[]

	def add_equation(self,name_or_sym:Union[str,Expression],eq:ExpressionOrNum):
		if not isinstance(name_or_sym,str):
			name_or_sym=str(name_or_sym.op(0))
		to_add=(name_or_sym,eq)
		self._eqs.append(to_add)

	def define_fields(self):
		for e in self._eqs:
			self.define_ode_variable(e[0])

	def define_residuals(self):
		for e in self._eqs:
			_,test=var_and_test(e[0])
			self.add_residual(test*e[1] )





