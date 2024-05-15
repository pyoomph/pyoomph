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
 
 
from .cb import CustomMathExpression
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline,SmoothBivariateSpline,RectBivariateSpline #type:ignore
import meshio
from  matplotlib.tri import Triangulation,LinearTriInterpolator
from .generic import Expression
from ..typings import *
if TYPE_CHECKING:
	from .generic import ExpressionOrNum




import numpy


class Interpolate1d(CustomMathExpression):
	def __init__(self,arr_or_fname:Union[str,NPFloatArray],kind:str='linear',axis:int=-1,bounds_error:bool=True,fill_value:Optional[float]=None):
		super().__init__()
		if isinstance(arr_or_fname,str):	
			arr_or_fname=numpy.loadtxt(arr_or_fname) #type:ignore
		x=numpy.array(arr_or_fname[:,0]) #type:ignore
		y=numpy.array(arr_or_fname[:,axis]) #type:ignore
		if fill_value is None:
			fill_value=numpy.nan
		self.interp=interp1d(x,y,kind=kind,bounds_error=bounds_error,fill_value=fill_value)

	def eval(self,arg_array:NPFloatArray)->float:
		return self.interp(arg_array[0]) #type:ignore


class InterpolateSpline1d(CustomMathExpression):
	def __init__(self,arr_or_fname:Union[str,NPFloatArray],k:int=3,w:Any=None,xaxis:int=0,yaxis:int=-1,ext:int=0,check_finite:bool=False):
		super().__init__()
		if isinstance(arr_or_fname,str):	
			arr_or_fname=numpy.loadtxt(arr_or_fname) #type:ignore
		x=arr_or_fname[:,xaxis]
		y=arr_or_fname[:,yaxis]
		self.interp=InterpolatedUnivariateSpline(x,y,w=w,k=k,ext=ext,check_finite=check_finite)


	def eval(self,arg_array:NPFloatArray)->float:
		return self.interp(arg_array[0]) #type:ignore

	def derivative(self,index:int) -> CustomMathExpression:
		return InterpolateSpline1dDerivative(self)


class InterpolateSpline1dDerivative(CustomMathExpression):
	def __init__(self,parent:InterpolateSpline1d):
		super().__init__()
		self.parent=parent
		self.interp=self.parent.interp.derivative()

	def eval(self,arg_array:NPFloatArray)->float:
		return self.interp(arg_array[0]) #type:ignore


#################


class InterpolateSmoothBivariateSpline2d(CustomMathExpression):
	def __init__(self,arr_or_fname:Union[str,NPFloatArray],kind:str='linear',xaxis:int=0,yaxis:int=1,zaxis:int=-1,w:Any=None,bbox:List[Optional[float]]=[None,None,None,None],kx:int=3,ky:int=3,s:Any=None,eps:float=1e-16):
		super().__init__()
		if isinstance(arr_or_fname,str):	
			arr_or_fname=numpy.loadtxt(arr_or_fname) #type:ignore
		x=arr_or_fname[:,xaxis]
		y=arr_or_fname[:,yaxis]
		z=arr_or_fname[:,zaxis]
		self.interp=SmoothBivariateSpline(x,y,z,w=w,kx=kx,ky=ky,bbox=bbox,s=s,eps=eps)


	def eval(self,arg_array:NPFloatArray)->float:
		return self.interp(arg_array[0],arg_array[1]) #type:ignore

	def derivative(self,index:int) -> CustomMathExpression:
		dx=[0,0]
		dx[index]=1
		return InterpolateSmoothBivariateSpline2dDerivative(self,tuple(dx))

class InterpolateSmoothBivariateSpline2dDerivative(CustomMathExpression):
	def __init__(self,parent:InterpolateSmoothBivariateSpline2d,dx:Tuple[int,int]):
		super().__init__()
		self.parent=parent
		self.interp=self.parent.interp
		self.dx=dx

	def eval(self,arg_array:NPFloatArray)->float:
		return self.interp(arg_array[0],arg_array[1],dx=self.dx[0],dy=self.dx[1]) #type:ignore

	def derivative(self,index:int) -> CustomMathExpression:
		dx=[i for i in self.dx]
		dx[index]=dx[index]+1
		return InterpolateSmoothBivariateSpline2dDerivative(self.parent,tuple(dx))		

	def get_id_name(self)->str:
		return  "D"+str(self.dx)+self.parent.get_id_name()
####



class InterpolateRectBivariateSpline2d(CustomMathExpression):
	def __init__(self,x:NPFloatArray,y:NPFloatArray,z:NPFloatArray,bbox:List[Optional[float]]=[None,None,None,None],kx:int=3,ky:int=3,s:Any=None):
		super().__init__()
		self.interp=RectBivariateSpline(x,y,z,s=s,kx=kx,ky=ky,bbox=bbox)


	def eval(self,arg_array:NPFloatArray)->float:
		return self.interp(arg_array[0],arg_array[1]) #type:ignore






class MeshFileInterpolation2d(CustomMathExpression):
	def __init__(self,meshfilename:str,fieldname:str,coordinate_unit:"ExpressionOrNum"=1, result_unit:"ExpressionOrNum"=1):
		super().__init__()
		mesh=meshio.read(meshfilename)
		points=mesh.points[:,0:2]
		if len(mesh.cells)!=1:
			raise RuntimeError("Does only work for one cell block at the moment")
		if mesh.cells[0].type!="triangle":
			raise RuntimeError("Does only work for triangle cell data at the moment")
		vector_dir=None
		if fieldname not in mesh.point_data.keys():
			if fieldname.endswith("_x") or fieldname.endswith("_y") or fieldname.endswith("_z"):
				if fieldname[:-2] in mesh.point_data.keys():
					vector_dir={"x":0,"y":1,"z":2}[fieldname[-1]]
					fieldname=fieldname[:-2]
				else:
					raise RuntimeError("Field "+str(fieldname)+" not in the point data of the mesh")
			else:
				raise RuntimeError("Field "+str(fieldname)+" not in the point data of the mesh")
		data=mesh.point_data[fieldname]
		if vector_dir is not None:
			data=data[:,vector_dir]
		triang=Triangulation(points[:,0], points[:,1], mesh.cells[0].data)
		self.interp=LinearTriInterpolator(triang, data)	    
		self.coordinate_unit=Expression(coordinate_unit)
		self.result_unit=Expression(result_unit)

	def get_argument_unit(self, index: int) -> "Expression":
		return self.coordinate_unit
    
	def get_result_unit(self) -> "Expression":
		return self.result_unit

	def eval(self, arg_array: NPFloatArray) -> float:
		x,y=arg_array[0],arg_array[1]
		return self.interp(x,y)
