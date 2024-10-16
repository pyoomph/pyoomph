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
 

import precice

from ..expressions import *
from ..generic.codegen import EquationTree, ODEStorageMesh, BaseEquations, Equations
from ..generic.problem import Problem

from scipy import spatial


# This is just a helper class which collects a few methods
class _PyoomphPreciceAdapater:
    def initialize_problem(self,problem:Problem):        
        problem._precice_interface=precice.Participant(problem.precice_participant,problem.precice_config_file,0,1)
        problem._equation_system._before_precice_initialise()
        problem._precice_interface.initialize()

    def coupled_run(self,problem:Problem,maxstep:Optional[float]=None,temporal_error:Optional[float]=None,output_initially:bool=True):
        problem._activate_solver_callback()
        
        if output_initially:
            problem.output()
        while problem._precice_interface.is_coupling_ongoing():
            if problem._precice_interface.requires_writing_checkpoint():
                problem.save_state(problem.get_output_directory("precice_checkpoint.dump"))
                
            precice_dt = problem._precice_interface.get_max_time_step_size()
            dt=precice_dt
            if maxstep:
                dt = min(precice_dt, maxstep)

            problem._equation_system._before_precice_solve(dt)
         
            if temporal_error is None:
                problem.unsteady_newton_solve(dt,True)
            else:
                current_time=problem.get_current_time(as_float=True,dimensional=False)
                problem.adaptive_unsteady_newton_solve(dt,temporal_error,True)
                dt=problem.get_current_time(as_float=True,dimensional=False)-current_time
            
            problem._equation_system._after_precice_solve(dt)
                                    
            problem._precice_interface.advance(dt)
                                    
            if problem._precice_interface.requires_reading_checkpoint():
                problem.load_state(problem.get_output_directory("precice_checkpoint.dump"))
            else:                    
                if problem._precice_interface.is_time_window_complete():
                    problem.actions_after_newton_solve()
                    problem.output()
            


class PreciceProvideMesh(BaseEquations):
    """This class exports the domain or interface where it is added to to preCICE. 
    Afterwards, it can be used in combination with :py:class:`~pyoomph.solvers.precice_adapter.PreciceWriteData` and :py:class:`~pyoomph.solvers.precice_adapter.PreciceReadData`.
    
    Args:
        precice_mesh_name: The name of the mesh in the preCICE config file
        use_lagrangian_coords: If True, the Lagrangian coordinates are used to map the nodes to preCICE, otherwise the Eulerian coordinates
    """
    def __init__(self,precice_mesh_name:str,use_lagrangian_coords=True):
        super().__init__()
        self.name=precice_mesh_name
        self.use_lagrangian_coords=use_lagrangian_coords

    def before_precice_initialise(self,eqtree:"EquationTree"):
        mesh=eqtree.get_mesh()
        pr=mesh.get_problem()
        interface=pr._precice_interface
        
        xml_dimension = interface.get_mesh_dimensions(self.name)
        my_nodes=[n for n in mesh.nodes()]
        
        
        mesh._precice_node_to_vertex_id={n:i for i,n in enumerate(my_nodes)}
        if isinstance(mesh,ODEStorageMesh):
            grid = numpy.zeros([1, xml_dimension])          
        else:
            grid = numpy.zeros([len(my_nodes), xml_dimension])
            if len(my_nodes)==0:
                raise ValueError("No nodes in mesh")
            iterate_dim=min(xml_dimension,my_nodes[0].ndim())
            if self.use_lagrangian_coords:
                for i,n in enumerate(my_nodes):
                    for j in range(iterate_dim):
                        grid[i,j] = n.x_lagr(j)
            else:
                for i,n in enumerate(my_nodes):
                    for j in range(iterate_dim):
                        grid[i,j] = n.x(j)
            
        mesh._precice_vertex_ids=interface.set_mesh_vertices(self.name, grid)
        if not isinstance(mesh._precice_vertex_ids,numpy.ndarray):
            mesh._precice_vertex_ids=numpy.array(mesh._precice_vertex_ids)
            
        if interface.requires_mesh_connectivity_for(self.name):
            for e in mesh.elements():
                all_nodes_have_vertex_id=True
                for ni in range(e.nnode()):
                    n=e.node_pt(ni)
                    if n not in mesh._precice_node_to_vertex_id:
                        all_nodes_have_vertex_id=False
                        break
                if not all_nodes_have_vertex_id:
                    raise RuntimeError("Not all nodes in element have vertex id...")
                typus=e.get_meshio_type_index()
                if typus==2: # LineC2
                    interface.set_mesh_edge(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)])
                    interface.set_mesh_edge(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)])                    
                elif typus==1: # LineC1
                    interface.set_mesh_edge(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)])
                elif typus==0: # Points have no edges
                    pass 
                elif typus==3: # TriC1
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)])
                elif typus==9: # TriC2
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(5)])                    
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                    
                    # Strange order, no matter how I permute the nodes, it is always backface in preCICE vtu out
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(2)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(1)])                                        
                elif typus==99: # TriC2TB
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                                            
                    # Strange order, no matter how I permute the nodes, it is always backface in preCICE vtu out
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(6)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                    
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                                            
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                                                                
                elif typus==66: # TriC1TB
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                                            
                    
                    
                elif typus==8: # QuadC2
                    if xml_dimension==3:
                        #interface.set_mesh_quad(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(3)])
                        raise RuntimeError("Check Quads here")
                    else:
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)])                        
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(5)])                                                
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(5)])                                                                        
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(6)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(3)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(8)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(4)])
                elif typus==6: # QuadC1
                    if xml_dimension==3:
                        #interface.set_mesh_quad(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(3)])
                        raise RuntimeError("Check Quads here")
                    else:
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)])                                        
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                                            
                else:
                    raise RuntimeError("Only mesh edges are implemented, not type index "+str(typus))
                
                    
                                
            
        


    def after_remeshing(self, eqtree: EquationTree):
        mesh=eqtree.get_mesh()
        pr=mesh.get_problem()
        interface=pr._precice_interface
        if not hasattr(mesh,"_precice_vertex_ids") or not hasattr(mesh,"_precice_node_to_vertex_id"):
            raise ValueError("Something strange. Did you remesh? This does not work with preCICE")
        my_nodes=[n for n in mesh.nodes()]
        if len(my_nodes)!=len(mesh._precice_node_to_vertex_id):
            raise ValueError("Number of nodes mismatch: "+str(len(my_nodes))+" vs. "+str(len(mesh._precice_node_to_vertex_id))+". Did you remesh? This does not work with preCICE")
        for n in my_nodes:
            if n not in mesh._precice_node_to_vertex_id:
                raise ValueError("Node not in preCICE mesh. Did you remesh? This does not work with preCICE")
        

            
            

class PreciceWriteData(BaseEquations):
    """
    Must be combined with :py:class:`~pyoomph.solvers.precice_adapter.PreciceProvideMesh` on the same domain or interface.
    
    Writes data to preCICE. Data can be an arbitrary scalar or vectorial expression. In the latter case, the vector_dim parameter must be set.

    Args:
        scaling: Optional scaling factor (e.g. units) for the data
        by_projection: If True, the data is first projected to nodal data and then written to preCICE. This is useful if the data is not available at the nodes.
        projection_space: If by_projection is True, the space to which the data is projected. If None, the coordinate space of the mesh is used.
        vector_dim: The dimension of the vector field. If None, a scalar field is assumed.
        kwargs: The data to be written. The keys are the names of the data in preCICE, the values are the expressions.
    """
    def __init__(self,scaling:ExpressionNumOrNone=None,by_projection:bool=False,projection_space:Optional[FiniteElementSpaceEnum]=None,vector_dim:Optional[int]=None,**kwargs:ExpressionOrNum):
        super().__init__()
        self.entries=kwargs.copy()
        self.by_projection=by_projection
        self.scaling=scaling
        self.projection_space=projection_space
        self.vector_dim=vector_dim
        
    def _sanitize_name(self,name:str):
        return "_precice_write_proj_"+name.replace(" ","_").replace("-","_").replace("/","_")

    def define_scalar_field(self,name:str,space:FiniteElementSpaceEnum,scale:Optional[ExpressionNumOrNone]=None):
        # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
        return Equations.define_scalar_field(self,name,space,scale=scale)

    def define_fields(self):
        if self.by_projection:
            if self.get_combined_equations()._is_ode():
                raise RuntimeError("TODO ODES")
            
            if not self.projection_space:
                space=self.get_current_code_generator()._coordinate_space
            else:
                space=self.projection_space
            for name in self.entries.keys():
                # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
                if self.vector_dim is None:
                    Equations.define_scalar_field(self,self._sanitize_name(name),space,scale=self.scaling)
                else:
                    Equations.define_vector_field(self,self._sanitize_name(name),space,scale=self.scaling,dim=self.vector_dim)                    

    def define_residuals(self):
        if self.by_projection:
            for name,expr in self.entries.items():
                vname=self._sanitize_name(name)
                v,vtest=var_and_test(vname)
                self.add_weak(v-expr,vtest)
        else:
            for n,v in self.entries.items():
                self.add_local_function("_precice_write_"+n,v)
    

    def _do_write_data(self,precice_dt:float):
        mesh=self.get_mesh()
        if not hasattr(mesh,"_precice_vertex_ids"):
            raise ValueError("PreciceProvideMesh not set, please add it before PreciceWriteData")
        interface=mesh.get_problem()._precice_interface
        provider=self.get_combined_equations().get_equation_of_type(PreciceProvideMesh,always_as_list=True)
        if len(provider)!=1:
            raise ValueError("PreciceProvideMesh not set or set multiple times on this domain, please add a single one")
        provider=provider[0]
        assert isinstance(provider,PreciceProvideMesh)
        vertex_ids=mesh._precice_vertex_ids
                
        if self.by_projection:
            for write_name in self.entries.keys():
                pyoomph_name=self._sanitize_name(write_name)
                
                mynodes=[n for n in mesh.nodes()]
                
                if self.vector_dim is None:
                
                    buffer=numpy.zeros([len(mynodes)])
                    if mesh.has_interface_dof_id(pyoomph_name)>=0:
                        index=mesh.has_interface_dof_id(pyoomph_name)
                        for i,n in enumerate(mynodes):
                            buffer[mesh._precice_node_to_vertex_id[n]]=n.value(n.additional_value_index(index))
                    else:
                        index=mesh.get_nodal_field_indices()[pyoomph_name]
                        for i,n in enumerate(mynodes):
                            buffer[mesh._precice_node_to_vertex_id[n]]=n.value(index,buffer)
                else:
                    buffer=[]
                    for vindex in range(self.vector_dim):
                        buffer_row=numpy.zeros([len(mynodes)])
                        component_name=pyoomph_name+"_"+["x","y","z"][vindex]
                        if mesh.has_interface_dof_id(component_name)>=0:
                            index=mesh.has_interface_dof_id(component_name)
                            for i,n in enumerate(mynodes):
                                buffer_row[mesh._precice_node_to_vertex_id[n]]=n.value(n.additional_value_index(index))
                        else:
                            index=mesh.get_nodal_field_indices()[component_name]
                            for i,n in enumerate(mynodes):
                                buffer_row[mesh._precice_node_to_vertex_id[n]]=n.value(index,buffer)
                        buffer.append(buffer_row)
                    buffer=numpy.array(buffer).T
                                
                interface.write_data(provider.name, write_name, vertex_ids, buffer)                                    
        else:
            nodemap=mesh.fill_node_index_to_node_map()
            inds=numpy.array([mesh._precice_node_to_vertex_id[n] for n in nodemap])            
            for write_name in self.entries.keys():
                if self.vector_dim is None:
                    expr_index=mesh.list_local_expressions().index("_precice_write_"+write_name)
                    buffer=mesh.evaluate_local_expression_at_nodes(expr_index,True,False)                
                    buffer=numpy.array(buffer)                                                            
                    interface.write_data(provider.name, write_name, vertex_ids[inds], buffer)
                else:                    
                    directs=["x","y","z"]
                    buffer=[]
                    for vindex in range(self.vector_dim):                    
                        expr_index=mesh.list_local_expressions().index("_precice_write_"+write_name+"_"+directs[vindex])
                        buffer_row=mesh.evaluate_local_expression_at_nodes(expr_index,True,False)                
                        buffer_row=numpy.array(buffer_row)                                        
                        buffer.append(buffer_row)
                    buffer=numpy.array(buffer).T                
                    interface.write_data(provider.name, write_name, vertex_ids[inds], buffer)

   
    def before_precice_initialise(self, eqtree: "EquationTree"):
        if self.get_problem()._precice_interface.requires_initial_data():
            self._do_write_data(0)
        

    def after_precice_solve(self, eqtree: "EquationTree",precice_dt:float):
        self._do_write_data(precice_dt)

            

class PreciceReadData(BaseEquations):
    """Must be combined with :py:class:`~pyoomph.solvers.precice_adapter.PreciceProvideMesh` on the same domain or interface.
    
    Reads data from preCICE. Data can be an arbitrary scalar or vectorial expression. In the latter case, the vector_dim parameter must be set.
    
    For each data to be read, a corresponding field is constructed in pyoomph on this domain. This field is pinned, i.e. not a degree of freedom, and its values are updated via preCICE.

    Args:
        scaling: Optional scaling factor (e.g. units) for the data
        vector_dim: The dimension of the vector field. If None, a scalar field is assumed.
        kwargs: The data to be read. The keys are the names of the data in pyoomph, the values are the names of the fields in preCICE.
    """
    def __init__(self,*,scaling:ExpressionNumOrNone=None,vector_dim:Optional[int]=None,**kwargs:str):
        super().__init__()
        self.entries=kwargs.copy()
        self.scaling=scaling
        self.vector_dim=vector_dim

    def define_scalar_field(self,name:str,space:FiniteElementSpaceEnum,scale:Optional[ExpressionNumOrNone]=None):
        # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
        return Equations.define_scalar_field(self,name,space,scale=scale)

    def define_fields(self):
        if self.get_combined_equations()._is_ode():
            raise RuntimeError("TODO ODES")
        for name in self.entries.keys():
            # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
            if self.vector_dim is None:
                Equations.define_scalar_field(self,name,self.get_current_code_generator()._coordinate_space,scale=self.scaling)
            else:
                Equations.define_vector_field(self,name,self.get_current_code_generator()._coordinate_space,scale=self.scaling,dim=self.vector_dim)

    def define_residuals(self):
        for name in self.entries.keys():
            if self.vector_dim is None:
                self.set_Dirichlet_condition(name,True) # Fix it
            else:
                directs=["x","y","z"]
                for i in range(self.vector_dim):
                    self.set_Dirichlet_condition(name+"_"+directs[i],True)

    def _do_read_data(self,precice_dt:float):
        mesh=self.get_mesh()
        if not hasattr(mesh,"_precice_vertex_ids"):
            raise ValueError("PreciceProvideMesh not set, please add it before PreciceReadData")
        interface=mesh.get_problem()._precice_interface
        provider=self.get_combined_equations().get_equation_of_type(PreciceProvideMesh,always_as_list=True)
        if len(provider)!=1:
            raise ValueError("PreciceProvideMesh not set or set multiple times on this domain, please add a single one")
        provider=provider[0]
        assert isinstance(provider,PreciceProvideMesh)
        vertex_ids=mesh._precice_vertex_ids        
                
        for pyoomph_name,precice_name in self.entries.items():
            buffer=interface.read_data(provider.name, precice_name, vertex_ids, precice_dt)
            if self.vector_dim is None:
                if len(buffer.shape)>1:
                    raise RuntimeError("Expected scalar, got vector")
                if mesh.has_interface_dof_id(pyoomph_name)>=0:
                    index=mesh.has_interface_dof_id(pyoomph_name)
                    for i,n in enumerate(mesh.nodes()):                    
                        n.set_value(n.additional_value_index(index) ,buffer[mesh._precice_node_to_vertex_id[n]])
                else:
                    index=mesh.get_nodal_field_indices()[pyoomph_name]
                    for i,n in enumerate(mesh.nodes()):
                        n.set_value(index,buffer[mesh._precice_node_to_vertex_id[n]])
            else:
                if len(buffer.shape)==1:
                    raise RuntimeError("Expected vector, got scalar")
                if buffer.shape[1]!=self.vector_dim:
                    raise RuntimeError("Expected vector of dimension "+str(self.vector_dim)+", got "+str(buffer.shape[1]))
                directs=["x","y","z"]
                for vindex in range(self.vector_dim):
                    if mesh.has_interface_dof_id(pyoomph_name+"_"+directs[vindex])>=0:
                        index=mesh.has_interface_dof_id(pyoomph_name+"_"+directs[vindex])
                        for i,n in enumerate(mesh.nodes()):                    
                            n.set_value(n.additional_value_index(index) ,buffer[mesh._precice_node_to_vertex_id[n],vindex])
                    else:
                        index=mesh.get_nodal_field_indices()[pyoomph_name+"_"+directs[vindex]]
                        for i,n in enumerate(mesh.nodes()):
                            n.set_value(index,buffer[mesh._precice_node_to_vertex_id[n],vindex])

    def before_precice_solve(self, eqtree: EquationTree, precice_dt: float):
        self._do_read_data(precice_dt)
             


# Instance of the helper class
_pyoomph_precice_adapter=_PyoomphPreciceAdapater()
# Access function to the helper class instance. Called by the Problem class
def get_pyoomph_precice_adapter():
    return _pyoomph_precice_adapter
    

