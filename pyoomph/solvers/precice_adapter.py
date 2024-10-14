import precice

from ..expressions import *
from ..generic.codegen import EquationTree, ODEStorageMesh, BaseEquations, Equations
from ..generic.problem import Problem

class _PyoomphPreciceAdapater:
    def initialize_problem(self,problem:Problem):        
        problem._precice_interface=precice.Participant(problem.precice_participant,problem.precice_config_file,0,1)
        problem._equation_system._before_precice_initialise()
        problem._precice_interface.initialize()

    def coupled_run(self,problem:Problem):
        problem._activate_solver_callback()
        
        while problem._precice_interface.is_coupling_ongoing():
            if problem._precice_interface.requires_writing_checkpoint():
                problem.save_state(problem.get_output_directory("precice_checkpoint.dump"))
                
            precice_dt = problem._precice_interface.get_max_time_step_size()
            dt = min(precice_dt, problem.outstep)

            problem._equation_system._before_precice_solve(dt)
         
            problem.unsteady_newton_solve(dt,True)
            
            problem._equation_system._after_precice_solve(dt)
            
                
            print("DT",dt)            
            problem._precice_interface.advance(dt)
            
                            
            if problem._precice_interface.requires_reading_checkpoint():
                problem.load_state(problem.get_output_directory("precice_checkpoint.dump"))
                    
            problem.output()
            


class PreciceProvideMesh(BaseEquations):
    def __init__(self,precice_mesh_name:str,use_lagrangian_coords=True):
        super().__init__()
        self.name=precice_mesh_name
        self.use_lagrangian_coords=use_lagrangian_coords

    def before_precice_initialise(self,eqtree:"EquationTree"):
        mesh=eqtree.get_mesh()
        pr=mesh.get_problem()
        interface=pr._precice_interface
        xml_dimension = interface.get_mesh_dimensions(self.name)

        if isinstance(mesh,ODEStorageMesh):
            grid = numpy.zeros([1, xml_dimension])          
        else:
            grid = numpy.zeros([1, xml_dimension])
            if self.use_lagrangian_coords:
                for i,n in enumerate(mesh.nodes()):
                    for j in range(min(xml_dimension,n.ndim())):
                        grid[i,j] = n.x_lagr(j)
            else:
                for i,n in enumerate(mesh.nodes()):
                    for j in range(min(xml_dimension,n.ndim())):
                        grid[i,j] = n.x(j)
            
        mesh._precice_vertex_ids=[interface.set_mesh_vertices(self.name, grid)]


    def after_remeshing(self, eqtree: EquationTree):
        print("WARNING: PreciceProvideMesh does not support remeshing")
        #raise ValueError("PreciceProvideMesh does not support remeshing") # TODO: Same for adaptivity

            
            

class PreciceWriteData(BaseEquations):
    def __init__(self,**kwargs:ExpressionOrNum):
        super().__init__()
        self.entries=kwargs.copy()

    def define_residuals(self):
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
        for write_name in self.entries.keys():
            expr_index=mesh.list_local_expressions().index("_precice_write_"+write_name)
            buffer=mesh.evaluate_local_expression_at_nodes(expr_index,True,False)
            print("WRITE",provider.name,write_name,buffer)
            interface.write_data(provider.name, write_name, vertex_ids, buffer)

   
    def before_precice_initialise(self, eqtree: "EquationTree"):
        if self.get_problem()._precice_interface.requires_initial_data():
            self._do_write_data(0)
        

    def after_precice_solve(self, eqtree: "EquationTree",precice_dt:float):
        self._do_write_data(precice_dt)

            

class PreciceReadData(BaseEquations):
    def __init__(self,**kwargs:str):
        super().__init__()
        self.entries=kwargs.copy()

    def define_fields(self):
        if self.get_combined_equations()._is_ode():
            raise RuntimeError("TODO ODES")
        for name in self.entries.keys():
            # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
            # TODO: Vector fields
            Equations.define_scalar_field(self,name,self.get_current_code_generator()._coordinate_space)

    def define_residuals(self):
        for name in self.entries.keys():
            self.set_Dirichlet_condition(name,True) # Fix it

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
            print("READ",provider.name,precice_name,buffer)
            if mesh.has_interface_dof_id(pyoomph_name)>=0:
                index=mesh.has_interface_dof_id(pyoomph_name)
                for i,n in enumerate(mesh.nodes()):
                    n.set_value(n.additional_value_index(index) ,buffer[i])
            else:
                index=mesh.get_nodal_field_indices()[pyoomph_name]
                for i,n in enumerate(mesh.nodes()):
                    n.set_value(index,buffer[i])

    def before_precice_solve(self, eqtree: EquationTree, precice_dt: float):
        self._do_read_data(precice_dt)
             

_pyoomph_precice_adapter=_PyoomphPreciceAdapater()
def get_pyoomph_precice_adapter():
    return _pyoomph_precice_adapter
    

