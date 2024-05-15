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


from pyoomph import *
from pyoomph.equations.multi_component import * # import the multi-component flow equations
from pyoomph.materials import * # import the material library core
import pyoomph.materials.default_materials # Add the default materials to the library
from pyoomph.equations.contact_angle import *
from pyoomph.expressions.units import *
from pyoomph.equations.ALE import *
from pyoomph.utils.dropgeom import *
from pyoomph.meshes.remesher import *

# THIS FILE IS MAINLY THE SAME AS THE evap_droplet.py
# HOWEVER, WE HAVE MERGED THE CONSIDERATION OF SURFACTANTS (evap_droplet_insoluble.py)
# AND FURTHERMORE ADD THERMAL EFFECTS AND GAS FLOW

############################
# As before in the multidom/evap_droplet example

# A mesh of an axisymmetric droplet surrounded by gas
class DropletWithGasMesh(GmshTemplate):
    def __init__(self,droplet,gas,cl_resolution_factor=0.1,gas_resolution_factor=50):
        super(DropletWithGasMesh, self).__init__()
        self.droplet,self.gas=droplet,gas
        self.cl_resolution_factor,self.gas_resolution_factor=cl_resolution_factor,gas_resolution_factor
        self.default_resolution=0.03
        self.remesher = Remesher2d(self)  # add remeshing possibility
        #self.mesh_mode="tris"

    def define_geometry(self):
        # finer resolution at the contact line, lower at the gas far field
        res_cl=self.cl_resolution_factor*self.default_resolution
        res_gas=self.gas_resolution_factor*self.default_resolution
        # droplet
        p00=self.point(0,0) # origin
        pr0 = self.point(self.droplet.base_radius, 0,size=res_cl) # contact line
        p0h=self.point(0,self.droplet.apex_height) # zenith
        self.circle_arc(pr0,p0h,through_point=self.point(-self.droplet.base_radius,0),name="droplet_gas") # curved interface
        self.create_lines(p0h,"droplet_axisymm",p00,"droplet_substrate",pr0)
        self.plane_surface("droplet_gas","droplet_axisymm","droplet_substrate",name="droplet") # droplet domain
        # gas dome
        pR0=self.point(self.gas.radius,0,size=res_gas)
        p0R = self.point(0,self.gas.radius,size=res_gas)
        self.circle_arc(pR0,p0R,center=p00,name="gas_infinity")
        self.line(p0h,p0R,name="gas_axisymm")
        self.line(pr0,pR0,name="gas_substrate")
        self.plane_surface("gas_substrate","gas_infinity","gas_axisymm","droplet_gas",name="gas") # gas domain

############################

# Class to unify all droplet specifications
class _DropletSpecs:
    def __init__(self):
        self.mixture=get_pure_liquid("water") # initial droplet mixture
        self.sliplength = 1 * micro * meter  # slip length at the substrate
        # Geometric properties: The user must set exactly two of them, rest is calculated in finalize()
        self.base_radius=None
        self.apex_height=None
        self.contact_angle=None
        self.volume=None
        self.curv_radius=None
        self.composition_space="C1" # Space to solve the liquid composition

    # Calculate all geometric properties from the two given properties
    def finalize(self):
        geom=DropletGeometry(base_radius=self.base_radius,apex_height=self.apex_height,contact_angle=self.contact_angle,volume=self.volume,curv_radius=self.curv_radius)
        self.base_radius,self.contact_angle,self.volume,self.apex_height,self.curv_radius=geom.base_radius,geom.contact_angle,geom.volume,geom.apex_height,geom.curv_radius


# Class to unify the gas properties
class _GasSpecs:
    def __init__(self):
        self.mixture=Mixture(get_pure_gas("air")+0.2*get_pure_gas("water"),quantity="relative_humidity",temperature=20*celsius)
        self.radius=5*milli*meter
        self.composition_space = "C1"  # Space to solve the gas composition
        # MODIFICATION: Add possibility to solve the flow
        self.solve_flow=False
        self.sliplength = 1 * micro * meter  # slip length at the substrate in case of flow




class EvaporatingDroplet(Problem):
    def __init__(self):
        super(EvaporatingDroplet, self).__init__()

        self.droplet=_DropletSpecs() # Droplet properties
        self.gas=_GasSpecs() # gas properties
        self.interface=None # Will be automatically determined if None
        self.contact_line=UnpinnedContactLine() # contact line dynamics

        self.temperature=20*celsius # global conditions
        self.absolute_pressure=1*bar
        self.gravity=9.81*vector(0,-1)*meter/second**2

        self.surfactants={}
        # MODIFICATION: Add a switch for controlling the consideration of thermal effects
        self.isothermal=True

    def _get_initial_surface_tension(self):
        sigma0 = self.droplet.mixture.evaluate_at_condition(self.interface.surface_tension,self.droplet.mixture.initial_condition,temperature=self.temperature,time=0)
        sigmaWithSurfs=self.interface.evaluate_at_initial_surfactant_concentrations(sigma0)
        return sigmaWithSurfs

    def define_problem(self):
        # Settings: Axisymmetric and typical scales
        self.set_coordinate_system("axisymmetric")
        self.droplet.finalize() # Calculate all geometric properties from the input
        self.set_scaling(temporal=1*second,spatial=self.droplet.base_radius)
        # Setting scalings for e.g. pressure from the liquid properties
        self.droplet.mixture.set_reference_scaling_to_problem(self,temperature=self.temperature)
        # Define temperature and pressure on a global level so that all fluid properties are evaluated at these conditions
        self.define_named_var(temperature=self.temperature,absolute_pressure=self.absolute_pressure)
        # Add the mesh
        self.add_mesh(DropletWithGasMesh(self.droplet,self.gas))

        # If not interface is set, get it from the library
        if self.interface is None:
            self.interface=get_interface_properties(self.droplet.mixture, self.gas.mixture, self.surfactants)
        # Laplace pressure scale
        sigma0=self._get_initial_surface_tension()
        self.set_scaling(pressure=sigma0/self.droplet.curv_radius)
        self.set_scaling(surface_concentration=1*micro*mol/meter**2)

        # Droplet equations
        d_eqs=MeshFileOutput() # Output
        d_eqs += PseudoElasticMesh() # Mesh motion

        # MODIFICATIONS:
        # Tell whether it should be isothermal or thermal flow equations and pass the initial temperature for the initial condition
        d_eqs += CompositionFlowEquations(self.droplet.mixture,gravity=self.gravity,compo_space=self.droplet.composition_space,isothermal=self.isothermal,initial_temperature=self.temperature) # flow

        if not self.isothermal:
            # Add a suitable temperature boundary condition at the substrate
            d_eqs+=DirichletBC(temperature=self.temperature)@"droplet_substrate"
            # Connect the temperature field
            d_eqs+=ConnectFieldsAtInterface("temperature")@"droplet_gas"

        d_eqs += DirichletBC(mesh_x=0,velocity_x=0)@"droplet_axisymm" # symmetry axis
        d_eqs += DirichletBC(mesh_y=0, velocity_y=0) @ "droplet_substrate" # allow slip, but fix mesh
        d_eqs += NavierStokesSlipLength(self.droplet.sliplength)@ "droplet_substrate" # limit slip by slip length
        d_eqs += MultiComponentNavierStokesInterface(self.interface)@"droplet_gas" # Free surface equation
        d_eqs += ConnectMeshAtInterface() @ "droplet_gas" # connect the gas mesh to co-move

        # Contact line dynamics: First tell the selected model some information which are useful if not set by hand
        self.contact_line.set_missing_information(initial_contact_angle=self.droplet.contact_angle,initial_surface_tension=sigma0,initial_contact_line_position=vector(self.droplet.base_radius,0))
        # Contact line dynamics with wall normal and tangent.
        d_eqs+=DynamicContactLineEquations(model=self.contact_line,wall_tangent=vector(-1,0),wall_normal=vector(0,1),with_observables=True)@"droplet_gas/droplet_substrate"
        d_eqs+=IntegralObservableOutput(filename="EVO_contact_line")@"droplet_gas/droplet_substrate" # Monitor the contact line over time

        # Gas equations
        g_eqs=MeshFileOutput() # output
        g_eqs+=PseudoElasticMesh() # mesh motion

        # MODIFICATION:
        # Tell whether it should be isothermal or thermal flow equations and pass the initial temperature for the initial condition
        # Also check if gas flow is demanded and switch to the CompositionFlow equations in that case
        if self.gas.solve_flow:
            g_eqs += CompositionFlowEquations(self.gas.mixture, compo_space=self.gas.composition_space,isothermal=self.isothermal, initial_temperature=self.temperature,gravity=self.gravity)
            g_eqs += DirichletBC(velocity_x=0)@"gas_axisymm" # Good boundary conditions
            g_eqs += (DirichletBC(velocity_y=0)+NavierStokesSlipLength(self.gas.sliplength))@"gas_substrate"
            g_eqs += BalanceGravityAtFarField(self.gravity) @ "gas_infinity"
        else:
            g_eqs+=CompositionDiffusionEquations(self.gas.mixture,space=self.gas.composition_space,isothermal=self.isothermal,initial_temperature=self.temperature)
        # Add a suitable temperature boundary condition at the substrate
        if not self.isothermal:
            g_eqs += DirichletBC(temperature=self.temperature)@"gas_substrate"
            g_eqs+=TemperatureInfinityEquations(self.temperature)@"gas_infinity" # Also mimic an infinite domain for the temperature


        g_eqs+=DirichletBC(mesh_x=0)@"gas_axisymm" # fixed mesh coordinates at the boundaries
        g_eqs+=DirichletBC(mesh_y=0)@"gas_substrate"
        g_eqs+=DirichletBC(mesh_x=True,mesh_y=True)@"gas_infinity"
        g_eqs+=CompositionDiffusionInfinityEquations()@"gas_infinity"

        # Control remeshing
        d_eqs += RemeshWhen(RemeshingOptions(max_expansion=1.5, min_expansion=0.7))
        g_eqs+=RemeshWhen(RemeshingOptions(max_expansion=1.5,min_expansion=0.7))

        d_eqs+=IntegralObservables(volume=1) # Calculate the volume evolution
        # Add all partial masses
        d_eqs += IntegralObservables(**{"mass_"+c:self.droplet.mixture.mass_density*var("massfrac_"+c) for c in self.droplet.mixture.components})
        d_eqs+=IntegralObservableOutput(filename="EVO_droplet") # Write all integral observables to file

        self.add_equations(d_eqs@"droplet"+g_eqs@"gas")


if __name__=="__main__":
    with EvaporatingDroplet() as problem:
        # Set exactly two geometric quantities
        #problem.set_c_compiler("tcc")
        problem.droplet.volume=100*nano*liter
        problem.droplet.contact_angle=60*degree

        problem.isothermal=False # Switch on thermal effects
        problem.gas.solve_flow=True # and solve the flow in the gas phase

        # Ambient conditions and fluid mixtures
        problem.temperature=20*celsius
        problem.droplet.mixture=Mixture(get_pure_liquid("water")) # load pure water from the material library
        problem.gas.mixture=Mixture(get_pure_gas("air")+20*percent*get_pure_gas("water"),quantity="relative_humidity",temperature=problem.temperature)

        problem.droplet.sliplength=1*nano*meter
        problem.contact_line=PinnedContactLine() # Simplest model, just pinned

        problem.run(500*second,startstep=0.1*second,maxstep=0.25*second,outstep=True,temporal_error=1)
