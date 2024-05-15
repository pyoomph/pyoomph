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


from evaporating_pure_droplet import * # Import the droplet evaporation problem without surfactants
from insoluble_surfactant_definition import * # and the previously defined insoluble surfactant


class EvapDropletWithSurfactant(EvaporatingDroplet):
    def __init__(self):
        super(EvapDropletWithSurfactant, self).__init__()
        self.surfactants={} # Here, the user can add surfactants to the interface

    def _get_initial_surface_tension(self):
        # This one will only evaluate the surface tension function at the liquid composition and the temperture
        sigma=super(EvapDropletWithSurfactant, self)._get_initial_surface_tension()
        # so we also must plug in the initial surfactant concentrations to get a constant value
        return self.interface.evaluate_at_initial_surfactant_concentrations(sigma)

    def define_problem(self):
        if self.interface is None: # when the interface properties were not set by hand, find them
            # but now potentially with surfactants
            self.interface=get_interface_properties(self.droplet.mixture,self.gas.mixture,surfactants=self.surfactants)
        self.set_scaling(surface_concentration=1*micro*mol/meter**2) # good scalings for the surface concentration fields
        super(EvapDropletWithSurfactant, self).define_problem() # define the problem as before


if __name__=="__main__":
    with EvapDropletWithSurfactant() as problem:
        # Set exactly two geometric quantities
        problem.droplet.volume=100*nano*liter
        problem.droplet.contact_angle=60*degree

        # Ambient conditions and fluid mixtures
        problem.temperature=20*celsius
        problem.droplet.mixture=get_pure_liquid("water")
        problem.gas.mixture=Mixture(get_pure_gas("air")+20*percent*get_pure_gas("water"),quantity="relative_humidity",temperature=problem.temperature)

        problem.surfactants["my_insoluble_surfactant"]=0.1*micro*mol/meter**2 # Add the surfactant

        # Set a pinned contact line to see the effect of the surfactants on the coffee-stain flow
        problem.contact_line = PinnedContactLine()

        problem.run(100*second,startstep=0.1*second,maxstep=5*second,outstep=True,temporal_error=1)


