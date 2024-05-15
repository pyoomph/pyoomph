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


from evaporating_droplet_insoluble_surfactants import * # Import the droplet evaporation problem with (insoluble) surfactants
from soluble_surfactants import * # and the previously defined soluble surfactant

if __name__=="__main__":
    with EvapDropletWithSurfactant() as problem:
        # problem.set_c_compiler("tcc")
        # Set exactly two geometric quantities
        problem.droplet.volume=100*nano*liter
        problem.droplet.contact_angle=60*degree

        # Ambient conditions and fluid mixtures
        problem.temperature = 20 * celsius
        problem.droplet.mixture = Mixture(get_pure_liquid("water") + 0.001 * get_pure_liquid("my_soluble_surfactant"))
        problem.gas.mixture = Mixture(get_pure_gas("air") + 20 * percent * get_pure_gas("water"),quantity="relative_humidity", temperature=problem.temperature)

        problem.surfactants["my_soluble_surfactant"] = 0.0 * micro * mol / meter ** 2  # No surfactants initially on the interface

        problem.contact_line = KwokNeumannContactLine(cl_speed_scale=None)  # Let it spread instantaneously due the surfactants


        problem.run(100*second,startstep=0.1*second,maxstep=5*second,outstep=True,temporal_error=1)


