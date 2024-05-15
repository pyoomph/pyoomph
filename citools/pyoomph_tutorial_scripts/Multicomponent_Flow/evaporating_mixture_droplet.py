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


from evaporating_pure_droplet import * # Load the previous example


if __name__=="__main__":
    with EvaporatingDroplet() as problem:
        #problem.set_c_compiler("tcc")
        # Set exactly two geometric quantities
        problem.droplet.volume=100*nano*liter
        problem.droplet.contact_angle=60*degree

        # Ambient conditions and fluid mixtures
        problem.temperature=20*celsius
        # Mix glycerol and water
        problem.droplet.mixture=Mixture(get_pure_liquid("water")+0.2*get_pure_liquid("glycerol"))
        problem.gas.mixture=Mixture(get_pure_gas("air")+20*percent*get_pure_gas("water"),quantity="relative_humidity",temperature=problem.temperature)
        problem.contact_line = KwokNeumannContactLine(cl_speed_scale=0.05 * meter / second)

        # Mix ethanol and water, in the gas phase, we have initial 0% ethanol water, but this is necessary for evaporation of ethanol
        #problem.droplet.mixture = Mixture(get_pure_liquid("ethanol") + 0.2 * get_pure_liquid("water"))
        #problem.gas.mixture = Mixture(get_pure_gas("air") + 60 * percent * get_pure_gas("water")+0*get_pure_gas("ethanol"),quantity="relative_humidity", temperature=problem.temperature)
        #problem.contact_line=YoungDupreContactLine(cl_speed_scale=10*milli*meter/second)


        problem.run(500*second,startstep=0.1*second,maxstep=2*second,outstep=True,temporal_error=1)

