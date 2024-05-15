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


from evaporating_droplet_thermal_marangoni import * # Import the problem
from pyoomph.output.plotting import *


# Specialize the MatplotlibPlotter for the droplet problem
class DropPlotter(MatplotlibPlotter):
    # This method defines the entire plot
    def define_plot(self):    
        p=self.get_problem() # we can get access to the problem by get_problem()
        r=p.droplet.base_radius # and hence can access the droplet base radius
        xrange=1.5*r # in x-direction, the image will be from [-1.5*r : 1.5*r ]
        self.background_color="darkgrey" # background color

        # First step: Set the field of view (xmin,ymin,xmax,ymax)
        self.set_view(-xrange,-0.165*xrange,xrange,0.9*xrange)

        # Second step: add colorbars with different colormaps at different positions
        # with 'factor', you can control the multiplicative factor (i.e. mm/s requires a factor of 1000 to convert the m/s to mm/s)
        # with 'offset', you can control an offset, e.g. for temperature K to C, offset=-273.15
        cb_v=self.add_colorbar("velocity [mm/s]",cmap="seismic",position="bottom right",factor=1000)
        cb_vap=self.add_colorbar("water vapor [wt%]",cmap="Blues",position="top right",factor=100)
        cb_T=self.add_colorbar(r"temperature [C]",cmap="coolwarm",position="bottom left",offset=-273.15)

        # Now, we can add all kinds of plots
        # plot the velocity (magnitude, since it is a vector) of the droplet domain
        self.add_plot("droplet/velocity",colorbar=cb_v)
        # plot the droplet and gas temperature, but mirrored at the x-axis
        self.add_plot("droplet/temperature", colorbar=cb_T,transform="mirror_x")
        self.add_plot("gas/temperature", colorbar=cb_T, transform="mirror_x")
        # add velocity arrows on both sides
        self.add_plot("droplet/velocity", mode="arrows",linecolor="green",transform=["mirror_x",None])

        # depending on the gas flow, add also the gas velocity and the arrows
        if self.get_problem().gas.solve_flow:
            self.add_plot("gas/velocity", colorbar=cb_v)
            self.add_plot("gas/velocity", mode="arrows", linecolor="green", transform=["mirror_x", None])
        # otherwise, just plot the vapor
        self.add_plot("gas/massfrac_water",colorbar=cb_vap,transform=None if self.get_problem().gas.solve_flow else ["mirror_x",None])

        # at the interface lines
        self.add_plot("droplet/droplet_gas",linecolor="yellow",transform=["mirror_x",None])
        self.add_plot("droplet/droplet_substrate",transform=["mirror_x",None])
        self.add_plot("gas/gas_substrate",transform=["mirror_x",None])

        # For the evaporation rate, we require an arrow key, again with a factor 1000, since we convert from kg to g per m^2s
        ak_evap=self.add_arrow_key(position="top right",title="evap. rate water [g/(m^2s)]",factor=1000)
        # We can hide instances by setting invisible=True:
        #       ak_evap.invisible=True
        # or we can move it a bit relative to the "position" by the xmargin and ymargin
        # or with xshift and yshift. All are in graph coordinates, i.e. 1 means the width/height of the entire image
        ak_evap.ymargin+=0.2
        ak_evap.xmargin *= 2

        # add the evaporation arrows at the interface, both sides
        arrs=self.add_plot("droplet/droplet_gas/masstrans_water",arrowkey=ak_evap,transform=["mirror_x",None])

        # and a time label and a scale bar
        self.add_time_label(position="top center")
        self.add_scale_bar(position="bottom center").textsize*=0.7


if __name__=="__main__":
    with EvaporatingDroplet() as problem:
        problem.plotter=DropPlotter(problem) # set the plotter and pass the problem itself
        # The rest is the same as before
        # .....
        #

        # Changing the file extension (also a list works, e.g. ["pdf","png"]
        problem.plotter.file_ext = "pdf"
        
        # Changing e.g. the dpi or default settings of the velocity arrows:
        problem.plotter.dpi *= 1.5
        # problem.plotter.defaults("arrows").arrowdensity /= 2
        # problem.plotter.defaults("arrows").arrowlength *= 1.5        

        # Set exactly two geometric quantities
        problem.set_c_compiler("tcc")
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

        problem.droplet.contact_angle = 60 * degree


        problem.run(500*second,startstep=0.1*second,maxstep=0.25*second,outstep=True,temporal_error=1)
