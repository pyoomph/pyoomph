.. _secinstallationfurther:

Further practical software
--------------------------

VSCode as Python IDE
~~~~~~~~~~~~~~~~~~~~

As an editor and IDE for the Python scripts, we highly recommend downloading and installing Visual Studio Code from `code.visualstudio.com <code.visualstudio.com>`__. After downloading and installing, make sure to install the Python extensions. If you have installed pyoomph in a *Conda Environment*, make sure to use this environment for the Python interpreter. One rather important setting is to activate the option *Python* :math:`>` *Terminal: Execute In File Dir*. Thereby, the output of each simulation will be written in a subdirectory with the same name as the simulation script, without the .py extension. To easily find it, search for *python.terminal.executeInFileDir* in the settings search bar.

You also might want to install the extension PyLance, which allows for type checking and error highlighting during development.

Paraview to visualize the output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While pyoomph can be used without any visualization tools, e.g. by plotting the resulting data by hand or using the plotting framework of pyoomph, it is beneficial to install a viewer for VTU/PVD files, which are the default output files. The typical free software to visualize these files is Paraview, which can be downloaded for free at `www.paraview.org <www.paraview.org>`__. You can open all .vtu and .pvd files you find in the output folder of a pyoomph simulation.

Typical problems with the installation
--------------------------------------

I the following, there are some typical errors listed, which may occur after installing and trying pyoomph:

.. container:: tcolorbox
	
	**distutils.errors.DistutilsPlatformError: Unable to find vcvarsall.bat**
			
	This happens on Windows if you have not installed MS Build Tools (cf. :numref:`secinstallationmsbuild`). Either install it or set the compiler to TinyC, which can be done by calling the method ``set_c_compiler("tcc")`` of your problem instance or by passing the command line argument *--tcc*.
   
   
