.. _installmumps:

Optional installation of MUMPS without PETSc (pyoomph_mumps)
------------------------------------------------------------

`MUMPS <https://mumps-solver.org/>`__ is a sparse direct solver that pivots, which is exactly what pyoomph's matrices usually require (Lagrange multipliers, incompressibility constraints, etc. put zeros on the diagonal). It is reachable through PETSc/SLEPc as ``petsc_mumps``/``slepc_mumps`` (see :numref:`petscslepc`), but that means building and maintaining a full PETSc+SLEPc stack -- and, for azimuthal or Floquet stability analysis, two of them, a real one and a complex one.

The separate package `pyoomph_mumps <https://github.com/pyoomph/pyoomph_mumps>`__ talks to ``dmumps`` and ``zmumps`` directly instead, without PETSc in between. Once it is installed, pyoomph registers two additional backends, both named ``mumps``:

*  the linear solver ``mumps``, i.e. ``problem.set_linear_solver("mumps")`` or the command line flag ``--mumps``. It is serial, or natively distributed under MPI (each rank supplying its own row block), and it reuses the analysis phase when the sparsity pattern does not move, so a transient run pays for the ordering only once.
*  the eigensolver ``mumps``, i.e. ``problem.set_eigensolver("mumps")`` or the command line flag ``--mumps_eigen``. This is pyoomph's built-in Spectra eigensolver with MUMPS supplying the shift-and-invert factorization, in real *and* complex arithmetic, i.e. azimuthal and Floquet stability analysis are covered as well.

Without the package, nothing changes: ``pyoomph.solvers.mumps`` just raises an ``ImportError`` and neither backend is registered.

.. note::

   This is particularly relevant on **Mac with Apple silicon (arm64)**, where Intel's ``mkl`` package -- and hence the fast MKL Pardiso solver -- is not available from ``pip`` at all, while Apple's deprecation of x86_64 makes the Rosetta 2 detour (see :numref:`installonmac`) progressively harder with each system update. The remaining native alternative, Apple's ``Accelerate`` framework, is too slow for larger problems. So on arm64 Macs you want either PETSc/MUMPS (:numref:`petscslepc`) or ``pyoomph_mumps``, the latter being considerably less to install if you do not need PETSc for anything else.

Installation
~~~~~~~~~~~~

The package is not on PyPI, so install it from its repository:

.. code:: bash

	git clone https://github.com/pyoomph/pyoomph_mumps.git
	cd pyoomph_mumps
	python -m pip install .

By default, this downloads and builds MUMPS (and a reference BLAS/LAPACK, if the machine has none) from source, which requires a **Fortran compiler**. On a Mac, you can get one via `Homebrew <https://brew.sh>`__:

.. code:: bash

	brew install gcc

If MUMPS is already present on your system (e.g. by ``libmumps-dev`` on Ubuntu or a module-loaded MUMPS on a cluster), the build is much faster when you point at it instead:

.. code:: bash

	python -m pip install . --config-settings=cmake.define.PYOOMPH_MUMPS_DOWNLOAD=OFF

Either way, the MUMPS tree must provide **both** arithmetics, i.e. ``libdmumps`` and ``libzmumps``, since the complex one is what the azimuthal and Floquet stability analyses invert. Further options (where to look for MUMPS and BLAS, OpenMP within the factorization, etc.) are listed in the README of the repository.

.. warning::

	The option ``PYOOMPH_MUMPS_USE_MPI`` must match the way pyoomph itself has been built, i.e. it must be ``ON`` for an MPI-enabled pyoomph and ``OFF`` otherwise. A serial MUMPS links a stub library defining its own ``MPI_Init``, which in the same process as an MPI-enabled pyoomph leads to a hang or a wrong communicator rather than to an error message. pyoomph therefore compares both settings when importing the backend and refuses a mismatched pair right away. The ``build_for_develop.sh`` script of ``pyoomph_mumps`` reads the setting from pyoomph and matches it automatically.

Usage
~~~~~

Nothing has to be activated: whenever MKL Pardiso and PETSc/MUMPS are absent, the automatic solver selection prefers the ``mumps`` backends over SuperLU, scipy and ``Accelerate``. To select them explicitly anyhow, e.g. to override a choice made in the driver code, either do it in python

.. code:: python

	problem.set_linear_solver("mumps")
	problem.set_eigensolver("mumps")

or pass the flags on the command line (the two registries are separate, so the solver names can coincide while the flags cannot):

.. code:: bash

	python my_simulation.py --mumps --mumps_eigen
