# Changelog

## [0.2.2] - 2026-10-04

79 commits since 0.2.1. Three themes: symbolic code generation that no longer treats an expression
DAG as a tree, an analytic Hessian that reaches through multi-return callbacks, and MUMPS reachable
without PETSc. Plus a long tail of remesh, continuation and `--distribute` fixes.

### Added

- **MUMPS without PETSc**, via the separate `pyoomph_mumps` package: the linear solver `--mumps`
  (serial or natively distributed) and the eigensolver `--mumps_eigen` (Spectra with MUMPS
  factorising the shifted matrix, real and complex), so azimuthal and Floquet stability no longer
  need a full PETSc+SLEPc stack.
- **Analytic Hessians through `CustomMultiReturnExpression`**, which brings the UNIFAC and AIOMFAC
  activity coefficients, the log-conformation decompositions, `InvertMatrix` and the Cahn-Hilliard
  potentials within reach of the eigenvalue, bifurcation, Floquet and orbit workflows.
- **The exact UNIFAC Jacobian is emitted into the generated C** instead of being finite-differenced
  (`analytic_c_jacobian=False` restores the old body).
- **Liquid-mixture diffusivity estimation from the activity coefficients**,
  `MixtureLiquidProperties.set_estimate_diffusivities()`.
- **Symbolic comparisons and branching**: `var("time") < 2*second` yields a held relational
  expression, `conditional(cond, iftrue, iffalse)` compiles it to a C ternary, and comparisons
  combine with `~`, `&`, `|` and `^`.
- **`is_zero(..., tensors=True)`** recognises a zero vector or matrix, which the plain test never did.
- **NSCH: a C1 clamp for density and viscosity**, plus an optional temporal error estimator; the old
  branch-wise clamp made Newton two-cycle once the phase field overshot `|phi|>1`.
- **`--where` now also picks the state a `--runmode c` resumes from**, taking a `.dump` file, `i=N`,
  `t=X` or a bool expression over `step` and `time`.
- **Tracer collections living on an interface are written to state files** (state format 0.1.6);
  they were previously lost on `--runmode c`.

### Changed / Improved

- **Code generation treats the residual as the DAG it is**, memoising or converting to DAG walks the
  residual mappers, `MakeResidualSteady`, the subexpression-to-struct mapper, the parameter
  substitution of the `dResidual/dParameter` loop, the two collectors and the last preorder scans of
  the emission path -- a UNIFAC mass-transfer residual under `azimuthal_stability=True` previously
  never finished at all.
- **The unit split skips `collect_common_factors` above a term count**, which is where an
  evaporating droplet's never-returning split spent all of its samples.
- **`write_code` reports where it spends its time**, per residual set, with mapper and memo counters.
- **GiNaC patches** for `real_part`/`imag_part` of an integer power, `pow(0,0)`, a real and
  commutative `subexpression()` marker, number lookups in `ex`-keyed caches, and a unit under a
  symbolic exponent; the patch step re-runs when the patch set changes.
- **A spline macro element is parametrised by arclength** rather than the raw spline parameter, so
  refinement puts new nodes at the geometric midpoint of an edge after a remesh.
- **Units written into a data file no longer contain a space**, which made them unreadable to
  anything but pyoomph's own reader.
- **"There is no domain X defined in this mesh" after a remesh now says what it means**: the mesher
  was handed a geometry it could not fill.
- The documentation builds against the current Sphinx generation.

### Fixed

- **The arclength continuation tangent is carried correctly across a remesh or an adapt** -- five
  defects that each left the arclength invariant satisfied while the tangent pointed elsewhere, so
  nothing reported them.
- **Two defects in the viscoelastic equations on a moving mesh**, both invisible on the static,
  dimensionless meshes the tests used.
- **Axisymmetric pinch-off and coalescence beside a wall survive their own surgery**, and the
  degenerate branch of the zeta invertibility check is now pinned by tests.
- **Remeshing on an inverted element asks for a different mesh on each retry**, scaling
  `default_resolution` so that the gmsh size fields move too.
- **Under `--distribute`**: a 1-D submesh survives a rank that gets none of it, an unowned interface
  dof is skipped during interface generation, the nearest-node fallback searches the old mesh
  globally, and fragment counting agrees across ranks.
- **State files**: tracer collections are restored when the file brings its own mesh, and a template
  rebuilt from a stored `.msh` keeps its boundary corner sizes.
- **A resumed run leaves the output files an uninterrupted one would write**, rather than a time
  column that runs forwards and then jumps back.
- **Two reference leaks**: a delayed expansion pinned both itself and its callable, and a mixture
  stayed uncollectible once it had a multi-return activity model.
- `get_real_part()`/`get_imag_part()` split an argument that no longer contains a field.
- `numouts` produces at least one output.
- Docstring markup that rendered as artifacts on Read the Docs, NSCH's forward references under
  Sphinx, the clone URL on the two source-install pages, and the macOS stubs.

### Packaging & CI

- **The source distribution ships the sources, not the whole checkout**: 0.2.1 was 9.6 MB, most of it
  a `docs/` tree that `.gitignore` had already punched holes in.
- arm64 Macs are pointed at `pyoomph_mumps`, and the deprecated `x86_64` arch is dropped for the
  current Command Line Tools.
- `Programming Language :: Python :: 3.14` is declared, which the cp312 Stable-ABI wheel already
  covers and is smoke-tested on.
- Test helper scripts in `tests/` are importable, so `pytest *.py` does not collect them as tests.

## [0.2.1]

Released as urgent patch of 0.2.0. 

### Fixed

- **A two-sided interface could exhaust the machine's memory during the first assembly.** The pruned sparsity pattern (`prune_structural_zeros_by_field_coupling`, on by
  default since 0.2.0) let the two sides of a coupled interface ask each other for an answer each was
  still computing, which recursed without bound. 

## [0.2.0] - 2026-08-30

About six weeks and 940+ commits since 0.1.9. Four themes: h-adaptivity generalised from quads and
bricks to every element shape, MPI support extended to nearly everything that was serial-only,
substantially faster assembly and code generation, and several new physics modules. The compiled core
also moved from pybind11 to nanobind.

### Added

- **Adaptive refinement for every element shape**: triangles, tetrahedra, wedges, pyramids and mixed
  forests of them (2d and 3d), with hanging nodes, 2:1 balancing, unrefinement and mixed-order
  (Taylor-Hood, Crouzeix-Raviart, MINI) spaces. Shared nodes are identified topologically, never by
  position. `dev_docs/adaptive_refinement.md`.
- **Curved boundaries for every element type**, through one shape-generic `GenericMacroElement`, in 2d
  and 3d, from templates and from gmsh, including under `--distribute`. `dev_docs/macro_elements.md`.
- **Adaptivity across coupled domain interfaces**: two domains sharing an interface are kept
  conforming automatically, serially and distributed. `dev_docs/interface_refinement_coupling.md`.
- **Precomputed CSR sparsity, value-only re-assembly and solver symbolic reuse** (on by default),
  covering the mass matrix, the multi-assembly, every bifurcation tracker and the distributed exchange
  plan. `dev_docs/structural_assembly.md`.
- **Threaded element assembly, `--omp N`** (off by default), bit-identical to the serial loop, and
  combinable with MPI. macOS uses a GCD backend. `dev_docs/openmp_assembly.md`.
- **Static condensation of element-local dofs** (`StaticCondensation`, `Problem.condense_dofs`);
  51-65 % faster factorisation of a Crouzeix-Raviart system. Serial and distributed, experimental.
  `dev_docs/static_condensation.md`.
- **Unknowns on the interior-facet skeleton** (`at_internal_facets=True`), surviving adaptation,
  remeshing, `--distribute` and state files; HDG example in the tutorial.
  `dev_docs/internal_facet_fields.md`.
- **Eigenvalue problems under MPI through SLEPc**, with gather-to-root for the serial solvers and
  scipy/ARPACK. `dev_docs/mpi_eigenproblems.md`.
- **Remeshing, state files and periodic boundary conditions under `--distribute`.**
- **Periodic orbits, Floquet multipliers, bifurcation tracking, branch switching and deflation under
  MPI**, replicated and distributed.
- **Floquet multipliers by structured condensation** (the default), returning exactly `ndof`
  multipliers with no shift or magnitude threshold, plus a shift-inverted matrix-free route for large
  problems. `dev_docs/floquet_multipliers.md`.
- **An interactive bifurcation GUI**, with parameter switching, two-parameter loci, field plots,
  deflation and periodic orbits. `dev_docs/bifurcation_loci.md`.
- **Axisymmetric pinch-off and coalescence**: `AxisymmetricReconnection(rmin=..., distmin=...)` lets a
  free surface change its topology, nearly volume conserving, under MPI too. Replaces
  `AxisymmetricPinchoffAndCoalescence`. `dev_docs/axisymmetric_topological_changes.md`.
- **Viscoelastic flow in the log-conformation representation** (Oldroyd-B, Giesekus, PTT, FENE-CR,
  FENE-P). `dev_docs/viscoelastic_log_conformation.md`.
- **Residual-based stabilization** of Navier-Stokes and of scalar transport, replacing
  `pyoomph.equations.SUPG`. `dev_docs/stabilized_navier_stokes.md`.
- **Electrostatics, electrolytes and electrohydrodynamics**: Poisson-Boltzmann, Ohmic conduction,
  Poisson-Nernst-Planck, surface charge, Maxwell stress, electroosmotic slip.
  `dev_docs/electrohydrodynamics.md`.
- **Dissolved salts**: an ion and salt library, salt transport with the ambipolar diffusivity, salt
  retention under evaporation, salt-induced Marangoni flow, and AIOMFAC electrolyte activity
  coefficients. `dev_docs/salt_transport.md`, `dev_docs/aiomfac_electrolytes.md`.
- **Liquids may be given by concentration** in `Mixture(...)`, e.g.
  `water + 1*milli*molar*get_pure_liquid("surfactant")`.
- **Surfactant transport in one class**, conservative by default. `dev_docs/surfactant_transport.md`.
- **Tracer particles, rewritten**: bulk and interface, moving meshes, trails, payloads, remeshing and
  `--distribute`. `dev_docs/tracers.md`.
- **A mesh point locator** replacing `MeshAsGeomObject`, with closest-point-projection transfer for
  interfaces without a usable zeta and periodic zeta on closed loops. `dev_docs/mesh_point_locator.md`.
- **A content-addressed JIT cache** over deterministic code generation.
- **A Spectra eigensolver**, so targeting an eigenvalue no longer needs PETSc/SLEPc.
- **TQMesh as a second 2d meshing backend**, and OCC geometry support in `GmshTemplate`.
- **Second-order spatial derivatives**, with complete Hessians.
- **Vector fields for `DirichletBC` and `InitialCondition` as a whole**, e.g.
  `DirichletBC(velocity=vector(1,0))`.
- **`RemeshWhen(on_inverted_element=True)`**: remeshing as the response to a folded mesh, where a
  smaller time step cannot help.
- **Spatial error estimators on interfaces**, per-criterion normalisation, and adaptation towards a
  dof budget. `dev_docs/spatial_error_estimators.md`.
- **Per-block Jacobian symmetry and constancy are proven** and used to select the symmetric solver and
  eigensolver paths. `dev_docs/jacobian_block_flags.md`.
- New tutorial chapters: parallelization (OpenMP and MPI), spatial adaptivity, viscoelastic flow, the
  electric double layer, salts and ions, tracers, facet fields and HDG, static condensation, branch
  switching, the bifurcation GUI, coordinate systems, and the agent guide (`AGENTS.md`).

### Changed / Improved

- **The core extension is built with nanobind**, `src/pybind/` is now `src/nanobind/`, and CPython
  3.12+ gets a single Stable-ABI wheel. Python 3.9 is no longer supported.
- **The reference cycles that kept every `Problem` and `Mesh` alive are gone**, along with a family of
  leaks the migration exposed.
- **The whole Python package type-checks clean** under pyright and mypy, which fixed a number of
  genuine defects along the way.
- **Elemental assembly and generated code got faster**: non-hanging elements are dispatched to a
  hang-free path, unused shape families and hanging bookkeeping are skipped, loop-invariant buffer
  reads are bound to locals (11-15 % of an elemental Jacobian), Jacobian and Hessian entries are
  hoisted, `subexpression()` now reaches the analytic Hessian, and `-fno-math-errno` is a default
  flag. `dev_docs/assembly_overhead.md`.
- **An adaptation that refines and unrefines nothing no longer touches the problem**, so the equation
  numbering and the sparsity pattern survive it.
- **`Problem.initialise()` is ~27 % cheaper per dof.**
- **Tensor index conventions are now self-consistent, and two of them changed**: `contract(A,b)` is
  `A_ij*b_j` (was `A_ji*b_j`), and `div(T)[i]` is `d_j T_ij` (was `d_j T_ji`), which makes `div` the
  adjoint of `grad`. Symmetric tensors are unaffected.
- **The coordinate-system keyword is `coordsys` everywhere**; the five old spellings warn.
- **The normal-mode coordinate systems** gained tensor divergence and directional derivative, so
  `GCL=True` and the viscoelastic module can be combined with normal-mode stability analysis.
- **Framework-only methods on the equation and mesh-template classes are underscore-prefixed**, and a
  number of dead methods and write-only attributes were removed.
- **Arclength continuation got a mesh-independent inner product**, and `quick_mode` continues without
  an eigensolve per step.
- **Anything the MPI ranks must agree on is independent of the hash seed and of heap addresses.**
- **Removed**: the PFEM prototype, the standalone MUMPS and PaStiX solvers, `pyoomph.equations.SUPG`,
  the `wrong_strain` option, and the `MeshAsGeomObject` backend.

### Fixed

- **The 3x3 Gauss-Legendre knot table for 2d quads had two transposed digits**, making the rule
  asymmetric: results on quad meshes were wrong by ~1e-9 however fine the mesh.
- **Refining across a quad/triangle interface could tear the mesh** (and segfault), because the quad
  neighbour lookup could match a triangle node by accident.
- **Distributed adaptivity could refine different elements on different ranks**, silently corrupting
  the mesh: pyoomph's per-element error overrides ran rank-locally. Errors are now synchronised
  owner-to-halo, the 3d 2:1 balancing pass is globally consistent, and there is an opt-in halo
  consistency check.
- **`ConstrainPositionsToC1Space`, `ConstrainFieldsToC1Space` and mixed-order spaces** aborted or
  produced an inconsistent Jacobian on non-uniformly refined 3d and simplex meshes.
- **A `RemeshWhen` firing inside a solve corrupted the dof vector** under temporal adaptivity; the
  remesh is now deferred until the C++ call returns.
- **Inverted-element detection hung under MPI**; the verdict is now reduced across ranks.
- **`activate_bifurcation_tracking(blocksolve=True)` segfaulted** rather than being refused.
- **MPI dof and residual accessors read a distributed vector by global equation number**, returning
  garbage or overrunning the heap; `create_pressure_fixation()` pinned a different dof on every rank.
- **Several bifurcation defects**: the first Lyapunov coefficient's diagnostics and guards, four in
  `NormalModeBifurcationTracker`, the sign of the azimuthal tracker's `M_imag` term, and the tracked
  eigenvalue, which flipped the reported stability at every bifurcation.
- **Pardiso could return a wrong solve silently**; a symmetric factorization is now a pivoted one.
  macOS Accelerate trapped on singular matrices and never applied the deflation rescale.
- **The Windows JIT produced DLLs with no libm calls**: `tcc` exits 0 on an undeclared symbol, and
  `strcpy` was never declared for it.
- **`expr += number` produced complex constants**, which broke code generation far from the cause.
- Several coordinate-system errors: the azimuthal row of the axisymmetric tensor divergence, its
  directional tensor derivative in all three branches, divergences summing over coordinates the mesh
  does not have, and a transposed `vector_gradient` in the differential-geometry base class.

### Packaging & CI

- Stable-ABI (abi3) wheels for CPython 3.12+; the macOS wheels ship their own OpenMP runtime.
- New workflows: wheel tests across Python 3.11-3.14 and all platforms, tutorial scripts run against a
  fresh wheel, prebuilt PETSc/SLEPc artifacts, and a nightly runner for `develop` that also builds the
  documentation.
- The test suite is split into a fast default run and a `--full` run.


## [0.1.9] 

Roughly five months and 250+ commits since the 0.1.8 release. The two biggest
themes are a new mesh-element family (pyramids, wedges, and their bubble-enriched
tetrahedral relatives, all with proper 3D facet support) and a substantially
reworked build/release pipeline (CMake + scikit-build-core and source distributions). 

Alongside that:
several new solver backends, and a long tail of correctness fixes in the FEM core.

### Added

- **New element types**: pyramids and wedges, including C1/C2 variants and the
  bubble-enriched `TetraC1TB`/`Tetra3dC1TB`/`Tetra3dC2TB` tetrahedra, with
  proper facet-based boundary/interface detection (replacing boundary-node-only
  identification) and 3D Gmsh facet support.
- **New solver backends**: a macOS Accelerate-framework linear solver and
  eigensolver. PETSc gained automatic field-split   index sets and general solver improvements.
- **`pyoomph check`** (`python -m pyoomph check solver|eigen|compiler|all`):
  reworked solver selection, checking, and reporting, including install hints
  for missing optional dependencies (MKL/Pardiso, PETSc/SLEPc).
- **Parallel/MPI groundwork**: basic METIS-based mesh partitioning, basic load
  balancing, Dirichlet-by-matrix-manipulation as an alternative to the classical
  implementation, and distributed Dirichlet index spreading over MPI.
- **New physics/numerics**:  latent heat support for `PrescribedMassTransfer`, 
  time derivatives of integrals, matrix-valued `IntegralExpression`s, 
  an `InvertSymmetricMatrix`  multi-return expression, additional local dof constraints 
  (C1 confinement /  ALE constraining), an adaptive bifurcation tracker, 
  and `RemesherViaRecreation`.
- **Source distribution (sdist)** generation, wired into CI and verified with a
  full fresh-environment install-from-source test.
- New GCL and Rayleigh-Plateau-instability tutorials; an inverse-problem
  tutorial; `AGENTS.md`/agent-facing docs for AI-assisted development.
- numerical-data-file loading as numpy array with column and parameter information

### Changed / Improved

- Extensive internal refactoring of hanging-dof handling (new space-information
  structures, restructured hang buffers, streamlined `fill_hang_info_with_equations`)
  and of DG field handling.
- The compiled core extension moved from a top-level `_pyoomph` module to
  `pyoomph._pyoomph_core`.
- Removed unused oomph-lib thirdparty code (FSI, multi-domain, spectral
  elements, DG elements, spines, triangle meshes, the LAPACK QZ eigensolver) —
  a meaningful source-tree size reduction.
- Solid mechanics performance improved; 1D axisymmetry coordinate (polar) range
  reworked to `2x2` matrices in the vector gradients instead of `3x3` with a zero row/column.
- All of `src/` (excluding thirdparty code) commented and documented, with
  pybind11 binding docstrings added throughout.
- Large tutorial-documentation pass: numerous code blocks converted from
  downloadable scripts to `literalinclude`, several documentation gaps filled,
  full spellcheck.

### Fixed

- Interface-dof bugs breaking adaptive multi-physics interfaces, C1TB
  interfaces, and edge cases on interfaces with opposite orientation.
- Hele-Shaw factors corrected
- `CSplineInterpolator` bug; Jacobian sanity checking added to catch a class of
  silent bug where a misnamed override (e.g. `define_residual` instead of the
  correct `define_residuals`) would otherwise just never get called.
- residual/Jacobian checking (e.g. "has residual but no Jacobian row/col"); 
  DG element sorting bug in Jacobian assembly (wrong comparator);
  an accidentally-commented line that left Jacobian codegen empty in some
  cases.
- Higher-codimension (codim-3) code paths; vector gradients on higher
  codimensions.
- hanging dofs for 2D facets on 3D meshes; 
  finite differences and 2D hanging interface dofs on 3D meshes; 
- a load_state issue on adaptive meshes fixed
- an unsymmetric-mass-matrix case now warns instead of
  silently producing wrong results on scipy/ARPACK-based eigensolvers.
- A segfault in `MPI_Init` when extra CLI arguments are passed.

### Windows support

- Fixed `WinError 32` ("file in use") crashes in `pyoomph check` and any
  script that tears down a `Problem` while its temp/output directory is being
  deleted: the log file and persistent output files (`ODEFileOutput`,
  `IntegralObservableOutput`) are now closed proactively in `Problem.release()`
  instead of waiting for eventual garbage collection, matching the existing
  proactive DLL-unload behavior.
- Fixed a related `ValueError` ("path is on mount 'C:', start on mount 'D:'")
  when the code/output directory and working directory are on different
  drives, by falling back to an absolute compiler source path.
- Windows wheels now build via MSYS2/MinGW + CMake instead of the old
  `setup.py`-based flow; added an on-demand CI workflow.

### Packaging & CI

- Migrated the build backend from `setup.py` to CMake + scikit-build-core.
- Wheels are now built via `cibuildwheel` across Linux (manylinux), macOS
  (x86_64 and arm64), and Windows, looping over Python 3.10-3.15 (3.15 via
  `cpython-prerelease`) in a single job per platform.
- Added a dedicated workflow to prebuild static CLN/GiNaC as reusable
  artifacts, with auto-detection of current CLN/GiNaC versions from
  ginac.de (falling back to known-good pinned versions if that lookup fails).
- Fixed a `.gitignore` bug (`*.txt` was silently excluding the tracked root
  `CMakeLists.txt`, among others, from the sdist) that had made the sdist
  fundamentally unbuildable.
