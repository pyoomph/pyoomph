# What a continued run leaves in its output files

`--runmode c` resumes from a state file. The state is restored correctly; what this document is about
is the *output* the interrupted run had already written past that point, and what happens to it.

## The problem

A run is killed, or is resumed from an earlier state with `--where`. Either way the output directory
holds files describing time steps the resumed run is about to compute again. They used to be left
alone and appended to:

```
box.txt, resumed with --where i=3:
  0.0 ... 0.9 1.0   0.4 0.5 0.6 0.7 0.8 0.9 1.0
                    ^^^ the time column jumps backwards mid-file
```

This was always reachable — a killed run's last output is normally past its last state — but it takes
one command to hit hard since `--where` can select any state to resume from.

## The contract

**A resumed run leaves the output files an uninterrupted run would have written.** For the growing
text files that is literal: `tests/test_state_file_restart.py::test_a_resumed_run_leaves_the_output_of_an_uninterrupted_one`
byte-compares them.

Three mechanisms, all driven from `continue_info`:

| what | where | how |
|---|---|---|
| growing text files | `_ODEFileOutput`, `_IntegralObservableOutput` | rewritten up to the resume time before appending |
| one file per step | `_TextOutput`, `_OutputTxtAlongLine`, `_GridFileOutput`, `_MeshFileOutput` | files from the resume step upwards are deleted |
| the PVD index | `_MeshFileOutput` | `DataSet` entries from the resume step upwards are dropped |

## Four traps

These are the things that are not obvious from the code and that a change here has to respect.

**1. `continue_info is not None` does not mean "continuing".** The same argument is handed over by
`Problem.redefine_problem` as `{"redefined":True}` and synthesised by
`_ODEFileOutput.change_output_directory` as `{"TODO":...}`. Neither is a resume and neither carries a
step or a time, so a trim keyed off `is None` would `KeyError`, or silently truncate a redefined
problem's output. Everything here goes through `_BaseOutputter.is_resuming()`, which tests for the
presence of `"outstep"`. `tests/test_state_file_restart.py::test_redefining_the_problem_is_not_a_continue`
pins it, and `tests/test_mpi_orbit_output.py` covers the `change_output_directory` path.

**2. `outstep` is the step that will be written NEXT.** `Problem.initialise` increments
`_output_step` in its continue branch before `_init_output()` runs. Stale therefore means
`step >= outstep`, not `>`.

**3. A step number appears in no file's contents, and rows are not one per step.** `output_every_step`
defaults to `True`, so extra rows are written between output steps without `_output_step` moving.
Counting rows cannot locate the resume point; only the time column can. That is why a file without one
can only warn.

**4. The time in these files is dimensional, in seconds.** `_BaseOutputter.get_time` writes
`get_current_time(dimensional=True, as_float=True)`, which reduces a dimensional time with
`float(t/second)`. `continue_info["floattime"]` used to be built with `as_float=False`, making it an
`Expression` and a mere duplicate of `dimtime` — harmless only because none of the four values was
read by anything. It is now the float it claims to be.

## The tie rule, and what cannot be trimmed

`trim_numerical_text_file` (`pyoomph/utils/num_text_out.py`) keeps every row up to and including the
**last** row at the resume time. A stationary or continuation run writes several rows at the same time
value, and all of them belong to the part being kept. Surviving rows are copied **verbatim**: going
through `numpy.loadtxt` would reformat every number and the byte-comparison above would be worthless.

An `ODEFileOutput(first_column=[])` has no time column at all. Nothing in such a file says when a row
was written, so there is no way to tell the stale rows from the rest. It **warns, names the file, and
appends** — which is what happened before any of this existed, and destroys nothing.

## Deleting the per-step files

`delete_files_from_previous_simulation(from_step)` lives on `_BaseOutputter` and walks
`get_filename(step)` upwards until a step has no files. Before this it existed in three identical
copies with no caller at all.

It runs **on rank 0 only**. A per-step name carries no rank, so it is the same on every rank: if all of
them deleted it, the one that lost the race would read its `FileNotFoundError` as "the run did not get
this far" and leave every later step behind. `_MeshFileOutput` is the exception — it writes
`<trunk>_<step>_<rank>.vtu` under `--distribute` — so its `get_filename` returns every rank's name and
rank 0 removes them all. Only paths an outputter names itself are ever removed; nothing globs the
output directory.

## Distributed outputs: the state of it

`TODO.txt` listed `TextFileOutput`, `ODEFileOutput`, `ExtremumObservables`, states and continuing as
open under `--distribute`. They are not — see `distributed_remeshing.md` §5 for the extremum and
text-file fixes, and `tests/test_mpi_observables.py`. What was actually broken:

* **`TextFileOutputAlongLine` and `GridFileOutput`** interpolate onto points given in the coordinates
  of the whole domain, but took this rank's partition and wrote to a file name carrying no rank. The
  ranks overwrote one another and whichever wrote last kept only the points inside its own share — at
  4 ranks the line output came out with **11 of its 21 points**, with no error and no warning. Both now
  request `global_mesh=True` and return on the ranks that only contributed to the merge, exactly as
  `_TextOutput` did already. `global_mesh=False` restores the old behaviour for a genuinely per-rank
  output, in which case give each rank its own file name.
* **`plot_in_dedicated_process`** had no rank guard at all. Every rank spawned its own replotter, all
  of them truncated the same `_dedicated_plotter_log.txt`, and under `--distribute` the ranks that do
  not write the state file handed their child a path that need not exist. Rank 0 spawns it now, and
  `_dedicated_plotter_active` — replicated, unlike the `Popen` handle — decides whether this process
  plots, because `perform_plot` is collective and a rank branching on its own handle would have rank 0
  skip the plot while the others waited for it forever.

Two further things about that child, both of which meant it produced **nothing at all**, in serial as
well as under MPI:

* Nobody ever sent the `__exit__` line it listens for, and nobody waited for it, so it was killed
  mid-queue. `Problem.release` now shuts it down and waits.
* It inherited the launcher's environment. `OMPI_COMM_WORLD_SIZE`, the ORTE contact URI and the PMIx
  equivalents make a child's own `MPI_Init` **join the parent's job**: it believed it was rank 0 of an
  n-rank world and blocked in its first collective waiting for peers that were busy solving. It is
  started with those variables stripped (`_environment_without_mpi`).
