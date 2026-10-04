# Tutorial failures under `--mpirun N --distribute`

Status: **diagnosed, not fixed.** Found while running the 0.2.2 release checklist's phase 6, which
asks for the tutorial scripts under `--omp N`, `--mpirun 4` and `--mpirun 4 --distribute`. The
complete results of that phase, on the 0.2.2 tree:

| configuration | result | wall clock |
|---|---|---|
| `--omp 4` | 139 passed, 1 skipped | 75 min |
| `--mpirun 4` (replicated) | 138 passed, 2 skipped | 116 min |
| `--mpirun 4 --distribute` | **133 passed, 5 failed**, 2 skipped | 373 min |

So this is specific to `--distribute`, not to MPI: the same scripts pass with four ranks as long as
every rank holds the whole mesh. (The 373 minutes are mostly the two 2 h timeouts in §5.1.)

These are **pre-existing** failures, not a 0.2.2 regression. `save_interface_state`, `_sorted_records`
and the guard that fires all predate `v0.2.1` (`cbe65852`), and nothing in 0.2.2's 79 commits touches
`Mesh::get_all_refinement_signatures` or the collective structure of `Problem.save_state`. The reason
they went unnoticed is coverage, not newness: `citools/nightly_develop.sh` runs the tutorial pass
serially and under `mpirun -n 4` **without** `--distribute` (`TUTORIAL_MPI_RANKS`), so nothing runs
these scripts in this configuration on any schedule.

The pytest suite is *not* blind to this area — `tests/mpi_state_file_worker.py` writes and reads state
files on an adaptively (and non-uniformly) refined distributed mesh, and passes. Whatever distinguishes
the failing scripts from that worker is therefore narrower than "distributed + adapted + state file";
the droplet_spread cases are moving-mesh problems with four nested interface meshes
(`domain/axis`, `domain/interface`, `domain/interface/substrate`, `domain/substrate`) driven through
`go_to_param` continuation, and the worker is a single rectangular template. Finding the minimal
difference is the first job of a fix, because that difference is what the regression test has to
contain.

Everything below was measured on draugr, 4 ranks, with the 0.2.2 tree.

---

## 1. What fails

All five failures of the completed `--mpirun 4 --distribute` pass, and the three distinct causes
behind them:

| script | symptom | cause |
|---|---|---|
| `Moving_Mesh/droplet_spread_hyperelastic_tangential_shift.py` | hangs forever at `t=8.0 s` | §2 + §3 |
| `Moving_Mesh/droplet_spread_marangoni_and_gravity.py` | `mpi4py.MPI.Exception: MPI_ERR_TRUNCATE` at `t=3.0 s` | §2 + §3 |
| `SpatioTemporal_PDEs/moffatt_eddies.py` | `ValueError: max() iterable argument is empty` on rank 0, while plotting | §5 |
| `Plotting_Interface/rising_bubble.py` | `TIMED OUT after 7200 s` | §5.1 |
| `Advanced_Linear_Dynamics/rising_bubble.py` | `TIMED OUT after 7200 s` | §5.1 |

Three causes, not five bugs. The two `droplet_spread` scripts are one defect in distributed state
saving; the two `rising_bubble` scripts are one defect reached through `refine_eigenfunction`; and
`moffatt_eddies` is its own thing in the plotting path.

Note that the pass ran while contending for the machine with other work, so the wall-clock times
here are not clean measurements of anything - only the pass/fail verdicts and the 7200 s timeouts
(which are a fixed budget, not a measurement) should be read as data.

The hang is the dangerous one, because it is silent and unbounded: all four ranks sit at 100 % CPU,
nothing is written, and there is no timeout. In the run that found it, `state_000008.dump` had been
created at 0 bytes while `state_000000`–`state_000007` were complete at ~26.8 KB each, and the log
ended at `Saving state .../state_000008.dump`. A user would see a job that never finishes.

## 2. The data defect: every unstamped root collapses onto `-1`

`Mesh::get_all_refinement_signatures` (`src/mesh.cpp:521`) keys its per-root map by
`global_base_index`:

```cpp
BulkElementBase *root = root_element_of(this->element_pt(ie));
if (!root) continue;
oomph::Tree *t = (re && re->tree_pt() ? re->tree_pt()->root_pt() : NULL);
auto it = tree_of_root.find(root->global_base_index);
```

`global_base_index` is the **root-only** field, documented in `src/elements.hpp:748` as "Only
meaningful on root elements; -1 until assigned". That is exactly the field
`Mesh::element_structural_key` (`src/mesh.cpp:474`) deliberately does **not** trust on a distributed
mesh — it prefers the stamped, inheritance-propagated `global_root_index`/`global_root_path` and only
falls back to the live root lookup:

```cpp
if (be && be->global_root_index >= 0 && be->global_root_path >= 0) { ... return true; }
BulkElementBase *r = root_element_of(e);
if (!r || r->global_base_index < 0) return false;
```

`elements.hpp:754` says why, in as many words: after `Problem::distribute()` the root element a leaf
belongs to "need not be present on this rank any more (its `object_pt()` is then NULL and the live
lookup returns -1), which made every element of a distributed mesh unaddressable".
`get_all_refinement_signatures` never got that treatment. So on a distributed, adapted mesh every
root whose `global_base_index` is unset lands in the single map entry `-1`, and the signatures of
unrelated trees are merged under one key.

The consistency check in `_sorted_records` (`pyoomph/meshes/meshstate.py:303`) then correctly notices
that the merged records disagree:

```
StateFileInconsistency: Two processes describe different refinement trees for root element -1.
A process must see the whole tree of every root it touches (its own elements plus the halo copies
of the others) for the refinement to be storable independently of the partition
```

The `-1` in that message is the tell. It is not two processes disagreeing about a real root; it is
several roots sharing the sentinel. Note that the earlier bulk guard at `meshstate.py:196`
(`elem_keys[:, 0] < 0`) does **not** fire first — the element keys are fine, because they go through
`element_structural_key` and its stamped path. Only the signature map is broken. Measured separately:
the interface path is clean too, `badkeys=0` on every rank for every interface mesh.

## 3. The structural defect: a rank-local raise inside a collective section

This is the part that turns a reportable inconsistency into a hang, and it is worth fixing on its own
even if §2 is fixed, because it will do the same for the next rank-local failure.

`_sorted_records` (`meshstate.py:286`) gathers, and then **only rank 0 keeps going**:

```python
gathered = _gather_blocks(...)      # collective: every rank calls it
if gathered is None:
    return None                     # every non-root rank leaves here
merged = ...
for i, r in enumerate(roots):
    ...
    elif check and not numpy.array_equal(previous, sig):
        raise StateFileInconsistency(...)   # ... so this can only ever raise on rank 0
```

The check therefore runs on rank 0 alone, and raises there. Meanwhile ranks 1..n-1 have already
returned and marched on through the rest of `Problem._define_state_file` — into the interface block,
whose `save_interface_state` calls `_gather_blocks` again, once per interface mesh. Rank 0 unwinds out
of `_define_state_file` instead and arrives at the rescue at the end of `save_state`:

```python
try:
    self._define_state_file(dump)
    ...
except BaseException as e:
    error=e
if distributed:
    mpi_share_any_failure(error,context="writing the state file")   # MPI_Allgather
```

So rank 0 is in `MPI_Allgather` while the others are in a later `MPI_Gather`. Measured, with the raise
replaced by a counter: rank 0 executed **12** `save_interface_state` calls where ranks 1, 2 and 3
executed **16** — exactly one round of the four interface meshes short.

Two collectives mismatched gives, depending on sizes and timing, either of the two observed symptoms:

* **hang** — `py-spy` on the four ranks of the first script:

  ```
  ranks A,B: mpi_share_any_failure (generic/mpi.py:251)        <- MPI_Allgather
             save_state (generic/problem.py:10934)
             error = StateFileInconsistency
  ranks C,D: _gather_blocks (meshes/meshstate.py:80)           <- MPI_Gather
             save_interface_state (meshes/meshstate.py:348)
             _define_state_file (generic/problem.py:10722)
             save_state (generic/problem.py:10924)
  ```

* **`MPI_ERR_TRUNCATE`** — the same mismatch, when the `allgather` happens to be matched against a
  `gather` payload of a different size. This is the better outcome of the two only because it stops.

The comment above `mpi_share_any_failure` claims this cannot happen — "a rank that failed would
otherwise leave the others waiting in the next collective - the run would hang instead of reporting
the failure". The net is real but it is in the wrong place: it wraps the section, while the section
itself contains collectives. It can only rescue a failure raised *between* collectives, not one raised
before a collective the other ranks are already committed to.

### 3.1 What a fix has to do

Not simply "move the raise". The requirement is that **the decision to fail is collective**: every
rank must learn of the inconsistency at the same point in the collective sequence, or none must.
Concretely, one of

* have `_sorted_records` return a verdict rather than raise, and agree on it with an `allreduce`
  immediately after the existing gather — same number of collectives on every rank, in the same order;
* or do the comparison on every rank (an `allgather` instead of `gather`) so the check is symmetric by
  construction, at the cost of the full records on every rank;
* or hoist the whole consistency check out of `_define_state_file` into a collective pre-pass.

Whatever the shape, **do not** just filter the inconsistent records away to make the symptom go
away. The records are the refinement of the mesh; writing a state file whose refinement is wrong
produces a file that loads without complaint and resumes a different problem. A hang is a bad failure;
a silently wrong state file is a worse one. That is also why this was not fixed under release time
pressure — see §6.

## 4. Reproducing

The isolated reproduction used for the measurements above, which leaves the checkout alone (the
editable install means editing `pyoomph/` in place would change the behaviour of a tutorial pass
running at the same time):

```bash
# a copy of the package plus the compiled core, so PYTHONPATH can shadow the editable install
cp -r $HOME/code/pyoomph/pyoomph /tmp/iso/pyoomph
cp $HOME/.local/lib/python3.12/site-packages/pyoomph/_pyoomph_core.abi3.so /tmp/iso/pyoomph/

# PYTHONNOUSERSITE keeps the editable-install .pth finder from winning; the user site-packages is
# put back on PYTHONPATH (as a plain directory, so its .pth files are NOT processed) for pygmsh etc.
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/iso:$HOME/.local/lib/python3.12/site-packages:$PETSC_DIR/$PETSC_ARCH_REAL/lib

cd <workdir with the script>
mpirun -n 4 python3 -u droplet_spread_marangoni_and_gravity.py --distribute
```

`droplet_spread_marangoni_and_gravity.py` is the better of the two reproducers: it fails at
`state_000003` (a few minutes) rather than `state_000008`, and it fails rather than hangs, so it needs
no `py-spy`. To see the raise site, print the traceback in `save_state`'s `except BaseException`; to
see the desynchronisation, count calls in `save_interface_state` per rank.

## 5. The plotting failure is separate

`moffatt_eddies.py` fails differently and has nothing to do with state files:

```
RuntimeError: MPI rank 0 failed while plotting (ValueError: max() iterable argument is empty).
```

from `perform_plot` → `plotting.py:156` → `run_with_global_mesh_data` (`meshdatamerge.py:519`). A
`max()` over something that is empty on rank 0 after the global mesh data merge, under `--distribute`
only. Here the rescue worked exactly as designed: `mpi_share_root_failure` reported which rank failed
and why, and the job ended instead of hanging — which is the contrast that makes §3 concrete. Not
investigated further.

### 5.1 `refine_eigenfunction` under `--distribute`: two hangs, one cause

**Both** scripts that call `Problem.refine_eigenfunction` under `--distribute` hang:
`Plotting_Interface/rising_bubble.py` and `Advanced_Linear_Dynamics/rising_bubble.py` (line 229,
`refine_eigenfunction(use_startvector=True)`). Each was killed by the harness's 2 h timeout, and
together they account for 4 of the 6 hours that pass took. Two independent scripts reaching the same
place makes this a property of the code path, not of either script.

Sampled on the first of them: 40+ minutes with all four ranks at 100 % CPU, inside **oomph-lib's
own** distributed equation-number synchronisation:

```
PMPI_Alltoall
oomph::Problem::copy_haloed_eqn_numbers_helper
oomph::Problem::synchronise_eqn_numbers      (src/thirdparty/oomph-lib/include/problem.cc:17220)
oomph::Problem::assign_eqn_numbers           (problem.cc:2374)
pyoomph::Problem::assign_eqn_numbers         (src/problem.cpp:1415)
  <- reapply_boundary_conditions (problem.py:5568)
  <- actions_before_stationary_solve -> solve -> refine_eigenfunction (problem.py:6754)
```

Two ranks sampled independently were in the *same* `Alltoall`, and its state files wrote correctly
(513 KB, so §2/§3 are not involved here).

This was first recorded as possibly benign: `refine_eigenfunction` drives `solve` in a loop, each
iteration re-numbering the equations, so an identical stack in two samples seconds apart is as
consistent with a hot loop of fast collectives as with one that never finishes. The harness settled
it — both scripts were killed by the 2 h timeout (`TIMED OUT after 7200 s`), the sampled one having
produced no output for the last 100 minutes. So it is a real hang, independent of §2/§3: the state
files here wrote correctly and the stack never leaves `assign_eqn_numbers`.

It is also the only one of the four that is inside **vendored oomph-lib** rather than pyoomph's own
code, which makes it the least likely to be fixable here and the most likely to need either an
upstream fix or a pyoomph-side avoidance of `refine_eigenfunction` under `--distribute`. Note that
`src/thirdparty/oomph-lib` is ~1500 commits behind upstream (see `INFO_oomph-lib`), so checking
whether upstream has already fixed this is the cheapest first step.

## 6. Why this is documented rather than fixed

Three reasons, recorded so the next person does not have to re-derive the decision:

1. It is pre-existing. 0.2.2 is no worse than 0.2.1 here, so shipping 0.2.2 does not make anything
   worse, and holding it does not make anything better.
2. §2 and §3 are both in the partition-stability machinery, where the failure mode of a wrong fix is a
   state file that loads fine and is wrong (§3.1). That is not work to do against a release deadline.
3. The pass is now complete and §1 is the whole list, so a fix has a fixed target: five scripts,
   three causes. Re-running it costs about six hours, four of which are the two timeouts in §5.1 —
   budget for that rather than assuming the pass has stalled.

## 7. Coverage, which is the actual root cause of the surprise

Whatever is done about §2 and §3, the gap that let a hang live in `--distribute` indefinitely is that
nothing runs it. Worth doing independently:

* a nightly tutorial pass with `--distribute`, or at least a subset of it (the `Moving_Mesh` bundle
  alone would have caught this);
* a pytest case covering whatever §1's scripts have that `tests/mpi_state_file_worker.py` does not —
  several nested interface meshes on a moving mesh is the obvious candidate. The existing worker
  already covers adapted + distributed + state file and passes, so "add a distributed state-file test"
  is not the gap; narrowing down which ingredient breaks it is;
* a guard against the general pattern: a rank-local `raise` inside a section that contains
  collectives. §3 is one instance; `grep` for `raise` between a `_gather_blocks` and a `return None`
  to find the others.
