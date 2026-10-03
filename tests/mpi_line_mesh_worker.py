# Worker for tests/test_mpi_line_mesh.py -- launched under `mpirun ... [--distribute]`.
#
# A 2-D mesh with a 1-D mesh beside it that shares no node with it. METIS minimises the edge cut
# over the global mesh, so a disconnected component of a few elements lands entirely on one rank,
# and the others retain none of it. Mesh::distribute() still calls setup_tree_forest() on every
# rank - deliberately, because the call is collective - and for a 1-D mesh that builds a
# BinaryTreeForest from an empty vector of tree roots.

import argparse
import json
import traceback

from pyoomph import Problem, DirichletBC
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh


class PoissonPlusLine(Problem):
    """A partitionable 2-D mesh, plus a small 1-D mesh that is not connected to it."""

    def __init__(self, N=8, Nline=5):
        super().__init__()
        self.N = N
        self.Nline = Nline

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        eqs = PoissonEquation(name="u", source=1)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")

        # Deliberately disconnected: it shares no node with "domain", which is what makes METIS put
        # all of it on one rank and leaves the others with an empty forest to set up.
        self.add_mesh(LineMesh(N=self.Nline, size=1.0, name="line"))
        line = PoissonEquation(name="v", source=1)
        line += DirichletBC(v=0) @ "left"
        line += DirichletBC(v=0) @ "right"
        self.add_equations(line @ "line")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        with PoissonPlusLine() as p:
            p.set_output_directory(args.outdir)
            p.quiet()
            p.max_refinement_level = 0
            p.solve()
            payload.update({
                "distributed": bool(p.is_distributed()),
                "ndof": int(p.ndof()),
                # How much of the 1-D mesh this rank ended up with. Zero on at least one rank is the
                # whole point of the test.
                "line_elements": int(p.get_mesh("line").nelement()),
                "domain_elements": int(p.get_mesh("domain").nelement()),
            })
    except BaseException as e:  # noqa: BLE001
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
