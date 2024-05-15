/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2024  Christian Diddens & Duarte Rocha

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. 

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/


#ifdef OOMPH_HAS_MPI
#include "mpi.h"
#endif

#include <iostream>

#include <functional>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../oomph_lib.hpp"

#include "../exception.hpp"


namespace py = pybind11;

namespace pyoomph
{

    typedef void *fptr;

    class GeneralSolverCallback
    {
    public:
        unsigned last_nrow_local;
        virtual int solve_la_system_serial(int op_flag, int n, int nnz, int nrhs,
                                           py::array_t<double> &values, py::array_t<int> &rowind, py::array_t<int> &colptr,
                                           py::array_t<double> &rhs, int ldb, int transpose) { return -1; };

        virtual void solve_la_system_distributed(int op_flag, int allow_permutations, int n, int nnz_local, int nrow_local, int first_row, py::array_t<double> &values, py::array_t<int> &col_index, py::array_t<int> &row_start, py::array_t<double> &b, int nprow, int npcol, int doc, py::array_t<size_t> &data, py::array_t<int> &info){}; //,MPI_Comm comm

        virtual void metis_partgraph_kway(int nvertex, py::array_t<int> &xadj_Py, py::array_t<int> &adjacency_vector_Py, py::array_t<int> &vwgt_Py, py::array_t<int> &adjwgt, int wgtflag, int numflag, int nparts, py::array_t<int> &options_Py, py::array_t<int> &edgecut_Py, py::array_t<int> &part_Py) {}
    };

    class PyGeneralSolverCallback : public GeneralSolverCallback
    {
    public:
        /* Inherit the constructors */
        using GeneralSolverCallback::GeneralSolverCallback;

        int solve_la_system_serial(int op_flag, int n, int nnz, int nrhs,
                                   py::array_t<double> &values, py::array_t<int> &rowind, py::array_t<int> &colptr,
                                   py::array_t<double> &rhs, int ldb, int transpose) override
        {
            PYBIND11_OVERLOAD(
                int,                                                               /* Return type */
                GeneralSolverCallback,                                             /* Parent class */
                solve_la_system_serial,                                            /* Name of function in C++ (must match Python name) */
                op_flag, n, nnz, nrhs, values, rowind, colptr, rhs, ldb, transpose /* Argument(s) */
            );
        }

        void solve_la_system_distributed(int op_flag, int allow_permutations, int n, int nnz_local, int nrow_local, int first_row, py::array_t<double> &values, py::array_t<int> &col_index, py::array_t<int> &row_start, py::array_t<double> &b, int nprow, int npcol, int doc, py::array_t<size_t> &data, py::array_t<int> &info) override
        { //,MPI_Comm comm
            PYBIND11_OVERLOAD(
                void,                                                                                                                              /* Return type */
                GeneralSolverCallback,                                                                                                             /* Parent class */
                solve_la_system_distributed,                                                                                                       /* Name of function in C++ (must match Python name) */
                op_flag, allow_permutations, n, nnz_local, nrow_local, first_row, values, col_index, row_start, b, nprow, npcol, doc, data, info); //,comm
        }

        void metis_partgraph_kway(int nvertex, py::array_t<int> &xadj_Py, py::array_t<int> &adjacency_vector_Py, py::array_t<int> &vwgt_Py, py::array_t<int> &adjwgt_Py, int wgtflag, int numflag, int nparts, py::array_t<int> &options_Py, py::array_t<int> &edgecut_Py, py::array_t<int> &part_Py) override
        {
            PYBIND11_OVERLOAD(
                void,                  /* Return type */
                GeneralSolverCallback, /* Parent class */
                metis_partgraph_kway,  /* Name of function in C++ (must match Python name) */
                nvertex, xadj_Py, adjacency_vector_Py, vwgt_Py, adjwgt_Py, wgtflag, numflag, nparts, options_Py, edgecut_Py, part_Py);
        }
    };

    GeneralSolverCallback *g_solver_cb = NULL;
    void set_Solver_callback(GeneralSolverCallback *cb) { g_solver_cb = cb; }

}

extern "C"
{
    typedef void *fptr;
    int superlu(int *op_flag, int *n, int *nnz, int *nrhs,
                double *values, int *rowind, int *colptr,
                double *b, int *ldb, int *transpose, int *doc,
                fptr *f_factors, int *info)
    {
        py::array_t<double> values_arr;
        if (values)
            values_arr = py::array_t<double>({*nnz}, {sizeof(double)}, values, py::capsule(values, [](void *f) {}));

        py::array_t<int> rowind_arr;
        if (rowind)
            rowind_arr = py::array_t<int>({*nnz}, {sizeof(int)}, rowind, py::capsule(rowind, [](void *f) {}));

        py::array_t<int> colptr_arr;
        if (colptr)
            colptr_arr = py::array_t<int>({*n + 1}, {sizeof(int)}, colptr, py::capsule(colptr, [](void *f) {}));

        py::array_t<double> b_arr;

        if (b)
            b_arr = py::array_t<double>({*n}, {sizeof(double)}, b, py::capsule(b, [](void *f) {}));

        int nrhs_val = 0;
        if (nrhs)
            nrhs_val = *nrhs;
        int nnz_val = 0;
        if (nnz)
            nnz_val = *nnz;
        int ldb_val = 0;
        if (ldb_val)
            ldb_val = *ldb;

        int res = pyoomph::g_solver_cb->solve_la_system_serial(*op_flag, *n, nnz_val, nrhs_val, values_arr, rowind_arr, colptr_arr, b_arr, ldb_val, (transpose ? 1 : 0));

        *info = 0; // XXX Hack. Really check for errors here

        return res;
    }

#ifdef OOMPH_HAS_MPI
    void superlu_dist_global_matrix(int opt_flag, int allow_permutations,
                                    int n, int nnz, double *values,
                                    int *row_index, int *col_start,
                                    double *b, int nprow, int npcol,
                                    int doc, void **data, int *info,
                                    MPI_Comm comm)
    {
        throw_runtime_error("SUPERLU DIST IMPLEM GLOBAL MATRIX");
    }

    void superlu_dist_distributed_matrix(int opt_flag, int allow_permutations,
                                         int n, int nnz_local,
                                         int nrow_local, int first_row,
                                         double *values, int *col_index,
                                         int *row_start, double *b,
                                         int nprow, int npcol,
                                         int doc, void **data, int *info,
                                         MPI_Comm comm)
    {
        py::array_t<double> Py_values;
        if (values)
            Py_values = py::array_t<double>({nnz_local}, {sizeof(double)}, values, py::capsule(values, [](void *f) {}));

        py::array_t<int> Py_col_index;
        if (col_index)
            Py_col_index = py::array_t<int>({nnz_local}, {sizeof(int)}, col_index, py::capsule(col_index, [](void *f) {})); // TODO: NS

        py::array_t<int> Py_row_start;
        if (row_start)
            Py_row_start = py::array_t<int>({nrow_local + 1}, {sizeof(int)}, row_start, py::capsule(row_start, [](void *f) {})); // TODO: NS

        if (opt_flag == 1)
            pyoomph::g_solver_cb->last_nrow_local = nrow_local;
        py::array_t<double> Py_b;
        if (b)
            Py_b = py::array_t<double>({pyoomph::g_solver_cb->last_nrow_local}, {sizeof(double)}, b, py::capsule(b, [](void *f) {}));

        size_t *conv_data = ((size_t **)(data))[0];
        py::array_t<size_t> Py_Data;
        if (conv_data)
            Py_Data = py::array_t<size_t>({1}, {sizeof(size_t)}, conv_data, py::capsule(conv_data, [](void *f) {}));

        py::array_t<int> Py_Info;
        if (info)
            Py_Info = py::array_t<int>({1}, {sizeof(int)}, info, py::capsule(info, [](void *f) {}));

        pyoomph::g_solver_cb->solve_la_system_distributed(opt_flag, allow_permutations, n, nnz_local, nrow_local, first_row, Py_values, Py_col_index, Py_row_start, Py_b, nprow, npcol, doc, Py_Data, Py_Info); //,comm


    }

    void superlu_cr_to_cc(int nrow, int ncol, int nnz, double *cr_values,
                          int *cr_index, int *cr_start, double **cc_values,
                          int **cc_index, int **cc_start)
    {
        throw_runtime_error("SUPERLU DIST IMPLEM CR TO CC");
    }

    void METIS_PartGraphKway(int *nvertex_pt, int *xadj, int *adjacency_vector, int *vwgt, int *adjwgt, int *wgtflag_pt, int *numflag_pt, int *nparts_pt, int *options, int *edgecut, int *part)
    {
        int nvertex = *nvertex_pt; //=total_number_of_root_elements

        py::array_t<int> xadj_Py; // xadj [total_number_of_root_elements+1]
        if (xadj)
            xadj_Py = py::array_t<int>({nvertex + 1}, {sizeof(int)}, xadj, py::capsule(xadj, [](void *f) {}));

        py::array_t<int> adjacency_vector_Py; // adjacency_vector // [xadj[-1]]
        if (adjacency_vector)
            adjacency_vector_Py = py::array_t<int>({xadj[nvertex]}, {sizeof(int)}, adjacency_vector, py::capsule(adjacency_vector, [](void *f) {}));

        py::array_t<int> vwgt_Py; // vwgt //Assembly times [total_number_of_root_elements]
        if (vwgt)
            vwgt_Py = py::array_t<int>({nvertex}, {sizeof(int)}, vwgt, py::capsule(vwgt, [](void *f) {}));

        py::array_t<int> adjwgt_Py; // vwgt //Assembly times [total_number_of_root_elements]
        if (adjwgt)
            adjwgt_Py = py::array_t<int>({xadj[nvertex]}, {sizeof(int)}, adjwgt, py::capsule(adjwgt, [](void *f) {}));

        int wgtflag = *wgtflag_pt; // 0 not weighted, 2: vertex weighted
        int numflag = *numflag_pt; // 0
        int nparts = *nparts_pt;   // mpi.nproc

        py::array_t<int> options_Py; // options 10
        if (options)
            options_Py = py::array_t<int>({10}, {sizeof(int)}, options, py::capsule(options, [](void *f) {}));

        py::array_t<int> edgecut_Py; // edgecut [ = total_number_of_root_elements]
        if (edgecut)
            edgecut_Py = py::array_t<int>({nvertex}, {sizeof(int)}, edgecut, py::capsule(edgecut, [](void *f) {}));

        py::array_t<int> part_Py; // part : partition
        if (part_Py)
            part_Py = py::array_t<int>({nvertex}, {sizeof(int)}, part, py::capsule(part, [](void *f) {}));
        pyoomph::g_solver_cb->metis_partgraph_kway(nvertex, xadj_Py, adjacency_vector_Py, vwgt_Py, adjwgt_Py, wgtflag, numflag, nparts, options_Py, edgecut_Py, part_Py);
    }

    void METIS_PartGraphVKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *)
    {
        throw_runtime_error("METIS IMPLEM: METIS_PartGraphVKway");
    }

#endif
}

void PyReg_Solvers(py::module &m)
{
    py::class_<pyoomph::GeneralSolverCallback, pyoomph::PyGeneralSolverCallback /* <--- trampoline*/>(m, "GeneralSolverCallback")
        .def(py::init<>())        
        .def("metis_partgraph_kway",&pyoomph::GeneralSolverCallback::metis_partgraph_kway,py::arg("nvertex"), py::arg("xadj"), py::arg("adjacency_vector"), py::arg("vwgt"), py::arg("adjwgt"), py::arg("wgtflag"), py::arg("numflag"), py::arg("nparts"), py::arg("options"), py::arg("edgecut"), py::arg("part"))
        .def("solve_la_system_distributed",&pyoomph::GeneralSolverCallback::solve_la_system_distributed,py::arg("op_flag"), py::arg("allow_permutations"), py::arg("n"), py::arg("nnz_local"), py::arg("nrow_local"), py::arg("first_row"), py::arg("values"), py::arg("col_index"),py::arg("row_start"), py::arg("b"), py::arg("nprow"), py::arg("npcol"), py::arg("doc"), py::arg("data"), py::arg("info"))
        .def("solve_la_system_serial", &pyoomph::GeneralSolverCallback::solve_la_system_serial,py::arg("op_flag"),py::arg("n"),py::arg("nnz"),py::arg("nrhs"),py::arg("values"),py::arg("rowind"),py::arg("colptr"),py::arg("b"),py::arg("ldb"),py::arg("transpose"));

    m.def("set_Solver_callback", [](pyoomph::GeneralSolverCallback *cb)
          { pyoomph::g_solver_cb = cb; });
    m.def(
        "get_Solver_callback", []()
        { return pyoomph::g_solver_cb; },
        py::return_value_policy::reference);
        
   m.def("csr_rows_to_coo_rows",[](const py::array_t<int> &csr_rows,unsigned nzz,unsigned first_row) {
     auto coo_rows=py::array_t<int>({nzz});
	  int *in_buf = (int *)csr_rows.request().ptr;
     int *res_buff=(int*)coo_rows.request().ptr;
     unsigned i_row=0;
     for (unsigned count=0;count<nzz;count++)
     {
         if (count<(unsigned)in_buf[i_row+1])
         {
           res_buff[count] = first_row+i_row;
         }
         else
         {
          i_row++;
          res_buff[count] = first_row+i_row;
         }
     }
     return coo_rows;
   });
   
   
}
