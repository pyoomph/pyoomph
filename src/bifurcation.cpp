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

/*
##################################
This file is strongly based  on the oomph-lib library (see thirdparty/oomph-lib/include/assembly_handler.h)
##################################
*/

// OOOMPH-LIB includes
#include "bifurcation.hpp"
#include "problem.hpp"
#include "elements.h"
#include "problem.h"
#include "mesh.h"

#include "elements.hpp"

using namespace oomph;

namespace pyoomph
{
  MyHopfHandler::MyHopfHandler(Problem *const &problem_pt,
                               double *const &parameter_pt) : Solve_which_system(0), Parameter_pt(parameter_pt), Omega(0.0)
  {

    call_param_change_handler = false;
    eigenweight = 1.0;
    // Set the problem pointer
    Problem_pt = problem_pt;
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();

    // create the linear algebra distribution for this solver
    // currently only global (non-distributed) distributions are allowed
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);

    // Resize the vectors of additional dofs
    Phi.resize(Ndof);
    Psi.resize(Ndof);
    C.resize(Ndof);
    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    // Calculate the value Phi by
    // solving the system JPhi = dF/dlambda

    // Locally cache the linear solver
    LinearSolver *const linear_solver_pt = problem_pt->linear_solver_pt();

    // Save the status before entry to this routine
    bool enable_resolve = linear_solver_pt->is_resolve_enabled();

    // We need to do a resolve
    linear_solver_pt->enable_resolve();

    // Storage for the solution
    DoubleVector x(dist_pt, 0.0);

    // Solve the standard problem, we only want to make sure that
    // we factorise the matrix, if it has not been factorised. We shall
    // ignore the return value of x.
    linear_solver_pt->solve(problem_pt, x);

    // Get the vector dresiduals/dparameter
    problem_pt->get_derivative_wrt_global_parameter(parameter_pt, x);

    // Copy rhs vector into local storage so it doesn't get overwritten
    // if the linear solver decides to initialise the solution vector, say,
    // which it's quite entitled to do!
    DoubleVector input_x(x);

    // Now resolve the system with the new RHS and overwrite the solution
    linear_solver_pt->resolve(input_x, x);

    // Restore the storage status of the linear solver
    if (enable_resolve)
    {
      linear_solver_pt->enable_resolve();
    }
    else
    {
      linear_solver_pt->disable_resolve();
    }

    // Normalise the solution x
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += x[n] * x[n];
    }
    length = sqrt(length);

    // Now add the real part of the null space components to the problem
    // unknowns and initialise it all
    // This is dumb at the moment ... fix with eigensolver?
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Phi[n]);
      C[n] = Phi[n] = -x[n] / length;
    }

    // Set the imaginary part so that the appropriate residual is
    // zero initially (eigensolvers)
    for (unsigned n = 0; n < Ndof; n += 2)
    {
      // Make sure that we are not at the end of an array of odd length
      if (n != Ndof - 1)
      {
        Psi[n] = C[n + 1];
        Psi[n + 1] = -C[n];
      }
      // If it's odd set the final entry to zero
      else
      {
        Psi[n] = 0.0;
      }
    }

    // Next add the imaginary parts of the null space components to the problem
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Psi[n]);
    }
    // Now add the parameter
    problem_pt->GetDofPtr().push_back(parameter_pt);
    // Finally add the frequency
    problem_pt->GetDofPtr().push_back(&Omega);

    // rebuild the dof dist
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof * 3 + 2, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);

    // delete the dist_pt
    delete dist_pt;
  }

  //====================================================================
  /// Constructor: Initialise the hopf handler,
  /// by setting initial guesses for Phi, Psi, Omega  and calculating Count.
  /// If the system changes, a new  handler must be constructed.
  //===================================================================
  MyHopfHandler::MyHopfHandler(Problem *const &problem_pt,
                               double *const &parameter_pt,
                               const double &omega,
                               const DoubleVector &phi,
                               const DoubleVector &psi) : Solve_which_system(0), Parameter_pt(parameter_pt), Omega(omega)
  {

    call_param_change_handler = false;
    eigenweight = 1.0;

    // Set the problem pointer
    Problem_pt = problem_pt;
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();

    // Resize the vectors of additional dofs
    Phi.resize(Ndof);
    Psi.resize(Ndof);
    C.resize(Ndof);
    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    // Normalise the guess for phi
    double length = 0.0, GrGr = 0.0, GrGi = 0.0, GiGi = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      GrGr += phi[n] * phi[n];
      GrGi += phi[n] * psi[n];
      GiGi += psi[n] * psi[n];
    }

    /// Correct the initial eigenvector guess
    double rotation_vector = atan2(GrGr - GiGi + sqrt(pow(GrGr - GiGi, 2) + 4 * pow(GrGi, 2)), 2 * GrGi);
    //  std::cout << "FROM C: Gs: " << GrGr << "  " << GiGi << " " << GrGi << "  " << rotation_vector << std::endl;

    // Now add the real part of the null space components to the problem
    // unknowns and initialise it all
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Phi[n]);
      C[n] = Phi[n] = phi[n] * cos(rotation_vector) - psi[n] * sin(rotation_vector);
      length += Phi[n] * Phi[n];
    }
    length = sqrt(length);

    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Psi[n]);
      Psi[n] = phi[n] * sin(rotation_vector) + psi[n] * cos(rotation_vector);
    }

    // Normalize it
    for (unsigned n = 0; n < Ndof; n++)
    {
      C[n] /= length;
      Phi[n] /= length;
      Psi[n] /= length;
    }

    // Now add the parameter
    problem_pt->GetDofPtr().push_back(parameter_pt);
    // Finally add the frequency
    problem_pt->GetDofPtr().push_back(&Omega);

    // rebuild the Dof_distribution_pt
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof * 3 + 2, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  //=======================================================================
  /// Destructor return the problem to its original (non-augmented) state
  //=======================================================================
  MyHopfHandler::~MyHopfHandler()
  {
    // If we are using the block solver reset the problem's linear solver
    // to the original one
    BlockHopfLinearSolver *block_hopf_solver_pt =
        dynamic_cast<BlockHopfLinearSolver *>(Problem_pt->linear_solver_pt());
    if (block_hopf_solver_pt)
    {
      // Reset the problem's linear solver
      Problem_pt->linear_solver_pt() = block_hopf_solver_pt->linear_solver_pt();
      // Delete the block solver
      delete block_hopf_solver_pt;
    }
    // Now return the problem to its original size
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  void MyHopfHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n] *= ew / eigenweight;
      Psi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  //=============================================================
  /// Get the number of elemental degrees of freedom
  //=============================================================
  unsigned MyHopfHandler::ndof(GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
      // Full augmented system
    case 0:
      return (3 * raw_ndof + 2);
      break;
      // Standard non-augmented system
    case 1:
      return raw_ndof;
      break;
      // Complex system
    case 2:
      return (2 * raw_ndof);
      break;

    default:
      throw OomphLibError("Solve_which_system can only be 0,1 or 2",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  //=============================================================
  /// Get the global equation number of the local unknown
  //============================================================
  unsigned long MyHopfHandler::eqn_number(GeneralisedElement *const &elem_pt,
                                          const unsigned &ieqn_local)
  {
    // Get the raw value
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn;
    if (ieqn_local < raw_ndof)
    {
      global_eqn = elem_pt->eqn_number(ieqn_local);
    }
    else if (ieqn_local < 2 * raw_ndof)
    {
      global_eqn = Ndof + elem_pt->eqn_number(ieqn_local - raw_ndof);
    }
    else if (ieqn_local < 3 * raw_ndof)
    {
      global_eqn = 2 * Ndof + elem_pt->eqn_number(ieqn_local - 2 * raw_ndof);
    }
    else if (ieqn_local == 3 * raw_ndof)
    {
      global_eqn = 3 * Ndof;
    }
    else
    {
      global_eqn = 3 * Ndof + 1;
    }
    return global_eqn;
  }

  //==================================================================
  /// Get the residuals
  //=================================================================
  void MyHopfHandler::get_residuals(GeneralisedElement *const &elem_pt,
                                    Vector<double> &residuals)
  {
    // Should only call get residuals for the full system
    if (Solve_which_system == 0)
    {
      // Need to get raw residuals and jacobian
      unsigned raw_ndof = elem_pt->ndof();

      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);
      // Get the basic residuals, jacobian and mass matrix
      elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);

      // Initialise the pen-ultimate residual
      residuals[3 * raw_ndof] = -1.0 /
                                (double)(Problem_pt->mesh_pt()->nelement()) * eigenweight;
      residuals[3 * raw_ndof + 1] = 0.0;

      // Now multiply to fill in the residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + i] = 0.0;
        residuals[2 * raw_ndof + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned global_unknown = elem_pt->eqn_number(j);
          // Real part
          residuals[raw_ndof + i] +=
              jacobian(i, j) * Phi[global_unknown] + Omega * M(i, j) * Psi[global_unknown];
          // Imaginary part
          residuals[2 * raw_ndof + i] +=
              jacobian(i, j) * Psi[global_unknown] - Omega * M(i, j) * Phi[global_unknown];
        }
        // Get the global equation number
        unsigned global_eqn = elem_pt->eqn_number(i);

        // Real part
        residuals[3 * raw_ndof] += (Phi[global_eqn] * C[global_eqn]) /
                                   Count[global_eqn];
        // Imaginary part
        residuals[3 * raw_ndof + 1] += (Psi[global_eqn] * C[global_eqn]) /
                                       Count[global_eqn];
      }
    }
    else
    {
      throw OomphLibError("Solve_which_system can only be 0",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyHopfHandler::debug_analytical_filling(oomph::GeneralisedElement *elem_pt, double eps)
  {
    if (!Problem_pt->are_hessian_products_calculated_analytically())
    {
      throw_runtime_error("Cannot do this without having analytical Hessian");
    }
    unsigned nd = this->ndof(elem_pt);
    Vector<double> fd_residuals(nd, 0.0);
    Vector<double> ana_residuals(nd, 0.0);
    DenseMatrix<double> fd_jacobian(nd, nd, 0.0);
    DenseMatrix<double> ana_jacobian(nd, nd, 0.0);
    Problem_pt->unset_analytic_hessian_products();
    this->get_jacobian(elem_pt, fd_residuals, fd_jacobian);
    Problem_pt->set_analytic_hessian_products();
    this->get_jacobian(elem_pt, ana_residuals, ana_jacobian);
    std::vector<std::string> dofnames = dynamic_cast<BulkElementBase *>(elem_pt)->get_dof_names();
    unsigned orig_ndof = dofnames.size();
    for (unsigned i = 0; i < orig_ndof; i++)
      dofnames.push_back("RE_eig[" + dofnames[i] + "]");
    for (unsigned i = 0; i < orig_ndof; i++)
      dofnames.push_back("IM_eig[" + dofnames[i] + "]");
    dofnames.push_back("PARAM");
    dofnames.push_back("OMEGA");
    std::cout << dofnames.size() << "  " << nd << std::endl;
    for (unsigned int i = 0; i < nd; i++)
    {
      double diff = fd_residuals[i] - ana_residuals[i];
      if (diff * diff > eps * eps)
      {
        std::cout << "RESIDUAL DIFF in component " << i << " of " << nd << "  :  " << diff << "  with FD/Ana " << fd_residuals[i] << " and " << ana_residuals[i] << "  ## " << dofnames[i] << std::endl;
      }
    }
    for (unsigned int i = 0; i < nd; i++)
    {
      for (unsigned int j = 0; j < nd; j++)
      {
        double diff = fd_jacobian(i, j) - ana_jacobian(i, j);
        if (diff * diff > eps * eps)
        {
          std::cout << "Jacobian DIFF at  (" << i << " , " << j << ") of " << nd << "  :  " << diff << "  with FD/Ana " << fd_jacobian(i, j) << " and " << ana_jacobian(i, j) << "  ## " << dofnames[i] << " ##wrt## " << dofnames[j] << std::endl;
        }
      }
    }
  }
  //===============================================================
  /// \short Calculate the elemental Jacobian matrix "d equation
  /// / d variable".
  //==================================================================
  void MyHopfHandler::get_jacobian(GeneralisedElement *const &elem_pt,
                                   Vector<double> &residuals,
                                   DenseMatrix<double> &jacobian)
  {

    bool ana_dparam = Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[3 * Ndof]);
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);

    // The standard case
    if (Solve_which_system == 0)
    {
      unsigned augmented_ndof = ndof(elem_pt);
      unsigned raw_ndof = elem_pt->ndof();

      if (!ana_hessian)
      {

        // Get the basic residuals and jacobian
        DenseMatrix<double> M(raw_ndof);
        elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);
        // Now fill in the actual residuals
        get_residuals(elem_pt, residuals);

        // Now the jacobian appears in other entries
        for (unsigned n = 0; n < raw_ndof; ++n)
        {
          for (unsigned m = 0; m < raw_ndof; ++m)
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian(n, m);
            jacobian(raw_ndof + n, 2 * raw_ndof + m) = Omega * M(n, m);
            jacobian(2 * raw_ndof + n, 2 * raw_ndof + m) = jacobian(n, m);
            jacobian(2 * raw_ndof + n, raw_ndof + m) = -Omega * M(n, m);
            unsigned global_eqn = elem_pt->eqn_number(m);
            jacobian(raw_ndof + n, 3 * raw_ndof + 1) += M(n, m) * Psi[global_eqn];
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) -= M(n, m) * Phi[global_eqn];
          }

          unsigned local_eqn = elem_pt->eqn_number(n);
          jacobian(3 * raw_ndof, raw_ndof + n) = C[local_eqn] / Count[local_eqn];
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = C[local_eqn] / Count[local_eqn];
        }

        const double FD_step = this->FD_step;

        Vector<double> newres_p(augmented_ndof), newres_m(augmented_ndof);

        //	 DenseMatrix<double> dJduPhi(raw_ndof,raw_ndof,0.0);
        //	 DenseMatrix<double> dJduPsi(raw_ndof,raw_ndof,0.0);

        // Loop over the dofs
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          // Just do the x's
          unsigned long global_eqn = eqn_number(elem_pt, n);
          double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
          double init = *unknown_pt;
          *unknown_pt += FD_step;

          // Get the new residuals
          get_residuals(elem_pt, newres_p);

          if (!this->symmetric_FD)
          {
            for (unsigned m = 0; m < raw_ndof; m++)
            {
              jacobian(raw_ndof + m, n) =
                  (newres_p[raw_ndof + m] - residuals[raw_ndof + m]) / (FD_step); // These are in fact second order derivatives, i.e. derivatives of the jacobian

              jacobian(2 * raw_ndof + m, n) =
                  (newres_p[2 * raw_ndof + m] - residuals[2 * raw_ndof + m]) / (FD_step);
            }
          }
          else
          {
            *unknown_pt = init;
            *unknown_pt -= FD_step;
            get_residuals(elem_pt, newres_m);
            for (unsigned m = 0; m < raw_ndof; m++)
            {
              jacobian(raw_ndof + m, n) =
                  (newres_p[raw_ndof + m] - newres_m[raw_ndof + m]) / (2 * FD_step); // These are in fact second order derivatives, i.e. derivatives of the jacobian

              jacobian(2 * raw_ndof + m, n) =
                  (newres_p[2 * raw_ndof + m] - newres_m[2 * raw_ndof + m]) / (2 * FD_step);
            }
          }
          // Reset the unknown
          *unknown_pt = init;
        }

        // PARAM DERIV

        if (ana_dparam)
        {
          Vector<double> dres_dparam(augmented_ndof, 0.0);
          this->get_dresiduals_dparameter(elem_pt, Problem_pt->GetDofPtr()[3 * Ndof], dres_dparam);
          for (unsigned m = 0; m < augmented_ndof - 2; m++)
          {
            jacobian(m, 3 * raw_ndof) = dres_dparam[m];
          }
        }
        else
        {
          // Now do the global parameter
          double *unknown_pt = Problem_pt->GetDofPtr()[3 * Ndof];
          double init = *unknown_pt;
          *unknown_pt += FD_step;

          Problem_pt->actions_after_change_in_bifurcation_parameter();
          // Get the new residuals
          get_residuals(elem_pt, newres_p);

          if (!this->symmetric_FD)
          {
            for (unsigned m = 0; m < augmented_ndof - 2; m++)
            {
              jacobian(m, 3 * raw_ndof) =
                  (newres_p[m] - residuals[m]) / FD_step;
            }
          }
          else
          {
            *unknown_pt = init;
            *unknown_pt -= FD_step;
            get_residuals(elem_pt, newres_m); // XXX MOD: IS NOT USED ANYHOW
            for (unsigned m = 0; m < augmented_ndof - 2; m++)
            {
              jacobian(m, 3 * raw_ndof) =
                  (newres_p[m] - residuals[m]) / FD_step;
            }
          }
          // Reset the unknown
          *unknown_pt = init;
          Problem_pt->actions_after_change_in_bifurcation_parameter();
        }
      }
      else // ANALYTIC HESSIAN AND PARAM DERIVS
      {
        pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
        std::vector<SinglePassMultiAssembleInfo> multi_assm;
        residuals.initialise(0.0);
        jacobian.initialise(0.0);
        oomph::DenseMatrix<double> M(raw_ndof, raw_ndof, 0.0);

        oomph::DenseMatrix<double> dJdU_Eig(2 * raw_ndof, raw_ndof, 0.0), dMdU_Eig(2 * raw_ndof, raw_ndof, 0.0);
        oomph::DenseMatrix<double> dJdParam(raw_ndof, raw_ndof, 0.0), dMdParam(raw_ndof, raw_ndof, 0.0);
        oomph::Vector<double> Eig_local(2 * raw_ndof);

        oomph::Vector<double> dRdParam(raw_ndof, 0.0);
        for (unsigned int i = 0; i < raw_ndof; i++)
        {
          unsigned global_eqn = elem_pt->eqn_number(i);

          Eig_local[i] = Phi[global_eqn];
          Eig_local[raw_ndof + i] = Psi[global_eqn];
        }
        multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian, &M));

        multi_assm.back().add_hessian(Eig_local, &dJdU_Eig, &dMdU_Eig);
        multi_assm.back().add_param_deriv(Parameter_pt, &dRdParam, &dJdParam, &dMdParam);
        pyoomph_elem_pt->get_multi_assembly(multi_assm);

        // Residuals
        residuals[3 * raw_ndof] = -1.0 / (double)(Problem_pt->mesh_pt()->nelement()) * eigenweight;
        residuals[3 * raw_ndof + 1] = 0.0;
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[raw_ndof + i] = 0.0;
          residuals[2 * raw_ndof + i] = 0.0;
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            // residuals[raw_ndof + i] += jacobian(i, j) * Phi_local[j] + Omega * M(i, j) * Psi_local[j];
            // residuals[2 * raw_ndof + i] += jacobian(i, j) * Psi_local[j] - Omega * M(i, j) * Phi_local[j];
            residuals[raw_ndof + i] += jacobian(i, j) * Eig_local[j] + Omega * M(i, j) * Eig_local[raw_ndof + j];
            residuals[2 * raw_ndof + i] += jacobian(i, j) * Eig_local[raw_ndof + j] - Omega * M(i, j) * Eig_local[j];
          }
          unsigned global_eqn = elem_pt->eqn_number(i);
          residuals[3 * raw_ndof] += (Phi[global_eqn] * C[global_eqn]) / Count[global_eqn];
          residuals[3 * raw_ndof + 1] += (Psi[global_eqn] * C[global_eqn]) / Count[global_eqn];
        }

        // Jacobian
        for (unsigned n = 0; n < raw_ndof; ++n)
        {
          jacobian(n, 3 * raw_ndof) = dRdParam[n];
          jacobian(raw_ndof + n, 3 * raw_ndof) = 0.0;
          jacobian(raw_ndof + n, 3 * raw_ndof + 1) = 0.0;
          jacobian(2 * raw_ndof + n, 3 * raw_ndof) = 0.0;
          jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) = 0.0;
          for (unsigned m = 0; m < raw_ndof; ++m)
          {
            jacobian(raw_ndof + n, m) = dJdU_Eig(n, m) + Omega * dMdU_Eig(raw_ndof + n, m);                                           // dR[Phi]/dU
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian(n, m);                                                                    // dR[Phi]/dPhi
            jacobian(raw_ndof + n, 2 * raw_ndof + m) = Omega * M(n, m);                                                               // dR[Phi]/dPsi
            jacobian(raw_ndof + n, 3 * raw_ndof) += dJdParam(n, m) * Eig_local[m] + Omega * dMdParam(n, m) * Eig_local[raw_ndof + m]; // dR[Phi]/dParam

            jacobian(2 * raw_ndof + n, m) = dJdU_Eig(raw_ndof + n, m) - Omega * dMdU_Eig(n, m);                                           // dR[Psi]/dU
            jacobian(2 * raw_ndof + n, 2 * raw_ndof + m) = jacobian(n, m);                                                                // dR[Psi]/dPsi
            jacobian(2 * raw_ndof + n, raw_ndof + m) = -Omega * M(n, m);                                                                  // dR[Psi]/dPhi
            jacobian(2 * raw_ndof + n, 3 * raw_ndof) += dJdParam(n, m) * Eig_local[raw_ndof + m] - Omega * dMdParam(n, m) * Eig_local[m]; // dR[Psi]/dParam

            jacobian(raw_ndof + n, 3 * raw_ndof + 1) += M(n, m) * Eig_local[raw_ndof + m]; // dR[Phi]/dOmega
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) -= M(n, m) * Eig_local[m];        // dR[Psi]/dOmega
          }

          unsigned local_eqn = elem_pt->eqn_number(n);
          jacobian(3 * raw_ndof, raw_ndof + n) = C[local_eqn] / Count[local_eqn];         // dR[Param]/dPhi
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = C[local_eqn] / Count[local_eqn]; // dR[Omega]/dPsi
        }
      }
    } // End of standard case

    // Normal case
    else if (Solve_which_system == 1)
    {
      // Just get the normal jacobian and residuals
      elem_pt->get_jacobian(residuals, jacobian);
    }
    // Otherwise the augmented complex case
    else if (Solve_which_system == 2)
    {
      unsigned raw_ndof = elem_pt->ndof();

      // Get the basic residuals and jacobian
      DenseMatrix<double> M(raw_ndof);
      elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);

      // We now need to fill in the other blocks
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(n, raw_ndof + m) = Omega * M(n, m);
          jacobian(raw_ndof + n, m) = -Omega * M(n, m);
          jacobian(raw_ndof + n, raw_ndof + m) = jacobian(n, m);
        }
      }

      // Now overwrite to fill in the residuals
      // The decision take is to solve for the mass matrix multiplied
      // terms in the residuals because they require no additional
      // information to assemble.
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        residuals[n] = 0.0;
        residuals[raw_ndof + n] = 0.0;
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          unsigned global_unknown = elem_pt->eqn_number(m);
          // Real part
          residuals[n] += M(n, m) * Psi[global_unknown];
          // Imaginary part
          residuals[raw_ndof + n] -= M(n, m) * Phi[global_unknown];
        }
      }
    } // End of complex augmented case
    else
    {
      throw OomphLibError("Solve_which_system can only be 0,1 or 2",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  //==================================================================
  /// Get the derivatives of the augmented residuals with respect to
  /// a parameter
  //=================================================================
  void MyHopfHandler::get_dresiduals_dparameter(
      GeneralisedElement *const &elem_pt,
      double *const &parameter_pt, Vector<double> &dres_dparam)
  {
    // Should only call get residuals for the full system
    if (Solve_which_system == 0)
    {
      // Need to get raw residuals and jacobian
      unsigned raw_ndof = elem_pt->ndof();

      DenseMatrix<double> djac_dparam(raw_ndof), dM_dparam(raw_ndof);
      // Get the basic residuals, jacobian and mass matrix
      elem_pt->get_djacobian_and_dmass_matrix_dparameter(
          parameter_pt, dres_dparam, djac_dparam, dM_dparam);

      // Initialise the pen-ultimate residual, which does not
      // depend on the parameter
      dres_dparam[3 * raw_ndof] = 0.0;
      dres_dparam[3 * raw_ndof + 1] = 0.0;

      // Now multiply to fill in the residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        dres_dparam[raw_ndof + i] = 0.0;
        dres_dparam[2 * raw_ndof + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned global_unknown = elem_pt->eqn_number(j);
          // Real part
          dres_dparam[raw_ndof + i] +=
              djac_dparam(i, j) * Phi[global_unknown] +
              Omega * dM_dparam(i, j) * Psi[global_unknown];
          // Imaginary part
          dres_dparam[2 * raw_ndof + i] +=
              djac_dparam(i, j) * Psi[global_unknown] -
              Omega * dM_dparam(i, j) * Phi[global_unknown];
        }
      }
    }
    else
    {
      throw OomphLibError("Solve_which_system can only be 0",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  //========================================================================
  /// Overload the derivative of the residuals and jacobian
  /// with respect to a parameter so that it breaks because it should not
  /// be required
  //========================================================================
  void MyHopfHandler::get_djacobian_dparameter(
      GeneralisedElement *const &elem_pt,
      double *const &parameter_pt,
      Vector<double> &dres_dparam,
      DenseMatrix<double> &djac_dparam)
  {
    std::ostringstream error_stream;
    error_stream << "This function has not been implemented because it is not required\n";
    error_stream << "in standard problems.\n";
    error_stream << "If you find that you need it, you will have to implement it!\n\n";

    throw OomphLibError(error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
  }

  void MyHopfHandler::get_hessian_vector_products(
      GeneralisedElement *const &elem_pt,
      Vector<double> const &Y,
      DenseMatrix<double> const &C,
      DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }

  //==========================================================================
  /// Return the eigenfunction(s) associated with the bifurcation that
  /// has been detected in bifurcation tracking problems
  //==========================================================================
  void MyHopfHandler::get_eigenfunction(
      Vector<DoubleVector> &eigenfunction)
  {
    // There is a real and imaginary part of the null vector
    eigenfunction.resize(2);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
    // Rebuild the vector
    eigenfunction[0].build(&dist, 0.0);
    eigenfunction[1].build(&dist, 0.0);
    // Set the value to be the null vector
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Phi[n];
      eigenfunction[1][n] = Psi[n];
    }
  }

  //====================================================================
  /// Set to solve the standard (underlying jacobian)  system
  //===================================================================
  void MyHopfHandler::solve_standard_system()
  {
    if (Solve_which_system != 1)
    {
      Solve_which_system = 1;
      // Restrict the problem to the standard variables only
      Problem_pt->GetDofPtr().resize(Ndof);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                                Ndof, false);
      // Remove all previous sparse storage used during Jacobian assembly
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    }
  }

  //====================================================================
  /// Set to solve the complex (jacobian and mass matrix)  system
  //===================================================================
  void MyHopfHandler::solve_complex_system()
  {
    // If we were not solving the complex system resize the unknowns
    // accordingly
    if (Solve_which_system != 2)
    {
      Solve_which_system = 2;
      // Resize to the first Ndofs (will work whichever system we were
      // solving before)
      Problem_pt->GetDofPtr().resize(Ndof);
      // Add the first (real) part of the eigenfunction back into the problem
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Phi[n]);
      }
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                                Ndof * 2, false);
      // Remove all previous sparse storage used during Jacobian assembly
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    }
  }

  //=================================================================
  /// Set to Solve full system system
  //=================================================================
  void MyHopfHandler::solve_full_system()
  {
    // If we are starting from another system
    if (Solve_which_system)
    {
      Solve_which_system = 0;

      // Resize to the first Ndofs (will work whichever system we were
      // solving before)
      Problem_pt->GetDofPtr().resize(Ndof);
      // Add the additional unknowns back into the problem
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Phi[n]);
      }
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Psi[n]);
      }
      // Now add the parameter
      Problem_pt->GetDofPtr().push_back(Parameter_pt);
      // Finally add the frequency
      Problem_pt->GetDofPtr().push_back(&Omega);

      //
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                                3 * Ndof + 2, false);
      // Remove all previous sparse storage used during Jacobian assembly
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    }
  }

  void MyHopfHandler::realign_C_vector()
  {

    double dot = 0.0;
    double doti = 0.0;
    double phisqr = 0.0;
    double psisqr = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      double phin = *(Problem_pt->GetDofPtr()[Ndof + n]);
      double psin = *(Problem_pt->GetDofPtr()[2 * Ndof + n]);
      dot += C[n] * phin;
      phisqr += phin * phin;
      doti += C[n] * psin;
      psisqr += psin * psin;
    }
    std::cerr << "DOT OF C and PHi is " << dot << " and PHi^2 = " << phisqr << std::endl;
    std::cerr << "DOT OF C and Psi is " << doti << " and Psi^2 = " << psisqr << std::endl;

    double lf = eigenweight / sqrt(phisqr);
    for (unsigned n = 0; n < Ndof; n++)
    {
      double *phin = (Problem_pt->GetDofPtr()[Ndof + n]);
      double *psin = (Problem_pt->GetDofPtr()[2 * Ndof + n]);
      (*phin) *= lf;
      (*psin) *= lf;
      C[n] = (*phin);
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();

    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);

    Phi.resize(Ndof);
    Y.resize(Ndof);
    Count.resize(Ndof, 0);

    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    LinearSolver *const linear_solver_pt = problem_pt->linear_solver_pt();

    bool enable_resolve = linear_solver_pt->is_resolve_enabled();
    linear_solver_pt->enable_resolve();
    DoubleVector x(dist_pt, 0.0);
    linear_solver_pt->solve(problem_pt, x);
    problem_pt->get_derivative_wrt_global_parameter(parameter_pt, x);
    DoubleVector input_x(x);
    linear_solver_pt->resolve(input_x, x);
    if (enable_resolve)
    {
      linear_solver_pt->enable_resolve();
    }
    else
    {
      linear_solver_pt->disable_resolve();
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += x[n] * x[n];
    }
    length = sqrt(length);

    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      Y[n] = Phi[n] = -x[n] / length;
    }
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 1, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    delete dist_pt;
  }

  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const DoubleVector &eigenvector) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);
    Phi.resize(Ndof);
    Y.resize(Ndof);
    Count.resize(Ndof, 0);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += eigenvector[n] * eigenvector[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      Y[n] = Phi[n] = eigenvector[n] / length;
    }
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 1, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);

    delete dist_pt;
  }

  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const DoubleVector &eigenvector, const DoubleVector &normalisation) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);
    Phi.resize(Ndof);
    Y.resize(Ndof);
    Count.resize(Ndof, 0);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += eigenvector[n] * normalisation[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      Y[n] = eigenvector[n] / length;
      Phi[n] = normalisation[n];
    }
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 1, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    delete dist_pt;
  }

  unsigned MyFoldHandler::ndof(GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Full_augmented:
      return (2 * raw_ndof + 1);
      break;

    case Block_augmented_J:
      return (raw_ndof + 1);
      break;

    case Block_J:
      return raw_ndof;
      break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  unsigned long MyFoldHandler::eqn_number(GeneralisedElement *const &elem_pt,
                                          const unsigned &ieqn_local)
  {
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn = 0;
    if (ieqn_local < raw_ndof)
    {
      global_eqn = elem_pt->eqn_number(ieqn_local);
    }
    else if (ieqn_local == raw_ndof)
    {
      global_eqn = Ndof;
    }
    else
    {
      global_eqn = Ndof + 1 + elem_pt->eqn_number(ieqn_local - 1 - raw_ndof);
    }
    return global_eqn;
  }

  void MyFoldHandler::get_residuals(GeneralisedElement *const &elem_pt,
                                    Vector<double> &residuals)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Block_J:
    {
      elem_pt->get_residuals(residuals);
    }
    break;
    case Block_augmented_J:
    {
      elem_pt->get_residuals(residuals);
      residuals[raw_ndof] = 0.0;
    }
    break;
    case Full_augmented:
    {
      DenseMatrix<double> jacobian(raw_ndof);
      elem_pt->get_jacobian(residuals, jacobian);
      residuals[raw_ndof] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y[elem_pt->eqn_number(j)];
        }
        unsigned global_eqn = elem_pt->eqn_number(i);
        residuals[raw_ndof] += (Phi[global_eqn] * Y[global_eqn]) / Count[global_eqn];
      }
    }
    break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyFoldHandler::get_jacobian(GeneralisedElement *const &elem_pt,
                                   Vector<double> &residuals,
                                   DenseMatrix<double> &jacobian)
  {
    bool ana_dparam = Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[Ndof]);
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<BulkElementBase *>(elem_pt);

    unsigned augmented_ndof = ndof(elem_pt);
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Block_J:
    {
      elem_pt->get_jacobian(residuals, jacobian);
    }
    break;
    case Block_augmented_J:
    {
      get_residuals(elem_pt, residuals);
      Vector<double> newres(augmented_ndof);
      elem_pt->get_jacobian(newres, jacobian);
      const double FD_step = 1.0e-8;
      {
        double *unknown_pt = Problem_pt->GetDofPtr()[Ndof];
        double init = *unknown_pt;
        *unknown_pt += FD_step;

        Problem_pt->actions_after_change_in_bifurcation_parameter();
        get_residuals(elem_pt, newres);

        for (unsigned n = 0; n < raw_ndof; n++)
        {
          jacobian(n, augmented_ndof - 1) = (newres[n] - residuals[n]) / FD_step;
        }
        *unknown_pt = init;

        Problem_pt->actions_after_change_in_bifurcation_parameter();
      }

      for (unsigned n = 0; n < raw_ndof; n++)
      {
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(augmented_ndof - 1, n) = Phi[local_eqn] / Count[local_eqn];
      }
    }
    break;

    case Full_augmented:
    {
      if (ana_hessian)
      {

        jacobian.initialise(0.0);
        residuals.initialise(0.0);
        DenseMatrix<double> djac_dparam(raw_ndof, raw_ndof, 0.0);
        Vector<double> dres_dparam(raw_ndof, 0.0);
        DenseMatrix<double> dJduPhiH(raw_ndof, raw_ndof, 0.0);
        Vector<double> Y_local(raw_ndof);
        for (unsigned _e = 0; _e < raw_ndof; _e++)
        {
          Y_local[_e] = Y[elem_pt->eqn_number(_e)];
        }

        pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
        std::vector<SinglePassMultiAssembleInfo> assemble_info;
        assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian));
        assemble_info.back().add_param_deriv(Parameter_pt, &dres_dparam, &djac_dparam);
        assemble_info.back().add_hessian(Y_local, &dJduPhiH);
        pyoomph_elem_pt->get_multi_assembly(assemble_info);

        // Fill augmented residuals
        residuals[raw_ndof] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[raw_ndof + 1 + i] = 0.0;
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y_local[j];
          }
          unsigned global_eqn = elem_pt->eqn_number(i);
          residuals[raw_ndof] += (Phi[global_eqn] * Y[global_eqn]) / Count[global_eqn];
        }

        // And the Jacobian
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          jacobian(n, raw_ndof) = dres_dparam[n];
          for (unsigned m = 0; m < raw_ndof; m++)
          {
            jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
            jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, m) * Y_local[m];
            jacobian(raw_ndof + 1 + m, n) = dJduPhiH(m, n);
          }
          unsigned global_eqn = elem_pt->eqn_number(n);
          jacobian(raw_ndof, raw_ndof + 1 + n) = Phi[global_eqn] / Count[global_eqn];
        }
      }
      else
      {
        get_residuals(elem_pt, residuals);
        Vector<double> newres(raw_ndof);
        DenseMatrix<double> newjac(raw_ndof);
        elem_pt->get_jacobian(newres, jacobian);

        for (unsigned n = 0; n < raw_ndof; n++)
        {
          for (unsigned m = 0; m < raw_ndof; m++)
          {
            jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
          }
        }

        if (ana_dparam)
        {
          DenseMatrix<double> djac_dparam(raw_ndof, raw_ndof, 0.0);
          Vector<double> dres_dparam(raw_ndof, 0.0);
          elem_pt->get_djacobian_dparameter(Problem_pt->GetDofPtr()[Ndof], dres_dparam, djac_dparam);
          for (unsigned n = 0; n < raw_ndof; n++)
          {
            jacobian(n, raw_ndof) = dres_dparam[n];
            for (unsigned l = 0; l < raw_ndof; l++)
            {
              jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, l) * Y[elem_pt->eqn_number(l)];
            }
          }
        }
        else
        {

          double FD_step = this->FD_step;
          {
            double *unknown_pt = Problem_pt->GetDofPtr()[Ndof];
            double init = *unknown_pt;
            *unknown_pt += FD_step;
            //            Problem_pt->actions_after_change_in_bifurcation_parameter();
            elem_pt->get_jacobian(newres, newjac);
            if (!this->symmetric_FD)
            {
              for (unsigned n = 0; n < raw_ndof; n++)
              {
                jacobian(n, raw_ndof) = (newres[n] - residuals[n]) / FD_step;
                for (unsigned l = 0; l < raw_ndof; l++)
                {
                  jacobian(raw_ndof + 1 + n, raw_ndof) += (newjac(n, l) - jacobian(n, l)) * Y[elem_pt->eqn_number(l)] /
                                                          FD_step;
                }
              }
            }
            else
            {
              *unknown_pt = init;
              *unknown_pt -= FD_step;
              Vector<double> newres_m(raw_ndof);
              DenseMatrix<double> newjac_m(raw_ndof);
              elem_pt->get_jacobian(newres_m, newjac_m);
              for (unsigned n = 0; n < raw_ndof; n++)
              {
                jacobian(n, raw_ndof) = (newres[n] - newres_m[n]) / (2 * FD_step);
                for (unsigned l = 0; l < raw_ndof; l++)
                {
                  jacobian(raw_ndof + 1 + n, raw_ndof) += (newjac(n, l) - newjac_m(n, l)) * Y[elem_pt->eqn_number(l)] /
                                                          (2 * FD_step);
                }
              }
            }
            *unknown_pt = init;
            Problem_pt->actions_after_change_in_bifurcation_parameter();
          }
        }

        for (unsigned n = 0; n < raw_ndof; n++)
        {
          unsigned long global_eqn = eqn_number(elem_pt, n);
          double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
          double init = *unknown_pt;
          *unknown_pt += FD_step;
          //          Problem_pt->actions_before_newton_convergence_check(); /// ALICE
          elem_pt->get_jacobian(newres, newjac);
          if (!this->symmetric_FD)
          {
            // Work out the differences
            for (unsigned k = 0; k < raw_ndof; k++)
            {
              for (unsigned l = 0; l < raw_ndof; l++)
              {
                jacobian(raw_ndof + 1 + k, n) += (newjac(k, l) - jacobian(k, l)) * Y[elem_pt->eqn_number(l)] / FD_step;
              }
            }
          }
          else
          {
            *unknown_pt = init;
            *unknown_pt -= FD_step;
            Vector<double> newres_m(raw_ndof);
            DenseMatrix<double> newjac_m(raw_ndof);
            elem_pt->get_jacobian(newres_m, newjac_m);
            // Work out the differences
            for (unsigned k = 0; k < raw_ndof; k++)
            {
              for (unsigned l = 0; l < raw_ndof; l++)
              {
                jacobian(raw_ndof + 1 + k, n) += (newjac(k, l) - newjac_m(k, l)) * Y[elem_pt->eqn_number(l)] / (2 * FD_step);
              }
            }
          }
          *unknown_pt = init;
          //        Problem_pt->actions_before_newton_convergence_check(); /// ALICE
        }

        // Fill in the row corresponding to the parameter
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          unsigned global_eqn = elem_pt->eqn_number(n);
          jacobian(raw_ndof, raw_ndof + 1 + n) = Phi[global_eqn] / Count[global_eqn];
        }
      }
    }
    break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyFoldHandler::get_dresiduals_dparameter(
      GeneralisedElement *const &elem_pt,
      double *const &parameter_pt,
      Vector<double> &dres_dparam)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Block_J:
    {
      elem_pt->get_dresiduals_dparameter(parameter_pt, dres_dparam);
    }
    break;

    case Block_augmented_J:
    {
      elem_pt->get_dresiduals_dparameter(parameter_pt, dres_dparam);
      dres_dparam[raw_ndof] = 0.0;
    }
    break;
    case Full_augmented:
    {
      DenseMatrix<double> djac_dparam(raw_ndof);
      elem_pt->get_djacobian_dparameter(parameter_pt, dres_dparam, djac_dparam);
      dres_dparam[raw_ndof] = 0.0;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        dres_dparam[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          dres_dparam[raw_ndof + 1 + i] += djac_dparam(i, j) * Y[elem_pt->eqn_number(j)];
        }
      }
    }
    break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyFoldHandler::get_djacobian_dparameter(GeneralisedElement *const &elem_pt, double *const &parameter_pt, Vector<double> &dres_dparam, DenseMatrix<double> &djac_dparam)
  {
    std::ostringstream error_stream;
    error_stream << "This function has not been implemented because it is not required\n";
    error_stream << "in standard problems.\n";
    error_stream << "If you find that you need it, you will have to implement it!\n\n";

    throw OomphLibError(error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
  }

  void MyFoldHandler::get_hessian_vector_products(GeneralisedElement *const &elem_pt, Vector<double> const &Y, DenseMatrix<double> const &C, DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }

  void MyFoldHandler::get_eigenfunction(Vector<DoubleVector> &eigenfunction)
  {
    eigenfunction.resize(1);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
    eigenfunction[0].build(&dist, 0.0);
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Y[n];
    }
  }

  MyFoldHandler::~MyFoldHandler()
  {
    AugmentedBlockFoldLinearSolver *block_fold_solver_pt = dynamic_cast<AugmentedBlockFoldLinearSolver *>(
        Problem_pt->linear_solver_pt());

    if (block_fold_solver_pt)
    {
      Problem_pt->linear_solver_pt() = block_fold_solver_pt->linear_solver_pt();
      delete block_fold_solver_pt;
    }
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  void MyFoldHandler::solve_augmented_block_system()
  {
    if (Solve_which_system != Block_augmented_J)
    {
      if (Solve_which_system == Block_J)
      {
        Problem_pt->GetDofPtr().push_back(Parameter_pt);
      }
      Problem_pt->GetDofPtr().resize(Ndof + 1);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof + 1, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Block_augmented_J;
    }
  }

  void MyFoldHandler::solve_block_system()
  {
    if (Solve_which_system != Block_J)
    {
      Problem_pt->GetDofPtr().resize(Ndof);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Block_J;
    }
  }

  void MyFoldHandler::solve_full_system()
  {
    if (Solve_which_system != Full_augmented)
    {
      if (Solve_which_system == Block_J)
      {
        Problem_pt->GetDofPtr().push_back(Parameter_pt);
      }
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Y[n]);
      }
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof * 2 + 1, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Full_augmented;
    }
  }

  void MyFoldHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  void MyFoldHandler::realign_C_vector()
  {

    double dot = 0.0;
    double phisqr = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      double phin = *(Problem_pt->GetDofPtr()[Ndof + 1 + n]);
      dot += Y[n] * phin;
      phisqr += phin * phin;
    }
    std::cerr << "DOT OF C and PHi is " << dot << " and PHi^2 = " << phisqr << std::endl;

    for (unsigned n = 0; n < Ndof; n++)
    {
      double phin = *(Problem_pt->GetDofPtr()[Ndof + 1 + n]);
      Y[n] = phin / phisqr; // Renormalize the c vector
    }
  }

  //////////////////////////////////////////////////

  MyPitchForkHandler::MyPitchForkHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector) : Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    symmetryweight = 1.0;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);
    Psi.resize(Ndof);
    Y.resize(Ndof);
    C.resize(Ndof);
    Count.resize(Ndof, 0);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    Nelement = n_element;
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += symmetry_vector[n] * symmetry_vector[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      C[n] = Y[n] = symmetry_vector[n] / length;
    }
    for (unsigned n = 0; n < Ndof; n++)
    {
      // problem_pt->GetDofPtr().push_back(&Psi[n]);
      Psi[n] = symmetry_vector[n] / length;
    }

    if (problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      setup_U_times_Psi_residual_indices();

    if (!problem_pt->is_quiet())
    {
      double initial_orthogonality = 0.0;
      if (problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        for (unsigned e = 0; e < n_element; e++)
        {
          GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
          DenseMatrix<double> psi_i_times_psi_j(elem_pt->ndof());
          initial_orthogonality += this->get_integrated_U_dot_Psi(elem_pt, psi_i_times_psi_j);
        }
      }
      else
      {
        for (unsigned n = 0; n < Ndof; n++)
        {
          initial_orthogonality += Psi[n] * (*problem_pt->GetDofPtr()[n]);
        }
      }
      std::cout << "Initial pitchfork symmetry breaking orthogonality <psi,u>=" << initial_orthogonality << std::endl;
    }
    problem_pt->GetDofPtr().push_back(&Sigma);
    Sigma = 0.0;

    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 2, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    delete dist_pt;
  }

  void MyPitchForkHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Psi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  MyPitchForkHandler::~MyPitchForkHandler()
  {
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  double MyPitchForkHandler::get_integrated_U_dot_Psi(oomph::GeneralisedElement *const &elem_pt, DenseMatrix<double> &psi_i_times_psi_j)
  {
    unsigned raw_ndof = elem_pt->ndof();
    psi_i_times_psi_j.initialise(0.0);
    Vector<double> residuals(raw_ndof);
    this->set_assembled_residual(elem_pt, 1);
    elem_pt->get_jacobian(residuals, psi_i_times_psi_j);
    double res = 0.0;
    for (unsigned int i = 0; i < raw_ndof; i++)
    {
      unsigned eqn_i = elem_pt->eqn_number(i);
      for (unsigned int j = 0; j < raw_ndof; j++)
      {
        unsigned eqn_j = elem_pt->eqn_number(j);
        res += (*Problem_pt->global_dof_pt(eqn_i)) * Psi[eqn_j] * psi_i_times_psi_j(i, j); // Contract with mass matrix to get integral(U*Psi)
      }
    }
    this->set_assembled_residual(elem_pt, 0);
    return res;
  }

  unsigned MyPitchForkHandler::ndof(oomph::GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    return (2 * raw_ndof + 2);
  }

  unsigned long MyPitchForkHandler::eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local)
  {
    unsigned raw_ndof = elem_pt->ndof();
    if (ieqn_local < raw_ndof)
    {
      return elem_pt->eqn_number(ieqn_local);
    }
    // The bifurcation parameter equation
    else if (ieqn_local == raw_ndof)
    {
      return Ndof;
    }
    else if (ieqn_local < (2 * raw_ndof + 1))
    {
      return Ndof + 1 + elem_pt->eqn_number(ieqn_local - 1 - raw_ndof);
    }
    else
    {
      return 2 * Ndof + 1;
    }
  }

  void MyPitchForkHandler::get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals)
  {
    unsigned raw_ndof = elem_pt->ndof();
    DenseMatrix<double> jacobian(raw_ndof, raw_ndof, 0.0);
    Vector<double> symmetryR(raw_ndof, 0.0);
    DenseMatrix<double> symmetryA(raw_ndof, raw_ndof, 0.0);
    if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
    {
      pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      std::vector<SinglePassMultiAssembleInfo> assemble_info;
      assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian));
      assemble_info.push_back(SinglePassMultiAssembleInfo(resolve_assembled_residual(elem_pt, 1), &symmetryR, &symmetryA));
      pyoomph_elem_pt->get_multi_assembly(assemble_info);
    }
    else
    {
      elem_pt->get_jacobian(residuals, jacobian);
    }
    residuals[raw_ndof] = 0.0;
    residuals[2 * raw_ndof + 1] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      unsigned local_eqn = elem_pt->eqn_number(i);
      residuals[raw_ndof + 1 + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned local_unknown = elem_pt->eqn_number(j);
        residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y[local_unknown];
      }
      residuals[2 * raw_ndof + 1] += (Y[local_eqn] * C[local_eqn]) / Count[local_eqn];
    }
    if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
    {
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned local_unknown = elem_pt->eqn_number(j);
          residuals[i] += Sigma * symmetryA(i, j) * Psi[local_unknown];
          residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * symmetryA(i, j) * Psi[local_unknown]);
        }
      }
    }
    else
    {
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        residuals[i] += Sigma * Psi[local_eqn] / Count[local_eqn];
        residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * Psi[local_eqn]) / Count[local_eqn];
      }
    }
  }

  void MyPitchForkHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    bool ana_dparam = Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[Ndof]);
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<BulkElementBase *>(elem_pt);

    unsigned augmented_ndof = ndof(elem_pt);
    unsigned raw_ndof = elem_pt->ndof();
    if (ana_hessian && ana_dparam)
    {
      jacobian.initialise(0.0);
      residuals.initialise(0.0);
      DenseMatrix<double> djac_dparam(raw_ndof, raw_ndof, 0.0);
      Vector<double> dres_dparam(raw_ndof, 0.0);
      DenseMatrix<double> dJduPhiH(raw_ndof, raw_ndof, 0.0);
      Vector<double> symmetryR(raw_ndof, 0.0);
      DenseMatrix<double> symmetryA(raw_ndof, raw_ndof, 0.0);
      DenseMatrix<double> symmetryDADU_times_Psi(raw_ndof, raw_ndof, 0.0);
      Vector<double> Y_local(raw_ndof);
      for (unsigned _e = 0; _e < raw_ndof; _e++)
      {
        Y_local[_e] = Y[elem_pt->eqn_number(_e)];
      }
      Vector<double> Psi_local(raw_ndof);
      for (unsigned _e = 0; _e < raw_ndof; _e++)
      {
        Psi_local[_e] = Psi[elem_pt->eqn_number(_e)];
      }

      pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      std::vector<SinglePassMultiAssembleInfo> assemble_info;
      assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian));
      assemble_info.back().add_param_deriv(Parameter_pt, &dres_dparam, &djac_dparam);
      assemble_info.back().add_hessian(Y_local, &dJduPhiH);
      if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        assemble_info.push_back(SinglePassMultiAssembleInfo(resolve_assembled_residual(elem_pt, 1), &symmetryR, &symmetryA));
        assemble_info.back().add_hessian(Psi_local, &symmetryDADU_times_Psi);
      }
      pyoomph_elem_pt->get_multi_assembly(assemble_info);

      residuals[2 * raw_ndof + 1] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
      // Fill augmented residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        residuals[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y_local[j];
        }

        residuals[2 * raw_ndof + 1] += (Y[local_eqn] * C[local_eqn]) / Count[local_eqn];
      }

      // And the Jacobian
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(n, raw_ndof) = dres_dparam[n];
        jacobian(2 * raw_ndof + 1, raw_ndof + 1 + n) = C[local_eqn] / Count[local_eqn];
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, m) * Y_local[m];
          jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
          jacobian(raw_ndof + 1 + n, m) = dJduPhiH(n, m);
        }
      }

      if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          unsigned eqn_i = elem_pt->eqn_number(i);
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof] += (*Problem_pt->global_dof_pt(eqn_i)) * symmetryA(i, j) * Psi_local[j];
            residuals[i] += Sigma * symmetryA(i, j) * Psi_local[j];
            jacobian(raw_ndof, i) += symmetryA(i, j) * Psi_local[j] + symmetryDADU_times_Psi(i, j) * (*Problem_pt->global_dof_pt(eqn_i));
            jacobian(i, j) += Sigma * symmetryDADU_times_Psi(i, j);
            jacobian(i, 2 * raw_ndof + 1) += symmetryA(i, j) * Psi_local[j];
          }
        }
      }
      else
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          unsigned local_eqn = elem_pt->eqn_number(i);
          residuals[i] += Sigma * Psi[local_eqn] / Count[local_eqn];
          jacobian(i, 2 * raw_ndof + 1) += Psi[local_eqn] / Count[local_eqn];
          residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * Psi[local_eqn]) / Count[local_eqn];
          jacobian(raw_ndof, i) = Psi[local_eqn] / Count[local_eqn];
        }
      }
    }
    else
    {
      elem_pt->get_jacobian(residuals, jacobian);
      get_residuals(elem_pt,residuals); // The full residuals

      for (unsigned n = 0; n < raw_ndof; n++)
      {
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
        }
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(2 * raw_ndof + 1, raw_ndof + 1 + n) = C[local_eqn] / Count[local_eqn];
      }

      if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        throw_runtime_error("Improved pitchfork tracking on nonsymmetric meshes only works with an analytically derived Hessian");
      }
      else
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          unsigned local_eqn = elem_pt->eqn_number(i);
          jacobian(i, 2 * raw_ndof + 1) = Psi[local_eqn] / Count[local_eqn];
          jacobian(raw_ndof, i) = Psi[local_eqn] / Count[local_eqn];
        }
      }
      const double FD_step = 1.0e-8;
      Vector<double> newres_p(augmented_ndof);

      for (unsigned n = 0; n < raw_ndof; ++n)
      {
        unsigned long global_eqn = Problem_pt->assembly_handler_pt()->eqn_number(elem_pt,n);
        double *unknown_pt = Problem_pt->global_dof_pt(global_eqn);
        double init = *unknown_pt;
        *unknown_pt += FD_step;
        newres_p.initialise(0.0);
        get_residuals(elem_pt, newres_p);
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + 1 + m, n) = (newres_p[raw_ndof + 1 + m] - residuals[raw_ndof + 1 + m]) / (FD_step);
        }
        *unknown_pt = init;
        //		  Problem_pt->actions_before_newton_convergence_check();
      }

      {

        double *unknown_pt = Parameter_pt;
        double init = *unknown_pt;
        *unknown_pt += FD_step;
        newres_p.initialise(0.0);        
        get_residuals(elem_pt, newres_p);
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(m, raw_ndof) = (newres_p[m] - residuals[m]) / FD_step;
        }
        for (unsigned m = raw_ndof + 1; m < augmented_ndof - 1; m++)
        {
          jacobian(m, raw_ndof) = (newres_p[m] - residuals[m]) / FD_step;
        }
        *unknown_pt = init;
        Problem_pt->actions_after_change_in_bifurcation_parameter();
      }
    }
  }

  void MyPitchForkHandler::get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam)
  {
    unsigned raw_ndof = elem_pt->ndof();
    DenseMatrix<double> djac_dparam(raw_ndof);
    elem_pt->get_djacobian_dparameter(parameter_pt, dres_dparam, djac_dparam);
    dres_dparam[raw_ndof] = 0.0;
    dres_dparam[2 * raw_ndof + 1] = 0.0;
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      dres_dparam[raw_ndof + 1 + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned local_unknown = elem_pt->eqn_number(j);
        dres_dparam[raw_ndof + 1 + i] +=
            djac_dparam(i, j) * Y[local_unknown];
      }
    }
  }
  void MyPitchForkHandler::get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam)
  {
    throw_runtime_error("implement");
  }
  void MyPitchForkHandler::get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }
  void MyPitchForkHandler::get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction)
  {
    eigenfunction.resize(1);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
    eigenfunction[0].build(&dist, 0.0);
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Y[n];
    }
  }
  void MyPitchForkHandler::solve_full_system()
  {
    // Is full system anyways.. Nothing to be done
  }

  void MyPitchForkHandler::setup_U_times_Psi_residual_indices()
  {
    pyoomph::Problem *prob = dynamic_cast<pyoomph::Problem *>(Problem_pt);
    if (!prob)
      throw_runtime_error("Not a pyoomph::Problem... Strange");
    auto codes = prob->get_bulk_element_codes();
    for (unsigned int i = 0; i < codes.size(); i++)
    {
      int orig_residual = codes[i]->get_func_table()->current_res_jac; // Store the initial residual (base state)
      int mass_matrix_residual = -1;
      if (codes[i]->_set_solved_residual("_simple_mass_matrix_of_defined_fields"))
      {
        mass_matrix_residual = codes[i]->get_func_table()->current_res_jac;
      }
      codes[i]->get_func_table()->current_res_jac = orig_residual; // Reset it
      residual_contribution_indices[codes[i]] = PitchForkResidualContributionList(codes[i], orig_residual, mass_matrix_residual);
    }
  }

  int MyPitchForkHandler::resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    return entry.residual_indices[residual_mode];
  }

  bool MyPitchForkHandler::set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    // Setup the solved residual by the index (-1 means no contribution)
    entry.code->get_func_table()->current_res_jac = entry.residual_indices[residual_mode];
    return entry.residual_indices[residual_mode] >= 0;
  }

  ///////////////////////////////////////

  ////////// IMPORTANT PART ////////////

  //////////////////////////////////////

  // Constructors. We must pass a problem, a parameter to optimize (i.e. to change in order to get Re(eigenvalue)=0)
  // and a guess of the eigenvector
  AzimuthalSymmetryBreakingHandler::AzimuthalSymmetryBreakingHandler(Problem *const &problem_pt, double *const &parameter_pt,
                                                                     const oomph::DoubleVector &real_eigen, const oomph::DoubleVector &imag_eigen, const double &Omega_guess,bool has_imag)
      : Omega(Omega_guess), Parameter_pt(parameter_pt), has_imaginary_part(has_imag)
  {
    if (!has_imaginary_part) Omega=0.0;
    call_param_change_handler = false; // These must be false at the moment
    FD_step = 1e-8;                    // Default parameter for finite difference step
    Problem_pt = problem_pt;           // Store the problem
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();

    // Resize the vectors of additional dofs
    real_eigenvector.resize(Ndof);
    imag_eigenvector.resize(Ndof);
    normalization_vector.resize(Ndof);
    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement(); // Number of elements
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    // Normalise the guess for the eigenvectors. First getting the lengths
    double length_eigen_real = 0.0, GrGr = 0.0, GiGi = 0.0, GrGi = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      GrGr += real_eigen[n] * real_eigen[n];
      GiGi += imag_eigen[n] * imag_eigen[n];
      GrGi += real_eigen[n] * imag_eigen[n];
    }

    /// Correct the initial eigenvector guess
    double phi=0.0;
    if (GiGi>1e-10*GrGr)  phi = atan2(GrGr - GiGi + sqrt(pow(GrGr - GiGi, 2) + 4 * pow(GrGi, 2)), 2 * GrGi); // TODO: Check this
     
    //std::cout << "Initial guess for the phase is " << phi << " RR " << GrGr << " " << " II " << GrGi << " RI " << GrGi << std::endl;
    //std::cout << "Has imaginary part " << has_imaginary_part << std::endl;

    // Next add the real and imaginary parts of the eigenvector to be determined to the problem
    // and initialise it all
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&real_eigenvector[n]);
      normalization_vector[n] = real_eigenvector[n] = real_eigen[n] * cos(phi) - imag_eigen[n] * sin(phi); // Set the real normalization vector and the initial real part of the eigenvector to length 1    length_eigen_real += real_eigenvector[n] * real_eigenvector[n];
      length_eigen_real += real_eigenvector[n] * real_eigenvector[n];
    }

    if (has_imaginary_part)
    {
      for (unsigned n = 0; n < Ndof; n++)
      {
        problem_pt->GetDofPtr().push_back(&imag_eigenvector[n]);
        imag_eigenvector[n] = real_eigen[n] * sin(phi) + imag_eigen[n] * cos(phi);
      }
    }

    for (unsigned n = 0; n < Ndof; n++)
    {
      normalization_vector[n] /= sqrt(length_eigen_real);
      real_eigenvector[n] /= sqrt(length_eigen_real);
      imag_eigenvector[n] /= sqrt(length_eigen_real);
    }

    // Now add the parameter as degree of freedom
    problem_pt->GetDofPtr().push_back(parameter_pt);
    // Finally add the unknown imaginary part of the eigenvalue as degree of freedom
    if (has_imaginary_part) problem_pt->GetDofPtr().push_back(&Omega);

    // rebuild the Dof_distribution_pt
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), (has_imaginary_part ? Ndof *  3 + 2 : Ndof*2+1), false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  // Destructor (used for cleaning up memory)
  AzimuthalSymmetryBreakingHandler::~AzimuthalSymmetryBreakingHandler()
  {
    // Now return the problem to its original size
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  // This will return the degrees of freedom of a single element of the augmented system
  // We will have to take the degrees of freedom of the original element and add a few more for the eigenvector values (Re and Im)
  unsigned AzimuthalSymmetryBreakingHandler::ndof(oomph::GeneralisedElement *const &elem_pt)
  {
    // This does not change if considering m contributions are incorporated already
    unsigned raw_ndof = elem_pt->ndof();
    {
      if (has_imaginary_part)
        return (3 * raw_ndof + 2);
      else
        return (2 * raw_ndof + 1);
    }
  }

  // This will cast the local equation number of an element to a global equation number.
  // Again, we have to consider the additional equations for the unknown eigenvector (Re and Im)
  unsigned long AzimuthalSymmetryBreakingHandler::eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local)
  {
    // Get the raw value
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn;
    if (ieqn_local < raw_ndof)
    {
      global_eqn = elem_pt->eqn_number(ieqn_local);
    }
    else if (ieqn_local < 2 * raw_ndof)
    {
      global_eqn = Ndof + elem_pt->eqn_number(ieqn_local - raw_ndof);
    }
    else if (has_imaginary_part)
    {
      if (ieqn_local < 3 * raw_ndof)
      {
        global_eqn = 2 * Ndof + elem_pt->eqn_number(ieqn_local - 2 * raw_ndof);
      }
      else if (ieqn_local == 3 * raw_ndof)
      {
        global_eqn = 3 * Ndof;
      }
      else
      {
        global_eqn = 3 * Ndof + 1;
      }
    }
    else
    {
      if (ieqn_local == 2 * raw_ndof)
      {
        global_eqn = 2 * Ndof;
      }      
    }
    return global_eqn;
  }

  // This will calculate the residual contribution of the original weak form by calling the function of the element
  void AzimuthalSymmetryBreakingHandler::get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals)
  {
    // Need to get raw residuals and jacobian
    unsigned raw_ndof = elem_pt->ndof();

    // Declare residuals, jacobian and mass matrix of real and imaginary contributions
    oomph::Vector<double> residuals_real(residuals.size(), 0);
    oomph::Vector<double> residuals_imag(residuals.size(), 0);
    DenseMatrix<double> jacobian_real(raw_ndof, raw_ndof, 0.0), M_real(raw_ndof, raw_ndof, 0.0);
    DenseMatrix<double> jacobian_imag(raw_ndof, raw_ndof, 0.0), M_imag(raw_ndof, raw_ndof, 0.0);

    // Get the base residuals, jacobian and mass matrix of real and imaginary parts
    set_assembled_residual(elem_pt, 1);
    elem_pt->get_jacobian_and_mass_matrix(residuals_real, jacobian_real, M_real);
    set_assembled_residual(elem_pt, 2);
    if (has_imaginary_part)
    {
      elem_pt->get_jacobian_and_mass_matrix(residuals_imag, jacobian_imag, M_imag);
    }
    set_assembled_residual(elem_pt, 0);
    elem_pt->get_residuals(residuals);

    // Initialise the pen-ultimate residual
    if (has_imaginary_part)
    {
      residuals[3 * raw_ndof] = -1.0 /(double)(Problem_pt->mesh_pt()->nelement());
      residuals[3 * raw_ndof + 1] = 0.0;
    }
    else
    {
      residuals[2 * raw_ndof] = -1.0 /(double)(Problem_pt->mesh_pt()->nelement());
    }

    // Now multiply to fill in the residuals
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      residuals[raw_ndof + i] = 0.0;
      if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned global_unknown = elem_pt->eqn_number(j);
        // First residual
        if (has_imaginary_part) 
        {
          residuals[raw_ndof + i] +=jacobian_real(i, j) * real_eigenvector[global_unknown] - jacobian_imag(i, j) * imag_eigenvector[global_unknown] + Omega * (M_real(i, j) * imag_eigenvector[global_unknown] + M_imag(i, j) * real_eigenvector[global_unknown]);        
          residuals[2 * raw_ndof + i] += jacobian_real(i, j) * imag_eigenvector[global_unknown] + jacobian_imag(i, j) * real_eigenvector[global_unknown] - Omega * (M_real(i, j) * real_eigenvector[global_unknown] + M_imag(i, j) * imag_eigenvector[global_unknown]);
        }
        else
        {
          residuals[raw_ndof + i] +=jacobian_real(i, j) * real_eigenvector[global_unknown];        
        }
      }
      // Get the global equation number
      unsigned global_eqn = elem_pt->eqn_number(i);

      if (has_imaginary_part) 
      {
        residuals[3 * raw_ndof] += (real_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
        // Imaginary eigenvector normalization
        residuals[3 * raw_ndof + 1] += (imag_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
      }
      else
      {
        residuals[2 * raw_ndof] += (real_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
      }
    }

    //=======Correct residuals according to boundary conditions dependent on m=======//

    // Loop through the RAW dofs
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      // Get global equation number to assess whether a boundary condition applies to it
      unsigned long global_eqn = eqn_number(elem_pt, i);
      // Assess whether a boundary condition applies to dof
      if (base_dofs_forced_zero.count(global_eqn))
      {
        // Correct residual value
        residuals[i] = 0; // Base residual values **PATCH
      }
      if (eigen_dofs_forced_zero.count(global_eqn))
      {
        residuals[raw_ndof + i] = 0; // Eigenvector residual values **PATCH
        if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0;
      }
    }
  }

  // Assembling the Jacobian matrix
  void AzimuthalSymmetryBreakingHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    bool ana_dparam = Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[3 * Ndof]);
    // Currently, we only calculate hessian analytically, if also ana_dparam is set (it is usually the case)
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);

    unsigned augmented_ndof = ndof(elem_pt);
    unsigned raw_ndof = elem_pt->ndof();

    // Declare residuals, jacobian and mass matrix of real and imaginary contributions
    oomph::Vector<double> residuals_real(residuals.size(), 0);
    oomph::Vector<double> residuals_imag(residuals.size(), 0);
    DenseMatrix<double> M(raw_ndof, raw_ndof, 0.0);
    DenseMatrix<double> jacobian_real(raw_ndof, raw_ndof, 0.0), M_real(raw_ndof, raw_ndof, 0.0);
    DenseMatrix<double> jacobian_imag(raw_ndof, raw_ndof, 0.0), M_imag(raw_ndof, raw_ndof, 0.0);

    if (!ana_hessian) // FD Hessian terms
    {

      // Get the base residuals, jacobian and mass matrix of real and imaginary parts
      set_assembled_residual(elem_pt, 1);
      elem_pt->get_jacobian_and_mass_matrix(residuals_real, jacobian_real, M_real);
      if (has_imaginary_part)
      {
        set_assembled_residual(elem_pt, 2);
        elem_pt->get_jacobian_and_mass_matrix(residuals_imag, jacobian_imag, M_imag);
      }
      set_assembled_residual(elem_pt, 0);
      elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);

      // Now fill in the actual residuals
      get_residuals(elem_pt, residuals);

      // Now the jacobian appears in other entries
      for (unsigned n = 0; n < raw_ndof; ++n)
      {
        for (unsigned m = 0; m < raw_ndof; ++m)
        {
          if (has_imaginary_part)
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian_real(n, m) + Omega * M_imag(n, m);
            jacobian(raw_ndof + n, 2 * raw_ndof + m) = -jacobian_imag(n, m) + Omega * M_real(n, m);
            jacobian(2 * raw_ndof + n, 2 * raw_ndof + m) = jacobian_real(n, m) + Omega * M_imag(n, m);
            jacobian(2 * raw_ndof + n, raw_ndof + m) = jacobian_imag(n, m) - Omega * M_real(n, m);
            unsigned global_eqn = elem_pt->eqn_number(m);
            jacobian(raw_ndof + n, 3 * raw_ndof + 1) += M_real(n, m) * imag_eigenvector[global_eqn] +
                                                        M_imag(n, m) * real_eigenvector[global_eqn];
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) -= M_real(n, m) * real_eigenvector[global_eqn] -
                                                            M_imag(n, m) * imag_eigenvector[global_eqn];
          }
          else
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian_real(n, m);
            unsigned global_eqn = elem_pt->eqn_number(m);
          }
        }

        unsigned local_eqn = elem_pt->eqn_number(n);
        if (has_imaginary_part)
        {
          jacobian(3 * raw_ndof, raw_ndof + n) = normalization_vector[local_eqn] / Count[local_eqn];
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = normalization_vector[local_eqn] / Count[local_eqn];
        }
        else
        {
          jacobian(2 * raw_ndof, raw_ndof + n) = normalization_vector[local_eqn] / Count[local_eqn];
        }
      }

      const double FD_step = 1.0e-8;

      Vector<double> newres_p(augmented_ndof), newres_m(augmented_ndof);

      // Loop over the dofs
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        // Just do the x's
        unsigned long global_eqn = eqn_number(elem_pt, n);
        double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
        double init = *unknown_pt;
        *unknown_pt += FD_step;

        // Get the new residuals
        get_residuals(elem_pt, newres_p);

        // Reset
        *unknown_pt = init;

        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + m, n) =
              (newres_p[raw_ndof + m] - residuals[raw_ndof + m]) / (FD_step);
          if (has_imaginary_part)
            jacobian(2 * raw_ndof + m, n) =
              (newres_p[2 * raw_ndof + m] - residuals[2 * raw_ndof + m]) /
              (FD_step);
        }
        // Reset the unknown
        *unknown_pt = init;
      }

      // Now do the global parameter
      // Either calculate the parameter derivatives analitically (ana_dparam=true) or calculate by finite difference.
      if (ana_dparam)
      {
        Vector<double> dres_dparam(augmented_ndof, 0.0);        
        this->get_dresiduals_dparameter(elem_pt, Problem_pt->GetDofPtr()[(has_imaginary_part ? 3 : 2) * Ndof], dres_dparam);
        for (unsigned m = 0; m < augmented_ndof - (has_imaginary_part ? 2 : 1); m++)
        {
          jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = dres_dparam[m];
        }
      }
      else
      {
        double *unknown_pt = Problem_pt->GetDofPtr()[(has_imaginary_part ? 3 : 2) * Ndof];
        double init = *unknown_pt;
        *unknown_pt += FD_step;

        Problem_pt->actions_after_change_in_bifurcation_parameter();
        // Get the new residuals
        get_residuals(elem_pt, newres_p);

        // Reset
        *unknown_pt = init;

        // Subtract
        *unknown_pt -= FD_step;
        get_residuals(elem_pt, newres_m);

        for (unsigned m = 0; m < augmented_ndof - (has_imaginary_part ? 2 : 1); m++)
        {
          jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = (newres_p[m] - residuals[m]) / FD_step;
        }
        // Reset the unknown
        *unknown_pt = init;
        Problem_pt->actions_after_change_in_bifurcation_parameter();
      }
    }
    else // Analytic Hessian version
    {

      // Cast to a pyoomph element, which has a function to calc Hessian*Vector of M and J simultaneosly. Pure oomph-lib does not have this
      pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);

      // Fill the real and imag eigenvector in local (elemental) indices
      // Vector<double> Yl(raw_ndof), Zl(raw_ndof);
      Vector<double> Eig((has_imaginary_part ? 2 : 1) * raw_ndof);
      for (unsigned _e = 0; _e < raw_ndof; _e++)
      {
        unsigned index = elem_pt->eqn_number(_e);
        Eig[_e] = real_eigenvector[index];
        if (has_imaginary_part) Eig[raw_ndof + _e] = imag_eigenvector[index];
      }

      DenseMatrix<double> JHess_real((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0), MHess_real((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0);
      DenseMatrix<double> JHess_imag((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0), MHess_imag((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0);

      DenseMatrix<double> dJ_real_dparam(raw_ndof, raw_ndof, 0.0), dJ_imag_dparam(raw_ndof, raw_ndof, 0.0);
      DenseMatrix<double> dM_real_dparam(raw_ndof, raw_ndof, 0.0), dM_imag_dparam(raw_ndof, raw_ndof, 0.0);
      Vector<double> dres_real_dparam(raw_ndof, 0.0), dres_imag_dparam(raw_ndof, 0.0);
      Vector<double> dres_dparam(raw_ndof, 0.0);

      residuals.initialise(0.0);
      jacobian.initialise(0.0);

      std::vector<SinglePassMultiAssembleInfo> assemble_info;
      int resindex;
      if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, 0)) >= 0)
      {
        assemble_info.push_back(SinglePassMultiAssembleInfo(resindex, &residuals, &jacobian));
        assemble_info.back().add_param_deriv(Parameter_pt, &dres_dparam);
      }
      if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, 1)) >= 0)
      {
        assemble_info.push_back(SinglePassMultiAssembleInfo(resindex, &residuals_real, &jacobian_real, &M_real));
        assemble_info.back().add_hessian(Eig, &JHess_real, &MHess_real);
        assemble_info.back().add_param_deriv(Parameter_pt, &dres_real_dparam, &dJ_real_dparam, &dM_real_dparam);
      }
      if (has_imaginary_part)
      {
        if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, 2)) >= 0)
        {
          assemble_info.push_back(SinglePassMultiAssembleInfo(resindex, &residuals_imag, &jacobian_imag, &M_imag));
          assemble_info.back().add_hessian(Eig, &JHess_imag, &MHess_imag);
          assemble_info.back().add_param_deriv(Parameter_pt, &dres_imag_dparam, &dJ_imag_dparam, &dM_imag_dparam);
        }
      }

      pyoomph_elem_pt->get_multi_assembly(assemble_info);

      // We will fill in the augmented residual vector once more by hand here. Otherwise, it we would call this->get_residual, we would assemble several matrices multiple times
      // Now multiply to fill in the residuals
      residuals[(has_imaginary_part ? 3 : 2) * raw_ndof] = -1.0 / (double)(Problem_pt->mesh_pt()->nelement());
      if (has_imaginary_part) residuals[3 * raw_ndof + 1] = 0.0;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + i] = 0.0;
        if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          if (has_imaginary_part) 
          {
            residuals[raw_ndof + i] += jacobian_real(i, j) * Eig[j] - jacobian_imag(i, j) * Eig[raw_ndof + j] + Omega * (M_real(i, j) * Eig[raw_ndof + j] + M_imag(i, j) * Eig[j]);
            residuals[2 * raw_ndof + i] += jacobian_real(i, j) * Eig[raw_ndof + j] + jacobian_imag(i, j) * Eig[j] - Omega * (M_real(i, j) * Eig[j] + M_imag(i, j) * Eig[raw_ndof + j]);
          }
          else
          {
            residuals[raw_ndof + i] += jacobian_real(i, j) * Eig[j] ;
          }
        }
        unsigned global_eqn = elem_pt->eqn_number(i);
        residuals[(has_imaginary_part ? 3 : 2) * raw_ndof] += (real_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
        if (has_imaginary_part) residuals[3 * raw_ndof + 1] += (imag_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
      }

      for (unsigned m = 0; m < raw_ndof; m++)
      {
        // First 'row' of the Jacobian:
        // raw Jacobian is already filled by elem_pt->get_jacobian()
        // parameter derivative of the residual
        jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = dres_dparam[m];

        // Second and third 'row' of the Jacobian: a lot of Hessian terms!
        jacobian(raw_ndof + m, (has_imaginary_part ? 3 : 2) * raw_ndof) = 0.0;
        if (has_imaginary_part)
        {
          jacobian(raw_ndof + m, 3 * raw_ndof + 1) = 0.0;
          jacobian(2 * raw_ndof + m, 3 * raw_ndof) = 0.0;
          jacobian(2 * raw_ndof + m, 3 * raw_ndof + 1) = 0.0;
        }
        for (unsigned int n = 0; n < raw_ndof; n++)
        {
          if (has_imaginary_part)
          {
            jacobian(raw_ndof + m, n) = JHess_real(m, n) - JHess_imag(raw_ndof + m, n) + Omega * (MHess_real(raw_ndof + m, n) + MHess_imag(m, n));
            jacobian(raw_ndof + m, raw_ndof + n) = jacobian_real(m, n) + Omega * M_imag(m, n);
            jacobian(raw_ndof + m, 2 * raw_ndof + n) = -jacobian_imag(m, n) + Omega * M_real(m, n);
            jacobian(raw_ndof + m, 3 * raw_ndof) += dJ_real_dparam(m, n) * Eig[n] - dJ_imag_dparam(m, n) * Eig[raw_ndof + n] + Omega * (dM_real_dparam(m, n) * Eig[raw_ndof + n] + dM_imag_dparam(m, n) * Eig[n]);
            jacobian(raw_ndof + m, 3 * raw_ndof + 1) += M_real(m, n) * Eig[raw_ndof + n] + M_imag(m, n) * Eig[n];

            jacobian(2 * raw_ndof + m, n) = JHess_real(raw_ndof + m, n) + JHess_imag(m, n) - Omega * (MHess_real(m, n) + MHess_imag(raw_ndof + m, n));
            jacobian(2 * raw_ndof + m, raw_ndof + n) = jacobian_imag(m, n) - Omega * M_real(m, n);
            jacobian(2 * raw_ndof + m, 2 * raw_ndof + n) = jacobian_real(m, n) - Omega * M_imag(m, n);
            jacobian(2 * raw_ndof + m, 3 * raw_ndof) += dJ_real_dparam(m, n) * Eig[raw_ndof + n] + dJ_imag_dparam(m, n) * Eig[n] - Omega * (dM_real_dparam(m, n) * Eig[n] + dM_imag_dparam(m, n) * Eig[raw_ndof + n]);
            jacobian(2 * raw_ndof + m, 3 * raw_ndof + 1) -= M_real(m, n) * Eig[n] + M_imag(m, n) * Eig[raw_ndof + n];
          }
          else
          {
            jacobian(raw_ndof + m, n) = JHess_real(m, n);
            jacobian(raw_ndof + m, raw_ndof + n) = jacobian_real(m, n);
            jacobian(raw_ndof + m, 2 * raw_ndof) += dJ_real_dparam(m, n) * Eig[n];
          }
        }
        unsigned local_eqn = elem_pt->eqn_number(m);
        jacobian((has_imaginary_part ? 3 : 2) * raw_ndof, raw_ndof + m) = normalization_vector[local_eqn] / Count[local_eqn];
        if (has_imaginary_part)   jacobian(3 * raw_ndof + 1, 2 * raw_ndof + m) = normalization_vector[local_eqn] / Count[local_eqn];
      }
    }

    //=======Correct jacobian according to boundary conditions dependent on m=======//

    // Get total number of dofs in the element
    unsigned ndof = elem_pt->ndof();
    // Loop through the dofs
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      // Get global equation number to assess whether a boundary condition applies to it
      unsigned long global_eqn = eqn_number(elem_pt, i);
      // Assess whether a boundary condition applies to dof
      if (base_dofs_forced_zero.count(global_eqn))
      {
        // Correct jacobian value
        for (unsigned j = 0; j < ndof; j++)
        {
          jacobian(i, j) = 0.0;
        }
        jacobian(i, i) = 1.0;
        if (ana_hessian)
        {
          residuals[i] = 0.0;
        }
      }
      if (eigen_dofs_forced_zero.count(global_eqn))
      {
        // Correct jacobian value
        for (unsigned j = 0; j < ndof; j++)
        {
          jacobian(raw_ndof + i, j) = 0.0;
          if (has_imaginary_part) jacobian(2 * raw_ndof + i, j) = 0.0;
        }
        jacobian(raw_ndof + i, raw_ndof + i) = 1.0;
        if (has_imaginary_part) jacobian(2 * raw_ndof + i, 2 * raw_ndof + i) = 1.0;
        if (ana_hessian)
        {
          residuals[raw_ndof + i] =0.0;
          if (has_imaginary_part)  residuals[2 * raw_ndof + i] = 0.0;
        }
      }
    }

    // DEBUG ANA
    /*
     if (ana_hessian)
     {
       oomph::Vector<double> fd_residuals(residuals.size(),0.0);
       oomph::DenseMatrix<double> fd_jacobian(residuals.size(),residuals.size(),0.0);
       Problem_pt->unset_analytic_hessian_products();
       this->get_jacobian(elem_pt,fd_residuals,fd_jacobian);
       Problem_pt->set_analytic_hessian_products();
       double delta=1e-8;
       std::string elem_info=dynamic_cast<pyoomph::BulkElementBase*>(elem_pt)->get_code_instance()->get_code()->get_file_name();
       for (unsigned int i=0;i<fd_residuals.size();i++)
       {
         std::string iwhat=(i<raw_ndof ? "raw" : (i<2*raw_ndof ? "Y" : (i<3*raw_ndof ? "Z" : (i==3*raw_ndof  ? "Param"  : "Omega" ) )));
         if (std::fabs(residuals[i]-fd_residuals[i])>delta)
         {
           std::cout << "ERROR in R : " << i << " of " << residuals.size() << " : " << residuals[i] << "  " << fd_residuals[i] << "   delta " << std::fabs(residuals[i]-fd_residuals[i]) << " in " << elem_info << std::endl;
         }
         for (unsigned int j=0;j<fd_residuals.size();j++)
         {
         std::string jwhat=(j<raw_ndof ? "raw" : (j<2*raw_ndof ? "Y" : (j<3*raw_ndof ? "Z" : (j==3*raw_ndof  ? "Param"  : "Omega" ) )));
          if (std::fabs(jacobian(i,j)-fd_jacobian(i,j))>delta)
          {
           std::cout << "ERROR in J : " << i << " , " << j << " of " << residuals.size() << " : " << jacobian(i,j) << "  " << fd_jacobian(i,j) << "   delta " << std::fabs(jacobian(i,j)-fd_jacobian(i,j))<<  " in " << elem_info  << " @ Deriv of " << iwhat << " wrto " << jwhat << std::endl;
          }
         }

       }
     }
     */
  }

  // Derivative of the augmented residuals with respect to a parameter

  void AzimuthalSymmetryBreakingHandler::get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt,
                                                                   oomph::Vector<double> &dres_dparam)
  {
    // Need to get raw residuals and jacobian
    unsigned raw_ndof = elem_pt->ndof();
    //    if (parameter_pt!=Parameter_pt)   std::cout << "PARAM DERIV " << parameter_pt << " " <<  Parameter_pt << std::endl;

    // Declare residuals, jacobian and mass matrix of real and imaginary contributions
    oomph::Vector<double> dres_real_dparam(raw_ndof, 0);
    oomph::Vector<double> dres_imag_dparam(raw_ndof, 0);
    DenseMatrix<double> djac_dparam(raw_ndof), dM_dparam(raw_ndof);
    DenseMatrix<double> djac_real_dparam(raw_ndof), dM_real_dparam(raw_ndof);
    DenseMatrix<double> djac_imag_dparam(raw_ndof), dM_imag_dparam(raw_ndof);

    // Get the dresiduals, djacobian and dmass_matrix for base, real and imaginary jacobians
    set_assembled_residual(elem_pt, 1);
    elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt, dres_real_dparam, djac_real_dparam, dM_real_dparam);
    if (has_imaginary_part)
    {
      set_assembled_residual(elem_pt, 2);
      elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt, dres_imag_dparam, djac_imag_dparam, dM_imag_dparam);
    }
    set_assembled_residual(elem_pt, 0);
    elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt, dres_dparam, djac_real_dparam, dM_real_dparam);

    // Initialise the pen-ultimate residual, which does not
    // depend on the parameter
    dres_dparam[(has_imaginary_part ? 3 : 2) * raw_ndof] = 0.0;
    if (has_imaginary_part)  dres_dparam[3 * raw_ndof + 1] = 0.0;

    // Now multiply to fill in the residuals
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      dres_dparam[raw_ndof + i] = 0.0;
      if (has_imaginary_part) dres_dparam[2 * raw_ndof + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned global_unknown = elem_pt->eqn_number(j);
        // Real part
        if (has_imaginary_part)
        {
          dres_dparam[raw_ndof + i] +=
              djac_real_dparam(i, j) * real_eigenvector[global_unknown] - djac_imag_dparam(i, j) * imag_eigenvector[global_unknown] +
              Omega * (dM_real_dparam(i, j) * imag_eigenvector[global_unknown] + dM_imag_dparam(i, j) * real_eigenvector[global_unknown]);
          // Imaginary part
          dres_dparam[2 * raw_ndof + i] +=
              djac_real_dparam(i, j) * imag_eigenvector[global_unknown] + djac_imag_dparam(i, j) * real_eigenvector[global_unknown] -
              Omega * (dM_real_dparam(i, j) * real_eigenvector[global_unknown] - dM_imag_dparam(i, j) * imag_eigenvector[global_unknown]);
        }
        else
        {
          dres_dparam[raw_ndof + i] +=djac_real_dparam(i, j) * real_eigenvector[global_unknown];
        }
      }
    }
  }

  // Derivative of the augmented Jacobian with respect to the parameter
  void AzimuthalSymmetryBreakingHandler::get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam)
  {
    throw_runtime_error("AzimuthalSymmetryBreakingHandler::get_djacobian_dparameter(oomph::GeneralisedElement* const &elem_pt,double* const &parameter_pt,oomph::Vector<double> &dres_dparam,oomph::DenseMatrix<double> &djac_dparam)");
    // TODO: Fill it
    // Is it required?
  }

  // Get the eigenfunction
  void AzimuthalSymmetryBreakingHandler::get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction)
  {
    // There is a real and imaginary part of the eigen vector
    eigenfunction.resize((has_imaginary_part ? 2 : 1)); // So we must return two real vectors (Re and Im)
    // build a distribution for the storage of the eigenvector parts
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false); // The eigenvectors have Ndof entries, i.e. the number of dofs of the original problem
    // Rebuild the vector
    eigenfunction[0].build(&dist, 0.0);
    if (has_imaginary_part) eigenfunction[1].build(&dist, 0.0);
    // Set the value to be the null vector
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = real_eigenvector[n];
      if (has_imaginary_part) eigenfunction[1][n] = imag_eigenvector[n];
    }
  }

  // Pyoomph has different residual contributions. The original residual along with its jacobian and the real and imag part of the azimuthal Jacobian and mass matrix. We get the indices of these contributions in beforehand.
  // We assume that all codes are initially set to the stage so that the original axisymmetric residual is solved
  void AzimuthalSymmetryBreakingHandler::setup_solved_azimuthal_contributions(std::string real_angular_J_and_M, std::string imag_angular_J_and_M)
  {
    pyoomph::Problem *prob = dynamic_cast<pyoomph::Problem *>(Problem_pt);
    if (!prob)
      throw_runtime_error("Not a pyoomph::Problem... Strange");
    // Each generated code may have different indices (i.e. not all contributions are present on each generated code)
    // Therefore, we must make a map from each generated code to the three residuals/jacobians/etc
    auto codes = prob->get_bulk_element_codes();
    for (unsigned int i = 0; i < codes.size(); i++)
    {
      int orig_residual = codes[i]->get_func_table()->current_res_jac; // Store the initial residual (base state)
      int real_azimuthal = -1;                                         // By default, no azimuthal residual present
      int imag_azimuthal = -1;
      if (codes[i]->_set_solved_residual(real_angular_J_and_M))
      {
        real_azimuthal = codes[i]->get_func_table()->current_res_jac; // Get the real residual index
      }
      if (codes[i]->_set_solved_residual(imag_angular_J_and_M))
      {
        imag_azimuthal = codes[i]->get_func_table()->current_res_jac; // Get he imaginary residual index
      }
      codes[i]->get_func_table()->current_res_jac = orig_residual; // Reset it

      // And store it in the mapping
      //std::cout << "MAPPING " << codes[i]->get_file_name() << " " << orig_residual << " " << real_azimuthal << " " << imag_azimuthal << std::endl;
      residual_contribution_indices[codes[i]] = AzimuthalSymmetryBreakingResidualContributionList(codes[i], orig_residual, real_azimuthal, imag_azimuthal);
    }
  }

  // Please reset it to the base state at the end via
  // set_assembled_residual(element,0)
  int AzimuthalSymmetryBreakingHandler::resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    return entry.residual_indices[residual_mode];
  }

  bool AzimuthalSymmetryBreakingHandler::set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    // Setup the solved residual by the index (-1 means no contribution)
    entry.code->get_func_table()->current_res_jac = entry.residual_indices[residual_mode];
    return entry.residual_indices[residual_mode] >= 0;
  }

}
