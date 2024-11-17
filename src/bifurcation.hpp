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

#pragma once

#include "assembly_handler.h"
#include <set>

namespace pyoomph
{
  class Problem;
  class DynamicBulkElementCode; // Forward decl.

  // Reimplementation of the Hopf-Handler with some changes
  class MyHopfHandler : public oomph::AssemblyHandler
  {
  protected:
    unsigned Solve_which_system;
    Problem *Problem_pt;
    double *Parameter_pt;
    unsigned Ndof;
    double Omega;
    oomph::Vector<double> Phi;
    oomph::Vector<double> Psi;
    oomph::Vector<double> C;
    oomph::Vector<int> Count;
    double eigenweight;

  public:
    bool call_param_change_handler;
    double FD_step = 1e-8;
    bool symmetric_FD = false;
    MyHopfHandler(Problem *const &problem_pt, double *const &parameter_pt);

    MyHopfHandler(Problem *const &problem_pt, double *const &paramter_pt,
                  const double &omega, const oomph::DoubleVector &phi,
                  const oomph::DoubleVector &psi);

    ~MyHopfHandler();

    void set_eigenweight(double ew);
    unsigned get_problem_ndof() { return Ndof; }
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt);

    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt,
                             const unsigned &ieqn_local);

    void get_residuals(oomph::GeneralisedElement *const &elem_pt,
                       oomph::Vector<double> &residuals);

    void get_jacobian(oomph::GeneralisedElement *const &elem_pt,
                      oomph::Vector<double> &residuals,
                      oomph::DenseMatrix<double> &jacobian);

    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt,
                                   double *const &parameter_pt,
                                   oomph::Vector<double> &dres_dparam);

    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt,
                                  double *const &parameter_pt,
                                  oomph::Vector<double> &dres_dparam,
                                  oomph::DenseMatrix<double> &djac_dparam);

    void get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt,
                                     oomph::Vector<double> const &Y,
                                     oomph::DenseMatrix<double> const &C,
                                     oomph::DenseMatrix<double> &product);

    void debug_analytical_filling(oomph::GeneralisedElement *elem_pt, double eps);
    int bifurcation_type() const { return 3; }

    double *bifurcation_parameter_pt() const
    {
      return Parameter_pt;
    }

    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction);

    const double &omega() const { return Omega; }

    void solve_standard_system();

    void solve_complex_system();

    void solve_full_system();

    void realign_C_vector(); // Reset the C-vector (which enforces the non-triviality of the eigenvector)
  };

  //////////////////////////////////////////////////////////

  // List of the tree residual contribution indices of each generated C code
  class PitchForkResidualContributionList
  {
  public:
    DynamicBulkElementCode *code;
    std::vector<int> residual_indices; // index 0 is base state, 1 is mass matrix residual
    PitchForkResidualContributionList(DynamicBulkElementCode *_code, int _base, int _massmat) : code(_code) { residual_indices = {_base, _massmat}; }
    PitchForkResidualContributionList() {}
  };

  class MyPitchForkHandler : public oomph::AssemblyHandler
  {
  protected:
    Problem *Problem_pt;
    unsigned Ndof;
    double Sigma;
    oomph::Vector<double> Y;
    oomph::Vector<double> Psi;
    oomph::Vector<double> C;
    oomph::Vector<int> Count;
    double *Parameter_pt;
    double eigenweight, symmetryweight;
    unsigned Nelement;
    std::map<const pyoomph::DynamicBulkElementCode *, PitchForkResidualContributionList> residual_contribution_indices;
    void setup_U_times_Psi_residual_indices();
    double get_integrated_U_dot_Psi(oomph::GeneralisedElement *const &elem_pt, oomph::DenseMatrix<double> &psi_i_times_psi_j);
    bool set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    int resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);

  public:
    bool call_param_change_handler;
    MyPitchForkHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector);
    ~MyPitchForkHandler();
    void set_eigenweight(double ew);
    unsigned get_problem_ndof() { return Ndof; }
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt);
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local);
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals);
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam);
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam);
    void get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product);
    int bifurcation_type() const { return 2; }
    double *bifurcation_parameter_pt() const { return Parameter_pt; }
    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction);
    void solve_full_system();
  };

  //////////////////////////////////////////////////////////

  class MyFoldHandler : public oomph::AssemblyHandler
  {
    enum
    {
      Full_augmented,
      Block_J,
      Block_augmented_J
    };

    unsigned Solve_which_system;
    Problem *Problem_pt;
    unsigned Ndof;
    oomph::Vector<double> Phi;
    oomph::Vector<double> Y;
    oomph::Vector<int> Count;
    double *Parameter_pt;
    double eigenweight;

  public:
    unsigned get_problem_ndof() { return Ndof; }
    bool call_param_change_handler;
    double FD_step = 1e-8;
    bool symmetric_FD = false;
    MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt);
    MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &eigenvector);
    MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const oomph::DoubleVector &normalisation);
    ~MyFoldHandler();

    void set_eigenweight(double ew);

    unsigned ndof(oomph::GeneralisedElement *const &elem_pt);

    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local);

    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals);

    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);

    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam);
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam);

    void get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product);

    int bifurcation_type() const { return 1; }

    double *bifurcation_parameter_pt() const { return Parameter_pt; }

    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction);

    void solve_augmented_block_system();
    void solve_block_system();
    void solve_full_system();
    void realign_C_vector(); // Reset the C-vector (which enforces the non-triviality of the eigenvector)
  };

  //////////////////////////////

  // List of the tree residual contribution indices of each generated C code
  class AzimuthalSymmetryBreakingResidualContributionList
  {
  public:
    DynamicBulkElementCode *code;
    std::vector<int> residual_indices; // index 0 is base state(axisymm), 1 is real azimuthal and 2 is imag azimuthal
    AzimuthalSymmetryBreakingResidualContributionList(DynamicBulkElementCode *_code, int _base, int _real, int _imag) : code(_code) { residual_indices = {_base, _real, _imag}; }
    AzimuthalSymmetryBreakingResidualContributionList() {}
  };

  // Actual assembly handler class for axial symmetry breaking systems
  class AzimuthalSymmetryBreakingHandler : public oomph::AssemblyHandler
  {
    Problem *Problem_pt;                    // Pointer to the problem class
    unsigned Ndof;                          // Degrees of freedom of the original problem (non-augmented)
    oomph::Vector<double> real_eigenvector; // Storage for the real and imaginary eigenvector (which will be a part of the unknowns)
    oomph::Vector<double> imag_eigenvector;

    // A vector used to normalize the eigenvector (i.e. to prevent that real_eigenvector=0,
    // which trivially solves the eigenproblem with eigenvalue 0)
    oomph::Vector<double> normalization_vector;

    // Vector counting the occurrence of equations (many equations will be accessed by different elements)
    // The contributions to the normalization constraints will be hence added multiple times for these degrees
    // of freedom by different elements. Therefore, we have to normalize by the number of elements contributing
    // to each equation
    oomph::Vector<int> Count;
    double Omega;         // Imaginary part of the eigenvalue to be determined
    double *Parameter_pt; // Pointer to the critical parameter to find by this bifurcation analysis

    // Each generated C code has three residual forms: These are stored in this mapping
    std::map<const pyoomph::DynamicBulkElementCode *, AzimuthalSymmetryBreakingResidualContributionList>
        residual_contribution_indices;
    // We setup this mapping in beforehand

    // Once the mapping is set up, you can call this function to set which residual form should be assembled by the get_jacobian/get_residual/...
    // Please reset it to the base state at the end via set_assembled_residual(element,0);
    // it will return false, if there is zero residual contribution (an element might have e.g. no imaginary contribution).
    // Then you could skip this contributions and the derivatives thereof.
    bool set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    int resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    // Indices (global equation numbers) of degrees of freedom which are forced to be zero due to the boundary conditions at the axis.
    // By default, all degrees at the axis are free (i.e. have an equation assigned). Depending on m, this set is filled.
    // If an equation is present in this set [ if (global_equations_forced_zero.count(global_eq)) {...} ], we do not add anything to
    // the residual (R_j=0) and we fill M_ij=0 and J_ij=0, except J_jj=1 (or any other nonzero value).
    std::set<unsigned> base_dofs_forced_zero;  // Base degrees of freedom forced to zero (e.g. velocity_x at the axis)
    std::set<unsigned> eigen_dofs_forced_zero; // Degrees of freedom of the eigenvector forced to zero (Note: the equation
                                               // indices are the base equations, not the indices of the eigendegrees!).
                                               // For e.g. m=1, equation of velocity_x at axis is part of base_dofs_forced_zero,
                                               // but not of eigen_dofs_forced_zero
  public:
    unsigned get_problem_ndof() { return Ndof; } // Returning the degrees of freedom of the original system (non-augmented)

    // Setter of the forced zero degrees
    void set_global_equations_forced_zero(const std::set<unsigned> &base, const std::set<unsigned> &eigen)
    {
      base_dofs_forced_zero = base;
      eigen_dofs_forced_zero = eigen;
    }

    // These two guys will be false all the time.
    // when call_param_change_handler==true, it will call a function which is used in oomph-lib, but never in pyoomph. So we can skip it
    bool call_param_change_handler;

    double FD_step; // Finite difference step (usually small)

    bool has_imaginary_part; // If the imaginary part of the jacobian or mass matrix is present

    // Constructors. We must pass a problem, a parameter to optimize (i.e. to change in order to get Re(eigenvalue)=0) and a guess of the eigenvector and the imaginary part of the eigenvector
    AzimuthalSymmetryBreakingHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &real_eigen, const oomph::DoubleVector &imag_eigen, const double &Omega_guess,bool has_imag);
    // Destructor (used for cleaning up memory)
    ~AzimuthalSymmetryBreakingHandler();

    // Pyoomph has different residual contributions. The original residual along with its
    // jacobian and the real and imag part of the azimuthal Jacobian and mass matrix.
    // We get the indices of these contributions in beforehand. We assume that all codes are
    // initially set to the stage whether the original axisymmetric residual is solved
    void setup_solved_azimuthal_contributions(std::string real_angular_J_and_M, std::string imag_angular_J_and_M);

    // This will return the degrees of freedom of a single element of the augmented system
    // We will have to take the degrees of freedom of the original element and add a few more
    // for the eigenvector values (Re and Im)
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt);

    // This will cast the local equation number of an element to a global equation number.
    // Again, we have to consider the additional equations for the unknown eigenvector (Re and Im)
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local);

    // This will calculate the residual contribution of the original weak form by calling the function of the element
    // However, we will also have to add the contributions to the augmented residual form, i.e. the ones determining the eigenvector and critical parameter
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals);

    // Same for the Jacobian: We must add the contributions for the augmented system
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);

    // Derivative of the augmented residuals with respect to the parameter
    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam);
    // Derivative of the augmented Jacobian with respect to the parameter
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam);

    int bifurcation_type() const { return 3; } // Internally used in oomph-lib. I assume it is best to return 3 (Hopf), since we have real and imag parts

    // Function to access the bifurcation parameter
    double *bifurcation_parameter_pt() const { return Parameter_pt; }

    // Get the eigenfunction
    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction);
    const double &omega() const { return Omega; } // and the value of the imaginary part of the eigenvalue

    void solve_full_system();

    //  void realign_C_vector(); //Reset the C-vector (which enforces the non-triviality of the eigenvector)
  };

}
