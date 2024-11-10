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

#include "problem.hpp"
#include "elements.hpp"
#include "jitbridge.h"
#include "exception.hpp"
#include "nodes.hpp"
#include "codegen.hpp"
#include "bifurcation.hpp"
#include "ccompiler.hpp"
#include "logging.hpp"

extern "C"
{
	void _pyoomph_check_compiler_size(unsigned long long jitsize, unsigned long long internal_size, char *name)
	{
		if (jitsize != internal_size)
		{
			std::ostringstream errmsg;
			std::string nam = name;
			errmsg << "Mismatch between compiler sizes. Test failed: " << nam << std::endl
				   << "Expected " << internal_size << ", but got " << jitsize;
			throw_runtime_error(errmsg.str());
		}
	}
}

namespace pyoomph
{

	void RequiredShapes_merge(JITFuncSpec_RequiredShapes_FiniteElement_t *src, JITFuncSpec_RequiredShapes_FiniteElement_t *dest)
	{
		dest->psi_C1 |= src->psi_C1;
		dest->psi_C2 |= src->psi_C2;
		dest->psi_C2TB |= src->psi_C2TB;
		dest->psi_C1TB |= src->psi_C1TB;
		dest->psi_DL |= src->psi_DL;
		dest->psi_D0 |= src->psi_D0;
		dest->dx_psi_C1 |= src->dx_psi_C1;
		dest->dx_psi_C2 |= src->dx_psi_C2;
		dest->dx_psi_C2TB |= src->dx_psi_C2TB;
		dest->dx_psi_C1TB |= src->dx_psi_C1TB;
		dest->dx_psi_DL |= src->dx_psi_DL;
		dest->dx_psi_D0 |= src->dx_psi_D0;
		dest->dX_psi_C1 |= src->dX_psi_C1;
		dest->dX_psi_C2 |= src->dX_psi_C2;
		dest->dX_psi_C2TB |= src->dX_psi_C2TB;
		dest->dX_psi_C1TB |= src->dX_psi_C1TB;
		dest->dX_psi_DL |= src->dX_psi_DL;
		dest->dX_psi_D0 |= src->dX_psi_D0;
		dest->psi_Pos |= src->psi_Pos;
		dest->dx_psi_Pos |= src->dx_psi_Pos;
		dest->dX_psi_Pos |= src->dX_psi_Pos;
		dest->normal_Pos |= src->normal_Pos;
		dest->elemsize_Eulerian_Pos |= src->elemsize_Eulerian_Pos;
		dest->elemsize_Lagrangian_Pos |= src->elemsize_Lagrangian_Pos;
		dest->elemsize_Eulerian_cartesian_Pos |= src->elemsize_Eulerian_cartesian_Pos;
		dest->elemsize_Lagrangian_cartesian_Pos |= src->elemsize_Lagrangian_cartesian_Pos;
		if (src->bulk_shapes)
		{
			if (!dest->bulk_shapes)
				dest->bulk_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
			RequiredShapes_merge(src->bulk_shapes, dest->bulk_shapes);
		}
		if (src->opposite_shapes)
		{
			if (!dest->opposite_shapes)
				dest->opposite_shapes = (JITFuncSpec_RequiredShapes_FiniteElement_t *)std::calloc(1, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
			RequiredShapes_merge(src->opposite_shapes, dest->opposite_shapes);
		}
	}

	void RequiredShapes_free(JITFuncSpec_RequiredShapes_FiniteElement_t *p)
	{
		if (p->bulk_shapes)
			RequiredShapes_free(p->bulk_shapes);
		if (p->opposite_shapes)
			RequiredShapes_free(p->opposite_shapes);
		std::free(p);
	}

	DynamicBulkElementCode::DynamicBulkElementCode(Problem *prob, CCompiler *ccompiler, std::string fnam, FiniteElementCode *elem) : problem(prob), compiler(ccompiler), filename(fnam), functable(NULL), element_class(elem), so_handle(NULL)
	{
		JIT_ELEMENT_init_SPEC initfunc = ccompiler->get_init_func();
		if (!initfunc)
		{
			throw_runtime_error("Cannot load the JIT code entry point");
		}
		so_handle = ccompiler->get_current_handle();
		ccompiler->reset_current_handle();
		functable = new JITFuncSpec_Table_FiniteElement_t;
		memset(functable, 0, sizeof(JITFuncSpec_Table_FiniteElement_t));

		functable->check_compiler_size = _pyoomph_check_compiler_size;
		initfunc(functable);

		// Merge the required shapes to add all external data
		JITFuncSpec_RequiredShapes_FiniteElement *merged = &(functable->merged_required_shapes);

		for (unsigned int i = 0; i < functable->num_res_jacs; i++)
		{
			RequiredShapes_merge(&functable->shapes_required_ResJac[i], merged);
			RequiredShapes_merge(&functable->shapes_required_Hessian[i], merged);
		}
		RequiredShapes_merge(&functable->shapes_required_IntegralExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_LocalExprs, merged);
		RequiredShapes_merge(&functable->shapes_required_Z2Fluxes, merged);
		RequiredShapes_merge(&functable->shapes_required_TracerAdvection, merged);

		functable->handle = so_handle;

		// Export the functions to call
		functable->get_element_size = _pyoomph_get_element_size;
		functable->invoke_callback = _pyoomph_invoke_callback;
		functable->invoke_multi_ret = _pyoomph_invoke_multi_ret;
		functable->fill_shape_buffer_for_point = _pyoomph_fill_shape_buffer_for_point;

		for (unsigned int i = 0; i < functable->numintegral_expressions; i++)
			integral_function_map[functable->integral_expressions_names[i]] = i;
	}

	DynamicBulkElementCode::~DynamicBulkElementCode()
	{
		// std::cout << "UNLOADING ELEMENT CODE " << filename << " FUNCTABLE " << functable << " SO HANDLE " << so_handle  << std::endl << std::flush;
		// std::cout << "COMPILER  " << compiler  << std::endl << std::flush;

		if (functable)
		{
			if (pyoomph_verbose)
			{
				std::cout << "Cleaning memory of functable" << std::endl << std::flush;
			}
			if (functable->clean_up) functable->clean_up(functable);
			delete functable; // TODO: Also delete the malloced subentries here
		}

		if (pyoomph_verbose)
		{
				std::cout << "Closing library handle " << this->get_file_name() << std::endl << std::flush;
		}
		compiler->close_handle(so_handle);
		if (pyoomph_verbose)
		{
				std::cout << "Closed library handle " << std::endl << std::flush;
		}
		so_handle = NULL;
		functable = NULL;
	}

	int DynamicBulkElementCode::get_integral_function_index(std::string n)
	{
		if (!integral_function_map.count(n))
			return -1;
		return integral_function_map[n];
	}

	unsigned DynamicBulkElementCode::_set_solved_residual(std::string name)
	{
		int res_jac_index = -1;
		for (unsigned int i = 0; i < functable->num_res_jacs; i++)
		{
			std::string n = functable->res_jac_names[i];
			if (n == name)
			{
				res_jac_index = i;
				break;
			}
		}
		functable->current_res_jac = res_jac_index;
		if (res_jac_index >= 0)
			return 1;
		else
			return 0;
	}
	DynamicBulkElementInstance *DynamicBulkElementCode::factory_instance(pyoomph::Mesh *bulkmesh)
	{
		return new DynamicBulkElementInstance(this, bulkmesh);
	}

	//////////////////////////////////////////

	void ExternalDataLinkVector::reindex_elemental_data()
	{
		elemental_data.clear();
		for (unsigned int i = 0; i < this->size(); i++)
		{
			int found = -1;
			for (unsigned int e = 0; e < elemental_data.size(); e++)
			{
				if (elemental_data[e] == this->at(i).data)
				{
					found = e;
					break;
				}
			}
			if (found < 0)
			{
				found = elemental_data.size();
				elemental_data.push_back(this->at(i).data);
			}
			this->at(i).elemental_index = found;
		}
	}

	///////////////////////////////////////////

	DynamicBulkElementInstance::DynamicBulkElementInstance(DynamicBulkElementCode *d, pyoomph::Mesh *bm) : dyn(d), // local_field_to_global_field_index_C1(d->functable->numfields_C1,-1),
																												   //		local_field_to_global_field_index_C2(d->functable->numfields_C2,-1),
																												   //		local_global_parameter_to_global_index(d->functable->numglobal_params,-1),
																										   linked_external_data(d->functable->numfields_ED0),
																										   bulkmesh(bm)
	{
		/*
		  for (unsigned int i=0; i<d->functable->num_nullified_bulk_residuals;i++)
		  {
			std::string fn=d->functable->nullified_bulk_residuals[i];
			int index;
			if (fn=="coordinate_x") index=-1;
			else if (fn=="coordinate_y") index=-2;
			else if (fn=="coordinate_z") index=-3;
			else
			{
			  index=get_nodal_field_index(fn);
			  if (index==-1) throw_runtime_error("Cannot nullify the bulk DoF " +fn);
			}
			nullify_bulk_residuals.insert(index);
		  }
		 */
	}

	void DynamicBulkElementInstance::link_external_data(std::string name, oomph::Data *data, int index)
	{
		int found = -1;
		for (unsigned int i = 0; i < dyn->functable->numfields_ED0; i++)
		{
			if (name == std::string(dyn->functable->fieldnames_ED0[i]))
			{
				found = i;
				break;
			}
		}
		if (found == -1)
			throw_runtime_error("Cannot link external data '" + name + "' since this is not required by the equation code");
		linked_external_data[found] = ExternalDataLink(data, index);
		linked_external_data.reindex_elemental_data();
	}

	std::map<std::string, unsigned> DynamicBulkElementInstance::get_nodal_field_indices()
	{
		std::map<std::string, unsigned> res;
		unsigned offs = 0;
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1[i]] = offs + i;
		}
		offs += dyn->functable->numfields_C1_basebulk;

		for (unsigned int i = 0; i < dyn->functable->numfields_D2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1TB[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1[i]] = offs + i;
		}
		offs += dyn->functable->numfields_D1_basebulk;

		// Now the additional ones
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB - dyn->functable->numfields_C2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2TB[i + dyn->functable->numfields_C2TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C2TB - dyn->functable->numfields_C2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C2 - dyn->functable->numfields_C2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C2[i + dyn->functable->numfields_C2_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C2 - dyn->functable->numfields_C2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1TB - dyn->functable->numfields_C1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1TB[i + dyn->functable->numfields_C1TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C1TB - dyn->functable->numfields_C1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_C1 - dyn->functable->numfields_C1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_C1[i + dyn->functable->numfields_C1_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_C1 - dyn->functable->numfields_C1_basebulk;

		// Now the additional ones
		for (unsigned int i = 0; i < dyn->functable->numfields_D2TB - dyn->functable->numfields_D2TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2TB[i + dyn->functable->numfields_D2TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_D2TB - dyn->functable->numfields_D2TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D2 - dyn->functable->numfields_D2_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D2[i + dyn->functable->numfields_D2_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_D2 - dyn->functable->numfields_D2_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1TB - dyn->functable->numfields_D1TB_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1TB[i + dyn->functable->numfields_D1TB_basebulk]] = offs + i;
		}
		offs += dyn->functable->numfields_D1TB - dyn->functable->numfields_D1TB_basebulk;
		for (unsigned int i = 0; i < dyn->functable->numfields_D1 - dyn->functable->numfields_D1_basebulk; i++)
		{
			res[dyn->functable->fieldnames_D1[i + dyn->functable->numfields_D1_basebulk]] = offs + i;
		}

		return res;
	}

	std::map<std::string, unsigned> DynamicBulkElementInstance::get_elemental_field_indices()
	{
		std::map<std::string, unsigned> res;
		for (unsigned int i = 0; i < dyn->functable->numfields_DL; i++)
		{
			res[dyn->functable->fieldnames_DL[i]] = i;
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_D0; i++)
		{
			res[dyn->functable->fieldnames_D0[i]] = i + dyn->functable->numfields_DL;
		}
		return res;
	}

	int DynamicBulkElementInstance::get_discontinuous_field_index(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->numfields_DL; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_DL[i]))
			{
				return i + dyn->functable->internal_offset_DL;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_D0; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_D0[i]))
			{
				return i + dyn->functable->internal_offset_D0;
			}
		}
		return -1;
	}

	int DynamicBulkElementInstance::get_nodal_field_index(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2TB[i]))
			{
				return i;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C2_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2[i]))
			{
				return i+ dyn->functable->numfields_C2TB_basebulk;
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C1_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C1[i]))
			{
				return i + dyn->functable->numfields_C2_basebulk+ dyn->functable->numfields_C2TB_basebulk;
			}
		}
		return -1;
	}

	unsigned DynamicBulkElementInstance::resolve_interface_dof_id(std::string n)
	{
		return this->get_bulk_mesh()->resolve_interface_dof_id(n);
	}

	std::string DynamicBulkElementInstance::get_space_of_field(std::string name)
	{
		for (unsigned int i = 0; i < dyn->functable->numfields_C2TB_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2TB[i]))
			{
				return "C2TB";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C2_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C2[i]))
			{
				return "C2";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_C1_basebulk; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_C1[i]))
			{
				return "C1";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_DL; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_DL[i]))
			{
				return "DL";
			}
		}
		for (unsigned int i = 0; i < dyn->functable->numfields_D0; i++)
		{
			if (!strcmp(name.c_str(), dyn->functable->fieldnames_D0[i]))
			{
				return "D0";
			}
		}
		return "";
	}

	void DynamicBulkElementInstance::sanity_check()
	{
		/*
		 for (unsigned int i=0;i<dyn->functable->numglobal_params;i++)
		 {
			if (local_global_parameter_to_global_index[i]<0) throw_runtime_error("Elemental parameter "+std::string(dyn->functable->global_paramnames[i])+" not bound");
		 }
		*/
		/*
		 for (unsigned int i=0;i<dyn->functable->numfields_C2;i++)
		 {
			if (local_field_to_global_field_index_C2[i]<0) throw_runtime_error("C2 field "+std::string(dyn->functable->fieldnames_C2[i])+" not bound");
		 }
		 for (unsigned int i=0;i<dyn->functable->numfields_C1;i++)
		 {
			if (local_field_to_global_field_index_C1[i]<0) throw_runtime_error("C1 field "+std::string(dyn->functable->fieldnames_C1[i])+" not bound");
		 }
		*/
	}

    bool DynamicBulkElementInstance::has_parameter_contribution(const std::string &param)
	{
		if (!this->get_problem()->has_global_parameter(param))
			return false;
		pyoomph::GlobalParameterDescriptor * parameter=this->get_problem()->get_global_parameter(param);
		for (unsigned int i = 0; i < dyn->functable->numglobal_params; i++)
		{
			if (dyn->functable->global_paramindices[i] == parameter->get_global_index())
				return true;
		}
		return false;
	}

    void CustomResJacInformation::set_custom_jacobian(const std::vector<double> &Jv, const std::vector<int> &col_index, const std::vector<int> &row_start)
	{
		Jvals.resize(Jv.size());
		Jcolumn_index.resize(col_index.size());
		Jrow_start.resize(row_start.size());
		for (unsigned int i = 0; i < Jv.size(); i++)
			Jvals[i] = Jv[i];
		for (unsigned int i = 0; i < col_index.size(); i++)
			Jcolumn_index[i] = col_index[i];
		for (unsigned int i = 0; i < row_start.size(); i++)
			Jrow_start[i] = row_start[i];
	}

	void Problem::unload_all_dlls()
	{
		if (pyoomph_verbose)
			std::cout << "Unloading all DLLs" << std::endl
					  << std::flush;
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			if (pyoomph_verbose)
				std::cout << "Unloading DLL " << bulk_element_codes[i]->get_file_name() << std::endl
						  << std::flush;
			delete bulk_element_codes[i];
		}
		if (pyoomph_verbose)
			std::cout << "DLLs unloaded " << std::endl
					  << std::flush;
		for (auto &gp : global_params_by_name)
		{

			delete gp.second;
		}

		bulk_element_codes.clear();

		global_params_by_name.clear();

		if (eigen_MassMatrixPt)
			delete eigen_MassMatrixPt;
		if (eigen_JacobianMatrixPt)
			delete eigen_JacobianMatrixPt;
	}

	Problem::~Problem()
	{
		// if (meshtemplate) delete meshtemplate; meshtemplate=NULL;
		// for (unsigned int i=0;i<fields_by_index.size();i++) delete fields_by_index[i];
		unload_all_dlls();
		// if (this->compiler) delete this->compiler;
		if (logfile)
		{
		  if (pyoomph::get_logging_stream()==logfile) pyoomph::set_logging_stream(NULL);
		  delete logfile;
		  logfile=NULL;
		  
		}
	}

	Problem::Problem() : oomph::Problem(), compiler(NULL), logfile(NULL), _is_quiet(false), bulk_element_codes(0) // , meshtemplate(new MeshTemplate(this))
	{
	}

	DynamicBulkElementCode *Problem::load_dynamic_bulk_element_code(std::string dynamic_lib, FiniteElementCode *element_class)
	{
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			if (bulk_element_codes[i]->get_file_name() == dynamic_lib)
				return bulk_element_codes[i];
		}
		CCompiler *ccompiler = this->get_ccompiler();
		bulk_element_codes.push_back(new DynamicBulkElementCode(this, ccompiler, dynamic_lib, element_class));
		element_class->fill_callback_info(bulk_element_codes.back()->functable);
		auto *ft = bulk_element_codes.back()->functable;
		for (unsigned int i = 0; i < ft->numglobal_params; i++)
		{
			//		std::cout << "LINKING GLOBAL PARAM " << i << " of " << functable->numglobal_params << std::endl;
			//		std::cout << "codeinst->get_problem()->get_global_parameter(functable->global_paramindices[i]) << std::endl;
			ft->global_parameters[i] = &(this->get_global_parameter(ft->global_paramindices[i])->value());
		}

		return bulk_element_codes.back();
	}

	/*
	const FieldDescriptor * Problem::assert_field(const std::string & name,const FieldSpace & space )
	{
	 if (!this->has_field(name))
	 {
	  FieldDescriptor *res=new FieldDescriptor(this,name,space,fields_by_index.size());
	  fields_by_name.insert(std::pair<std::string,FieldDescriptor *>(name,res));
	  fields_by_index.push_back(res);
		return res;
	 }
	 else
	 {
	  const FieldDescriptor * res=get_field(name);
	  if (res->get_space()!=space) throw_runtime_error("Field '"+name+"' is defined on different spaces");
	  return res;
	 }
	}

	*/

	void Problem::_set_solved_residual(std::string name)
	{
		unsigned numfound = 0;
		for (unsigned int i = 0; i < bulk_element_codes.size(); i++)
		{
			numfound += bulk_element_codes[i]->_set_solved_residual(name);
		}
		if (!numfound)
		{
			throw_runtime_error("Cannot activate the residual-Jacobian pair named '" + name + "', since it is defined in no equations at all");
		}
		this->_solved_residual = name;
	}

	double &Problem::global_parameter(const std::string &n)
	{
		GlobalParameterDescriptor *res = assert_global_parameter(n);
		return res->value();
	}

	GlobalParameterDescriptor *Problem::assert_global_parameter(const std::string &name)
	{
		if (!this->has_global_parameter(name))
		{
			GlobalParameterDescriptor *res = new GlobalParameterDescriptor(this, name, global_params_by_index.size());
			global_params_by_name.insert(std::pair<std::string, GlobalParameterDescriptor *>(name, res));
			global_params_by_index.push_back(res);
			double *valptr = &(res->value());
			this->set_analytic_dparameter(valptr); // Default to analytic derivative
			return res;
		}
		else
		{
			GlobalParameterDescriptor *res = get_global_parameter(name);
			return res;
		}
	}

	double Problem::global_temporal_error_norm()
	{
		double global_error = 0.0;
		for (unsigned int ns = 0; ns < this->nsub_mesh(); ns++)
		{
			global_error += dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ns))->get_temporal_error_norm_contribution();
		}
		if (!_is_quiet)
			std::cout << "GLOBAL TEMPORAL ERROR " << sqrt(global_error) << std::endl;
		return sqrt(global_error);
	}

	void Problem::ensure_dummy_values_to_be_dummy()
	{
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			unsigned nelem = mesh_pt(nmi)->nelement();
			//		std::cout << "ENSURE PINNING NEL " << nelem << std::endl;
			for (unsigned n = 0; n < nelem; n++)
			{
				auto el = dynamic_cast<BulkElementBase *>(mesh_pt(nmi)->element_pt(n));
				if (el)
					el->unpin_dummy_values();
			}
		}
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			unsigned nelem = mesh_pt(nmi)->nelement();
			for (unsigned n = 0; n < nelem; n++)
			{
				auto el = dynamic_cast<BulkElementBase *>(mesh_pt(nmi)->element_pt(n));
				if (el)
					el->pin_dummy_values();
			}
		}
	}

	void Problem::actions_after_adapt()
	{
		for (unsigned nmi = 0; nmi < this->nsub_mesh(); nmi++)
		{
			if (dynamic_cast<Mesh *>(this->mesh_pt(nmi)))
			{
				dynamic_cast<Mesh *>(this->mesh_pt(nmi))->invalidate_lagrangian_kdtree();
			}
		}
		ensure_dummy_values_to_be_dummy();
		setup_pinning();
	}

	void Problem::set_initial_condition()
	{
		oomph::Problem::set_initial_condition();
	}

	void Problem::assemble_eigenproblem_matrices(oomph::CRDoubleMatrix *&M, oomph::CRDoubleMatrix *&J, double sigma_r)
	{
		if (!M)
		{
			if (eigen_MassMatrixPt)
			{
				delete eigen_MassMatrixPt;
			}
			eigen_MassMatrixPt = new oomph::CRDoubleMatrix(this->dof_distribution_pt());
			M = eigen_MassMatrixPt;
		}
		if (!J)
		{
			if (eigen_JacobianMatrixPt)
			{
				delete eigen_JacobianMatrixPt;
			}
			eigen_JacobianMatrixPt = new oomph::CRDoubleMatrix(this->dof_distribution_pt());
			J = eigen_JacobianMatrixPt;
		}
		this->get_eigenproblem_matrices(*M, *J, sigma_r);
	}

	std::vector<double> Problem::get_history_dofs(unsigned t)
	{
		std::vector<double> res(this->ndof(), 0.0);
		oomph::DoubleVector dofs;
		if (t == 0)
			this->get_dofs(dofs);
		else
			this->get_dofs(t, dofs);
		for (unsigned int i = 0; i < this->ndof(); i++)
			res[i] = dofs[i];
		return res;
	}

	std::tuple<std::vector<double>, std::vector<bool>> Problem::get_current_dofs()
	{
		std::vector<double> res(this->ndof(), 0.0);
		std::vector<bool> is_positional(this->ndof(), false);
		oomph::DoubleVector dofs;
		this->get_dofs(dofs);
		for (unsigned int i = 0; i < this->ndof(); i++)
			res[i] = dofs[i];

		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = dynamic_cast<pyoomph::Node *>(m->node_pt(in));
				auto *vp = n->variable_position_pt();
				for (unsigned int iv = 0; iv < vp->nvalue(); iv++)
				{
					if (vp->eqn_number(iv) >= 0)
						is_positional[vp->eqn_number(iv)] = true;
				}
			}
		}

		return std::make_tuple(res, is_positional);
	}

	void Problem::set_current_dofs(const std::vector<double> &inp)
	{
		oomph::DoubleVector dofs;
		dofs.build(this->dof_distribution_pt(), 0.0);
		if (inp.size() != this->ndof())
			throw_runtime_error("Mismatch in dof vector size");
		for (unsigned int i = 0; i < this->ndof(); i++)
			dofs[i] = inp[i];
		this->set_dofs(dofs);
	}

	void Problem::set_history_dofs(unsigned t, const std::vector<double> &inp)
	{
		oomph::DoubleVector dofs;
		dofs.build(this->dof_distribution_pt(), 0.0);
		if (inp.size() != this->ndof())
			throw_runtime_error("Mismatch in dof vector size");
		if (t>=this->time_stepper_pt()->ntstorage()) 
		        throw_runtime_error("Wrong history offset");
		for (unsigned int i = 0; i < this->ndof(); i++)
			dofs[i] = inp[i];
		this->set_dofs(t, dofs);
	}

	std::vector<double> Problem::get_current_pinned_values(bool with_pos)
	{
		std::vector<double> res;
		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = m->node_pt(in);
				for (unsigned int iv = 0; iv < n->nvalue(); iv++)
				{
					if (n->is_pinned(iv))
						res.push_back(n->value(iv));
				}
				if (with_pos)
				{
					for (unsigned int iv = 0; iv < n->ndim(); iv++)
					{
						if (dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->is_pinned(iv))
							res.push_back(dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->value(iv));
					}
				}
			}
			for (unsigned int ie = 0; ie < m->nelement(); ie++)
			{
				auto *e = m->element_pt(ie);
				for (unsigned int iid = 0; iid < e->ninternal_data(); iid++)
				{
					auto *id = e->internal_data_pt(iid);
					for (unsigned int iv = 0; iv < id->nvalue(); iv++)
					{
						if (id->is_pinned(iv))
							res.push_back(id->value(iv));
					}
				}
			}
		}
		return res;
	}

	void Problem::get_residuals(oomph::DoubleVector &residuals)
	{
		if (!use_custom_residual_jacobian)
		{
			get_residuals_by_elemental_assembly(residuals);
		}
		else
		{
			CustomResJacInformation info(false);
			get_custom_residuals_jacobian(&info);
			if (!residuals.built())
			{
				oomph::LinearAlgebraDistribution dist(this->communicator_pt(), info.residuals.size(), false);
				residuals.build(&dist, 0.0);
			}
			for (unsigned int i = 0; i < info.residuals.size(); i++)
				residuals[i] = info.residuals[i];
		}
	}

	void Problem::get_jacobian(oomph::DoubleVector &residuals, oomph::CRDoubleMatrix &jacobian)
	{
		if (!use_custom_residual_jacobian)
		{
			get_jacobian_by_elemental_assembly(residuals, jacobian);
		}
		else
		{
			CustomResJacInformation info(true);
			get_custom_residuals_jacobian(&info);
			//       std::cout << "RET FROM PYTH" << std::endl;

			if (!residuals.built())
			{
				oomph::LinearAlgebraDistribution dist(this->communicator_pt(), info.residuals.size(), false);
				residuals.build(&dist, 0.0);
			}
			for (unsigned int i = 0; i < info.residuals.size(); i++)
				residuals[i] = info.residuals[i];

			//       std::cout << "BUILD J  "<< info.residuals.size() << "  " << info.Jcolumn_index.size() << "  " << info.Jrow_start.size() << std::endl;
			jacobian.build(info.residuals.size(), info.Jvals, info.Jcolumn_index, info.Jrow_start);
			//       std::cout << "DONE BUILD J" << std::endl;
		}
	}

	int Problem::resolve_parameter_value_ptr(double *ptr)
	{
		for (const auto &a : global_params_by_name)
		{
			if ((&a.second->value()) == ptr)
				return a.second->get_global_index();
		}
		throw_runtime_error("Cannot resolve the double pointer of a global parameter to this problem");
		return -1;
	}

	void Problem::set_arclength_parameter(std::string nam, double val)
	{
		if (nam == "Desired_proportion_of_arc_length")
			Desired_proportion_of_arc_length = val;
		else if (nam == "Scale_arc_length")
			Scale_arc_length = (val > 0.5 ? true : false);
		else if (nam == "Use_finite_differences_for_continuation_derivatives")
			Use_finite_differences_for_continuation_derivatives = (val > 0.5 ? true : false);
		else if (nam == "Use_continuation_timestepper")
			Use_continuation_timestepper = (val > 0.5 ? true : false);
		else if (nam == "Desired_newton_iterations_ds")
		   Desired_newton_iterations_ds=val;
		else
			throw_runtime_error("Unknown param to set " + nam);
	}

	void Problem::_replace_RJM_by_param_deriv(std::string name, bool active)
	{
		if (!active)
			__replace_RJM_by_param_deriv = NULL;
		else
		{
			if (!global_params_by_name.count(name))
				throw_runtime_error("Cannot replace residuals/jacobian/mass matrix by parameter derivatives for global parameter " + name + ", since it is not present in the problem");
			auto *p = global_params_by_name[name];
			__replace_RJM_by_param_deriv = &(p->value());
		}
	}

	double Problem::arc_length_step(const std::string param, const double &ds, unsigned max_adapt)
	{
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot continue in the global parameter " + param + ", since it is not present in the problem");
		auto *p = global_params_by_name[param];
		double *valptr = &(p->value());
		//		this->set_analytic_dparameter(valptr);
		return this->arc_length_step_solve(valptr, ds, max_adapt);
	}

	std::vector<double> Problem::get_arclength_dof_derivative_vector()
	{
		std::vector<double> res(Dof_derivative.size());
		for (unsigned i = 0; i < res.size(); i++)
			res[i] = dof_derivative(i);
		return res;
	}

	void Problem::activate_my_fold_tracking(double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt, eigenvector);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::AugmentedBlockFoldLinearSolver(this->linear_solver_pt());
		}
	}

	void Problem::activate_my_fold_tracking(double *const &parameter_pt, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyFoldHandler(this, parameter_pt);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::AugmentedBlockFoldLinearSolver(this->linear_solver_pt());
		}
	}

	void Problem::activate_my_hopf_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, const bool &block_solve)
	{
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyHopfHandler(this, parameter_pt, omega, null_real, null_imag);
		if (block_solve)
		{
			this->linear_solver_pt() = new oomph::BlockHopfLinearSolver(this->linear_solver_pt());
		}
	}

	void Problem::actions_after_change_in_global_parameter(double *const &parameter_pt)
	{
		for (auto &p : this->global_params_by_index)
		{
			if (&(p->value()) == parameter_pt)
			{
				this->actions_after_change_in_global_parameter(p->get_name());
			}
		}
	}

	void Problem::actions_after_parameter_increase(double *const &parameter_pt)
	{
		for (auto &p : this->global_params_by_index)
		{
			if (&(p->value()) == parameter_pt)
			{
				this->actions_after_parameter_increase(p->get_name());
			}
		}
	}

	void Problem::activate_my_azimuthal_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, std::map<std::string, std::string> special_residual_forms)
	{
		reset_assembly_handler_to_default();
		AzimuthalSymmetryBreakingHandler *azi = new AzimuthalSymmetryBreakingHandler(this, parameter_pt, null_real, null_imag, omega);
		if (!special_residual_forms.count("azimuthal_real_eigen"))
		{
			throw_runtime_error("You have not specified a azimuthal_real_eigen as special residual");
		}
		if (!special_residual_forms.count("azimuthal_imag_eigen"))
		{
			throw_runtime_error("You have not specified a azimuthal_imag_eigen as special residual");
		}
		azi->setup_solved_azimuthal_contributions(special_residual_forms["azimuthal_real_eigen"], special_residual_forms["azimuthal_imag_eigen"]);
		this->assembly_handler_pt() = azi;
	}

	void Problem::activate_my_pitchfork_tracking(double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector, const bool &block_solve)
	{
		//	this->activate_pitchfork_tracking(parameter_pt, symmetry_vector, block_solve);
		reset_assembly_handler_to_default();
		this->assembly_handler_pt() = new MyPitchForkHandler(this, parameter_pt, symmetry_vector);
	}

	std::vector<double> Problem::get_parameter_derivative(const std::string param)
	{
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot derive wrt unknown global parameter " + param);
		auto *p = global_params_by_name[param];
		double *valptr = &(p->value());
		//		this->set_analytic_dparameter(valptr);
		oomph::DoubleVector resdv(this->dof_distribution_pt());
		resdv.clear();
		get_derivative_wrt_global_parameter(valptr, resdv);
		std::vector<double> res(this->ndof());
		for (unsigned int i = 0; i < res.size(); i++)
			res[i] = resdv[i];
		return res;
	}

	void Problem::after_bifurcation_tracking_step()
	{
		if (dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt()))
		{
			dynamic_cast<MyFoldHandler *>(this->assembly_handler_pt())->realign_C_vector();
		}
	}

	void Problem::set_dof_direction_arclength(std::vector<double> ddir)
	{
		this->reset_arc_length_parameters();
		const unsigned long ndof_local = this->Dof_distribution_pt->nrow_local();
		if (ddir.size() != ndof_local)
			throw_runtime_error("Mismatching size in the dof direction vector and the actual number of DoFs:" + std::to_string(ddir.size()) + " vs " + std::to_string(ndof_local));
		this->Arc_length_step_taken = true;
		if (!this->Use_continuation_timestepper)
		{
			if (this->Dof_derivative.size() != ndof_local)
			{
				this->Dof_derivative.resize(ndof_local, 0.0);
			}
		}
		for (unsigned int i = 0; i < ddir.size(); i++)
			dof_derivative(i) = ddir[i];
	}

	double Problem::get_bifurcation_omega()
	{
		if (bifurcation_tracking_mode == "hopf" && (dynamic_cast<MyHopfHandler *>(this->assembly_handler_pt())))
		{
			return dynamic_cast<MyHopfHandler *>(this->assembly_handler_pt())->omega();
		}
		else if (bifurcation_tracking_mode == "azimuthal" && (dynamic_cast<AzimuthalSymmetryBreakingHandler *>(this->assembly_handler_pt())))
		{
			return dynamic_cast<AzimuthalSymmetryBreakingHandler *>(this->assembly_handler_pt())->omega();
		}
		else
		{
			return 0.0;
		}
	}

	std::vector<std::complex<double>> Problem::get_bifurcation_eigenvector()
	{
		if (bifurcation_tracking_mode == "")
			return std::vector<std::complex<double>>();
		oomph::Vector<oomph::DoubleVector> be;
		this->get_bifurcation_eigenfunction(be);
		std::vector<std::complex<double>> res(be[0].nrow());
		if (be.size() == 1)
		{
			for (unsigned int i = 0; i < be[0].nrow(); i++)
				res[i] = std::complex<double>(be[0][i], 0.0);
		}
		else
		{
			for (unsigned int i = 0; i < be[0].nrow(); i++)
				res[i] = std::complex<double>(be[0][i], be[1][i]);
		}
		return res;
	}

	void Problem::start_bifurcation_tracking(const std::string param, const std::string typus, const bool &blocksolve, const std::vector<double> &eigenv1, const std::vector<double> &eigenv2, const double &omega, std::map<std::string, std::string> special_residual_forms)
	{
		if (param == "" || typus == "" || typus == "none")
		{
			bifurcation_tracking_mode = "";
			this->deactivate_bifurcation_tracking();
			return;
		}
		if (!global_params_by_name.count(param))
			throw_runtime_error("Cannot track a bifuraciton in the global parameter " + param + ", since it is not present in the problem");
		auto *p = global_params_by_name[param];
		double *valptr = &(p->value());
		//		this->set_analytic_dparameter(valptr);
		oomph::DoubleVector ev1(this->dof_distribution_pt());
		for (unsigned i = 0; i < std::min((size_t)eigenv1.size(), (size_t)this->ndof()); i++)
		{
			ev1[i] = eigenv1[i];
		}
		oomph::DoubleVector ev2(this->dof_distribution_pt());
		for (unsigned i = 0; i < std::min((size_t)eigenv2.size(), (size_t)this->ndof()); i++)
		{
			ev2[i] = eigenv2[i];
		}
		if (typus == "fold")
		{
			bifurcation_tracking_mode = "fold";
			if (eigenv1.empty())
				this->activate_my_fold_tracking(valptr, blocksolve);
			else
				this->activate_my_fold_tracking(valptr, ev1, blocksolve);
		}
		else if (typus == "hopf")
		{
			bifurcation_tracking_mode = "hopf";
			this->activate_my_hopf_tracking(valptr, omega, ev1, ev2, blocksolve);
			//    this->activate_hopf_tracking(valptr,omega,ev1,ev2,blocksolve);
		}
		else if (typus == "azimuthal")
		{
			bifurcation_tracking_mode = "azimuthal";
			this->activate_my_azimuthal_tracking(valptr, omega, ev1, ev2, special_residual_forms);
		}
		else if (typus == "cartesian_normal_mode")
		{
			bifurcation_tracking_mode = "cartesian_normal_mode";
			this->activate_my_azimuthal_tracking(valptr, omega, ev1, ev2, special_residual_forms);
		}		
		else if (typus == "pitchfork")
		{
			bifurcation_tracking_mode = "pitchfork";
			this->activate_my_pitchfork_tracking(valptr, ev1, blocksolve);
			//    this->activate_hopf_tracking(valptr,omega,ev1,ev2,blocksolve);
		}
		else
			throw_runtime_error("Cannot track unknown bifurcation type: " + typus);
	}

	void Problem::set_current_pinned_values(const std::vector<double> &inp, bool with_pos)
	{
		unsigned int pos = 0;
		unsigned mpos = inp.size();
		for (unsigned int ism = 0; ism < this->nsub_mesh(); ism++)
		{
			pyoomph::Mesh *m = dynamic_cast<pyoomph::Mesh *>(this->mesh_pt(ism));
			for (unsigned int in = 0; in < m->nnode(); in++)
			{
				auto *n = m->node_pt(in);
				for (unsigned int iv = 0; iv < n->nvalue(); iv++)
				{
					if (n->is_pinned(iv))
					{
						n->set_value(iv, inp[pos++]);
						if (pos > mpos)
							throw_runtime_error("Mismatch in value vector size: " + std::to_string(mpos) + " given, but reached index " + std::to_string(pos));
					}
				}
				if (with_pos)
				{
					for (unsigned int iv = 0; iv < n->ndim(); iv++)
					{
						if (dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->is_pinned(iv))
							dynamic_cast<pyoomph::Node *>(n)->variable_position_pt()->set_value(iv, inp[pos++]);
					}
				}
			}
			for (unsigned int ie = 0; ie < m->nelement(); ie++)
			{
				auto *e = m->element_pt(ie);
				for (unsigned int iid = 0; iid < e->ninternal_data(); iid++)
				{
					auto *id = e->internal_data_pt(iid);
					for (unsigned int iv = 0; iv < id->nvalue(); iv++)
					{
						if (id->is_pinned(iv))
						{
							id->set_value(iv, inp[pos++]);
							if (pos > mpos)
								throw_runtime_error("Mismatch in value vector size: " + std::to_string(mpos) + " given, but reached index " + std::to_string(pos));
						}
					}
				}
			}
		}
	}
	
	
	void Problem::open_log_file(const std::string &fname,const bool & activate_logging)
	{

		if (fname=="")
		{
			if (activate_logging) pyoomph::set_logging_stream(this->logfile);
			else 
			{
				if (pyoomph::get_logging_stream()==logfile) pyoomph::set_logging_stream(NULL);
				if (logfile) delete logfile;
				logfile=NULL;
			}
			return;
		}
		if (activate_logging && logfile)
		{
			if (pyoomph::get_logging_stream()==logfile) pyoomph::set_logging_stream(NULL);
			delete logfile;
			logfile=NULL;
		}
		logfile=new std::ofstream(fname.c_str());
		if (!logfile->is_open()) throw_runtime_error("Cannot open log file "+fname);
		if (activate_logging) pyoomph::set_logging_stream(logfile);
	}

	void Problem::quiet(bool _quiet)
	{
		_is_quiet = _quiet;
		Shut_up_in_newton_solve = _quiet;
		if (_quiet)
		{
			this->linear_solver_pt()->disable_doc_time();
			oomph::oomph_info.stream_pt() = &oomph::oomph_nullstream;
		}
		else
		{
			this->linear_solver_pt()->enable_doc_time();
			oomph::oomph_info.stream_pt() = &std::cout;
		}
	}

	SparseRank3Tensor Problem::assemble_hessian_tensor(bool symmetric)
	{
		SparseRank3Tensor result(this->ndof(), symmetric);
		const unsigned long n_elements = mesh_pt()->nelement();
		for (unsigned int ne = 0; ne < n_elements; ne++)
		{
			BulkElementBase *elem_pt = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(ne));
			const unsigned nvar = assembly_handler_pt()->ndof(elem_pt);
			oomph::DenseMatrix<double> hessian_buffer(nvar, nvar * nvar, 0.0);
			elem_pt->assemble_hessian_tensor(hessian_buffer);
			for (unsigned int i = 0; i < nvar; i++)
			{
				unsigned iG = assembly_handler_pt()->eqn_number(elem_pt, i);
				for (unsigned int j = 0; j < nvar; j++)
				{
					unsigned jG = assembly_handler_pt()->eqn_number(elem_pt, j);
					for (unsigned int k = 0; k < nvar; k++)
					{
						double hval = hessian_buffer(i, k * nvar + j);
						if (std::fabs(hval) > Numerical_zero_for_sparse_assembly)
						{
							unsigned kG = assembly_handler_pt()->eqn_number(elem_pt, k);
							result.accumulate(iG, jG, kG, hval);
						}
					}
				}
			}
		}
		return result;
	}

	void GlobalParameterDescriptor::set_analytic_derivative(bool active)
	{
		if (active)
			problem->set_analytic_dparameter(&Value);
		else
			problem->unset_analytic_dparameter(&Value);
	}
	bool GlobalParameterDescriptor::get_analytic_derivative()
	{
		return problem->is_dparameter_calculated_analytically(&Value);
	}

}
