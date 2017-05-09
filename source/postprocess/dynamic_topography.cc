/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/simulator.h>
#include <aspect/postprocess/dynamic_topography.h>

#include <aspect/postprocess/boundary_pressures.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    DynamicTopography<dim>::execute (TableHandler &)
    {
      Postprocess::BoundaryPressures<dim> *boundary_pressures =
        this->template find_postprocessor<Postprocess::BoundaryPressures<dim> >();
      AssertThrow(boundary_pressures != NULL,
                  ExcMessage("Could not find the BoundaryPressures postprocessor") );
      const double surface_pressure = boundary_pressures->pressure_at_top();

      const unsigned int quadrature_degree = this->get_fe().base_element(this->introspection().base_elements.velocities).degree+1;
      //Gauss quadrature in the interior for best accuracy
      const QGauss<dim> quadrature_formula(quadrature_degree);
      //GLL quadrature on the surface to get a diagonal mass matrix
      const QGaussLobatto<dim-1> quadrature_formula_face(quadrature_degree);

      const unsigned int dofs_per_cell = this->get_fe().dofs_per_cell;
      const unsigned int dofs_per_face = this->get_fe().dofs_per_face;
      const unsigned int n_q_points = quadrature_formula.size();
      const unsigned int n_face_q_points = quadrature_formula_face.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values |
                               update_gradients |
                               update_q_points |
                               update_JxW_values);

      FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                        this->get_fe(),
                                        quadrature_formula_face,
                                        update_JxW_values | 
                                        update_values |
                                        update_gradients |
                                        update_q_points);

      MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points, this->n_compositional_fields());

      MaterialModel::MaterialModelInputs<dim> in_face(fe_face_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out_face(fe_face_values.n_quadrature_points, this->n_compositional_fields());

      std::vector<std::vector<double> > composition_values (this->n_compositional_fields(),std::vector<double> (quadrature_formula.size()));
      std::vector<std::vector<double> > face_composition_values (this->n_compositional_fields(),std::vector<double> (quadrature_formula_face.size()));

      //Storage for shape function values in solving CBF system
      std::vector<Tensor<1,dim> > phi_u (dofs_per_cell);
      std::vector<SymmetricTensor<2,dim> > epsilon_phi_u (dofs_per_cell);
      std::vector<double> div_phi_u (dofs_per_cell);
      std::vector<double> div_solution(n_q_points);

      //Vectors for solving CBF system
      Vector<double> local_vector(dofs_per_cell);
      Vector<double> local_mass_matrix(dofs_per_cell);

      LinearAlgebra::BlockVector rhs_vector(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
      LinearAlgebra::BlockVector mass_matrix(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());

      LinearAlgebra::BlockVector distributed_stress_vector(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
      LinearAlgebra::BlockVector distributed_topo_vector(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());

      LinearAlgebra::BlockVector surface_stress_vector(this->introspection().index_sets.system_partitioning,
                                   this->introspection().index_sets.system_relevant_partitioning,
                                   this->get_mpi_communicator());
      topo_vector.reinit(this->introspection().index_sets.system_partitioning,
                         this->introspection().index_sets.system_relevant_partitioning,
                         this->get_mpi_communicator());
      surface_stress_vector = 0.;
      distributed_topo_vector = 0.;
      topo_vector = 0.;

      // have a stream into which we write the data. the text stream is then
      // later sent to processor 0
      std::ostringstream output;
      std::vector<std::pair<Point<dim>,double> > stored_values;

      // loop over all of the surface cells and if one less than h/3 away from
      // the top surface, evaluate the stress at its center
      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();

      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          if (cell->at_boundary())
            {
              // see if the cell is at the *top* boundary, not just any boundary
              unsigned int top_face_idx = numbers::invalid_unsigned_int;
              {
                for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                  if (cell->at_boundary(f) && this->get_geometry_model().depth (cell->face(f)->center()) < cell->face(f)->minimum_vertex_distance()/3)
                    {
                      top_face_idx = f;
                      break;
                    }

                if (top_face_idx == numbers::invalid_unsigned_int)
                  continue;
              }
              fe_values.reinit (cell);
              fe_face_values.reinit (cell, top_face_idx);

              local_vector = 0.;
              local_mass_matrix = 0.;

              // get the various components of the solution, then
              // evaluate the material properties there
              fe_values[this->introspection().extractors.temperature]
              .get_function_values (this->get_solution(), in.temperature);
              fe_values[this->introspection().extractors.pressure]
              .get_function_values (this->get_solution(), in.pressure);
              fe_values[this->introspection().extractors.velocities]
              .get_function_values (this->get_solution(), in.velocity);
              fe_values[this->introspection().extractors.velocities]
              .get_function_symmetric_gradients (this->get_solution(), in.strain_rate);
              fe_values[this->introspection().extractors.pressure]
              .get_function_gradients (this->get_solution(), in.pressure_gradient);
              fe_values[this->introspection().extractors.velocities]
              .get_function_divergences (this->get_solution(), div_solution);

              in.position = fe_values.get_quadrature_points();

              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                fe_values[this->introspection().extractors.compositional_fields[c]]
                .get_function_values(this->get_solution(),
                                     composition_values[c]);
              for (unsigned int i=0; i<n_q_points; ++i)
                {
                  for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                    in.composition[i][c] = composition_values[c][i];
                }
              in.cell = &cell;

              this->get_material_model().evaluate(in, out);

              // get the various components of the solution, then
              // evaluate the material properties there
              fe_face_values[this->introspection().extractors.temperature]
              .get_function_values (this->get_solution(), in_face.temperature);
              fe_face_values[this->introspection().extractors.pressure]
              .get_function_values (this->get_solution(), in_face.pressure);
              fe_face_values[this->introspection().extractors.velocities]
              .get_function_values (this->get_solution(), in_face.velocity);
              fe_face_values[this->introspection().extractors.velocities]
              .get_function_symmetric_gradients (this->get_solution(), in_face.strain_rate);
              fe_face_values[this->introspection().extractors.pressure]
              .get_function_gradients (this->get_solution(), in_face.pressure_gradient);

              in_face.position = fe_face_values.get_quadrature_points();

              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                fe_face_values[this->introspection().extractors.compositional_fields[c]]
                .get_function_values(this->get_solution(),
                                     face_composition_values[c]);
              for (unsigned int i=0; i<n_face_q_points; ++i)
                {
                  for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                    in_face.composition[i][c] = face_composition_values[c][i];
                }
              in_face.cell = &cell;

              this->get_material_model().evaluate(in_face, out_face);

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double eta = out.viscosities[q];
                  const double density = out.densities[q];
                  bool is_compressible = this->get_material_model().is_compressible();
                  Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(in.position[q]);

                  //Set up shape function values
                  for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                      phi_u[k] = fe_values[this->introspection().extractors.velocities].value(k,q);
                      epsilon_phi_u[k] = fe_values[this->introspection().extractors.velocities].symmetric_gradient(k,q);
                      div_phi_u[k] = fe_values[this->introspection().extractors.velocities].divergence (k, q);
                    }

                  for (unsigned int i = 0; i<dofs_per_cell; ++i)
                    {
                      //Viscous stress part
                      local_vector(i) += 2.0 * eta * ( epsilon_phi_u[i] * in.strain_rate[q]
                                                       - (is_compressible ? 1./3. * div_phi_u[i] * div_solution[q] : 0.0) ) * fe_values.JxW(q);
                      //Pressure and compressibility parts
                      local_vector(i) -= div_phi_u[i] * in.pressure[q] * fe_values.JxW(q);
                      //Force part
                      local_vector(i) -= density * gravity * phi_u[i] * fe_values.JxW(q);
                    }
                }
              for (unsigned int q=0; q < n_face_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  local_mass_matrix(i) += fe_face_values[this->introspection().extractors.velocities].value(i,q) *
                                          fe_face_values[this->introspection().extractors.velocities].value(i,q) *
                                          fe_face_values.JxW(q);

              cell->distribute_local_to_global( local_vector, rhs_vector );
              cell->distribute_local_to_global( local_mass_matrix, mass_matrix );
            }

      rhs_vector.compress(VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);

      //Since the mass matrix is diagonal, we can just solve for the stress vector by dividing
      const IndexSet local_elements = mass_matrix.locally_owned_elements();
      for (unsigned int k=0; k<local_elements.n_elements(); ++k)
        {
          const unsigned int global_index = local_elements.nth_index_in_set(k);
          if ( mass_matrix[global_index] > 1.e-15)
            distributed_stress_vector[global_index] = rhs_vector[global_index]/mass_matrix[global_index];
        }
      distributed_stress_vector.compress(VectorOperation::insert);
      surface_stress_vector = distributed_stress_vector;

      //Now loop over the cells again and solve for the dynamic topography
      std::vector< Point<dim-1> > face_support_points = this->get_fe().base_element( this->introspection().base_elements.temperature ).get_unit_face_support_points();
      Quadrature<dim-1> transfer_quadrature(face_support_points);
      FEFaceValues<dim> fe_transfer_values (this->get_mapping(),
                                            this->get_fe(),
                                            transfer_quadrature,
                                            update_values | update_q_points);

      std::vector<Tensor<1,dim> > stress_transfer_values( transfer_quadrature.size() );
      std::vector<double> topo_values( transfer_quadrature.size() );
      std::vector<types::global_dof_index> face_dof_indices (dofs_per_face);

      cell = this->get_dof_handler().begin_active();
      endc = this->get_dof_handler().end();
      for ( ; cell != endc; ++cell)
        if (cell->is_locally_owned())
          if (cell->at_boundary())
            {
              // see if the cell is at the *top* boundary, not just any boundary
              unsigned int top_face_idx = numbers::invalid_unsigned_int;
              {
                for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                  if (cell->at_boundary(f) && this->get_geometry_model().depth (cell->face(f)->center()) < cell->face(f)->minimum_vertex_distance()/3)
                    {
                      top_face_idx = f;
                      break;
                    }

                if (top_face_idx == numbers::invalid_unsigned_int)
                  continue;
              }

              const double density = 3000.;
              fe_transfer_values.reinit (cell, top_face_idx);
              fe_transfer_values[this->introspection().extractors.velocities].get_function_values( surface_stress_vector, stress_transfer_values );
              cell->face(top_face_idx)->get_dof_indices (face_dof_indices);
              for( unsigned int i = 0; i < face_dof_indices.size(); ++i)

              for (unsigned int i = 0; i < dofs_per_face; ++i)
                {
                  const std::pair<unsigned int, unsigned int> component_index = this->get_fe().face_system_to_component_index(i);
                  const unsigned int component = component_index.first;
                  const unsigned int support_index = component_index.second;
                  if (component == this->introspection().component_indices.temperature)
                    {
                      const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(fe_transfer_values.quadrature_point(support_index));
                      const double gravity_norm = gravity.norm();
                      const Tensor<1,dim> gravity_direction = gravity/gravity.norm();
                      const double dynamic_topography = (stress_transfer_values[support_index] * gravity_direction - surface_pressure)/density/gravity_norm;

                      distributed_topo_vector[ face_dof_indices[i] ] = dynamic_topography;
                    }
                }
            }
      distributed_topo_vector.compress(VectorOperation::insert);
      topo_vector = distributed_topo_vector;
      std::string filename = this->get_output_directory() +
                             "dynamic_topography." +
                             Utilities::int_to_string(this->get_timestep_number(), 5);

      return std::pair<std::string,std::string>("Writing dynamic topography:",
                                                filename);
    }

    template <int dim>
    const LinearAlgebra::BlockVector &
    DynamicTopography<dim>::
    get_topography_vector() const
    {
      return topo_vector;
    }

    template <int dim>
    std::list<std::string>
    DynamicTopography<dim>::required_other_postprocessors() const
    {
      std::list<std::string> deps;
      deps.push_back("boundary pressures");
      return deps;
    }

    template <int dim>
    void
    DynamicTopography<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Dynamic Topography");
        {
          prm.declare_entry ("Density above","0",
                             Patterns::Double (0),
                             "Dynamic topography is calculated as the excess or lack of mass that is supported by mantle flow. "
                             "This value depends on the density of material that is moved up or down, i.e. crustal rock, and the "
                             "density of the material that is displaced (generally water or air). While the density of crustal rock "
                             "is part of the material model, this parameter `Density above' allows the user to specify the density "
                             "value of material that is displaced above the solid surface. By default this material is assumed to "
                             "be air, with a density of 0. "
                             "Units: $kg/m^3$.");
          prm.declare_entry ("Density below","9900.",
                             Patterns::Double (0),
                             "Dynamic topography is calculated as the excess or lack of mass that is supported by mantle flow. "
                             "This value depends on the density of material that is moved up or down, i.e. crustal rock, and the "
                             "density of the material that is displaced (for the CMB it is typically iron). While the density of crustal rock "
                             "is part of the material model, this parameter `Density below' allows the user to specify the density "
                             "value of material that is displaced above the solid surface. By default this material is assumed to "
                             "be iron, with a density of 9900. "
                             "Units: $kg/m^3$.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    DynamicTopography<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Dynamic Topography");
        {
          density_above = prm.get_double ("Density above");
          density_below = prm.get_double ("Density below");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(DynamicTopography,
                                  "dynamic topography",
                                  "A postprocessor that computes a measure of dynamic topography "
                                  "based on the stress at the surface. The data is written into text "
                                  "files named 'dynamic\\_topography.NNNNN' in the output directory, "
                                  "where NNNNN is the number of the time step."
                                  "\n\n"
                                  "The exact approach works as follows: At the centers of all cells "
                                  "that sit along the top surface, we evaluate the stress and "
                                  "evaluate the component of it in the direction in which "
                                  "gravity acts. In other words, we compute "
                                  "$\\sigma_{rr}={\\hat g}^T(2 \\eta \\varepsilon(\\mathbf u)- \\frac 13 (\\textrm{div}\\;\\mathbf u)I)\\hat g - p_d$ "
                                  "where $\\hat g = \\mathbf g/\\|\\mathbf g\\|$ is the direction of "
                                  "the gravity vector $\\mathbf g$ and $p_d=p-p_a$ is the dynamic "
                                  "pressure computed by subtracting the adiabatic pressure $p_a$ "
                                  "from the total pressure $p$ computed as part of the Stokes "
                                  "solve. From this, the dynamic "
                                  "topography is computed using the formula "
                                  "$h=\\frac{\\sigma_{rr}}{(\\mathbf g \\cdot \\mathbf n)  \\rho}$ where $\\rho$ "
                                  "is the density at the cell center. Note that this implementation takes "
                                  "the direction of gravity into account, which means that reversing the flow "
                                  "in backward advection calculations will not reverse the instantaneous topography "
                                  "because the reverse flow will be divided by the reverse surface gravity.  "
                                  "\n"
                                  "The file format then consists of lines with Euclidean coordinates "
                                  "followed by the corresponding topography value."
                                  "\n\n"
                                  "(As a side note, the postprocessor chooses the cell center "
                                  "instead of the center of the cell face at the surface, where we "
                                  "really are interested in the quantity, since "
                                  "this often gives better accuracy. The results should in essence "
                                  "be the same, though.)")
  }
}
