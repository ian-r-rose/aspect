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

#include <aspect/simulator.h>
#include <aspect/postprocess/visualization/dynamic_topography.h>
#include <aspect/postprocess/dynamic_topography.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      /**
       * Execute the visualization postprocessor.
       */
      template <int dim>
      std::pair<std::string, Vector<float> *>
      DynamicTopography<dim>::execute() const
      {
        std::pair<std::string, Vector<float> *>
        return_value ("dynamic_topography",
                      new Vector<float>(this->get_triangulation().n_active_cells()));

        Postprocess::DynamicTopography<dim> *dynamic_topography =
          this->template find_postprocessor<Postprocess::DynamicTopography<dim> >();
        AssertThrow(dynamic_topography != NULL,
                    ExcMessage("Could not find the DynamicTopography postprocessor."));
        const LinearAlgebra::BlockVector &topography_vector = dynamic_topography->topography_vector();

        const QTrapez<dim-1> quadrature_formula;
        std::vector<Tensor<1,dim> > stress_values( quadrature_formula.size() );
        std::vector<double> topo_values( quadrature_formula.size() );

        FEFaceValues<dim> fe_face_values (this->get_mapping(),
                                          this->get_fe(),
                                          quadrature_formula,
                                          update_JxW_values | update_values | update_quadrature_points);

        // loop over all of the surface cells and if one less than h/3 away from
        // the top surface, evaluate the stress at its center
        typename DoFHandler<dim>::active_cell_iterator
        cell = this->get_dof_handler().begin_active(),
        endc = this->get_dof_handler().end();

        unsigned int cell_index = 0;
        for (; cell!=endc; ++cell,++cell_index)
          if (cell->is_locally_owned())
            if (cell->at_boundary())
              {

                // see if the cell is at the *top* or *bottom* boundary, not just any boundary
                unsigned int face_idx = numbers::invalid_unsigned_int;
                for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                  {
                    const double depth_face_center = this->get_geometry_model().depth (cell->face(f)->center());
                    const double upper_depth_cutoff = cell->face(f)->minimum_vertex_distance()/3.0;
                    const double lower_depth_cutoff = this->get_geometry_model().maximal_depth() - cell->face(f)->minimum_vertex_distance()/3.0;

                    // Check if cell is at upper and lower surface at the same time
                    if (depth_face_center < upper_depth_cutoff && depth_face_center > lower_depth_cutoff)
                      AssertThrow(false, ExcMessage("Your geometry is model so small that the upper and lower boundary of "
                                                    "the domain are bordered by the same cell. "
                                                    "Consider using a higher mesh resolution.") );

                    // Check if the face is at the top or bottom boundary
                    if (depth_face_center < upper_depth_cutoff || depth_face_center > lower_depth_cutoff)
                      {
                        face_idx = f;
                        break;
                      }
                  }

                if (face_idx == numbers::invalid_unsigned_int)
                  {
                    (*return_value.second)(cell_index) = 0.;
                    continue;
                  }

                fe_face_values.reinit (cell, face_idx);
                fe_face_values[this->introspection().extractors.temperature].get_function_values( topography_vector, topo_values );

                std::vector<types::global_dof_index> face_dof_indices (this->get_fe().dofs_per_face);
                cell->face(face_idx)->get_dof_indices (face_dof_indices);
                double cell_surface_area = 0.0;
                double dynamic_topography = 0.0;
                for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                  {
                    dynamic_topography += topo_values[q] * fe_face_values.JxW(q);
                    cell_surface_area += fe_face_values.JxW(q);
                  }
                dynamic_topography /= cell_surface_area;
                (*return_value.second)(cell_index) = dynamic_topography;
              }

        return return_value;
      }

      /**
       * Register the other postprocessor that we need: DynamicTopography
       */
      template <int dim>
      std::list<std::string>
      DynamicTopography<dim>::required_other_postprocessors() const
      {
        std::list<std::string> deps;
        deps.push_back("dynamic topography");
        return deps;
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(DynamicTopography,
                                                  "dynamic topography",
                                                  "A visualization output object that generates output "
                                                  "for the dynamic topography. The approach to determine the "
                                                  "dynamic topography requires us to compute the stress tensor and "
                                                  "evaluate the component of it in the direction in which "
                                                  "gravity acts. In other words, we compute "
                                                  "$\\sigma_{rr}={\\hat g}^T(2 \\eta \\varepsilon(\\mathbf u)-\\frac 13 (\\textrm{div}\\;\\mathbf u)I)\\hat g - p_d$ "
                                                  "where $\\hat g = \\mathbf g/\\|\\mathbf g\\|$ is the direction of "
                                                  "the gravity vector $\\mathbf g$ and $p_d=p-p_a$ is the dynamic "
                                                  "pressure computed by subtracting the adiabatic pressure $p_a$ "
                                                  "from the total pressure $p$ computed as part of the Stokes "
                                                  "solve. From this, the dynamic "
                                                  "topography is computed using the formula "
                                                  "$h=\\frac{\\sigma_{rr}}{(\\mathbf g \\cdot \\mathbf n)  \\rho}$ where $\\rho$ "
                                                  "is the density at the cell center. Note that this implementation takes "
                                                  "the direction of gravity into account, which means that reversing the flow "
                                                  "in backward advection calculations will not reverse the intantaneous topography "
                                                  "because the reverse flow will be divided by the reverse surface gravity."
                                                  "\n\n"
                                                  "Strictly speaking, the dynamic topography is of course a "
                                                  "quantity that is only of interest at the surface. However, "
                                                  "we compute it everywhere to make things fit into the framework "
                                                  "within which we produce data for visualization. You probably "
                                                  "only want to visualize whatever data this postprocessor generates "
                                                  "at the surface of your domain and simply ignore the rest of the "
                                                  "data generated.")
    }
  }
}
