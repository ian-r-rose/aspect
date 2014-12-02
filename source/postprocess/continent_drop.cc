/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


#include <aspect/postprocess/continent_drop.h>

#include <aspect/simulator.h>
#include <aspect/global.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    void  
    ContinentDrop<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Continent drop");
        {
          prm.declare_entry ("Number of continents", "0",
                             Patterns::Integer (0), "");
          prm.declare_entry ("Drop time", "0.0",
                             Patterns::Double (0), "");
          prm.declare_entry ("Angular size", "30.0",
                             Patterns::Double (0), "");
          prm.declare_entry ("Depth", "100.e3",
                             Patterns::Double (0), "");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
    template <int dim>
    void
    ContinentDrop<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Continent drop");
        {
          n_continents = prm.get_integer("Number of continents");
          drop_time = prm.get_double("Drop time");
          angular_size = prm.get_double("Angular size");
          depth = prm.get_double("Depth");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      
      centers.assign(n_continents, Point<dim>() );
      for( unsigned int i=0; i < n_continents; ++i)
      {
        double theta = double(i) * 2.0*M_PI/ double(n_continents);
        
        centers[i][0] = std::cos(theta);
        centers[i][1] = std::sin(theta);
      }
      
      dropped = false;
    }

    template <int dim>
    bool 
    ContinentDrop<dim>::in_continent (Point<dim> p)
    {
      if ( this->get_geometry_model().depth(p) > depth) return false;

      for ( unsigned int i=0; i<n_continents; ++i)
      {
        Point<dim> p2 = p/p.norm();
        double theta = p2[0]*centers[i][0] + p2[1]*centers[i][1];
        theta = std::acos(theta)*180.0/M_PI;
        if(theta < angular_size/2.0) return true;
      }

      return false;
    }
      

    template <int dim>
    std::pair<std::string,std::string>
    ContinentDrop<dim>::execute (TableHandler &statistics)
    {
      if (dropped || this->get_time()/(this->convert_output_to_years() ? year_in_seconds : 1.0) < drop_time)
        return std::pair<std::string, std::string> ( "", "" );  

      Assert( this->n_compositional_fields() >= 1 , ExcMessage("Need at least one compositional field"));
      LinearAlgebra::BlockVector temporary_solution(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
      LinearAlgebra::BlockVector& solution = const_cast<LinearAlgebra::BlockVector&>(this->get_solution());
      temporary_solution = solution;

      // get the temperature/composition support points
      const unsigned int base_element = this->introspection().base_elements.compositional_fields;

      const std::vector<Point<dim> > support_points
        = this->get_fe().base_element(base_element).get_unit_support_points();
      Assert (support_points.size() != 0,
              ExcInternalError());

      // create an FEValues object with just the temperature/composition element
      FEValues<dim> fe_values (this->get_mapping(), this->get_fe(),
                               support_points,
                               update_quadrature_points);

      std::vector<types::global_dof_index> local_dof_indices (this->get_fe().dofs_per_cell);

      for (typename DoFHandler<dim>::active_cell_iterator cell = this->get_dof_handler().begin_active();
           cell != this->get_dof_handler().end(); ++cell)
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);

            // go through the temperature/composition dofs and set their global values
            // to the temperature/composition field interpolated at these points
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<this->get_fe().base_element(base_element).dofs_per_cell; ++i)
              {
                const unsigned int system_local_dof
                  = this->get_fe().component_to_system_index(this->introspection().component_indices.compositional_fields[0],
                                                             /*dof index within component=*/i);

                  const double value = ( in_continent( fe_values.quadrature_point(i) ) ? 1.0 : 0.0  ); 
                  temporary_solution(local_dof_indices[system_local_dof]) = value;

              }
          }


      temporary_solution.compress(VectorOperation::insert);
      solution = temporary_solution;
  
      dropped = true;

      return std::pair<std::string, std::string> ("",
                                                  "");
    }
  }
}


namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(ContinentDrop,
                                  "continent drop",
                                  "")
  }
}
