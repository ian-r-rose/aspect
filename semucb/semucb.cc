/*
  Copyright (C) 2015 by the authors of the ASPECT code.

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


#ifndef __aspect__initial_conditions_semucb_h
#define __aspect__initial_conditions_semucb_h

#include <deal.II/base/std_cxx11/array.h>

#include <aspect/utilities.h>
#include <aspect/initial_conditions/interface.h>
#include <aspect/simulator.h>

extern "C" { 
#include <ucb_A3d.h>
}

namespace aspect
{
  namespace InitialConditions
  {
    using namespace dealii;

    /**
     *
     * @ingroup InitialConditionsModels
     */
    template <int dim>
    class SEMUCB : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Return the initial temperature as a function of position.
         */
        virtual
        double initial_temperature (const Point<dim> &position) const;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
    };

    template <int dim>
    void
    SEMUCB<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("SEMUCB");
        {

        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }



    template <int dim>
    void
    SEMUCB<dim>::parse_parameters (ParameterHandler &prm)
    {
      AssertThrow( dim == 3 , ExcMessage("SEMUCB must be used with a 3D model"));
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("SEMUCB");
        {
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

      init_ucb_ref_();
      init_ucb_a3d_();
      init_moho_();
    }

    template <int dim>
    double
    SEMUCB<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      const double r2d = 180.0/M_PI;

      const std_cxx11::array<double,dim> scoord = aspect::Utilities::spherical_coordinates( position );
      double lat = 90.-scoord[2]*r2d; //convert to latitude
      double lon = scoord[1]*r2d;
      double radius = scoord[0]/1000.; //convert to km

      double vs_perturbation;
      get_a3d_perturbation_(&lat, &lon, &radius, "S", &vs_perturbation);
 
/*      const double dvsdT = -1.e-4;
      const double reference_T = 1000.;

      double dT = vs_perturbation/dvsdT;
      return reference_T + dT;*/
      return vs_perturbation;
    }      
      

  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialConditions
  {
    ASPECT_REGISTER_INITIAL_CONDITIONS(SEMUCB,
                                       "SEMUCB",
                                       "")
  }
}
#endif
