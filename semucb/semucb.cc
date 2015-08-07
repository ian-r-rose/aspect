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
#include <deal.II/base/function.h>
#include <boost/container/flat_map.hpp>

#include <aspect/utilities.h>
#include <aspect/initial_conditions/interface.h>
#include <aspect/simulator.h>

extern "C" {
#include <ucb_A3d.h>
}


static void initialize_semucb()
{
  static bool semucb_initialized = false;
  if ( semucb_initialized ) return;

  init_ucb_ref_();
  init_ucb_a3d_();
  init_moho_();

  semucb_initialized = true;
}

namespace aspect
{

  class LinearInterpolatedFunction
  {
      LinearInterpolatedFunction( const std::vector<double> &x, const std::vector<double> &y )
      {
        AssertThrow( x.size() == y.size(), ExcMessage("Vectors for interpolation must be the same size") );
        mapping.reserve(x.size());

        std::vector<double>::const_iterator xit = x.begin();
        std::vector<double>::const_iterator yit = y.begin();

        for ( ; xit != x.end(); ++xit, ++yit )
          mapping.insert( std::pair<double, double>( *xit, *yit ) );
      }

      virtual double value( const double x )
      {
        boost::container::flat_map<double,double>::iterator entry = mapping.upper_bound(x);
        if ( entry == mapping.end() ) return (--entry)->second;
        if ( entry == mapping.begin() ) return (entry)->second;

        boost::container::flat_map<double,double>::iterator upper = entry, lower = --entry;
        const double dx = (x - lower->first)/(upper->first - lower->first);
        return dx*upper->second + (1.-dx)*lower->second;
      }


    private:
      boost::container::flat_map<double, double> mapping;
  };

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
 
      initialize_semucb();
    }

    template <int dim>
    double
    SEMUCB<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      const double r2d = 180.0/M_PI;

      //Get the position in geographic coordinates
      const std_cxx11::array<double,dim> scoord = aspect::Utilities::spherical_coordinates( position );
      double lat = 90.-scoord[2]*r2d; //convert to latitude
      double lon = scoord[1]*r2d;
      double radius = scoord[0];
      double radius_km = radius/1000.;

      //Get the 1D reference model at this radius
      double ref_rho, ref_vpv, ref_vsv, ref_qk, ref_qmu, ref_vph, ref_vsh, ref_eta;
      if( radius > 6370999.) radius = 6370999.;  //Ref model has problems at RE
      get_ucb_ref_(&radius, &ref_rho, &ref_vpv, &ref_vsv, &ref_qk, &ref_qmu, &ref_vph, &ref_vsh, &ref_eta);

      //Get the 1D reference temperature profile at this radius
      const double reference_temperature = 0.;//adiabatic_initial_conditions.initial_temperature( position );
 
      //Calculate reference moduli
      const double ref_vs = 0.5 * (ref_vsv + ref_vsh); //Don't care about anisotropy at the moment, so average these
      const double ref_G = ref_vs*ref_vs*ref_rho;

      //Shortcut if ref_vs = 0
      if ( ref_vs == 0.0 ) return reference_temperature;

      const double dGdT = 3.e7;
      const double alpha = 3.e-5;
      const double dvsdT = 0.5 * ref_vs * ( dGdT/ref_G + alpha );
      this->get_pcout()<<"dvsdt "<<dvsdT<<"\t"<<ref_G<<"\t"<<ref_vs<<std::endl;

      double vs_perturbation;
      get_a3d_perturbation_(&lat, &lon, &radius_km, const_cast<char *>("S"), &vs_perturbation);

      const double dT = vs_perturbation*ref_vs / dvsdT;

      return reference_temperature + dT;
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
