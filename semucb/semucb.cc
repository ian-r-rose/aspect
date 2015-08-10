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

#include <fstream>

extern "C" {
#include <ucb_A3d.h>
}

//Class for taking two std::vector<double> objects, and constructing a linearly interpolated
//function from the first to the second.
class LinearInterpolatedFunction
{
  public:

    //Do nothing constructor
    LinearInterpolatedFunction() {}

    //Constructor that calls initialize
    LinearInterpolatedFunction( const std::vector<double> &x, const std::vector<double> &y )
    {
      initialize(x,y);
    }

    //Fill a boost::container::flat_map with the x,y values.
    //The flat map has the property that the items in it are ordered,
    //so the input vectors do not necessarily need to be ordered, just 
    //paired such that x[i]->y[i].  The flat map also does binary searches
    //according to a key, but unlike std::map, is implemented in a contiguous
    //block of memory, so should have better speed and cache locality (at least in theory!)
    void initialize( const std::vector<double> &x, const std::vector<double> &y )
    {
      if( x.size() != y.size() ) std::cerr<<"Vectors given to LinearInterpolatedFunction are of different lengths!"<<std::endl;
      mapping.reserve(x.size());


      std::vector<double>::const_iterator xit = x.begin();
      std::vector<double>::const_iterator yit = y.begin();

      for ( ; xit != x.end(); ++xit, ++yit )
	mapping.insert( std::pair<double, double>( *xit, *yit ) );
    }

    //Evaluate the interpolation
    virtual double operator()( const double x ) const
    {
      //get an iterator to the left of the requested value
      boost::container::flat_map<double,double>::const_iterator entry = mapping.upper_bound(x);
 
      //If we are after the end of the map, just return the rightmost value
      if ( entry == mapping.end() ) return (--entry)->second;
      //If we are before the beginning of the map, just return the leftmost value
      if ( entry == mapping.begin() ) return (entry)->second;

      //Otherwise, interpolate between the two relevant values
      boost::container::flat_map<double,double>::const_iterator upper = entry, lower = --entry;
      const double dx = (x - lower->first)/(upper->first - lower->first);
      return dx*upper->second + (1.-dx)*lower->second;
    }

  private:
    boost::container::flat_map<double, double> mapping;
};

namespace MantleModel
{

  LinearInterpolatedFunction gravity;
  LinearInterpolatedFunction pressure;
  LinearInterpolatedFunction reference_temperature;
  LinearInterpolatedFunction reference_density; 
  LinearInterpolatedFunction P_wave_speed; 
  LinearInterpolatedFunction S_wave_speed; 
  LinearInterpolatedFunction bulk_modulus; 
  LinearInterpolatedFunction shear_modulus; 
  LinearInterpolatedFunction thermal_expansivity;
  LinearInterpolatedFunction heat_capacity;
  LinearInterpolatedFunction dGdT;

  void initialize()
  {
    static bool mantle_model_initialized = false; 
    if (mantle_model_initialized ) return;

    std::ifstream model_file( "mantle_model.txt" );

    std::vector<double> r_vec, g_vec, p_vec, T_vec, rho_vec, vp_vec, 
                        vs_vec, K_vec, G_vec, a_vec, c_vec, dgdt_vec;

    char tmpline[1024];
    model_file.getline(tmpline, 1024);  // read in the header
    while ( ! model_file.eof() )
      {
	double r, g, p, T, rho, vp, vs, K, G, a, c, dgdt;
	model_file >> r >> g >> p >> T >> rho >> vp >> vs >> K >> G >> a >> c >> dgdt;

	//Fill the vectors of interest
	r_vec.push_back(r);
	g_vec.push_back(g);
	p_vec.push_back(p);
	T_vec.push_back(T);
	rho_vec.push_back(rho);
	vp_vec.push_back(vp);
	vs_vec.push_back(vs);
	K_vec.push_back(K);
	G_vec.push_back(G);
	a_vec.push_back(a);
	c_vec.push_back(c);
	dgdt_vec.push_back(dgdt);
      }

    //Initialize the LinearInterpolatedFunctions
    gravity.initialize( r_vec, g_vec );
    pressure.initialize( r_vec, p_vec );
    reference_temperature.initialize( r_vec, T_vec );
    reference_density.initialize( r_vec, rho_vec );
    P_wave_speed.initialize( r_vec, vp_vec );
    S_wave_speed.initialize( r_vec, vs_vec );
    bulk_modulus.initialize( r_vec, K_vec );
    shear_modulus.initialize( r_vec, G_vec );
    thermal_expansivity.initialize( r_vec, a_vec );
    heat_capacity.initialize( r_vec, c_vec );
    dGdT.initialize( r_vec, dgdt_vec );

    mantle_model_initialized = true;
  }
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
      MantleModel::initialize();
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
      const double reference_T = MantleModel::reference_temperature( radius );
 
      //Calculate reference moduli
      const double ref_vs = 0.5 * (ref_vsv + ref_vsh); //Don't care about anisotropy at the moment, so average these
      const double ref_G = ref_vs*ref_vs*ref_rho;

      //Shortcut if ref_vs = 0
      if ( ref_vs == 0.0 ) return reference_T;

      const double dvsdT = 0.5 * ref_vs * ( MantleModel::dGdT(radius) /ref_G + MantleModel::thermal_expansivity(radius) );

      double vs_perturbation;
      get_a3d_perturbation_(&lat, &lon, &radius_km, const_cast<char *>("S"), &vs_perturbation);

      const double dT = vs_perturbation*ref_vs / dvsdT;

      return reference_T + dT;
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
