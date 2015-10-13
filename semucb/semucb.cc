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


#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <aspect/utilities.h>
#include <aspect/simulator.h>

#include <aspect/initial_conditions/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/postprocess/visualization.h>


#include <boost/container/flat_map.hpp>
#include <fstream>

extern "C" {
#include <ucb_A3d.h>
}

const double radius_earth = 6371.e3;
const double radius_cmb = 3481.e3;

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
      if ( x.size() != y.size() ) std::cerr<<"Vectors given to LinearInterpolatedFunction are of different lengths!"<<std::endl;
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

class PiecewiseConstantFunction
{
  public:

    //Do nothing constructor
    PiecewiseConstantFunction() {}

    //Constructor that calls initialize
    PiecewiseConstantFunction( const std::vector<double> &x, const std::vector<double> &y )
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
      if ( x.size() != y.size() ) std::cerr<<"Vectors given to PiecewiseConstantFunction are of different lengths!"<<std::endl;
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

      //Otherwise, choose the value of the lower entry
      boost::container::flat_map<double,double>::const_iterator lower = --entry;
      return lower->second;
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
  PiecewiseConstantFunction  viscosity_profile;

  double thermal_conductivity( double /*radius*/, double pressure, double temperature )
  {
    if( pressure < 14.e9 ) // Upper mantle
    {
      //Parameterization from Xu et al (2004) for olivine
      const double k298 = 4.13;
      const double a = 0.032;
      return k298 * std::pow( 298./temperature, 0.5)*(1. + a*pressure/1.e9);
    }
    else if ( pressure <= 23.5e9 && pressure >= 14.e9 ) //transition zone
    {
      //Parameterization from Xu et al (2004) for wadsleyite
      const double k298 = 8.10;
      const double a = 0.023;
      return k298 * std::pow( 298./temperature, 0.5)*(1. + a*pressure/1.e9);
    }
    else //lower mantle
    {
      //Simple parameterization of thermal conductivity for pyrolite
      //from Stackhouse, Stixrude, and Karki (2015)
      const double x = temperature/1200.;
      const double f = (x > 1.0 ? 2./3. * std::pow(x, -0.5) + x/3. : 1.0);
      return (4.9 + 0.105 * pressure/1.e9)*f/x;
    }
  }

  void initialize()
  {
    static bool mantle_model_initialized = false;
    if (mantle_model_initialized ) return;

    std::ifstream model_file( "mantle_model.txt" );

    std::vector<double> r_vec, g_vec, p_vec, T_vec, rho_vec, vp_vec,
        vs_vec, K_vec, G_vec, a_vec, c_vec, dgdt_vec;

    char tmpline[1024];
    model_file.getline(tmpline, 1024);  // read in the header
    double r, g, p, T, rho, vp, vs, K, G, a, c, dgdt;
    while( model_file >> r >> g >> p >> T >> rho >> vp >> vs >> K >> G >> a >> c >> dgdt)
      {
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

    std::ifstream viscosity_file("V2.txt"); //V2 does not have the low viscosity channel
    std::vector<double> viscosity_radius, viscosity_value;
    for( unsigned int i=0; i < 4; ++i ) viscosity_file.getline(tmpline, 1024); //read in header lines
    double depth, visc;
    while(  viscosity_file >> depth >> visc )
      {
        viscosity_radius.push_back( radius_earth - depth*1.e3 );
        viscosity_value.push_back( visc * 1.e21 );
      }
    //Initialize the PiecewiseConstantFunction
    viscosity_profile.initialize( viscosity_radius, viscosity_value );

    //No need to reinitialize
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
  using namespace dealii;

  namespace GravityModel
  {

    template <int dim>
    class SEMUCB : public Interface<dim>
    {
      public:
        SEMUCB()
        {
          MantleModel::initialize();
        }

        //Return a gravity vector that is radial, with magnitude
        //given by the gravity function in the MantleModel
        virtual Tensor<1,dim>
        gravity_vector (const Point<dim> &p) const
        {
          const double r = p.norm();
          if ( r == 0.0 ) return Tensor<1,dim>();
          else return -MantleModel::gravity(r) * p / r;
        }
    };
  }

  namespace BoundaryTemperature
  {
    template <int dim>
    class SEMUCB : public Interface<dim>
    {
      public:
        virtual
        double temperature (const GeometryModel::Interface<dim> &,
                            const types::boundary_id,
                            const Point<dim> &location) const
        {
          return MantleModel::reference_temperature( location.norm() );
        }

        virtual
        double minimal_temperature (const std::set<types::boundary_id> &) const
        {
          return MantleModel::reference_temperature( radius_earth );
        }

        virtual
        double maximal_temperature (const std::set<types::boundary_id> &) const
        {
          return MantleModel::reference_temperature( radius_cmb );
        }
    };
  }

  namespace MaterialModel
  {

    template <int dim>
    class SEMUCB : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:

        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                              MaterialModel::MaterialModelOutputs<dim> &out) const;

        virtual bool is_compressible () const
        {
          return true;
        }

        virtual double reference_viscosity () const
        {
          return eta;
        }
        virtual double reference_density () const
        {
          return MantleModel::reference_density(reference_radius);
        }

        virtual double reference_thermal_expansion_coefficient () const
        {
          return MantleModel::thermal_expansivity(reference_radius);
        }

        double reference_thermal_diffusivity () const
        {
          return reference_thermal_conductivity/reference_density()/reference_cp();
        }

        double reference_cp () const
        {
          return MantleModel::heat_capacity( reference_radius );
        }

        static
        void
        declare_parameters (ParameterHandler &prm);

        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
        double reference_radius;
        double eta;
        double reference_thermal_conductivity;
    };

    template <int dim>
    void
    SEMUCB<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int i=0; i < in.position.size(); ++i)
        {
          const double radius = in.position[i].norm();

          out.viscosities[i] = MantleModel::viscosity_profile(radius);
          out.densities[i] = MantleModel::reference_density(radius)
                             * (1.0 + MantleModel::thermal_expansivity(radius) *
                                (MantleModel::reference_temperature(radius) - in.temperature[i]) );

          out.thermal_expansion_coefficients[i] = MantleModel::thermal_expansivity(radius);
          out.specific_heat[i] = MantleModel::heat_capacity(radius);
          out.compressibilities[i] = 1.0/MantleModel::bulk_modulus(radius);
          out.thermal_conductivities[i] = MantleModel::thermal_conductivity(radius, in.pressure[i], in.temperature[i]);


          // Pressure derivative of entropy at the given positions.
          out.entropy_derivative_pressure[i] = 0.0;
          // Temperature derivative of entropy at the given positions.
          out.entropy_derivative_temperature[i] = 0.0;
          // Change in composition due to chemical reactions at the
          // given positions. The term reaction_terms[i][c] is the
          // change in compositional field c at point i.
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;
        }
    }

    template <int dim>
    void
    SEMUCB<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("SEMUCB");
        {
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    SEMUCB<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("SEMUCB");
        {
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      reference_radius = 4.9e6; //mid-mantle-ish type depth
      eta = 1.e21;
      reference_thermal_conductivity = 3.0;

      initialize_semucb();
      MantleModel::initialize();

      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::temperature;
      this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
      this->model_dependence.compressibility = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
      this->model_dependence.specific_heat = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
      this->model_dependence.thermal_conductivity = NonlinearDependence::temperature | NonlinearDependence::pressure;
    }
  }

  namespace InitialConditions
  {
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
      if ( radius > 6370999.) radius = 6370999.; //Ref model has problems at RE
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

  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      class SEMUCBDensityAnomaly
        : public DataPostprocessorScalar<dim>,
          public SimulatorAccess<dim>,
          public Interface<dim>
      {
        public:
          SEMUCBDensityAnomaly ()
            :  DataPostprocessorScalar<dim> ("density_anomaly", update_values | update_q_points | update_gradients)
          {
            MantleModel::initialize();
          }

          virtual
          void
          compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                             const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                             const std::vector<std::vector<Tensor<2,dim> > > &,
                                             const std::vector<Point<dim> >                  &,
                                             const std::vector<Point<dim> >                  &evaluation_points,
                                             std::vector<Vector<double> >                    &computed_quantities) const
	  {
	    const unsigned int n_quadrature_points = uh.size();
	    Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
	    Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
	    Assert (uh[0].size() == this->introspection().n_components,           ExcInternalError());

	    MaterialModel::MaterialModelInputs<dim> in(n_quadrature_points,
						       this->n_compositional_fields());
	    MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
							 this->n_compositional_fields());

	    in.position = evaluation_points;
	    in.strain_rate.resize(0); // we do not need the viscosity
	    for (unsigned int q=0; q<n_quadrature_points; ++q)
	      {
		in.pressure[q]=uh[q][this->introspection().component_indices.pressure];
		in.temperature[q]=uh[q][this->introspection().component_indices.temperature];
		for (unsigned int d = 0; d < dim; ++d)
		  {
		    in.velocity[q][d]=uh[q][this->introspection().component_indices.velocities[d]];
		    in.pressure_gradient[q][d] = duh[q][this->introspection().component_indices.pressure][d];
		  }

		for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
		  in.composition[q][c] = uh[q][this->introspection().component_indices.compositional_fields[c]];
	      }

	    this->get_material_model().evaluate(in, out);

	    for (unsigned int q=0; q<n_quadrature_points; ++q)
	      computed_quantities[q](0) = out.densities[q] - MantleModel::reference_density(in.position[q].norm());
	  }
      };

      template <int dim>
      class SEMUCBTemperatureAnomaly
        : public DataPostprocessorScalar<dim>,
          public SimulatorAccess<dim>,
          public Interface<dim>
      {
        public:
          SEMUCBTemperatureAnomaly ()
            :  DataPostprocessorScalar<dim> ("temperature_anomaly", update_values | update_q_points)
          {
            MantleModel::initialize();
          }

          virtual
          void
          compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                             const std::vector<std::vector<Tensor<1,dim> > > &,
                                             const std::vector<std::vector<Tensor<2,dim> > > &,
                                             const std::vector<Point<dim> >                  &,
                                             const std::vector<Point<dim> >                  &evaluation_points,
                                             std::vector<Vector<double> >                    &computed_quantities) const
	  {
	    const unsigned int n_quadrature_points = uh.size();
	    Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
	    Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
	    Assert (uh[0].size() == this->introspection().n_components,   ExcInternalError());

	    for (unsigned int q=0; q<n_quadrature_points; ++q)
	      computed_quantities[q](0) = uh[q][this->introspection().component_indices.temperature] 
                                          - MantleModel::reference_temperature(evaluation_points[q].norm());
	  }
      };
      template <int dim>
      class AdvectiveHeatFluxAnomaly
        : public DataPostprocessorScalar<dim>,
          public SimulatorAccess<dim>,
          public Interface<dim>
      {
        public:
          AdvectiveHeatFluxAnomaly ()
            : DataPostprocessorScalar<dim> ("advective_heat_flux_anomaly",
                                             update_values | update_q_points | update_gradients)
          {}

          virtual
          void
          compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                             const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                             const std::vector<std::vector<Tensor<2,dim> > > &,
                                             const std::vector<Point<dim> >                  &,
                                             const std::vector<Point<dim> >                  &evaluation_points,
                                             std::vector<Vector<double> >                    &computed_quantities) const
          {
	    const unsigned int n_quadrature_points = uh.size();
	    Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
	    Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
	    Assert (uh[0].size() == this->introspection().n_components,           ExcInternalError());

	    MaterialModel::MaterialModelInputs<dim> in(n_quadrature_points,
						       this->n_compositional_fields());
	    MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
							 this->n_compositional_fields());

	    //Create vector for the temperature gradients.  All the other things
	    //we need are in MaterialModelInputs/Outputs
	    std::vector<Tensor<1,dim> > temperature_gradient(n_quadrature_points);

	    in.position = evaluation_points;
	    in.strain_rate.resize(0); // we do not need the viscosity
	    for (unsigned int q=0; q<n_quadrature_points; ++q)
	      {
		in.pressure[q]=uh[q][this->introspection().component_indices.pressure];
		in.temperature[q]=uh[q][this->introspection().component_indices.temperature];
		for (unsigned int d = 0; d < dim; ++d)
		  {
		    in.velocity[q][d]=uh[q][this->introspection().component_indices.velocities[d]];
		    in.pressure_gradient[q][d] = duh[q][this->introspection().component_indices.pressure][d];
		    temperature_gradient[q][d] = duh[q][this->introspection().component_indices.temperature][d];
		  }

		for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
		  in.composition[q][c] = uh[q][this->introspection().component_indices.compositional_fields[c]];
	      }

	    this->get_material_model().evaluate(in, out);

	    for (unsigned int q=0; q<n_quadrature_points; ++q)
	      {
		const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(in.position[q]);
		const Tensor<1,dim> vertical = -gravity/( gravity.norm() != 0.0 ?
							  gravity.norm() : 1.0 );
		const double advective_flux = (in.velocity[q] * vertical) * in.temperature[q] *
					      out.densities[q]*out.specific_heat[q];
		const double ref_advective_flux = (in.velocity[q] * vertical) *  
                                                  MantleModel::reference_temperature(evaluation_points[q].norm()) *
                                                  out.densities[q]*out.specific_heat[q];
		computed_quantities[q](0) = advective_flux - ref_advective_flux;
	      }
	    }
      };
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
  namespace GravityModel
  {
    ASPECT_REGISTER_GRAVITY_MODEL(SEMUCB,
                                  "SEMUCB",
                                  "")
  }
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(SEMUCB,
                                   "SEMUCB",
                                   "")
  }
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(SEMUCBDensityAnomaly,
                                                  "SEMUCB density anomaly",
                                                  "")
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(SEMUCBTemperatureAnomaly,
                                                  "SEMUCB temperature anomaly",
                                                  "")
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(AdvectiveHeatFluxAnomaly,
                                                  "advective heat flux anomaly",
                                                  "")
    }
  }
  namespace BoundaryTemperature
  {
    ASPECT_REGISTER_BOUNDARY_TEMPERATURE_MODEL(SEMUCB,
                                               "SEMUCB",
                                               "")
  }
}
