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
#include <aspect/material_model/interface.h>
#include <aspect/postprocess/visualization.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/gravity_model/radial.h>


const double radius_iapetus = 736.e3;
const double radius_cmb = 295.e3;

namespace aspect
{

  namespace GravityModel
  {
    using namespace dealii;

    template <int dim>
    class Centrifugal : public virtual GravityModel::Interface<dim>, public virtual SimulatorAccess<dim>
    {
      public:
        virtual Tensor<1,dim> gravity_vector (const Point<dim> &position) const
        {
          return centrifugal_vector(position);
        }

        Tensor<1,dim> centrifugal_vector(const Point<dim> &p) const
	{
	  //Not the usual form, but works equally well in 2d and 3d
	  const Point<dim> r = p - (p*rotation_axis)*rotation_axis;
	  return Omega*Omega*r;
	}

        virtual void initialize_simulator(const Simulator<dim> &sim)
        {
          SimulatorAccess<dim>::initialize_simulator(sim);
        }

	static void declare_parameters (ParameterHandler &prm)
	{
	  prm.enter_subsection("Gravity model");
	  {
	    prm.enter_subsection("Centrifugal");
	    {
	      prm.declare_entry ("Omega", "7.29e-5",
				 Patterns::Double (0),
				 "Angular velocity of the planet, rad/s.");
	    }
	    prm.leave_subsection ();
	  }
	  prm.leave_subsection ();
	}

	void parse_parameters (ParameterHandler &prm)
	{
	  prm.enter_subsection("Gravity model");
	  {
	    prm.enter_subsection("Centrifugal");
	    {
	      Omega = prm.get_double ("Omega");
	    }
	    prm.leave_subsection ();
	  }
	  prm.leave_subsection ();

          rotation_axis = Tensor<1,dim>();
          rotation_axis[dim-1] = 1.0;
	}

      private:

        double Omega;  //angular velocity of the planet
	Tensor<1,dim> rotation_axis; //axis of rotation unit vector
        
    };

    template <int dim>
    class RadialConstantWithCentrifugal : public virtual GravityModel::Interface<dim>, public virtual SimulatorAccess<dim>
    {
      public:
        virtual Tensor<1,dim> gravity_vector (const Point<dim> &p) const
        {
          return gravity.gravity_vector(p) + centrifugal.gravity_vector(p);
        }

        virtual void initialize_simulator(const Simulator<dim> &sim)
        {
          centrifugal.initialize_simulator(sim);
        }
        
        void parse_parameters( ParameterHandler &prm )
	{
	  gravity.parse_parameters(prm);
	  centrifugal.parse_parameters(prm);
        }

      private:
         Centrifugal<dim> centrifugal;
         RadialConstant<dim> gravity;
    };

    template <int dim>
    class RadialLinearWithCentrifugal : public virtual GravityModel::Interface<dim>, public virtual SimulatorAccess<dim>
    {
      public:
        virtual Tensor<1,dim> gravity_vector (const Point<dim> &p) const
        {
          return gravity.gravity_vector(p) + centrifugal.gravity_vector(p);
        }

        virtual void initialize_simulator(const Simulator<dim> &sim)
        {
          centrifugal.initialize_simulator(sim);
        }
        
        void parse_parameters( ParameterHandler &prm )
	{
	  gravity.parse_parameters(prm);
	  centrifugal.parse_parameters(prm);
        }

      private:
         Centrifugal<dim> centrifugal;
         RadialLinear<dim> gravity;
    };

  }

  namespace BoundaryTemperature
  {
    template <int dim>
    class Iapetus : public Interface<dim>
    {
      public:
        virtual
        double temperature (const GeometryModel::Interface<dim> &,
                            const types::boundary_id,
                            const Point<dim> &location) const
        {
          return 220.;
        }

        virtual
        double minimal_temperature (const std::set<types::boundary_id> &) const
        {
          return 220.;
        }

        virtual
        double maximal_temperature (const std::set<types::boundary_id> &) const
        {
          return 220.;
        }
    };
  }

  namespace MaterialModel
  {

    template <int dim>
    class Ice_1h : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:

        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                              MaterialModel::MaterialModelOutputs<dim> &out) const;

        double calculate_viscosity( const double pressure,
                                     const double temperature,
                                     const SymmetricTensor<2,dim> &strain_rate) const;

        virtual bool is_compressible () const
        {
          return false;
        }

        virtual double reference_viscosity () const
        {
          return 1.e15;
        }
        virtual double reference_density () const
        {
          return density;
        }

        virtual double reference_thermal_expansion_coefficient () const
        {
          return thermal_expansivity;
        }

        double reference_thermal_diffusivity () const
        {
          return thermal_conductivity/density/heat_capacity;
        }

        double reference_cp () const
        {
          return heat_capacity;
        }

        static
        void
        declare_parameters (ParameterHandler &prm);

        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
        //Thermodynamic parameters
        double density;
        double heat_capacity;
        double thermal_expansivity;
        double thermal_conductivity;
        double reference_temperature;

        //Parameters for stability of viscosity calculations
        double minimum_viscosity;
        double maximum_viscosity;
        double minimum_strain_rate;
        double relative_strain_rate_tolerance;
        double absolute_strain_rate_tolerance;
        unsigned int max_stress_iterations;

        //Parameters for dislocation creep
        double A_dis;
        double Q_dis;
        double dislocation_stress_exponent;
        
        //Parameters for diffusion creep
        double A_diff;
        double Q_diff;
        double diffusion_grain_size_exponent;
        double grain_size;

        //Parameters for basal slip
        double A_bs;
        double basal_slip_stress_exponent;
        double Q_bs;

        //Parameters for grain boundary sliding
        double A_gbs;
        double Q_gbs;
        double grain_boundary_sliding_stress_exponent;
        double grain_boundary_sliding_grain_size_exponent;

    };

    template <int dim>
    double
    Ice_1h<dim>::
    calculate_viscosity(const double pressure, const double temperature,
                        const SymmetricTensor<2,dim> &strain_rate) const
    {
      const double edot_ii = std::max( std::sqrt( std::fabs( second_invariant( deviator(strain_rate)))),
                                       minimum_strain_rate);

      //Everything but the stress dependence for:
      
      //diffusion creep      
      const double diffusion_creep_factor = A_diff * 
                                            std::pow( grain_size, -diffusion_grain_size_exponent ) *
                                            std::exp( -Q_diff / constants::gas_constant / temperature);
      //dislocation creep
      const double dislocation_creep_factor = A_dis * 
                                              std::exp( -Q_dis / constants::gas_constant / temperature);
      //basal slip
      const double basal_slip_factor = A_bs * 
                                       std::exp( -Q_bs / constants::gas_constant / temperature );
      //grain boundary sliding
      const double grain_boundary_sliding_factor = A_gbs * 
                                                   std::pow(grain_size, -grain_boundary_sliding_grain_size_exponent) *
                                                   std::exp( -Q_gbs / constants::gas_constant / temperature );

      //initial guess at stress: all accommodated by diffusion creep
      double stress_ii = edot_ii/diffusion_creep_factor;
      //strain rates for the specific mechanisms
      double edot_diff, edot_dis, edot_bs, edot_gbs, edot_superplastic, strain_rate_residual;
      //derivative of strain rate with respect to stress
      double d_edot_d_stress;
      //iterations for Newton's method
      unsigned int stress_iteration = 0;

      do
        {
          //Calculate strain rates based on the current guess for stress
          edot_diff = diffusion_creep_factor * stress_ii;
          edot_dis = dislocation_creep_factor * std::pow(stress_ii, dislocation_stress_exponent); 
          edot_bs = basal_slip_factor * std::pow(stress_ii, basal_slip_stress_exponent); 
          edot_gbs = grain_boundary_sliding_factor * std::pow(stress_ii, grain_boundary_sliding_stress_exponent); 
          edot_superplastic = 1./(1./edot_bs + 1./edot_gbs);

          //evaluate the error on the strain rate
          strain_rate_residual = edot_diff + edot_dis + edot_superplastic - edot_ii;

          d_edot_d_stress = ( edot_diff + 
                              edot_dis*dislocation_stress_exponent + 
                              (grain_boundary_sliding_stress_exponent/edot_gbs + basal_slip_stress_exponent/edot_bs) * 
                               edot_superplastic * edot_superplastic) / stress_ii;

          stress_ii -= strain_rate_residual/d_edot_d_stress;
          stress_iteration++;
        }
      while (std::abs(strain_rate_residual/edot_ii) > relative_strain_rate_tolerance 
             && stress_iteration < max_stress_iterations);

      const double viscosity = std::min(std::max(stress_ii/2./edot_ii, minimum_viscosity), maximum_viscosity);
      return viscosity;
    }

    template <int dim>
    void
    Ice_1h<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int i=0; i < in.position.size(); ++i)
        {
          if( in.strain_rate.size() != 0 )
            out.viscosities[i] = calculate_viscosity(in.pressure[i], in.temperature[i], in.strain_rate[i]);

          out.densities[i] = density * (1.0 - thermal_expansivity * (in.temperature[i] - reference_temperature) );

          out.thermal_expansion_coefficients[i] = thermal_expansivity;
          out.specific_heat[i] = heat_capacity;
          out.compressibilities[i] = 0.;
          out.thermal_conductivities[i] = thermal_conductivity;


          //Things having to do with phase transitions
          out.entropy_derivative_pressure[i] = 0.0;
          out.entropy_derivative_temperature[i] = 0.0;
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;
        }
    }

    template <int dim>
    void
    Ice_1h<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Ice_1h");
        {
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    Ice_1h<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Ice_1h");
        {
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      //Thermodynamic parameters
      density = 935.;
      thermal_expansivity = 1.e-4;
      thermal_conductivity = 3.;
      heat_capacity = 2000.;
      reference_temperature = 273.;

      //Viscosity calculation parameters
      minimum_viscosity = 1.e13;
      maximum_viscosity = 1.e20;
      minimum_strain_rate = 1.e-20;
      max_stress_iterations = 10;
      relative_strain_rate_tolerance = 1.e-4;
      absolute_strain_rate_tolerance = 1.e-16;

      //Dislocation
      A_dis = 4.e-19;
      Q_dis = 60.e3;
      dislocation_stress_exponent = 4.;

      //Diffusion
      grain_size = 1.e-3;
      diffusion_grain_size_exponent = 2.;
      A_diff = 3.5e-10;
      Q_diff = 59.4e3;

      //Basal slip
      A_bs = 2.2e-7;
      Q_bs = 60.e3;
      basal_slip_stress_exponent = 2.4;
      
      //Grain boundary sliding
      A_gbs = 6.2e-14;
      Q_gbs = 49.e3;
      grain_boundary_sliding_stress_exponent = 1.8;
      grain_boundary_sliding_grain_size_exponent = 1.4;
      

      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::temperature | NonlinearDependence::strain_rate;
      this->model_dependence.density = NonlinearDependence::temperature;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
    }
  }

  namespace InitialConditions
  {
    template <int dim>
    class Iapetus : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Return the initial temperature as a function of position.
         */
        virtual
        double initial_temperature (const Point<dim> &position) const
        {
          return 220.;
        }

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm)
	{
	  prm.enter_subsection("Initial conditions");
	  {
	    prm.enter_subsection("Iapetus");
	    {
	    }
	    prm.leave_subsection ();
	  }
	  prm.leave_subsection ();
	}

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm)
        {
          prm.enter_subsection("Initial conditions");
          {
            prm.enter_subsection("Iapetus");
            {
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }

      private:
    };

  }

}

// explicit instantiations
namespace aspect
{
  namespace InitialConditions
  {
    ASPECT_REGISTER_INITIAL_CONDITIONS(Iapetus,
                                       "iapetus",
                                       "")
  }
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(Ice_1h,
                                   "ice_1h",
                                   "")
  }
  namespace BoundaryTemperature
  {
    ASPECT_REGISTER_BOUNDARY_TEMPERATURE_MODEL(Iapetus,
                                               "iapetus",
                                               "")
  }

  namespace GravityModel
  {
    ASPECT_REGISTER_GRAVITY_MODEL(Centrifugal,
                                  "centrifugal",
                                  "what it sounds like.")
    ASPECT_REGISTER_GRAVITY_MODEL(RadialConstantWithCentrifugal,
                                  "radial constant with centrifugal",
                                  "what it sounds like.")
    ASPECT_REGISTER_GRAVITY_MODEL(RadialLinearWithCentrifugal,
                                  "radial constant with centrifugal",
                                  "what it sounds like.")

  }
}
