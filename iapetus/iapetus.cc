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

#include <fstream>


const double radius_iapetus = 736.e3;
const double radius_cmb = 295.e3;
const double initial_temperature = 220.;
const double surface_temperature = 90.;

class Radioisotope
{
  public:
    Radioisotope( const double initial_heating, const double decay_constant )
      : q0(initial_heating), lambda(decay_constant) {}

    double heat ( const double time ) const
    {
      return q0 * std::exp( -lambda * time );
    }
  private:
    const double q0;
    const double lambda;
};

class Chondrite
{
  public:
    Chondrite()
      : U238( 1.6e-12, 1.55e-10/aspect::year_in_seconds), 
        U235( 2.99e-12, 9.84e-10/aspect::year_in_seconds),
        Th232( 1.00e-12, 4.95e-11/aspect::year_in_seconds),
        K40Ca( 1.43e-11, 4.96e-10/aspect::year_in_seconds),
        K40Ar( 6.36e-13, 5.81e-10/aspect::year_in_seconds)
    {}

    double heat( const double time ) const
    {
      return U238.heat(time) + U235.heat(time) + Th232.heat(time) + K40Ca.heat(time) + K40Ar.heat(time);
    }

  private:
    const Radioisotope U238;
    const Radioisotope U235;
    const Radioisotope Th232;
    const Radioisotope K40Ca;
    const Radioisotope K40Ar;
};

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
    class Iapetus : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:

        Iapetus ()
          : R(radius_cmb), 
            N(100),
            dr( R/(N-1) ),
            kappa(1.e-6),
            conductivity(4.)
        {
          T.assign( N, initial_temperature);
          r.assign( N, 0.0 );

          for( unsigned int i = 0; i<N; ++i)
            r[i] = dr*i;
        }

        virtual
        double temperature (const GeometryModel::Interface<dim> &gm,
                            const types::boundary_id id,
                            const Point<dim> &) const
        {
          const std::string boundary_name = gm.translate_id_to_symbol_name(id);

          if (boundary_name == "inner")
            return T.back();
          else if (boundary_name =="outer")
            return surface_temperature;
          else
            this->get_pcout()<<"Unknown boundary indicator ??"<<std::endl;
          return T.back();
        }

        virtual
        double minimal_temperature (const std::set<types::boundary_id> &) const
        {
          return surface_temperature;
        }

        virtual
        double maximal_temperature (const std::set<types::boundary_id> &) const
        {
          return T.back();
        }

        virtual
        void update()
        {
          //scale the heat flux to an equivalent 3D value
          const double Q = -1.e-2;
          update_temperature( Q, this->get_time(), this->get_timestep() );
          output_temperature_profile();
        }

      private:

        void update_temperature( const double heat_flux, const double time, const double dt )
        {
          const double eta = kappa*dt/2./dr/dr; 
          std::vector<double> a(N, 0.);
          std::vector<double> b(N, 0.);
          std::vector<double> c(N, 0.);
          std::vector<double> rhs(N, 0.);

          //Setup the tridiagonal system, where `a` is the lower diagonal,
          //`b`	is the diagonal, and `c` is the upper diagonal

          //Handle the lower bc, dTdr = 0
          b[0] = 1.;
          c[0] = -1.;
          //handle the internal nodes
          for (unsigned int i=1; i < N-1; ++i)
            {
              //assemble lhs matrix
              a[i] = -eta*r[i-1]/r[i];
              b[i] = 1. + 2.*eta;
              c[i] = -eta*r[i+1]/r[i];

              //diffusion part of rhs
              rhs[i] = (eta*r[i-1]/r[i]) * T[i-1] + 
                       (1. - 2.*eta) * T[i] + 
                       (eta*r[i+1]/r[i]) * T[i+1] ;
              //heating part of rhs
              rhs[i] += 0.5*dt*(rock.heat(time)+rock.heat(time+dt));
            }
          //handle the upper bc, k dTdr = Q
          a[N-1] = -1.;
          b[N-1] = 1.;
          rhs[N-1] = heat_flux * dr / conductivity;

          //Solve the system using the tridiagonal Thomas algorithm
          std::vector<double> c_prime(N, 0.);
          std::vector<double> d_prime(N, 0.);
          std::vector<double> Tnew(N, 0.);

          c_prime[0] = c[0]/b[0];
          d_prime[0] = rhs[0]/b[0];

          for (unsigned int i=1; i < N-1; ++i)
            {
              c_prime[i] = c[i]/(b[i] - a[i]*c_prime[i-1]);
              d_prime[i] = (rhs[i] - a[i]*d_prime[i-1])/(b[i] - a[i]*c_prime[i-1]);
            }
          d_prime[N-1] = (rhs[N-1] - a[N-1]*d_prime[N-2])/(b[N-1] - a[N-1]*c_prime[N-2]);

          Tnew[N-1] = d_prime[N-1];
          for (int i=N-2; i >= 0; --i)
            Tnew[i] = d_prime[i] - c_prime[i]*Tnew[i+1];

          //Update the temperature
          T = Tnew;
        }

        void output_temperature_profile()
        {
          if( dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
            {
              std::ofstream outfile;
              if (this->get_timestep_number() == 0)
                outfile.open("core_temperature.txt", std::ios::out);
              else
                outfile.open("core_temperature.txt", std::ios::app);

              const double time = this->get_time();
              for (unsigned int i=0; i<N; ++i)
                outfile<<time<<"\t"<<r[i]<<"\t"<<T[i]<<std::endl;
              outfile.close();
            }
          }

        std::vector<double> T; //temperature
        std::vector<double> r;

        const double R;
        const unsigned int N;
        const double dr;
        const double kappa;
        const double conductivity;
        const Chondrite rock;
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
    calculate_viscosity(const double, const double temperature,
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
        double initial_temperature (const Point<dim> &) const
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
