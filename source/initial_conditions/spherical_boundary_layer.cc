#include <aspect/initial_conditions/spherical_boundary_layer.h>
#include <aspect/postprocess/interface.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/sphere.h>
#include <cmath>
#include <ctime>
#include <boost/math/special_functions/erf.hpp>

namespace aspect
{
  namespace InitialConditions
  {
    template <int dim>
    double
    SphericalBoundaryLayer<dim>::
    initial_temperature (const Point<dim> &position) const
    {

      Assert ((dynamic_cast<const GeometryModel::SphericalShell<dim>*>
              (&this->get_geometry_model()) != 0) ||
              (dynamic_cast<const GeometryModel::Sphere<dim>*>
              (&this->get_geometry_model()) != 0), 
              ExcMessage ("This initial condition can only be used if the geometry "
                          "is spherical."));

      //get radii of the planet
      double R1, R0, Rm;
 
      double T_surface = this->get_boundary_temperature().minimal_temperature(this->get_fixed_temperature_boundary_indicators());
      double T_bottom = this->get_boundary_temperature().maximal_temperature(this->get_fixed_temperature_boundary_indicators());

      double p_bottom = this->get_adiabatic_conditions().pressure(position);
      double p_top = 0.0;
               
      const Point<dim> surface_point = this->get_geometry_model().representative_point(0.0);
      const Point<dim> bottom_point = this->get_geometry_model().representative_point(this->get_geometry_model().maximal_depth());
      double depth = this->get_geometry_model().depth(position);

      R0 = bottom_point.norm();
      R1 = surface_point.norm();
      Rm = (R1+R0)/2.0;

      //get the angle of the point (2d only)
      double phi = std::atan2(position[0],position[1]);
      double mantle_temp;
      double top_boundary_thickness, bottom_boundary_thickness;

      // look up material properties
      typename MaterialModel::Interface<dim>::MaterialModelInputs in(2, this->n_compositional_fields());
      typename MaterialModel::Interface<dim>::MaterialModelOutputs out(2, this->n_compositional_fields());
      in.position[0]=surface_point;
      in.position[1]=bottom_point;
      in.temperature[0]=T_surface;
      in.temperature[1]=T_bottom;
      in.pressure[0]=p_top;
      in.pressure[1]=p_bottom;
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
      {
        in.composition[0][c] = 0.;
        in.composition[1][c] = 0.;
      }
      in.strain_rate[0] = SymmetricTensor<2,dim>();
      in.strain_rate[1] = SymmetricTensor<2,dim>(); 

      this->get_material_model().evaluate(in, out);

      const double rho_top = out.densities[0];
      const double rho_bottom = out.densities[1];
  
      const double kappa_top = out.thermal_conductivities[0] / (out.densities[0] * out.specific_heat[0]);
      const double kappa_bottom = out.thermal_conductivities[1] / (out.densities[1] * out.specific_heat[1]);

      const double eta_top = out.viscosities[0];
      const double eta_bottom = out.viscosities[1];

      const double alpha_top = out.thermal_expansion_coefficients[0];
      const double alpha_bottom = out.thermal_expansion_coefficients[1];

      const double g_top = this->get_gravity_model().gravity_vector(surface_point).norm();
      const double g_bottom = this->get_gravity_model().gravity_vector(bottom_point).norm();


      if(dynamic_cast<const GeometryModel::Sphere<dim>*>(&this->get_geometry_model()))
      {
        R0 = -R1;
        Rm = 0.0;
        mantle_temp = backup_temperature;
        T_bottom = backup_temperature;

        top_boundary_thickness = 1./2. * std::pow(5.e4 /*Ra_c*/ * eta_top * kappa_top / ( rho_top * alpha_top * g_top * (mantle_temp-T_surface) ) , 1./3.); 
      }
      else
      {
        double area_top, area_bottom;
        if(dim==3)
        {
          area_top = 4. * M_PI * R1*R1;
          area_bottom = 4. * M_PI * R0*R0;
        }
        else if(dim==2)
        {
          area_top = 2. * M_PI * R1;
          area_bottom = 2. * M_PI * R0;
        }
        
        double factor = std::pow( (eta_bottom/eta_top)*std::pow(area_top/area_bottom,3.), 0.25);
        mantle_temp = 2./3.*(T_bottom + factor * T_surface)/(1. + factor);
        top_boundary_thickness = 1./20. * std::pow(5.e4 /*Ra_c*/ * eta_top * kappa_top / ( rho_top * alpha_top * g_top * (mantle_temp-T_surface) ) , 1./3.); 
        bottom_boundary_thickness = 1./20. * std::pow(5.e4 /*Ra_c*/ * eta_bottom * kappa_bottom / ( rho_bottom * alpha_bottom * g_top * (T_bottom-mantle_temp) ) , 1./3.); 
      }
      if(numbers::is_finite(bottom_boundary_thickness) == false) bottom_boundary_thickness = (Rm-R0)/5.0;;
      if(numbers::is_finite(top_boundary_thickness) == false) top_boundary_thickness = (R1-Rm)/5.0;;

      double temp;
      //Top boundary
      if(position.norm() >= Rm)
      {
         double z = R1-position.norm();
         double pert = std::sin(phi*order)/10.0;
         z = (z < 0.0 ? 0.0 : z);
         temp = (mantle_temp - T_surface)*boost::math::erf(z*(1.0+pert)/top_boundary_thickness) + T_surface;
         if(temp > mantle_temp) temp=mantle_temp;
      }
      //bottom boundary
      else if(position.norm() < Rm)
      {
         double z = position.norm() - R0;
         double pert = std::cos(phi*order)/10.0;
         z = (z < 0.0 ? 0.0 : z);
         temp = (mantle_temp - T_bottom)*boost::math::erf(z*(1.0+pert)/bottom_boundary_thickness) + T_bottom;
         if(temp < mantle_temp) temp=mantle_temp;
      }
      else temp=mantle_temp;

      for(int i=0; i<blobs.size(); ++i)
        temp -= 0.5*(i%2==0 ? (mantle_temp-T_surface) : (mantle_temp-T_bottom))*std::exp( - Point<dim>(position-R1*blobs[i]).square() / 2.0 / blob_r/blob_r);
      
      if(temp < T_surface) temp = T_surface;
      if(temp > T_bottom) temp = T_bottom;

      return temp;
    }
    template <int dim>
    void
    SphericalBoundaryLayer<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Spherical boundary layer");
        {
          prm.declare_entry ("Perturbation order", "0.0e0",
                             Patterns::Double (0.0),
                             "The order of the perturbation to the boundary layer.");
          prm.declare_entry ("Number of blobs", "0",
                             Patterns::Integer (0),
                             "The number of random blobs.");
          prm.declare_entry ("Blob radius", "0.0e0",
                             Patterns::Double (0.0),
                             "The radius of the random blobs.");
          prm.declare_entry ("Backup temperature", "0.0e0",
                             Patterns::Double (0.0),
                             "Mantle temperature if there is no bottom boundary layer.");
         
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    SphericalBoundaryLayer<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("Spherical boundary layer");
        {
          order = prm.get_double("Perturbation order");
          blob_r = prm.get_double("Blob radius");
          backup_temperature = prm.get_double("Backup temperature");
          n_blobs = prm.get_integer("Number of blobs");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      blobs.clear();
      //Fill the blobs list if we need it.      
      std::srand((unsigned)time(0));
      while(blobs.size() < n_blobs)
      {
        Point<dim> p;
        for(int i=0; i<dim; ++i)
          p[i] = 2.0*(double(std::rand())/RAND_MAX - 0.5);

        if( p.norm() > 1.0)
          continue;
        else blobs.push_back(p);
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialConditions
  {
    ASPECT_REGISTER_INITIAL_CONDITIONS(SphericalBoundaryLayer,
                                       "spherical boundary layer",
                                        "DESCRIPTION")

  }
}
