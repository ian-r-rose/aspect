#ifndef __aspect__initial_conditions_spherical_boundary_layer_h
#define __aspect__initial_conditions_spherical_boundary_layer_h

#include <aspect/initial_conditions/interface.h>
#include <aspect/simulator.h>


namespace aspect
{
  namespace InitialConditions
  {
    using namespace dealii;

    /**
     * A class that describes a perturbed initial temperature field for
     * the spherical shell.
     *
     * @ingroup InitialConditionsModels
     */
    template <int dim>
    class SphericalBoundaryLayer : public Interface<dim> ,  public ::aspect::SimulatorAccess<dim> 
    {
      public:
        /**
         * Return the initial temperature as a function of position.
         */
        virtual
        double initial_temperature (const Point<dim> &position) const;
  
        static
        void
        declare_parameters (ParameterHandler &prm);

        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
        double order;        
        double backup_temperature;
         
        unsigned int n_blobs;
        std::vector<Point<dim> > blobs;
        double blob_r;


    };

  }
}

#endif
