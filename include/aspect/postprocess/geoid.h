/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

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


#ifndef __aspect__postprocess_geoid_h
#define __aspect__postprocess_geoid_h

#include <aspect/simulator.h>
#include <aspect/simulator_access.h>
#include <aspect/postprocess/interface.h>
#include <aspect/postprocess/boundary_pressures.h>
#include <aspect/postprocess/boundary_densities.h>

namespace aspect
{
  namespace Postprocess
  {
    using namespace dealii;

    namespace internal
    {
      /**
       * A struct that contains a representation of the coefficients of a
       * harmonic expansion.
       */
      template <int dim>
      struct HarmonicCoefficients
      {
        HarmonicCoefficients(const unsigned int max_degree);

        std::vector<double> sine_coefficients;
        std::vector<double> cosine_coefficients;
      };

      /**
       * A class to expand arbitrary fields of doubles into multipole
       * moments.  In 3D this is spherical multipole moments, and in
       * 2D it is cylindrical multipole moments.
       */
      template <int dim>
      class MultipoleExpansion
      {
        public:
          MultipoleExpansion(const unsigned int max_degree);

          /**
           *Do the multipole expansion at a particular quadrature point.
           *
           *@param position The location of the quadrature point.
           *
           *@param value  The value of the function being expanded.
           *
           *@param JxW The Jacobian and weight given to the quadrature point
           *
           *@param evaluation_radius The radius at which we expand the multipole.
           */
          void add_quadrature_point (const Point<dim> &position, const double value,
                                     const double JxW, const double evaluation_radius);

          /**
           * Return a reference to the internal representation of the mulipole expansion.
           */
          const HarmonicCoefficients<dim> &get_coefficients () const;

          /*
           * Set all the multipole coefficients to zero.
           */
          void clear();

          /*
           * Scalar add another multipole expansion to this one.
           * Computes this = s*this + a*M for each coefficient.
           */
          void sadd( double s, double a, const MultipoleExpansion &M);

          /*
           * Scalar add another multipole expansion to this one.
           * Each degree can have a different scalar factor, so
           * the size of s and a are expected to be max_degree+1.
           * Computes this = s[l]*this + a[l]*M for each coefficient.
           */
          void sadd( const std::vector<double> &s, const std::vector<double> &a, const MultipoleExpansion &M);

          void

          /**
           * Perform an MPI sum on the coefficents.
           */
          mpi_sum_coefficients (MPI_Comm mpi_communicator);

        private:
          const unsigned int max_degree;  //expansion degree
          HarmonicCoefficients<dim> coefficients; //sine and cosine coefficients

      };
    }

    /**
     * A postprocessor that computes the geoid height at the surface.
     *
     * @ingroup Postprocessing
     */
    template <int dim>
    class Geoid : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate the solution for the geoid height at the surface.
         */
        virtual
        std::pair<std::string,std::string>
        execute (TableHandler &statistics);


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

        /**
         * Let the postprocessor manager know about the other postprocessors
         * which this one depends on.  Specifically, BoundaryPressures and
         * BoundaryDensities.
         */
        virtual
        std::list<std::string>
        required_other_postprocessors() const;

      private:

        /**
         * Compute the multipole expansion of the internal density structure.
         */
        void compute_internal_density_expansions();

        /**
         * Compute the harmonic expansion of the dynamic topography on the
         * top and bottom boundaries.
         */
        void compute_topography_expansions();

        /**
         * Compute the geoid at the top and bottom of the domain.  This has
         * contributions from internal density structure, the bottom topography,
         * and the top topography.
         */
        void compute_geoid_expansions();

        /**
         * Write the geoid and associated information to an output file.
         */
        void output_geoid_information();

        /**
         * A parameter that we read from the input file that denotes whether
         * we should include the contribution by dynamic topography.
         */
        bool include_topography_contribution;

        /**
         * The density below the bottom boundary (typically the density of
         * liquid iron).
         */
        double density_below;

        /**
         * The density above the top boundary (typically the density of
         * air or water).
         */
        double density_above;

        /**
         * The maximum harmonic degree for the expansion.
         */
        unsigned int max_degree;

        /**
         * A pointer to the postprocessor for computing boundary
         * pressures.
         */
        BoundaryPressures<dim> *boundary_pressure_postprocessor;

        /**
         * A pointer to the postprocessor for computing boundary
         * densities.
         */
        BoundaryDensities<dim> *boundary_density_postprocessor;

        /**
         * The multipole expansion of internal density anomalies, evaluated at the bottom.
         */
        std_cxx11::shared_ptr< internal::MultipoleExpansion<dim> > internal_density_expansion_bottom;

        /**
         * The multipole expansion of internal density anomalies, evaluated at the surface.
         */
        std_cxx11::shared_ptr< internal::MultipoleExpansion<dim> > internal_density_expansion_surface;

        /**
         * The harmonic expansion of surface topography.
         */
        std_cxx11::shared_ptr< internal::MultipoleExpansion<dim> > surface_topography_expansion;

        /**
         * The harmonic expansion of bottom topography.
         */
        std_cxx11::shared_ptr< internal::MultipoleExpansion<dim> > bottom_topography_expansion;

        /**
         * The harmonic expansion of the gravitational potential at the bottom.
         */
        std_cxx11::shared_ptr< internal::MultipoleExpansion<dim> > bottom_potential_expansion;

        /**
         * The harmonic expansion of the gravitational potential at the surface.
         */
        std_cxx11::shared_ptr< internal::MultipoleExpansion<dim> > surface_potential_expansion;
    };
  }
}


#endif
