/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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


#ifndef _aspect_postprocess_visualization_seismic_anomalies_h
#define _aspect_postprocess_visualization_seismic_anomalies_h

#include <aspect/postprocess/visualization.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      /**
       * A class derived that implements a function that provides the computed
       * computed seismic anomaly in $V_s$ for graphical output.
       */
      template <int dim>
      class SeismicVsAnomaly
        :
        public CellDataVectorCreator<dim>,
        public SimulatorAccess<dim>
      {
        public:
          /**
           * The function classes have to implement that want to output
           * cellwise data.
           * @return A pair of values with the following meaning: - The first
           * element provides the name by which this data should be written to
           * the output file. - The second element is a pointer to a vector
           * with one element per active cell on the current processor.
           * Elements corresponding to active cells that are either artificial
           * or ghost cells (in deal.II language, see the deal.II glossary)
           * will be ignored but must nevertheless exist in the returned
           * vector. While implementations of this function must create this
           * vector, ownership is taken over by the caller of this function
           * and the caller will take care of destroying the vector pointed
           * to.
           */
          virtual
          std::pair<std::string, Vector<float> *>
          execute () const;

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

          /**
           * Scheme chosen to define the average seismic velocity as
          * a function of depth. Reference profile evaluates the
          * material model using the P-T profile defined by the reference
          * adiabatic conditions and the lateral average option calculates
          * the average velocity within a number n_slices of depth slices.
           */
          enum VelocityScheme
          {
            reference_profile,
            lateral_average
          } average_velocity_scheme;

          /**
           * Number of depth slices used to define average
           * seismic shear wave velocities from which anomalies
           * are calculated.
           */
          unsigned int n_slices;
      };


      /**
       * A class derived that implements a function that provides the computed
       * computed seismic anomaly in $V_p$ for graphical output.
       */
      template <int dim>
      class SeismicVpAnomaly
        :
        public CellDataVectorCreator<dim>,
        public SimulatorAccess<dim>
      {
        public:
          /**
           * The function classes have to implement that want to output
           * cellwise data.
           * @return A pair of values with the following meaning: - The first
           * element provides the name by which this data should be written to
           * the output file. - The second element is a pointer to a vector
           * with one element per active cell on the current processor.
           * Elements corresponding to active cells that are either artificial
           * or ghost cells (in deal.II language, see the deal.II glossary)
           * will be ignored but must nevertheless exist in the returned
           * vector. While implementations of this function must create this
           * vector, ownership is taken over by the caller of this function
           * and the caller will take care of destroying the vector pointed
           * to.
           */
          virtual
          std::pair<std::string, Vector<float> *>
          execute () const;

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

          /**
           * Scheme chosen to define the average seismic velocity as
          * a function of depth. Reference profile evaluates the
          * material model using the P-T profile defined by the reference
          * adiabatic conditions and the lateral average option calculates
          * the average velocity within a number n_slices of depth slices.
           */
          enum VelocityScheme
          {
            reference_profile,
            lateral_average
          } average_velocity_scheme;

          /**
           * Number of depth slices used to define average
           * seismic compressional wave velocities from which anomalies
           * are calculated.
           */
          unsigned int n_slices;
      };
    }
  }
}

#endif
