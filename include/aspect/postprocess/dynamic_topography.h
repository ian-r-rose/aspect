/*
  Copyright (C) 2011-2015 by the authors of the ASPECT code.

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


#ifndef __aspect__postprocess_dynamic_topography_h
#define __aspect__postprocess_dynamic_topography_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {

    /**
     * A postprocessor that computes dynamic topography at the surface.
     *
     * @ingroup Postprocessing
     */
    template <int dim>
    class DynamicTopography : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate the solution for the dynamic topography.
         */
        virtual
        std::pair<std::string,std::string>
        execute (TableHandler &statistics);

        /**
         * Let the postprocessor manager know about the other postprocessors
         * which this one depends on.  Specifically, we will need BoundaryPressures
         * for computing the dynamic pressure at the surface.
         */
        virtual
        std::list<std::string>
        required_other_postprocessors() const;
    };
  }
}


#endif
