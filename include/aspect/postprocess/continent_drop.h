/*
  Copyright (C) 2014 by the authors of the ASPECT code.

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


#ifndef __aspect__postprocess_continent_drop_h
#define __aspect__postprocess_continent_drop_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace Postprocess
  {

    template <int dim>
    class ContinentDrop : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        virtual
        std::pair<std::string,std::string>
        execute (TableHandler &statistics);

        static void declare_parameters (ParameterHandler &prm);

        virtual void parse_parameters (ParameterHandler &prm);

      private:
        bool in_continent( Point<dim> );

        unsigned int n_continents;
        double drop_time;
        double angular_size;
        double depth;
        bool dropped;
        std::vector< Point<dim> > centers;
    };
  }
}


#endif
