#include <deal.II/base/parameter_handler.h>
#include <aspect/global.h>
#include <aspect/simulator_signals.h>

namespace aspect
{
  using namespace dealii;

  // Global variables (to be set by parameters)
  double unmooring_time;
  std::set< types::boundary_id > free_surface_boundary_indicators;

  /**
   * Declare additional parameters.
   */
  void declare_parameters(const unsigned int,
                          ParameterHandler &prm)
  {
    prm.enter_subsection ("Free surface");
    {
      prm.declare_entry("Unmooring time", "0.",
                        Patterns::Double(0.),
                        "");
    }
    prm.leave_subsection ();
  }

  template <int dim>
  void parse_parameters(const Parameters<dim>,
                        ParameterHandler &prm)
  {
    prm.enter_subsection ("Free surface");
    {
      unmooring_time = prm.get_double("Unmooring time");
      AssertThrow( unmooring_time >= 0.0, ExcMessage("Unmooring time must be greater than zero") );
    }
    prm.leave_subsection ();

  }

  template <int dim>
  void unmoor_free_surface (const SimulatorAccess<dim> &simulator_access,
                            Parameters<dim> &parameters)
  {
    double time = simulator_access.get_time()/(simulator_access.convert_output_to_years() ? year_in_seconds : 1.0);
    if( simulator_access.get_timestep_number() == 0 && unmooring_time != 0.0)
      {
        simulator_access.get_pcout()<<"Mooring surface!"<<std::endl;
        //store the list of free surface indicators
        free_surface_boundary_indicators = parameters.free_surface_boundary_indicators;

        //make the free surface indicators free slip for now
        parameters.tangential_velocity_boundary_indicators.insert(free_surface_boundary_indicators.begin(),
                                                                  free_surface_boundary_indicators.end() );
        parameters.free_surface_boundary_indicators.clear();
//        for ( auto c : parameters.tangential_velocity_boundary_indicators)
 //         simulator_access.get_pcout()<<int(c)<<std::endl;
      }
    else if ( time > unmooring_time && parameters.free_surface_boundary_indicators.empty() )
      {
        simulator_access.get_pcout()<<"Unmooring surface!"<<std::endl;
        parameters.free_surface_boundary_indicators = free_surface_boundary_indicators;

        for ( std::set<types::boundary_id>::iterator it = free_surface_boundary_indicators.begin(); 
              it != free_surface_boundary_indicators.end(); ++it)
          parameters.tangential_velocity_boundary_indicators.erase(*it);
      }
    else return;
  }

  // Connect declare_parameters and parse_parameters to appropriate signals.
  void parameter_connector ()
  {
    SimulatorSignals<2>::declare_additional_parameters.connect (&declare_parameters);
    SimulatorSignals<3>::declare_additional_parameters.connect (&declare_parameters);

    SimulatorSignals<2>::parse_additional_parameters.connect (&parse_parameters<2>);
    SimulatorSignals<3>::parse_additional_parameters.connect (&parse_parameters<3>);
  }

  template <int dim>
  void signal_connector (SimulatorSignals<dim> &signals)
  {
    signals.edit_parameters_pre_setup_dofs.connect (&unmoor_free_surface<dim>);
  }

  // Tell Aspect to send signals to the connector functions
  ASPECT_REGISTER_SIGNALS_PARAMETER_CONNECTOR(parameter_connector)
  ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>, signal_connector<3>)
}
