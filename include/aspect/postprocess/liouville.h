#ifndef __aspect__postprocess_liouville_h
#define __aspect__postprocess_liouville_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace Postprocess
  {

    template <int dim>
    class Liouville : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        virtual
        std::pair<std::string,std::string>
        execute (TableHandler &statistics);

        static void declare_parameters (ParameterHandler &prm);

        virtual void parse_parameters (ParameterHandler &prm);
      private:
 
        void calculate_convective_moment();
        Tensor<1,dim> solve_eigenvalue_problem();
        void setup();
        void integrate_spin_axis();
        Tensor<1,dim> dOmega_dt( const SymmetricTensor<2,dim> &, const Tensor<1,dim> & );
         
        SymmetricTensor<2,dim> convective_moment;
        Tensor<1,dim> spin_axis;
        double factor;
        double Omega;
        double tau;
        double reference_moment;
        double Fr;
        double eigenvalues[dim];
    };
  }
}


#endif
