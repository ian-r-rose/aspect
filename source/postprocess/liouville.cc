#include <aspect/postprocess/liouville.h>
#include <aspect/simulator.h>
#include <aspect/global.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <cmath>

namespace aspect
{
  //Useful utility for doing global MPI reductions on SymmetricTensor objects
  template<int dim>
  SymmetricTensor<2,dim> reduce_tensor(const SymmetricTensor<2,dim> &local, const MPI_Comm &mpi_communicator)
  {
    double entries[6] = {0,0,0,0,0,0};
    double global_entries[6] = {0,0,0,0,0,0};
    SymmetricTensor<2,dim> global;

    for(unsigned int i=0; i<dim; ++i)
      for(unsigned int j=0; j<=i; ++j)
        entries[i*dim + j] = local[i][j];

    MPI_Allreduce (&entries, &global_entries, 6, MPI_DOUBLE,
                       MPI_SUM, mpi_communicator);

    for(unsigned int i=0; i<dim; ++i)
      for(unsigned int j=0; j<=i; ++j)
        global[i][j] = global_entries[i*dim + j];

    return global;
  }

  //Useful utility for doing global MPI reductions on Tensor<1,dim> objects
  template<int dim>
  Tensor<1,dim> reduce_vector(const Tensor<1,dim> &local, const MPI_Comm &mpi_communicator)
  {
    double entries[3] = {0,0,0};
    double global_entries[3] = {0,0,0};
    Tensor<1,dim> global;

    for(unsigned int i=0; i<dim; ++i)
      entries[i] = local[i];

    MPI_Allreduce (&entries, &global_entries, 3, MPI_DOUBLE,
                       MPI_SUM, mpi_communicator);

    for(unsigned int i=0; i<dim; ++i)
      global[i] = global_entries[i];
 
    return global;
  }

  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    Liouville<dim>::execute (TableHandler &statistics)
    {
      Tensor<1,dim> principle_axis;
      calculate_convective_moment();
      principle_axis = solve_eigenvalue_problem();
      if (this->get_timestep_number() == 0)
        setup();
      else
        integrate_spin_axis();

      double angle = std::atan2( spin_axis[1], spin_axis[0])*180.0/M_PI;
      if (angle < 0.0) angle += 180.0;
      
      double angular_mismatch = std::acos(spin_axis*principle_axis)*180.0/M_PI;
      if(angular_mismatch > 90.0) angular_mismatch = std::abs(angular_mismatch-180.0);

      statistics.add_value ("Spin angle", angle);
      statistics.set_precision ("Spin angle", 8);
      statistics.set_scientific ("Spin angle", false);

      statistics.add_value ("Axis mismatch", angular_mismatch);
      statistics.set_precision ("Axis mismatch", 8);
      statistics.set_scientific ("Axis mismatch", false);

      statistics.add_value ("Eigenvalue 1", eigenvalues[0]);
      statistics.set_precision ("Eigenvalue 1", 15);
      statistics.set_scientific ("Eigenvalue 1", true);

      statistics.add_value ("Eigenvalue 2", eigenvalues[1]);
      statistics.set_precision ("Eigenvalue 2", 15);
      statistics.set_scientific ("Eigenvalue 2", true);

      if (dim == 3)
      {
        statistics.add_value ("Eigenvalue 3", eigenvalues[2]);
        statistics.set_precision ("Eigenvalue 3", 15);
        statistics.set_scientific ("Eigenvalue 3", true);
      }

      std::ostringstream screen_text;
      screen_text.precision(8);
      screen_text << angle;

      return std::pair<std::string, std::string> ("Spin angle: ",
                                                  screen_text.str());
    }

    template <int dim>
    void
    Liouville<dim>::setup()
    {
      const Point<dim> surface_point = this->get_geometry_model().representative_point(0.0);
      const double radius = surface_point.norm();
      tau = this->get_material_model().reference_viscosity() / this->get_material_model().reference_density() /
            this->get_gravity_model().gravity_vector(surface_point).norm() / radius;
      reference_moment = trace(convective_moment)/dim;
      Fr = Omega*Omega*radius/this->get_gravity_model().gravity_vector(surface_point).norm();

      spin_axis = solve_eigenvalue_problem();
    }

      
    
    template <int dim>
    void
    Liouville<dim>::calculate_convective_moment()
    {
      QGauss<dim> quadrature(this->get_dof_handler().get_fe().base_element(2).degree+1);
      FEValues<dim> fe(this->get_mapping(), this->get_dof_handler().get_fe(), quadrature,
                              UpdateFlags(update_quadrature_points | update_JxW_values | update_values));

      typename DoFHandler<dim>::active_cell_iterator cell;
      std::vector<Point<dim> > q_points(quadrature.size());
      std::vector<Vector<double> > fe_vals(quadrature.size(), Vector<double>(this->get_dof_handler().get_fe().n_components()));

      //allocate the local and global moments
      SymmetricTensor<2,dim> local_moment;

      //loop over all local cells
      for (cell = this->get_dof_handler().begin_active(); cell != this->get_dof_handler().end(); ++cell)
        if (cell->is_locally_owned())
        {
          fe.reinit (cell);
          q_points = fe.get_quadrature_points();
          fe.get_function_values(this->get_solution(), fe_vals);

          //get the density at each quadrature point
          typename MaterialModel::Interface<dim>::MaterialModelInputs in(q_points.size(), this->n_compositional_fields());
          typename MaterialModel::Interface<dim>::MaterialModelOutputs out(q_points.size(), this->n_compositional_fields());
          for(int i=0; i< q_points.size(); i++)
          {
             in.pressure[i] = fe_vals[i][dim];
             in.temperature[i] = fe_vals[i][dim+1];
             for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
               in.composition[i][c] = fe_vals[i][dim+2+c];
             in.position[i] = q_points[i];

          }
          this->get_material_model().evaluate(in, out);
  
          //actually compute the moment of inertia
          Point<dim> r_vec;
          for (unsigned int k=0; k< quadrature.size(); ++k)
            {
              r_vec = q_points[k];
              if(dim == 2)
              {
                local_moment[0][0] += fe.JxW(k) *(r_vec.square() - r_vec[0]*r_vec[0])*out.densities[k];
                local_moment[1][1] += fe.JxW(k) *(r_vec.square() - r_vec[1]*r_vec[1])*out.densities[k];
                local_moment[0][1] += -fe.JxW(k) *( r_vec[1]*r_vec[0])*out.densities[k];
              }
              else if(dim==3)
              {
                local_moment[0][0] += fe.JxW(k) *(r_vec.square() - r_vec[0]*r_vec[0])*out.densities[k];
                local_moment[1][1] += fe.JxW(k) *(r_vec.square() - r_vec[1]*r_vec[1])*out.densities[k];
                local_moment[2][2] += fe.JxW(k) *(r_vec.square() - r_vec[2]*r_vec[2])*out.densities[k];
                local_moment[0][1] += -fe.JxW(k) *( r_vec[0]*r_vec[1])*out.densities[k];
                local_moment[0][2] += -fe.JxW(k) *( r_vec[0]*r_vec[2])*out.densities[k];
                local_moment[1][2] += -fe.JxW(k) *( r_vec[1]*r_vec[2])*out.densities[k];
       
              }
            }
        }
      convective_moment = reduce_tensor(local_moment, this->get_mpi_communicator());
    }
     
    template<>
    Tensor<1,3> Liouville<3>::solve_eigenvalue_problem()
    {
      double m = trace(convective_moment)/3.0;
      SymmetricTensor<2,3> tmp = (convective_moment - m*unit_symmetric_tensor<3>());
      double q = determinant(tmp)/2.0;
      if (q == 0.0) //degenerate, just return z hat
      {
        Tensor<1,3> ax;
        ax[2] = 1.0;
        return ax;
      }

      double p = tmp.norm()*tmp.norm()/6.0;
      double arg = (p*p*p - q*q);  if(arg < 0.0) arg = 0.0;
      double phi = std::atan(std::sqrt(arg)/q)/3.0;
      if( phi < 0.0) phi = 0.0;
      if( phi > M_PI) phi = M_PI;

      eigenvalues[0] = m + 2.0* std::sqrt(p)*std::cos(phi);
      eigenvalues[1] = m - std::sqrt(p)*(std::cos(phi) + std::sqrt(3.0)*std::sin(phi));
      eigenvalues[2] = m - std::sqrt(p)*(std::cos(phi) - std::sqrt(3.0)*std::sin(phi));
      
      Tensor<2,3> tmp2 = Tensor<2,3>(convective_moment - eigenvalues[1]*unit_symmetric_tensor<3>()) *
                         Tensor<2,3>(convective_moment - eigenvalues[2]*unit_symmetric_tensor<3>());
      for(int i = 0; i <3; ++i)
      {
        Tensor<1,3> test, product; test = 0.0;  test[i] = 1.0;
        product = tmp2*test;
        if ( product.norm() > 1e-10)
        {
          return product/product.norm();
        }
      }
    }


    template<>
    Tensor<1,2> Liouville<2>::solve_eigenvalue_problem()
    {
      double tr = trace(convective_moment);
      double det = determinant(convective_moment);
      double disc = tr*tr - 4.0*det;
      if (disc < 0.0) disc = 0.0;   

      //compute the eigenvalues
      eigenvalues[0] = (tr + std::sqrt(disc))/2.0;
      eigenvalues[1] = (tr - std::sqrt(disc))/2.0;

      //annihilate the eigenvector for the second eigenvalue
      SymmetricTensor<2,2> tmp = convective_moment - eigenvalues[1]*unit_symmetric_tensor<2>();
      for(int i = 0; i <2; ++i)
      {
        Tensor<1,2> test, product; test = 0.0;  test[i] = 1.0;
        product = tmp*test;
        if ( product.norm() > 1e-10)
        {
          return product/product.norm();
        }
      }
    }
    
    template <int dim>
    void
    Liouville<dim>::integrate_spin_axis()
    {
      double dt = this->get_timestep();
      Tensor<1,dim> k1, k2, k3, k4;
     
      k1 = dOmega_dt(convective_moment, spin_axis);
      k2 = dOmega_dt(convective_moment, spin_axis + dt/2.0 * k1);
      k3 = dOmega_dt(convective_moment, spin_axis + dt/2.0 * k2);
      k4 = dOmega_dt(convective_moment, spin_axis + dt * k3);

      spin_axis = spin_axis + dt / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
      spin_axis = spin_axis/spin_axis.norm();
    }
 
    template <int dim>
    Tensor<1,dim>
    Liouville<dim>::dOmega_dt( const SymmetricTensor<2,dim> &moment, const Tensor<1,dim> &axis)
    {
      Tensor<1,dim> Omega_dot;
      const double dashpot = factor / Fr / reference_moment / tau;
      Omega_dot = dashpot * ( moment * axis - (axis * (moment*axis) )*axis);
      return Omega_dot;
    }

    template <int dim>
    void
    Liouville<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Spin axis");
        {
          prm.declare_entry ("Omega", "7.29e-5",
                             Patterns::Double (0), "");
          prm.declare_entry ("Factor", "1.0",
                             Patterns::Double (0), "");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

    template <int dim>
    void
    Liouville<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Spin axis");
        {
          Omega = prm.get_double("Omega");
          factor = prm.get_double("Factor");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(Liouville,
                                  "liouville",
                                  "")
  }
}
