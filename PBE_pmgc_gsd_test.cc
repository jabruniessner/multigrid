#include "Convolution.h"
#include "Debye_Hueckel_functions.h"
#include "MultigridDomain.h"
#include "bitshift_lib.h"
#include "create_charge_distribution.h"
#include "cycles.h"
#include "dot_finder.h"
#include "fileio.h"
#include "level_transition.h"
#include "pmgc/gsd.h"
#include "pmgc/precision.h"
#include "scientific_quantities.h"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

using namespace cycles;
using namespace convolution;

constexpr Dimension Dim = 3;
constexpr std::size_t nlev = 1u;
constexpr std::size_t base_length = 1;
constexpr DataType omega = 1.;
constexpr DataType box_length = 16;
constexpr DataType ionic_strength = 0.15;
constexpr DataType kappa = KappaA(ionic_strength);
constexpr DataType kappa_2 = kappa * kappa;
constexpr DataType ionradius = 1.5;
constexpr DataType grid_step =
    box_length / (base_length * utils::power_off(2, nlev));
constexpr DataType delta_epsilon =
    (epsilon_p - epsilon_r); // Difference in epsilon
                             //

using Domain_Type =
    Multigrid_domain<Dim, nlev, base_length, base_length, base_length>;

using Domain_Type_upper = decltype(Domain_Type::domain_t_v)::domain_t;

DataType sqr(double val) { return val * val; }

template <std::size_t level = nlev>
void Set_boundary_conditions(Atom<DataType> *atoms, std::size_t num_atoms,
                             const Domain_Type_upper &domain, std::size_t x,
                             std::size_t y, std::size_t z) {
  DataType buffer_value = 0;
  for (int i = 0; i < num_atoms; i++) {
    const DataType distance =
        std::sqrt(sqr(x * grid_step - atoms[i].Position[0]) +
                  sqr(y * grid_step - atoms[i].Position[1]) +
                  sqr(z * grid_step - atoms[i].Position[2]));

    buffer_value +=
        DH_Sphere(atoms[i].radius, atoms[i].charge, distance, kappa);
  }
  domain(x, y, z) = buffer_value;
}

template <std::size_t level = nlev> void coarsen_domains(Domain_Type domain) {
  if constexpr (level <= 1) {
    return;
  } else {
    using Offset_Type = std::array<DataType, Dim>;
    constexpr std::array<Offset_Type, 1> offsets{};
    constexpr std::array<DataType, 1> values{1};

    auto &dest = domain.template get_domain<level - 1>();
    auto &src = domain.template get_domain<level>();
    level_transition::coarsening(dest, src, values, offsets);
    coarsen_domains<level - 1>(domain);
  }
}

int main(int argc, char *argv[]) {

  using OffsetType = std::array<int, Dim>;

  if (argc < 3) {
    std::cout
        << "Usage: ./this_program in_file out_file x_min y_min z_min num_iters"
        << std::endl;
    return 0;
  }
  // int num_iters = std::stoi(argv[3]);

#ifdef DEBUGMODE
  sycl::cpu_selector selector;
#else
  sycl::gpu_selector selector;
#endif

  sycl::queue q{selector,
                sycl::property_list{sycl::property::queue::in_order{}}};

  std::string filename_in{argv[1]};
  std::string filename_out{argv[2]};
  std::list<Atom<DataType>> atom_list;
  read_pqr_file(filename_in, atom_list);

  const auto num_atoms = atom_list.size();

  std::vector<Atom<DataType>> atoms_vector;
  atoms_vector.reserve(num_atoms);

  auto x_min = std::stod(argv[3]);
  auto y_min = std::stod(argv[4]);
  auto z_min = std::stod(argv[5]);
  int num_iters = std::stoi(argv[6]);

  for (auto &atom : atom_list) {
    atom.Position[0] -= x_min;
    atom.Position[1] -= y_min;
    atom.Position[2] -= z_min;
    // atom.radius += ionradius;
    atoms_vector.push_back(atom);
  }

  Atom<DataType> *atoms_device =
      sycl::malloc_device<Atom<DataType>>(atom_list.size(), q);

  q.memcpy(atoms_device, atoms_vector.data(),
           atoms_vector.size() * sizeof(Atom<DataType>))
      .wait();

  Domain_Type sol(q), sol2(q), lhs_domain1(q), lhs_domain2(q), rhs_domain(q),
      epsilonx_map(q), epsilony_map(q), epsilonz_map(q), epsilonc_map(q),
      epsx_map(q), epsy_map(q), epsz_map(q), kappa_(q), kappa_second_map(q);

  q.fill(epsilonx_map.get_domain().values_buff,
         (DataType)epsilon_r / (grid_step * grid_step),
         epsilonx_map.get_domain().num_values);

  q.fill(epsilony_map.get_domain().values_buff,
         (DataType)epsilon_r / (grid_step * grid_step),
         epsilony_map.get_domain().num_values);

  q.fill(epsilonz_map.get_domain().values_buff,
         (DataType)epsilon_r / (grid_step * grid_step),
         epsilonz_map.get_domain().num_values);

  constexpr auto length = Domain_Type::length;

  {
    auto &boundary_domain = sol.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1], 0);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], I[1],
                                         std::get<2>(length) + 1);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, 0, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain,
                                         std::get<0>(length) + 1, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0], 0, I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain, I[0],
                                         std::get<1>(length) + 1, I[1]);
         })
        .wait();

    auto &boundary_domain2 = sol2.template get_domain<nlev>();
    q.parallel_for(
         sycl::range<2>(std::get<1>(length) + 2, std::get<2>(length) + 2),
         [=](sycl::id<2> I) {
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0], I[1], 0);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0], I[1],
                                         std::get<2>(length) + 1);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, 0, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2,
                                         std::get<0>(length) + 1, I[0], I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0], 0, I[1]);
           Set_boundary_conditions<nlev>(atoms_device, num_atoms,
                                         boundary_domain2, I[0],
                                         std::get<1>(length) + 1, I[1]);
         })
        .wait();

    auto &epsilonx_domain = epsilonx_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonx_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonx2_domain = epsilonx_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonx2_domain,
                          static_cast<DataType>(grid_step));
    });

    // std::cout << "The x-domain is: " << std::endl;
    // epsilonx_domain.print_domain();

    auto &epsilony_domain = epsilony_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[1] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilony_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilony2_domain = epsilonx_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilony2_domain,
                          static_cast<DataType>(grid_step));
    });

    // std::cout << "The y-domain is: " << std::endl;
    // epsilony_domain.print_domain();

    auto &epsilonz_domain = epsilonz_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[2] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonz_domain,
                          static_cast<DataType>(grid_step),
                          (DataType)epsilon_p / (grid_step * grid_step));
    });

    auto &epsilonz2_domain = epsilonx_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      Sphere<DataType, Dim> Atom = atoms_device[I];
      Atom.Position[0] -= 0.5 * grid_step;
      find_dots_in_sphere(Atom, epsilonz2_domain,
                          static_cast<DataType>(grid_step));
    });

    //  std::cout << "The z-domain is: " << std::endl;
    //  epsilonz_domain.print_domain();

    auto &epsilonc_domain = epsilonc_map.template get_domain<nlev>();
    q.parallel_for(
        sycl::range<1>(epsilonc_domain.num_values), [=](sycl::id<1> I) {
          epsilonc_domain.values_buff[I] = 2 * epsilonx_domain.values_buff[I] +
                                           2 * epsilony_domain.values_buff[I] +
                                           2 * epsilonz_domain.values_buff[I];
        });

    q.wait();

    auto &kappa_domain = kappa_.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      auto atom = atoms_device[I];
      atom.radius += 1.5;
      //                      diff_operator.get_offsets(), 1e-2);
      find_dots_in_sphere(atom, kappa_domain, static_cast<DataType>(grid_step));
    });

    auto &kappa_domain2 = kappa_second_map.template get_domain<nlev>();
    q.parallel_for(sycl::range<1>(atoms_vector.size()), [=](sycl::id<1> I) {
      auto atom = atoms_device[I];
      atom.radius += 1.5;
      //                      diff_operator.get_offsets(), 1e-2);
      find_dots_in_sphere(atom, kappa_domain2,
                          static_cast<DataType>(grid_step));
    });

    // Inverting the kappa domain because the original functions marks the
    // points inside the protein with 1. And assigning the right value
    q.parallel_for(sycl::range<1>(kappa_domain.num_values), [=](sycl::id<1> I) {
       kappa_domain.values_buff[I] != 0
           ? kappa_domain.values_buff[I] = 0
           : kappa_domain.values_buff[I] = -kappa_2 * epsilon_r;
     }).wait();

    std::cout << "The kappa domain is: " << std::endl;
    // kappa_domain.print_domain();

    q.parallel_for(sycl::range<1>(kappa_domain2.num_values),
                   [=](sycl::id<1> I) {
                     kappa_domain2.values_buff[I] != 0
                         ? kappa_domain2.values_buff[I] = 0
                         : kappa_domain2.values_buff[I] = 1;
                   })
        .wait();

    auto &rhs = rhs_domain.template get_domain<nlev>();
    std::array<Domain<Dim, std::get<0>(length), std::get<1>(length),
                      std::get<2>(length)>,
               Dim>
        epsilon_domains{epsilonx_domain, epsilony_domain, epsilonz_domain};

    q.submit([=](sycl::handler &h) {
       h.single_task([=]() {
         for (int I = 0; I < num_atoms; I++)
           add_charges_to_distribution(
               rhs, atoms_device[I].Position,
               static_cast<DataType>(atoms_device[I].charge / epsilon),
               spacing<DataType, static_cast<DataType>(grid_step)>{});
       });
     }).wait();

    // auto &defect_p = lhs_domain1.get_domain();
    // auto &defect_r = lhs_domain2.get_domain();

    DataType *DT_null = nullptr;

    int nx = std::get<0>(length) + 2;
    int ny = std::get<1>(length) + 2;
    int nz = std::get<2>(length) + 2;

    std::cout << "nx: " << nx << std::endl;
    std::cout << "ny: " << ny << std::endl;
    std::cout << "nz: " << nz << std::endl;

    int i = 1;

    cycles::Gauss_Seidel_PBE j_smoother{sol};
    //  auto *a = &sol;
    //  auto *b = &lhs_domain1;
    std::cout << "Before the iterations: " << std::endl;
    sol.get_domain().print_domain();

    for (int i = 0; i < num_iters; i++) {

      // cg_solver(init_guess, rhs, kappa_map, epsilon_domains, kappa_2,
      // grid_step,
      //           epsilon_r, delta_epsilon);

      DataType const residual =
          compute_residual_PBE(rhs_domain.template get_domain<nlev>(),
                               sol.template get_domain<nlev>(),
                               lhs_domain2.template get_domain<nlev>(),
                               kappa_second_map.template get_domain<nlev>(),
                               epsx_map.template get_domain<nlev>(),
                               epsy_map.template get_domain<nlev>(),
                               epsz_map.template get_domain<nlev>(), kappa_2,
                               grid_step, epsilon_r, delta_epsilon);

      //   std::cout << "The residual after " << i << " iterations is " <<
      //   residual
      //             << std::endl;

      std::index_sequence<1> iter_nums{};
      j_smoother(Integer<nlev>{}, iter_nums, sol, rhs_domain, kappa_second_map,
                 epsx_map, epsy_map, epsz_map, kappa_2, grid_step, epsilon_r,
                 delta_epsilon, omega);

      // std::swap(a, b);
    }

    std::cout << "After the iterations: " << std::endl;
    sol.template get_domain<nlev>().print_domain();

    std::cout << "Before the iterations: " << std::endl;
    auto &init_guess = sol2.get_domain();
    init_guess.print_domain();

    pmgc::Vgsrb7x(&nx, &ny, &nz, (int *)nullptr, DT_null,
                  epsilonc_map.get_domain().values_buff,
                  kappa_domain.values_buff, rhs.values_buff,
                  epsilonx_domain.values_buff, epsilony_domain.values_buff,
                  epsilonz_domain.values_buff, init_guess.values_buff,
                  &num_iters, q);

    std::cout << "After the iterations: " << std::endl;
    init_guess.print_domain();

    //  cycles::Gauss_Seidel_PBE{sol};

    //  q.wait();

    //  DataType const residual = compute_residual_PBE(
    //      rhs_domain.template get_domain<nlev>(), sol.template
    //      get_domain<nlev>(), lhs_domain2.template get_domain<nlev>(),
    //      kappa_second_map.template get_domain<nlev>(),
    //      epsx_map.template get_domain<nlev>(),
    //      epsy_map.template get_domain<nlev>(),
    //      epsz_map.template get_domain<nlev>(), kappa_2, grid_step, epsilon_r,
    //      delta_epsilon);

    //  q.wait();

    //  std::cout << "The residual after " << num_iters << " is " << residual
    //            << std::endl;

    // sol.get_domain().print_domain();

    // std::ofstream outfile{filename_out};
    //  init_guess.print_dx_to_stream(outfile, x_min, y_min, z_min, 96);
  }

  std::cout << "The length is: " << std::get<0>(length) << std::endl;

  return 0;
}
