#include "CG_Solver.h"
#include "Convolution.h"
#include <chrono>

int main() {

  using namespace cg_solver;
  using namespace convolution;

  using OffsetType = std::array<int, 3>;
  sycl::gpu_selector selector;
  sycl::queue q(selector,
                sycl::property_list{sycl::property::queue::in_order{}});

  Domain<3, 353, 353, 353> sol(Paddings::PERIODIC, q, 1);
  Domain<3, 353, 353, 353> init_guess(Paddings::PERIODIC, q, 1);
  Domain<3, 353, 353, 353> rhs(Paddings::PERIODIC, q, 1);
  Domain<3, 353, 353, 353> defect_r(Paddings::PERIODIC, q, 1);
  Domain<3, 353, 353, 353> defect_p(Paddings::PERIODIC, q, 1);

  constexpr std::array<DataType, 7> values = {6, -1, -1, -1, -1, -1, -1};
  constexpr std::array<OffsetType, 7> offsets{{{0, 0, 0},
                                               {1, 0, 0},
                                               {-1, 0, 0},
                                               {0, 1, 0},
                                               {0, -1, 0},
                                               {0, 0, 1},
                                               {0, 0, -1}}};

  q.parallel_for(sycl::range<2>(353 + 2, 353 + 2), [=](sycl::id<2> I) {
     sol(0, I[0], I[1]) = -(DataType)I[1] / (353 + 1.);
     sol(353 + 1, I[0], I[1]) = -(DataType)I[1] / (353 + 1.);
     sol(I[0], 0, I[1]) = -(DataType)I[1] / (353 + 1.);
     sol(I[0], 353 + 1, I[1]) = -(DataType)I[1] / (353 + 1.);
     sol(I[0], I[1], 353 + 1) = -1.;
   }).wait();

  //	for(int k = 0; k<5; k++){
  //		for(int j = 0; j<5; j++){
  //			for(int i = 0; i<5; i++)
  //			{
  //				std::cout<<sol.get_value(i, j, k)<<" ";
  //			}
  //			std::cout<<std::endl;
  //		}
  //		std::cout<<std::endl;
  //	}

  Convolve(rhs, sol, values, offsets);

  auto start = std::chrono::high_resolution_clock::now();
  CG_solver(init_guess, rhs, defect_r, defect_p, values, offsets,
            static_cast<DataType>(1e-2));
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "The required time was: " << duration.count() << " seconds"
            << std::endl;

  subtract_domains(init_guess, init_guess, sol);

  std::cout << "The output Matrix is given by: " << std::endl;

  //	for(int k = 0; k<5; k++){
  //		for(int j = 0; j<5; j++){
  //			for(int i = 0; i<5; i++)
  //			{
  //				std::cout<<init_guess.get_value(i, j, k)<<" ";
  //			}
  //			std::cout<<std::endl;
  //		}
  //		std::cout<<std::endl;
  //	}

  //       std::cout<<"The defect is given by: "<<std::endl;
}
