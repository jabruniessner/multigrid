#include "CG_Solver.h"
#include "Convolution.h"
#include <chrono>

int main()
{
	sycl::gpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Domain<2, 6355u, 6355u> sol(Paddings::PERIODIC, q, 1);
	Domain<2, 6355u, 6355u> init_guess(Paddings::PERIODIC, q, 1);
	Domain<2, 6355u, 6355u> rhs(Paddings::PERIODIC, q, 1);
	Domain<2, 6355u, 6355u> defect_r(Paddings::PERIODIC, q, 1);
	Domain<2, 6355u, 6355u> defect_p(Paddings::PERIODIC, q, 1); 
	 

	constexpr std::array<DataType, 5> values = {4, -1, -1, -1, -1};
	constexpr std::array<OffsetType, 5> offsets ={{{0,0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

	const int m = 16;
	
	for( int i = 0; i<6355+2; i++)
	{
		sol.set_value( - (DataType) i / 6356., i, 0);
		sol.set_value( - (DataType) i / 6356., i, 6356);
		sol.set_value( - 1., 6356, i);	
	}
	

	std::cout<< "#============= The initial set up is ==============#" << std::endl;

//	std::cout<<"The inhomogenous part is given by: "<<std::endl;
//	sol.print_domain();

	Convolve(rhs, sol, values, offsets);
//	std::cout << "The right hand side is given by: "<<std::endl;
//	rhs.print_domain();
	
	auto start = std::chrono::high_resolution_clock::now();
	CG_solver(init_guess, rhs, defect_r, defect_p, values, offsets, 1000);
	auto end = std::chrono::high_resolution_clock::now();
	
	std::chrono::duration<double> duration = end-start;

	std::cout << "The required time was: "<< duration.count()<<" seconds"<<std::endl;

//	subtract_domains(init_guess, init_guess, sol);
//
//
//	std::cout<<"The output Matrix is given by: "<< std::endl; 
//        init_guess.print_domain();
//
//	std::cout<<"The defect is given by: "<<std::endl;
//
//
//	Convolve(defect_r, init_guess, values, offsets);
//
//	std::cout<<"The convolution of the output is given by: "<< std::endl;
//	defect_r.print_domain();
//	subtract_domains(init_guess, sol, rhs);
//
//	init_guess.print_domain();

}
