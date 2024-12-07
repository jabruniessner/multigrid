#include "MultigridDomain.h"


int main()
{
	
	sycl::cpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Multigrid_domain<2, 1, 4u> mult_domain(q);

	auto& dom = mult_domain.get_domain<1u>();


	auto a = dom.get_value(0, 0);

	//std::cout << "The value of a is: "<< a << std::endl;

	std::cout<<std::endl;	

	std::cout<<"The value for the length in one direction is: "<< values_1D << std::endl;
}
