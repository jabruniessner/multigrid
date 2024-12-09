#include "MultigridDomain.h"


//This function will be called inside the sycl::queue. It sets the values at the boundary correctly on the Domain which is passed in by reference
template<Dimension Dim, std::size_t size ,std::size_t... strides_all, std::size_t... previous, std::size_t current, std::size_t... following>
void coarsen_boundary_condition(std::array<DataType, Dim-1>& values, 
                            	std::array<OffsetType, Dim-1>& offsets,
				sycl::id<Dim-1> I
                            	Domain<Dim, ((strides_all+1)/2-1)...>& dest, 
                            	Domain<Dim, strides_all...>& src,
                            	std::index_sequence<previous...>,  
                            	std::sequence<following...>)
{       
	DataType value = 0;
	auto index_tuple = std::forward_as_tuple(std::forward(I[previous])...,
						 current,
						 std::forward(I[following])...);

	//for(std::size_t i = 0; i<
}



int main()
{
	sycl::gpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Multigrid_domain<2, 1, 4u> lhs_domain(q);
	Multigrid_domain<2, 1, 4u> rhs_domain(q);

	std::array<OffsetType, 4> offsets{{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}};
	std::array<DataType, 4>{1,1,1,1};
}
