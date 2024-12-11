#include "MultigridDomain.h"
#include "Convolution.h"
#include "level_transition.h"
#include "cycles.h"


template<Dimension Dim, std::size_t base_length, std::size_t nlev, std::size_t level=nlev>
void Initializing_all_rhs(Multigrid_domain<Dim, base_length, nlev>& MultDomain)
{
	if constexpr (level == 1)
	{
		return;
	}
	else
	{

		constexpr std::array<OffsetType, 1> offsets_coarsening{{{0, 0}}};
		constexpr std::array<DataType, 1> values_coarsening{1};

		coarsening(MultDomain.template get_domain<level-1>(), 
			   MultDomain.template get_domain<level>(),
		   	   values_coarsening,
		           offsets_coarsening);
	        Initializing_all_rhs<Dim, base_length, nlev, level-1>(MultDomain);
	}
}





int main()
{
	sycl::gpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Multigrid_domain<2, 1, 4u> lhs_domain1(q);
	Multigrid_domain<2, 1, 4u> lhs_domain2(q);
	Multigrid_domain<2, 1, 4u> rhs_domain(q);
	Multigrid_domain<2, 1, 4u> boundary_values(q);

	constexpr auto length = Multigrid_domain<2, 1, 4u>::length;


	for(int i = 0; i < length+2; i++)
	{
		boundary_values.set_value(0., 0, i);
		boundary_values.set_value(1., length+1, i);
		boundary_values.set_value((double) i / (double) (length+1), i, 0);
		boundary_values.set_value((double) i / (double) (length+1), i, length+1);
	}	
	

	//boundary_values.domain.print_domain();
	

	
	std::array<OffsetType, 5> offsets_op{{{-1, 0}, {1, 0}, {0, 0}, {0, -1}, {0, 1}}};
	std::array<DataType, 5> values_op{-1, -1, 4, -1, -1};


	Convolve(rhs_domain.domain, boundary_values.domain, values_op, offsets_op);

	print_multigrid_domain(rhs_domain);


	
	
	std::array<OffsetType, 4> offsets{{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}};
	std::array<DataType, 4> values{-1, -1, -1, -1};
}
