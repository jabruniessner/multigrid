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

	constexpr std::size_t nlev=2u;

	Multigrid_domain<2, 1, nlev> lhs_domain1(q);
	Multigrid_domain<2, 1, nlev> lhs_domain2(q);
	Multigrid_domain<2, 1, nlev> rhs_domain(q);
	Multigrid_domain<2, 1, nlev> boundary_values(q);

	constexpr auto length = Multigrid_domain<2, 1, nlev>::length;


	for(int i = 0; i < length+2; i++)
	{
		boundary_values.set_value(0., 0, i);
		boundary_values.set_value(1., length+1, i);
		boundary_values.set_value((double) i / (double) (length+1), i, 0);
		boundary_values.set_value((double) i / (double) (length+1), i, length+1);
	}	
	
	
	
	std::array<OffsetType, 5> offsets_op{{{-1, 0}, { 1, 0}, {0, 0}, {0, -1}, {0, 1}}};
	std::array<DataType, 5> values_op{-1./4., -1./4., 1, -1./4., -1./4.}; //Dividing the original operator by the Diagonal 
							      		      //as it is only applied to the right hand side anyways


	Convolve(rhs_domain.domain, boundary_values.domain, values_op, offsets_op);

	
	std::array<OffsetType, 4u> offsets{{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}};
	std::array<float, 4u> values{-1./4., -1./4., -1./4., -1./4.};

	

	Jacobi_Smoother j_smoother(
			Integer<3>{},
			rhs_domain.domain, 
			values, 
			offsets);

//	j_smoother(lhs_domain1.domain,
//		   lhs_domain2.domain,
//		   rhs_domain.domain,
//		   values, 
//		   offsets);

	Multi_Level_operator mult_level(Integer<nlev>{},
					values,
					offsets);
	mult_level.print_operator();

	V_Cycle_base v_cycle(
			j_smoother,
			j_smoother,
			lhs_domain1, 
			mult_level);

	v_cycle.iteration(lhs_domain1,
			  lhs_domain2,
			  rhs_domain,
			  mult_level);

	std::cout<< "lhs_domain_1:"<< std::endl;
	print_multigrid_domain(lhs_domain1);

	std::cout<< "lhs_domain_2:"<< std::endl;
	print_multigrid_domain(lhs_domain2);
}
