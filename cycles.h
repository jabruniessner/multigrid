#include "level_transition.h"
#include <array>

template<std::size_t Num_Iters, Dimension Dim, DataType, OffsetType, std::size_t length, 
	Length... strides_all>
struct Jacobi_Smoother{

	void operator()(Domain<Dim, strides_all...>& dest,
		 Domain<Dim, strides_all...>& src,
		 Domain<Dim, strides_all...>& rhs,
		 std::array<DataType, length>& values,
		 std::array<OffsetType, length>& offsets)
	{	
		for(int i = 0; i< Num_Iters; i++)
		{		
			Convolve(dest, src, values, offsets);
			subtract_domain(dest, rhs, dest);
			
			auto& temp = dest;
		        dest = src;
			src = temp;
		}

		if constexpr (Num_Iters%2 != 0)
		{
			auto& temp = dest;
			dest = src;
			src = temp;
		}
	}
};



template<typename Pre_Smoother, typename Post_Smoother, Dimension Dim, std::size_t base_length, std::size_t length,std::size_t nlev, std::size_t level=nlev>
struct V_Cycle_base
{

	static Pre_Smoother pre_smoother;
	static Post_Smoother post_smoother;
	

	template<std::size_t iter_level=level>
	static void iteration(
			Multigrid_domain<Dim, base_length, nlev>& current,
			Multigrid_domain<Dim, base_length, nlev>& next,
			Multigrid_domain<Dim, base_length, nlev>& rhs_domain,
			Multi_Level_operator<Dim, DataType, length, nlev>& Diff_operator
			)
	{
		pre_smoother(next.template get_domain<iter_level>(), 
			     current.template get_domain<iter_level>(), 
			     rhs_domain.template get_domain<iter_level>(), 
			     Diff_operator.template get_values<iter_level>(), 
			     Diff_operator.template get_offsets<iter_level>());
		
		iteration<iter_level-1>();
		
		post_smoother(next.template get_domain<iter_level>(), 
			      current.template get_domain<iter_level>(), 
			      rhs_domain.template get_domain<iter_level>(), 
			      Diff_operator.template get_values<iter_level>(), 
			      Diff_operator.template get_offsets<iter_level>());
	}

	template<>
	static void iteration<0u>(){}
};
