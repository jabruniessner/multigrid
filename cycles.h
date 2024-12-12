#include "level_transition.h"
#include <array>



template<std::size_t Num_Iters, Dimension Dim, typename DataType, typename OffsetType, std::size_t length, 
	Length... strides_all>
struct Jacobi_Smoother
{

	Jacobi_Smoother(Integer<Num_Iters> integer,
			Domain<Dim, strides_all...> domain, 
			std::array<DataType, length> values, 
			std::array<OffsetType, length> offsets){}

	void operator()(Domain<Dim, strides_all...>& dest,
		 Domain<Dim, strides_all...>& src,
		 Domain<Dim, strides_all...>& rhs,
		 std::array<DataType, length>& values,
		 std::array<OffsetType, length>& offsets)
	{	
		for(int i = 0; i< Num_Iters; i++)
		{		
			Convolve(dest, src, values, offsets);
			subtract_domains(dest, rhs, dest);
			
			DataType* temp = dest.values_buff;
		        dest.values_buff = src.values_buff;
			src.values_buff = temp;
		}

	
		DataType* temp = dest.values_buff;
		dest.values_buff = src.values_buff;
		src.values_buff = temp;
	}


};



template<typename Pre_Smoother, typename Post_Smoother, 
	Dimension Dim, std::size_t base_length, 
	std::size_t length, typename DataType,
	std::size_t nlev, std::size_t level=nlev>
struct V_Cycle_base
{

	V_Cycle_base(Pre_Smoother& presmoother, 
		     Post_Smoother& post_smoother,
		     Multigrid_domain<Dim, base_length, nlev>&,
		     Multi_Level_operator<Dim, DataType, length, nlev>&
		     ){}

	

//	template<std::size_t iter_level=level>
//	static void iteration(
//			Multigrid_domain<Dim, base_length, nlev>& current,
//			Multigrid_domain<Dim, base_length, nlev>& next,
//			Multigrid_domain<Dim, base_length, nlev>& rhs_domain,
//			Multi_Level_operator<Dim, DataType, length, nlev>& Diff_operator
//			)
//	{
//		pre_smoother(next.template get_domain<iter_level>(), 
//			     current.template get_domain<iter_level>(), 
//			     rhs_domain.template get_domain<iter_level>(), 
//			     Diff_operator.template get_values<iter_level>(), 
//			     Diff_operator.template get_offsets<iter_level>());
//		
//		iteration<iter_level-1>();
//		
//		post_smoother(next.template get_domain<iter_level>(), 
//			      current.template get_domain<iter_level>(), 
//			      rhs_domain.template get_domain<iter_level>(), 
//			      Diff_operator.template get_values<iter_level>(), 
//			      Diff_operator.template get_offsets<iter_level>());
//	}
//
//	template<>
//	static void iteration<0u>(){}

	static Pre_Smoother pre_smoother;
	static Post_Smoother post_smoother;

};
