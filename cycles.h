#include "level_transition.h"

template<std::size_t Num_Iters, Dimension Dim, DataType, OffsetType, Length... strides_all>
struct Jacobi_Smoother{

	operator(Domain<Dim, strides_all...>& dest,
		 Domain<Dim, strdies_all...>& src,
		 Domain<Dim, strdies_all...>& rhs,
		 std::array<DataType, length>& values,
		 std::array<OffsetType, length>& offsets)
	{	
		for(int i = 0; i< Num_Iters)
		{		
			Convolve(dest, src, values, offsets);
			subtract_domains(dest, rhs, dest);
			
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
}



template<Dimension Dim, std::size_t base_length, std::size_t nlev, std::size_t level=nlev>
struct V_Cycle_base()
{
	Multigrid_domain<Dim, base_length, nlev>& current;
	Multigrid_domain<Dim, base_length, nlev>& next;
	Multigrid_domain<Dim, base_length, nlev>& rhs_domain;
	

	template<std::size_t iter_level=level>
	static void iteration()
	{


	}

	template<>
	static void iteration<0u>()
	{
		std::cout<<"Hello World!"<<std::endl;	
	}
}
