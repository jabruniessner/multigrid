#include <utility>
#include <array>
#include "level_transition.h"
#include "Convolution.h"


#ifndef CYCLES_H
#define CYCLES_H

namespace cycles{

using namespace multigrid_domain;


//template<std::size_t Num_Iters, Dimension Dim, 
//	typename DataType, typename OffsetType, 
//	std::size_t length, Length... strides_all>
//struct Jacobi_Smoother
//{
//
//	Jacobi_Smoother(){};
//
//	Jacobi_Smoother(Integer<Num_Iters> integer,
//			Domain<Dim, strides_all...> domain, 
//			std::array<DataType, length> values, 
//			std::array<OffsetType, length> offsets){}
//
//	void operator()(Domain<Dim, strides_all...>& dest,
//		 Domain<Dim, strides_all...>& src,
//		 Domain<Dim, strides_all...>& rhs,
//		 std::array<DataType, length>& values,
//		 std::array<OffsetType, length>& offsets)
//	{	
//		for(int i = 0; i< Num_Iters; i++)
//		{		
//			convolution::Convolve(dest, src, values, offsets);
//			subtract_domains(dest, rhs, dest);
//			
//			DataType* temp = dest.values_buff;
//		        dest.values_buff = src.values_buff;
//			src.values_buff = temp;
//		}
//
//	
//		DataType* temp = dest.values_buff;
//		dest.values_buff = src.values_buff;
//		src.values_buff = temp;
//	}
//
//
//};


template<std::size_t Num_Iters, Dimension Dim, 
	typename DataType, typename OffsetType, 
	std::size_t length, std::size_t base_length, std::size_t nlev>
struct Jacobi_Smoother
{
	Jacobi_Smoother(){};
	
	Jacobi_Smoother(Integer<Num_Iters> integer,
			Multigrid_domain<Dim, base_length, nlev>,
			std::array<DataType, length> values,
			std::array<OffsetType, length> offsets){};

	template<std::size_t level = nlev>
	void operator()(  Integer<level>,
			  Multigrid_domain<Dim, base_length, nlev>& dest,
		    	  Multigrid_domain<Dim, base_length, nlev>& src,
		     	  Multigrid_domain<Dim, base_length, nlev>& rhs,
		     	  std::array<DataType, length>& values,
		     	  std::array<OffsetType, length>& offsets)

	{
		
		auto& dest_domain = dest.template get_domain<level>();
		auto&  src_domain = src.template get_domain<level>();	
		auto&  rhs_domain = rhs.template get_domain<level>();
		
		for(int i = 0; i< Num_Iters; i++)
		{		
			convolution::Convolve(dest_domain, src_domain, values, offsets);
			subtract_domains(dest_domain, rhs_domain, dest_domain);

			std::swap(dest_domain.values_buff, 
					src_domain.values_buff);

		}

		
		std::swap(dest_domain.values_buff, 
				src_domain.values_buff);
	}

	void operator()(  Multigrid_domain<Dim, base_length, nlev>& dest,
		    	  Multigrid_domain<Dim, base_length, nlev>& src,
		     	  Multigrid_domain<Dim, base_length, nlev>& rhs,
		     	  std::array<DataType, length>& values,
		     	  std::array<OffsetType, length>& offsets)

	{
		this->operator()(Integer<nlev>{},dest, src, rhs, values, offsets);	
	}

	
};



template<typename Pre_Smoother, typename Post_Smoother, 
	Dimension Dim, std::size_t base_length, 
	std::size_t length, std::size_t length_diff_op, 
	std::size_t length_coarsening_op, typename DataType,
	std::size_t nlev, std::size_t level=nlev>
struct V_Cycle_base
{
	V_Cycle_base(Pre_Smoother& presmoother, 
		     Post_Smoother& post_smoother,
		     Multigrid_domain<Dim, base_length, nlev>&,
		     Multi_Level_operator<Dim, DataType, length, nlev>&,
		     Multi_Level_operator<Dim, DataType, length_diff_op, nlev>&,
		     Multi_Level_operator<Dim, DataType, length_coarsening_op, nlev>&
		     ){}

	template<std::size_t iter_level=level>
	static void iteration(
			Multigrid_domain<Dim, base_length, nlev>& next,
			Multigrid_domain<Dim, base_length, nlev>& current,
			Multigrid_domain<Dim, base_length, nlev>& rhs_domain,
			Multi_Level_operator<Dim,
			DataType, length, nlev>& Smooth_operator,
			Multi_Level_operator<Dim, 
			DataType, length_diff_op, nlev>& Diff_operator,
			Multi_Level_operator<Dim, 
			DataType, length_coarsening_op, nlev>& coarsening_operator)
	{


		if constexpr (iter_level==1) {
			return;
		}
		else
		{
			pre_smoother(Integer<iter_level>{},
   				     next, 
			     	     current, 
			     	     rhs_domain, 
			     	     Smooth_operator.template get_values<iter_level>(), 
			     	     Smooth_operator.template get_offsets<iter_level>());
		
			//Computing offsets
	        	convolution::Convolve(current.template get_domain<iter_level>(),
			 	     next.template get_domain<iter_level>(),
			 	     Diff_operator.template get_values<iter_level>(),
			 	     Diff_operator.template get_offsets<iter_level>());

		
			subtract_domains(next.template get_domain<iter_level>(),
				 rhs_domain.template get_domain<iter_level>(),
				 current.template get_domain<iter_level>());

			level_transition::coarsening(
				 rhs_domain.template get_domain<iter_level-1>(),
				 next.template get_domain<iter_level>(),
				 coarsening_operator.template get_values<iter_level>(),
				 coarsening_operator.template get_offsets<iter_level>()
				);


			//Now we need to coarsen the domain.
			iteration<iter_level-1>(next, 
					current, 
					rhs_domain, 
					Smooth_operator,
					Diff_operator, 
					coarsening_operator);

		
			post_smoother(Integer<iter_level>{},
				      next, 
				      current, 
			     	      rhs_domain, 
			      	      Smooth_operator.template get_values<iter_level>(), 
			      	      Smooth_operator.template get_offsets<iter_level>());
		}
	}


	inline static Pre_Smoother pre_smoother{};
	inline static Post_Smoother post_smoother{};

};

}

#endif
