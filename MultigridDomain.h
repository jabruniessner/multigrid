#include "Domain.h"
#include "predefinitions.h"
#include "utils.h"

#ifndef MULTIGRIDDOMAIN_H
#define MULTIGRIDDOMAIN_H

constexpr Length nlev = 5;
constexpr Length values_1D = utils::Power<2u, nlev>::value - 1;

template<Dimension Dim, std::size_t base_length, std::size_t nlev>
struct Multigrid_domain : public Multigrid_domain<Dim, base_length, nlev-1>
{	
	
	Multigrid_domain(sycl::queue& q) : 
		domain(Paddings::PERIODIC, q, 1), 
		Multigrid_domain<Dim, base_length, nlev-1>(q) {};
	
	template<std::size_t lev = nlev>
	DataType get_value(Position1D i, Position1D j)
	{
		static_assert(lev <= nlev, "level too large!");
		return Multigrid_domain<Dim, base_length, lev>::domain.get_value(i, j);
	}

	template<std::size_t lev = nlev>
	void set_value(DataType val, Position1D i, Position1D j)
	{
		static_assert(lev <= nlev, "level too large!");
		Multigrid_domain<Dim, base_length, lev>::domain.set_value(val, i, j);
	}

	template<std::size_t lev = nlev>
	decltype(Multigrid_domain<Dim, base_length, lev>::domain)& get_domain()
	{
		return Multigrid_domain<Dim, base_length, lev>::domain;
	}

	void print_level()
	{
		std::cout<< nlev << std::endl;
	}

	void print_length()
	{
		std::cout << length << std::endl;
	}
	
	constexpr static Length length = base_length* utils::Power<2u, nlev>::value-1;
	Domain<Dim, length, length> domain;
};


template<Dimension Dim, std::size_t base_length>
struct Multigrid_domain<Dim, base_length, 0u>
{	
	Multigrid_domain(sycl::queue& q) :
		domain(Paddings::PERIODIC, q, 1) {}

	
	DataType get_value(Position1D i, Position1D j)
	{
		return domain.get_value(i, j);
	}
	
	void set_value(DataType val, Position1D i, Position1D j)
	{
		domain.set_value(val, i, j);
	}

	void print_level()
	{
		std::cout<< 0u << std::endl;
	}

	std::size_t get_base_length()
	{
		return base_length;
	}	
	
	constexpr static Length length = base_length-1;
	Domain<Dim, length, length> domain;
};

template<Dimension Dim, std::size_t base_length, std::size_t nlev, std::size_t level=nlev>
void print_multigrid_domain(Multigrid_domain<Dim, base_length, nlev>& MultDomain)
{
	if constexpr (level==0)
	{
		return;
	}
	else
	{

		std::cout<<"\n\n\n";
		std::cout << "Level: "<< level<<std::endl;
		MultDomain.template get_domain<level>().print_domain();
		print_multigrid_domain<Dim, base_length, nlev, level-1>(MultDomain);
		
	}
}


template <Dimension Dim, typename DataType ,std::size_t length, std::size_t nlev>
struct Multi_Level_operator : public Multi_Level_operator<Dim, DataType, length, nlev-1>
{
	using OffsetType = std::array<std::size_t, Dim>;

	Multi_Level_operator(std::array<DataType, length>& values, 
			     std::array<OffsetType, length>& offsets) :
		values(values),
		offsets(offsets) {};

	Multi_Level_operator(std::array<DataType, length>&& values,
		             std::array<OffsetType, length>&& offsets) :
		values(values),
		offsets(offsets) {};
	
	template<std::size_t access_level = nlev>
	auto& get_values()
	{
		return 
		  Multi_Level_operator<
		  	Dim, 
			DataType, 
			length, 
			access_level>::values;
	}

	template<std::size_t access_level= nlev>
	auto& get_offsets()
	{
		return 
		 Multi_Level_operator<
		 	Dim, 
			DataType, 
			length, 
			access_level>::offsets; 
	}
	
	std::array<DataType, length> values;
	std::array<OffsetType, length> offsets;
};


template <Dimension Dim, typename DataType, std::size_t length>
struct Multi_Level_operator<Dim, DataType, length, 0u>
{
	using OffsetType = std::array<std::size_t, Dim>;

	Multi_Level_operator(std::array<DataType, length>& values, 
			     std::array<OffsetType, length>& offsets) :
		values(values),
		offsets(offsets) {};

	Multi_Level_operator(std::array<DataType, length>&& values,
		             std::array<OffsetType, length>&& offsets) :
		values(values),
		offsets(offsets) {};

	auto get_values()
	{
		return values;
	}

	auto get_offsets()
	{
		return offsets;
	}
	
	std::array<DataType, length> values;
	std::array<OffsetType, length> offsets;
};

#endif


