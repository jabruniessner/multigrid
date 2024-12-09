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
	
	constexpr static Length length = base_length-1;
	Domain<Dim, length, length> domain;
};

#endif


