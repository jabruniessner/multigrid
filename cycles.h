#include "level_transition.h"

template<Dimension Dim, std::size_t base_length, std::size_t nlev, std::size_t level=nlev>
struct V_Cycle_base()
{
	Multigrid_domain<Dim, base_length, nlev>& Multdomain_1;
	Multigrid_domain<Dim, base_length, nlev>& Multdomain_2;
	Multigrid_domain<Dim, base_length, nlev>& rhs_domain;
}
