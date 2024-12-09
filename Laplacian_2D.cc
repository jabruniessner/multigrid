#include "MultigridDomain.h"

int main()
{
	sycl::gpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Multigrid_domain<2, 1, 4u> lhs_domain(q);
	Multigrid_domain<2, 1, 4u> rhs_domain(q);

	std::array<OffsetType, 4> offsets{{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}};
	std::array<DataType, 4>{1,1,1,1};
}
