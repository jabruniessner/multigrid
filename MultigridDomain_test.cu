#include "MultigridDomain.h"

template <typename T> struct TD;

using namespace multigrid_domain;

int main() {

  Multigrid_domain<2, 4u, 4u, 4u> mult_domain{};

  //	auto& dom0 = mult_domain.get_domain<0u>();
  //	auto& dom1 = mult_domain.get_domain<1u>();
  //	auto& dom2 = mult_domain.get_domain<2u>();
  //	auto& dom3 = mult_domain.get_domain<3u>();
  //	auto& dom4 = mult_domain.get_domain<4u>();

  auto a = mult_domain.get_value<0u>(0, 0);
  auto b = mult_domain.get_value<1u>(0, 0);
  auto c = mult_domain.get_value<2u>(0, 0);
  auto d = mult_domain.get_value<3u>(0, 0);
  auto e = mult_domain.get_value<4u>(0, 0);

  // TD<decltype(dom0)> td;
  // TD<decltype(dom1)> td1;
  // TD<decltype(dom2)> td2;
  // TD<decltype(dom3)> td3;
  // TD<decltype(dom4)> td4;

  std::cout << "The values are: " << a << " " << b << " " << c << " " << d
            << " " << e << std::endl;

  std::cout << std::endl;
}
