#include<tuple>
#include<utility>
#include<iostream>
#include<vector>
#include<array>

namespace utils{

  template<typename Arg1, typename... Args>
  Arg1&& get_first(Arg1&& arg1, Args&&...){
	return std::forward<Arg1>(arg1);
  }


  template<typename Arg1, typename... Args>
  auto get_tail(Arg1&& arg1, Args... args){
	auto tail = std::forward_as_tuple(args...);
       	return tail;	
  }

  template <std::size_t Offset, std::size_t ... Is>
  std::index_sequence<(Offset + Is)...> add_offset(std::index_sequence<Is...>)
  {
         return {};
  }

  template <std::size_t Offset, std::size_t N>
  auto make_index_sequence_with_offset()
  {
          return add_offset<Offset>(std::make_index_sequence<N>{});
  }


  template<typename Tuple, std::size_t... Ints>
  std::tuple<std::tuple_element_t<Ints, Tuple>...>
  extract_tuple(Tuple&& tuple, std::index_sequence<Ints...>) {
   return { std::get<Ints>(std::forward<Tuple>(tuple))... };
  }

  template<std::size_t base, std::size_t power>
  struct Power
  {
	  constexpr static std::size_t value = base * Power<base, power-1>::value;
  };

  template<std::size_t base>
  struct Power<base, 0u>
  {
	  constexpr static std::size_t value=1;
  };

}

