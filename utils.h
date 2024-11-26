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

  template<typename Tuple, std::size_t... Ints>
  std::tuple<std::tuple_element_t<Ints, Tuple>...>
  extract_tuple(Tuple&& tuple, std::index_sequence<Ints...>) {
   return { std::get<Ints>(std::forward<Tuple>(tuple))... };
  }

}

