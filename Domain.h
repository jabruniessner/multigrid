#include <sycl/sycl.hpp>
#include "predefinitions.h"
#include <cassert>
#include <format>

#ifndef DOMAIN_H
#define DOMAIN_H

template <Length FirstStride, Length... RestStrides, typename Padding, typename Position, typename... PositionRest>
size_t flatten_index(Padding padding, Position i, PositionRest... rest_positions)
{
	//initialize i with the first index value
	int index = i;

	//Use a fold expression to process all Strides and indices
	((index *= RestStrides+2*padding, index+= rest_positions), ...);

	return index;
}

template<Dimension Dim, Length... strides_all>
struct Domain {
		template<typename... Length>
		Domain(Paddings padding, sycl::queue& q, int padding_width) : 
			strides{strides_all...}, 
			padding(padding),
			padding_width(padding_width),	
			q(q)

		{	
			static_assert(sizeof... (strides_all) == Dim);

			
			num_values=1;
			((num_values*= strides_all + 2*padding_width), ...);

			values_buff = sycl::malloc_device<DataType>(num_values*sizeof(DataType), q);
			q.memset(values_buff, 0, num_values*sizeof(DataType)).wait();
		}

		template<typename... Positions>
		DataType& operator()(Positions... positions) const
		{
			static_assert(sizeof... (Positions) == Dim);
			return values_buff[flatten_index<strides_all...>(padding_width, positions...)];
		}


		template<typename... Positions>
		void set_value(DataType val, Positions... position)
		{
			q.memcpy(&values_buff[flatten_index<strides_all...>(padding_width, position...)], &val, sizeof(DataType)).wait();
		}


		template<typename... Positions>
		DataType get_value(Positions... position)
		{
			DataType k;
			std::size_t flat_index = flatten_index<strides_all...>(padding_width, position...);
			q.memcpy( &k, &values_buff[flat_index], sizeof(DataType)).wait();
			
			return k;
		}


		template< typename... Indices>
		void print_domain(Indices... indices)
		{
			if constexpr (sizeof...(Indices) < Dim){
				for(Position1D i = 0; i < strides[sizeof...(Indices)]+2*padding_width; i++)
					print_domain(indices..., i);
				std::cout<<std::endl;
			}
			else
			{
				std::cout<<std::format("{:6.3f} ", this->get_value(indices...));
			}

		};


	
		DataType* values_buff;
		Length strides[Dim];
		Length num_values;
		Length padding_width;
		Paddings padding;
		sycl::queue& q;
};


template<Dimension Dim, Length... strides_all, std::size_t... dims>
int domain_compute_norm_squared(DataType& result,  Domain<Dim, strides_all...>& a, std::index_sequence<dims...>)
{
        DataType* result_device = sycl::malloc_device<DataType>(sizeof(DataType), a.q);

        a.q.memset(result_device, 0, sizeof(DataType));

	std::cout<<"The strides are: ";
	((std::cout<<strides_all<<" "), ...);
	std::cout<<std::endl;

        a.q.parallel_for(sycl::range<Dim>(strides_all...), sycl::reduction(result_device, sycl::plus<>()), [=](sycl::id<Dim> I, auto& acc)
                        { 
                                acc += a((I[dims]+a.padding_width)...)*a((I[dims]+a.padding_width)...);
                        }).wait();

        a.q.memcpy(&result, result_device, sizeof(DataType)).wait();

        return 0;
}

template<Dimension Dim, Length... strides_all>
int domain_compute_norm_squared(DataType& result,  Domain<Dim, strides_all...>& a)
{	
	return
	domain_compute_norm_squared(result,a, std::make_index_sequence<Dim>());
}

template<Dimension Dim, Length... strides_all>
int domain_scalar_multiply(Domain<Dim, strides_all...>& dest, Domain<Dim, strides_all...>&a, const DataType& scalar)
{
        assert(dest.q == a.q);
        assert(dest.num_values == a.num_values);
        assert(dest.padding_width == a.padding_width);

        dest.q.parallel_for(sycl::range<1>(a.num_values), [=](sycl::id<1> i){
                        dest.values_buff[i] = a.values_buff[i]*scalar;
                        }).wait();

        return 0;
}

template<Dimension Dim, Length... strides_all>
int divide_domains(Domain<Dim, strides_all...>& dest, Domain<Dim, strides_all...>& a, Domain<Dim, strides_all...> &b)
{

        assert(dest.q==a.q && a.q == b.q);
        assert(dest.num_values == a.num_values && b.num_values == a.num_values);
        assert(dest.padding_width == a.padding_width && a.padding_width == b.padding_width);


        dest.q.parallel_for(sycl::range<1>(a.num_values), [=](sycl::id<1> i){

                        dest.values_buff[i] = a.values_buff[i]/b.values_buff[i];
        }).wait();

        return 0;
}

template<Dimension Dim, Length... strides_all>
int multiply_domains(Domain<Dim, strides_all...>& dest, Domain<Dim, strides_all...>& a, Domain<Dim, strides_all...> &b)
{

        assert(dest.q==a.q && a.q == b.q);
        assert(dest.num_values == a.num_values && b.num_values == a.num_values);
        assert(dest.padding_width == a.padding_width && a.padding_width == b.padding_width);


        dest.q.parallel_for(sycl::range<1>(a.num_values), [=](sycl::id<1> i){

                        dest.values_buff[i] = a.values_buff[i]*b.values_buff[i];
        }).wait();

        return 0;
}


template<Dimension Dim, Length... strides_all>
int subtract_domains(Domain<Dim, strides_all...>& dest, 
		     Domain<Dim, strides_all...>& a, 
		     Domain<Dim, strides_all...> &b)
{

        assert(dest.q==a.q && a.q == b.q);
        assert(dest.num_values == a.num_values 
	       && b.num_values == a.num_values);
        
	assert(dest.padding_width == a.padding_width 
	       && a.padding_width == b.padding_width);


        dest.q.parallel_for(sycl::range<1>(a.num_values), [=](sycl::id<1> i){
                        dest.values_buff[i] = a.values_buff[i]-b.values_buff[i];
        }).wait();

        return 0;
}


template<Dimension Dim, Length... strides_all>
int add_domains(Domain<Dim, strides_all...>& dest, Domain<Dim, strides_all...>& a, Domain<Dim, strides_all...> &b)
{

        assert(dest.q==a.q && a.q == b.q);
        assert(dest.num_values == a.num_values && b.num_values == a.num_values);
        assert(dest.padding_width == a.padding_width && a.padding_width == b.padding_width);

        dest.q.parallel_for(sycl::range<1>(a.num_values), [=](sycl::id<1> i){

                        dest.values_buff[i] = a.values_buff[i]+b.values_buff[i];
        }).wait();

        return 0;
}







#endif

