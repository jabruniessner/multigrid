#include "Convolution.h"
#include "CG_Solver.h"

int main()
{
	sycl::cpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Domain<2, 3u, 3u> domain_src(Paddings::PERIODIC, q, 1);
	Domain<2, 3u, 3u> domain_dest(Paddings::PERIODIC, q, 1);

	//===== First Trial ======//
	
	std::cout<<"The input matrix is given by: "<<std::endl;
	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			domain_src.set_value(1,  i, j);
		}

		std::cout<<std::endl;
	}

	domain_src.set_value(1, 1, 1);

	for(Position1D i = 0; i<3+2; i++)
	{
		for(Position1D j = 0; j<3+2; j++)
		{
			//domain_src.set_value(1,  i, j);
			std::cout<< domain_src.get_value(i, j) << " ";
		}

		std::cout<<std::endl;
	}



	std::cout<<"The norm squared is given by: "<< std::endl;
	DataType result = 1;
	
	domain_compute_norm_squared(result, domain_src);

	std::cout<<"The result is given by: "<< result << std::endl;


	Convolve(domain_dest, domain_src, vec_val, vec_offsets);

	std::cout<<"The output Matrix is given by: "<< std::endl;

	for(Position1D i = 1; i<3+1; i++){
		for(Position1D j = 1; j<3+1; j++)
		{
			std::cout<<domain_dest.get_value(i, j)<< " ";
		}
		std::cout<<std::endl;
	}

	//===== Second trial ======//

	std::cout<<"The input matrix is given by: "<<std::endl;
	for(Position1D i = 0; i<3+2; i++)
	{
		domain_src.set_value(1,  i, 4u);
		domain_src.set_value(1,  i, 0u);

		domain_src.set_value(1, 4u, i);
		domain_src.set_value(1, 0u, i);

	}
	

	std::cout << "(Boundary set to 1)"<<std::endl;
	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			domain_src.set_value(0, i, j);
			std::cout<<domain_src.get_value(i, j)<< " ";
		}
		std::cout<<std::endl;
	}


	Convolve(domain_dest, domain_src, vec_val, vec_offsets);

	std::cout<<"The output Matrix is given by: "<< std::endl;

	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			std::cout<<domain_dest.get_value(i, j)<< " ";
		}
		std::cout<<std::endl;
	}

	//======= Trial scalar multiplication =======
	std::cout<<"The initial matrix is: "<<std::endl;
	DataType count = 0;
	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			domain_src.set_value(++count, i, j);
			std::cout<< count <<" ";
		}
		std::cout<<std::endl;
	}

	std::cout<<"The result is: "<<std::endl;

	std::cout<<"Scalar multiplication with factor 3"<<std::endl;

	domain_scalar_multiply(domain_dest, domain_src, 3);

	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
			std::cout<<domain_dest.get_value(i, j)<<" ";

		std::cout<<std::endl;
	}

	//===== Trial Addition ======/
	

	std::cout<<"Component-wise addition Domain with itself"<<std::endl;
	add_domains(domain_dest, domain_src, domain_src);	

	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
			std::cout<<domain_dest.get_value(i, j)<<" ";

		std::cout<<std::endl;
	}

	//===== Trial Subtraction ======/
	
	std::cout<<"Component-wise subtraction Domain with itself"<<std::endl;
	subtract_domains(domain_dest, domain_src, domain_src);	

	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
			std::cout<<domain_dest.get_value(i, j)<<" ";

		std::cout<<std::endl;
	}


	//===== Trial Multiplication ======/
	
	std::cout<<"Component-wise multiplication Domain with itself" << std::endl;
	multiply_domains(domain_dest, domain_src, domain_src);	

	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
			std::cout<<domain_dest.get_value(i, j)<<" ";

		std::cout<<std::endl;
	}

	//===== Trial Division ======/
	std::cout<<"Component-wise division Domain with itself" << std::endl;
	divide_domains(domain_dest, domain_src, domain_src);	

	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
			std::cout<<domain_dest.get_value(i, j)<<" ";

		std::cout<<std::endl;
	}
		

}




//int main()
//{
//	gpu_selector selector;
//	queue q(selector, property_list{property::queue::in_order{}});
//
//	std::cout << "Hello World!" << std::endl;
//	Domain2D domain_src(2344, 2344, Paddings::PERIODIC, q);
//	Domain2D domain_dest(2344, 2344, Paddings::PERIODIC, q);
//
//	unsigned int i= 0+domain_src.padding_width, j=0+domain_src.padding_width;
//
//	domain_src.set_value(1., i, j);
//
//	constexpr std::array<OffsetType, 5> vec_offsets={{{0, 0}, {1,0}, {-1, 0}, {0, 1}, {0, -1}}};
//	constexpr std::array<DataType, 5> vec_val{4 , -1, -1, -1, -1};
//	
//	
//	//clock_t starttime = clock();
//	auto start = std::chrono::high_resolution_clock::now();
//	Convolve2D(domain_dest, domain_src, vec_val, vec_offsets);
//	auto end = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double> duration = end-start;
//	//clock_t endtime = clock();
//
//	std::cout<< "The required time for the convolution is: " << duration.count() << " seconds."<<std::endl;
//}
