#include "Convolution.h"
#include "CG_Solver.h"
#include<iostream>



int main()
{
	
	using OffsetType = std::array<int, 3>;
	sycl::gpu_selector selector;
	sycl::queue q(selector, sycl::property_list{sycl::property::queue::in_order{}});

	Domain<3, 3u, 3u, 3u> domain_src(Paddings::PERIODIC, q, 1);
	Domain<3, 3u, 3u, 3u> domain_dest(Paddings::PERIODIC, q, 1);

	constexpr std::array<DataType, 7> values = {6, -1, -1, -1, -1, -1, -1};
	constexpr std::array<OffsetType, 7> offsets{{{0,0,0}, {1, 0,0}, {-1,0,0}, 
						     {0,1,0}, {0,-1,0},  {0,0,1}, 
						     {0,0,-1}}};


	//====== First Trial ====//
	
	std::cout<<"The input matrix is given by: "<<std::endl;
	for(Position1D i = 1; i<3+1; i++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D k = 1; k < 3+1; k++)
			{
				domain_src.set_value(1, i, j, k);
			}
		}
	}

	for(Position1D k = 0; k<3+2; k++)
	{
		for(Position1D j = 0; j<3+2; j++)
		{
			for(Position1D i = 0; i<3+2; i++)
			{
				std::cout<< domain_src.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}


	std::cout<<"The norm squared is given by: "<<std::endl;
	DataType result = 1;

	domain_compute_norm_squared(result, domain_src);

	std::cout<<"The result is given by: "<< result<<std::endl;

	Convolve(domain_dest, domain_src, values, offsets);

	std::cout<<"The output Matrix is given by: "<<std::endl;

	for(Position1D k = 0; k<3+2; k++)
	{
		for(Position1D j = 0; j<3+2; j++)
		{
			for(Position1D i = 0; i<3+2; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}


	//======== Second trial ========//
	
	std::cout<<"The input matrix is given by: "<<std::endl;
	for(Position1D j = 0; j<3+2; j++)
		for(Position1D i = 0; i<3+2; i++)
		{
			domain_src.set_value(1, i, j, 4u);
			domain_src.set_value(1, i, j, 0u);

			domain_src.set_value(1, i, 4u, j);
			domain_src.set_value(1, i, 0u, j);

			domain_src.set_value(1, 4u, i, j);
			domain_src.set_value(1, 0u, i, j);
		}

	std::cout << "(Boundary set to 1)"<<std::endl;

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_src.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}


	Convolve(domain_dest, domain_src, values, offsets);


	std::cout<<"The output matrix is given by: "<<std::endl;

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}

	//============== Trial scalar multiplication ===============
	std::cout<<"The initial matrix is: "<<std::endl;
	DataType count = 0;

	for(Position1D k = 1; k<3+1; k++){
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				domain_src.set_value(++count, i, j, k);
				std::cout<< count <<" ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}	

	std::cout<<"The result is: "<<std::endl;

	std::cout<<"Scalar multiplication with factor 3"<<std::endl;

	domain_scalar_multiply(domain_dest, domain_src, 3);	

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}


	//======== Trial Addition ======//
	

	std::cout<<"Component-wise addition Domain with itself"<<std::endl;
	add_domains(domain_dest, domain_src, domain_src);

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}


	//======= Trial Subtraction =======/
	

	std::cout<<"Component-wise subtraction Domain with itself" << std::endl;
	subtract_domains(domain_dest, domain_src, domain_src);

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}

	//===== Trial Multiplication ======/
	
	std::cout<<"Component-wise multiplication Domain with itself" << std::endl;
	multiply_domains(domain_dest, domain_src, domain_src);	

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}


	//===== Trial Division ======/
	std::cout<<"Component-wise division Domain with itself" << std::endl;
	divide_domains(domain_dest, domain_src, domain_src);	

	for(Position1D k = 1; k<3+1; k++)
	{
		for(Position1D j = 1; j<3+1; j++)
		{
			for(Position1D i = 1; i<3+1; i++)
			{
				std::cout<< domain_dest.get_value(i, j, k) << " ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}

}
