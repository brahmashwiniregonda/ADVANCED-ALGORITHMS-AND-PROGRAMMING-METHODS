// Compiled using MSVC v19.29.30136 for x86 cl /std:c++20 tensor.cpp /EHsc

#include "tensor.hpp"
#include <iostream>
#include <iomanip>
#include <tuple>

int main()
{
    // Tester code for the tensor class
    std::cout << "Tensor Testing" << std::endl;

    // Tensor multiplication test
    tensor<float,3> a(2,3,4);
    tensor<float,4> b(2,3,5,2);

    tensor<float,3> c = tensor_product(a,b,std::array<std::size_t, 4>{0,3,1,1});

    // Display results
    std::cout << "c: ";
    for (std::size_t i = 0; i < c.size().size(); i++)
    {
        std::cout << c.size()[i] << ",";
    }
    std::cout << std::endl;

    // Matrix test (Rank 2 tensor)
    tensor<float, 2> m(3, 3); // Create a 3x3 matrix

    // Fill with data
    m(0, 0) = 1.1;
    m(0, 1) = 1.2;
    m(0, 2) = 1.3;
    m(1, 0) = 2.1;
    m(1, 1) = 2.2;
    m(1, 2) = 2.3;
    m(2, 0) = 3.1;
    m(2, 1) = 3.2;
    m(2, 2) = 3.3;

    // Get the dimensions
    auto m_size = m.size();

    // Print the matrix
    std::cout << "m: " << m_size[0] << "," << m_size[1] << std::endl;
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++)
        {
            std::cout << m(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    // Slice the matrix to get row 0
    tensor<float, 1> r0 = m.slice(0, 0);

    // Get the dimensions
    auto r0_size = r0.size();

    std::cout << std::endl;
    std::cout << "r0: " << r0_size[0] << std::endl;

    // Print r0 to cout
    for (auto i = 0; i < 3; i++)
    {
        std::cout << r0(i) << ' ';
    }
    std::cout << std::endl;

    // Slice the matrix to get item 1,1
    auto c11 = m.slice(0, 1, 1, 1);

    std::cout << std::endl;
    std::cout << "c11: " << std::endl;

    // Print the value
    std::cout << c11() << std::endl;

    // Slice column 2
    std::cout << std::endl;
    auto c2 = m.slice(1, 2);

    auto c2_size = c2.size();
    std::cout << "c2: " << c2_size[0] << std::endl;

    for (auto i = 0; i < 3; i++)
    {
        std::cout << c2(i) << ' ';
    }
    std::cout << std::endl;

    // Create a 3 x 2 matrix
    tensor<float, 2> m2(3, 2);

    // Initialize values of m2
    m2(0, 0) = 1.0;
    m2(0, 1) = 1.0;
    m2(1, 0) = 2.0;
    m2(1, 1) = 2.0;
    m2(2, 0) = 3.0;
    m2(2, 1) = 3.0;

    std::cout << std::endl;
    std::cout << "m2: " << m2.size()[0] << "," << m2.size()[1] << std::endl;
    for (std::size_t i = 0; i < 3; i++)
    {
        for (std::size_t j = 0; j < 2; j++)
        {
            std::cout << m2(i, j) << ' ';
        }

        std::cout << std::endl;
    }

    // Use tensor_product to multiply the matrices
    // In matrix multiplication, the column (dim 1) of matrix 1 
    // and the row (dim 0) of matrix 2 are summed over
    auto product = tensor_product(m, m2, std::array<std::size_t, 2>{1, 0});

    // Print the sizes
    auto product_size = product.size();
    std::cout << std::endl;
    std::cout << "product = m x m2" << std::endl;
    std::cout << "product: " << product_size[0] << "," << product_size[1] << std::endl;

    // Print out the contents
    for (std::size_t i = 0; i < product_size[0]; i++)
    {
        for (std::size_t j = 0; j < product_size[1]; j++)
        {
            std::cout << product(i, j) << ' ';
        }

        std::cout << std::endl;
    } 
    
    return 0;
}