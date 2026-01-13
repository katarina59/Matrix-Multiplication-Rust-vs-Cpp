#pragma once
#include <vector>
#include <cstddef>

using Matrix = std::vector<std::vector<double>>;

Matrix create_random_matrix(size_t n);
Matrix standard_implementation(const Matrix& A, const Matrix& B);

Matrix divide_and_conquer_implementation(
    const Matrix& A,
    const Matrix& B,
    size_t depth,
    size_t max_parallel_depth
);

Matrix strassen_implementation(
    const Matrix& A,
    const Matrix& B,
    size_t depth,
    size_t max_parallel_depth
);

Matrix extract_block(const Matrix& M, size_t row, size_t col, size_t size);
void insert_block(Matrix& M, const Matrix& block, size_t row, size_t col);

Matrix add_blocks(const Matrix& A, const Matrix& B);
Matrix sub_blocks(const Matrix& A, const Matrix& B);

bool compare_matrices(const Matrix& A, const Matrix& B);

// g++ -O3 -std=c++17 -pthread matrix.cpp -o matrix_mul
// ./matrix_mul 512 3