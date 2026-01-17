#pragma once
#include <vector>
#include <cstddef>

using Matrix = std::vector<std::vector<double>>;

Matrix random_matrix(size_t n);
Matrix standard(const Matrix& A, const Matrix& B);

Matrix dc_mul(
    const Matrix& A,
    const Matrix& B
);

Matrix strassen(
    const Matrix& A,
    const Matrix& B
);

Matrix view(const Matrix& M, size_t row, size_t col, size_t size);

Matrix add(const Matrix& A, const Matrix& B);
Matrix sub(const Matrix& A, const Matrix& B);

bool approx_equal(const Matrix& A, const Matrix& B);
