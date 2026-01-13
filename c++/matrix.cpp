#include "matrix.hpp"
#include <random>
#include <cmath>
#include <future>
#include <iostream>
#include <chrono>

constexpr size_t BASE_CASE = 128;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  program <n> [max_parallel_depth] [test]\n";
        return 0;
    }

    size_t n;
    try {
        n = std::stoul(argv[1]);
    } catch (...) {
        std::cerr << "Error: matrix dimension must be an integer\n";
        return 1;
    }

    size_t max_parallel_depth = (argc >= 3) ? std::stoul(argv[2]) : 3;
    bool test_mode = (argc >= 4 && std::string(argv[3]) == "test");

    auto A = create_random_matrix(n);
    auto B = create_random_matrix(n);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto C1 = standard_implementation(A, B);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto C2 = divide_and_conquer_implementation(A, B, 0, max_parallel_depth);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto C3 = strassen_implementation(A, B, 0, max_parallel_depth);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto t_standard =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto t_dc =
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto t_strassen =
        std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    if (test_mode) {
        std::cout << t_standard << " "
                  << t_dc << " "
                  << t_strassen << "\n";
        return 0;
    }

    std::cout <<
        "Correctness check:\n"
        " - Divide & Conquer: " << (compare_matrices(C1, C2) ? "OK" : "ERROR") << "\n"
        " - Strassen: " << (compare_matrices(C1, C3) ? "OK" : "ERROR") << "\n\n";

    std::cout << "Matrix size: " << n << " × " << n << "\n\n";

    std::cout <<
        "Execution times (ms):\n"
        " - Standard: " << t_standard << "\n"
        " - Divide & Conquer: " << t_dc << "\n"
        " - Strassen: " << t_strassen << "\n";

    return 0;
}

Matrix standard_implementation(
    const Matrix& A, 
    const Matrix& B
) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k)
            for (size_t j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

Matrix divide_and_conquer_implementation(
    const Matrix& A,
    const Matrix& B,
    size_t depth,
    size_t max_parallel_depth
) {
    size_t n = A.size();
    if (n <= BASE_CASE)
        return standard_implementation(A, B);

    size_t h = n / 2;
    Matrix C(n, std::vector<double>(n));

    auto A11 = extract_block(A, 0, 0, h);
    auto A12 = extract_block(A, 0, h, h);
    auto A21 = extract_block(A, h, 0, h);
    auto A22 = extract_block(A, h, h, h);

    auto B11 = extract_block(B, 0, 0, h);
    auto B12 = extract_block(B, 0, h, h);
    auto B21 = extract_block(B, h, 0, h);
    auto B22 = extract_block(B, h, h, h);

    Matrix C11a, C11b, C12a, C12b, C21a, C21b, C22a, C22b;

    if (depth < max_parallel_depth) {
        auto f11 = std::async(std::launch::async, divide_and_conquer_implementation, A11, B11, depth+1, max_parallel_depth);
        auto f12 = std::async(std::launch::async, divide_and_conquer_implementation, A12, B21, depth+1, max_parallel_depth);
        auto f21 = std::async(std::launch::async, divide_and_conquer_implementation, A11, B12, depth+1, max_parallel_depth);
        auto f22 = std::async(std::launch::async, divide_and_conquer_implementation, A12, B22, depth+1, max_parallel_depth);

        C11a = f11.get();
        C11b = f12.get();
        C12a = f21.get();
        C12b = f22.get();

        C21a = divide_and_conquer_implementation(A21, B11, depth+1, max_parallel_depth);
        C21b = divide_and_conquer_implementation(A22, B21, depth+1, max_parallel_depth);
        C22a = divide_and_conquer_implementation(A21, B12, depth+1, max_parallel_depth);
        C22b = divide_and_conquer_implementation(A22, B22, depth+1, max_parallel_depth);
    } else {
        C11a = divide_and_conquer_implementation(A11, B11, depth+1, max_parallel_depth);
        C11b = divide_and_conquer_implementation(A12, B21, depth+1, max_parallel_depth);
        C12a = divide_and_conquer_implementation(A11, B12, depth+1, max_parallel_depth);
        C12b = divide_and_conquer_implementation(A12, B22, depth+1, max_parallel_depth);
        C21a = divide_and_conquer_implementation(A21, B11, depth+1, max_parallel_depth);
        C21b = divide_and_conquer_implementation(A22, B21, depth+1, max_parallel_depth);
        C22a = divide_and_conquer_implementation(A21, B12, depth+1, max_parallel_depth);
        C22b = divide_and_conquer_implementation(A22, B22, depth+1, max_parallel_depth);
    }

    insert_block(C, add_blocks(C11a, C11b), 0, 0);
    insert_block(C, add_blocks(C12a, C12b), 0, h);
    insert_block(C, add_blocks(C21a, C21b), h, 0);
    insert_block(C, add_blocks(C22a, C22b), h, h);

    return C;
}

Matrix strassen_implementation(
    const Matrix& A,
    const Matrix& B,
    size_t depth,
    size_t max_parallel_depth
) {
    size_t n = A.size();
    if (n <= BASE_CASE)
        return standard_implementation(A, B);

    size_t h = n / 2;
    Matrix C(n, std::vector<double>(n));

    auto A11 = extract_block(A, 0, 0, h);
    auto A12 = extract_block(A, 0, h, h);
    auto A21 = extract_block(A, h, 0, h);
    auto A22 = extract_block(A, h, h, h);

    auto B11 = extract_block(B, 0, 0, h);
    auto B12 = extract_block(B, 0, h, h);
    auto B21 = extract_block(B, h, 0, h);
    auto B22 = extract_block(B, h, h, h);

    Matrix M1, M2, M3, M4, M5, M6, M7;

    if (depth < max_parallel_depth) {
        auto f1 = std::async(std::launch::async, strassen_implementation,
            add_blocks(A11, A22), add_blocks(B11, B22), depth+1, max_parallel_depth);

        auto f2 = std::async(std::launch::async, strassen_implementation,
            add_blocks(A21, A22), B11, depth+1, max_parallel_depth);

        auto f3 = std::async(std::launch::async, strassen_implementation,
            A11, sub_blocks(B12, B22), depth+1, max_parallel_depth);

        auto f4 = std::async(std::launch::async, strassen_implementation,
            A22, sub_blocks(B21, B11), depth+1, max_parallel_depth);

        auto f5 = std::async(std::launch::async, strassen_implementation,
            add_blocks(A11, A12), B22, depth+1, max_parallel_depth);

        auto f6 = std::async(std::launch::async, strassen_implementation,
            sub_blocks(A21, A11), add_blocks(B11, B12), depth+1, max_parallel_depth);

        auto f7 = std::async(std::launch::async, strassen_implementation,
            sub_blocks(A12, A22), add_blocks(B21, B22), depth+1, max_parallel_depth);

        M1 = f1.get();
        M2 = f2.get();
        M3 = f3.get();
        M4 = f4.get();
        M5 = f5.get();
        M6 = f6.get();
        M7 = f7.get();
    } else {
        M1 = strassen_implementation(add_blocks(A11, A22), add_blocks(B11, B22), depth+1, max_parallel_depth);
        M2 = strassen_implementation(add_blocks(A21, A22), B11, depth+1, max_parallel_depth);
        M3 = strassen_implementation(A11, sub_blocks(B12, B22), depth+1, max_parallel_depth);
        M4 = strassen_implementation(A22, sub_blocks(B21, B11), depth+1, max_parallel_depth);
        M5 = strassen_implementation(add_blocks(A11, A12), B22, depth+1, max_parallel_depth);
        M6 = strassen_implementation(sub_blocks(A21, A11), add_blocks(B11, B12), depth+1, max_parallel_depth);
        M7 = strassen_implementation(sub_blocks(A12, A22), add_blocks(B21, B22), depth+1, max_parallel_depth);
    }

    auto C11 = add_blocks(sub_blocks(add_blocks(M1, M4), M5), M7);
    auto C12 = add_blocks(M3, M5);
    auto C21 = add_blocks(M2, M4);
    auto C22 = add_blocks(sub_blocks(add_blocks(M1, M3), M2), M6);

    insert_block(C, C11, 0, 0);
    insert_block(C, C12, 0, h);
    insert_block(C, C21, h, 0);
    insert_block(C, C22, h, h);

    return C;
}

Matrix create_random_matrix(size_t n) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    Matrix m(n, std::vector<double>(n));
    for (auto& row : m)
        for (auto& v : row)
            v = dist(gen);

    return m;
}

Matrix extract_block(
    const Matrix& M, 
    size_t r, 
    size_t c, 
    size_t size
) {
    Matrix block(size, std::vector<double>(size));
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            block[i][j] = M[r + i][c + j];
    return block;
}

void insert_block(
    Matrix& M, 
    const Matrix& block, 
    size_t r, 
    size_t c
) {
    size_t size = block.size();
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            M[r + i][c + j] = block[i][j];
}

Matrix add_blocks(
    const Matrix& A, 
    const Matrix& B
) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix sub_blocks(
    const Matrix& A, 
    const Matrix& B
) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

bool compare_matrices(
    const Matrix& A, 
    const Matrix& B
) {
    const double EPS = 1e-6;
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (std::abs(A[i][j] - B[i][j]) > EPS)
                return false;
    return true;
}
