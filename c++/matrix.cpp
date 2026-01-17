#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <future>
#include <chrono>
#include <algorithm>  

constexpr size_t BASE_CASE = 128;

struct Matrix {
    size_t n;
    std::vector<double> data;

    Matrix(size_t n_) : n(n_), data(n_ * n_, 0.0) {}

    inline double& operator()(size_t i, size_t j) {
        return data[i * n + j];
    }

    inline double operator()(size_t i, size_t j) const {
        return data[i * n + j];
    }
};


Matrix random_matrix(size_t n) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    Matrix m(n);
    for (auto& v : m.data)
        v = dist(gen);

    return m;
}

template<typename F>
auto measure(F&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = f();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return std::make_pair(result, ms);
}


Matrix standard(const Matrix& A, const Matrix& B) {
    Matrix C(A.n);

    for (size_t i = 0; i < A.n; ++i)
        for (size_t k = 0; k < A.n; ++k) {
            double aik = A(i, k);
            for (size_t j = 0; j < A.n; ++j)
                C(i, j) += aik * B(k, j);
        }

    return C;
}


Matrix add(const Matrix& A, const Matrix& B) {
    Matrix C(A.n);
    for (size_t i = 0; i < C.data.size(); ++i)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    Matrix C(A.n);
    for (size_t i = 0; i < C.data.size(); ++i)
        C.data[i] = A.data[i] - B.data[i];
    return C;
}

Matrix view(const Matrix& M, size_t r, size_t c, size_t size) {
    Matrix V(size);
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            V(i, j) = M(r + i, c + j);
    return V;
}


Matrix dc_mul(const Matrix& A, const Matrix& B) {
    if (A.n <= BASE_CASE)
        return standard(A, B);

    size_t h = A.n / 2;

    auto A11 = view(A, 0, 0, h);
    auto A12 = view(A, 0, h, h);
    auto A21 = view(A, h, 0, h);
    auto A22 = view(A, h, h, h);

    auto B11 = view(B, 0, 0, h);
    auto B12 = view(B, 0, h, h);
    auto B21 = view(B, h, 0, h);
    auto B22 = view(B, h, h, h);

    auto f1 = std::async(std::launch::async, dc_mul, A11, B11);
    auto f2 = std::async(std::launch::async, dc_mul, A12, B21);
    auto f3 = std::async(std::launch::async, dc_mul, A11, B12);
    auto f4 = std::async(std::launch::async, dc_mul, A12, B22);

    Matrix C11 = add(f1.get(), f2.get());
    Matrix C12 = add(f3.get(), f4.get());
    Matrix C21 = add(dc_mul(A21, B11), dc_mul(A22, B21));
    Matrix C22 = add(dc_mul(A21, B12), dc_mul(A22, B22));

    Matrix C(A.n);
    for (size_t i = 0; i < h; ++i)
        for (size_t j = 0; j < h; ++j) {
            C(i, j) = C11(i, j);
            C(i, j + h) = C12(i, j);
            C(i + h, j) = C21(i, j);
            C(i + h, j + h) = C22(i, j);
        }

    return C;
}


Matrix strassen(const Matrix& A, const Matrix& B) {
    if (A.n <= BASE_CASE)
        return standard(A, B);

    size_t h = A.n / 2;

    auto A11 = view(A, 0, 0, h);
    auto A12 = view(A, 0, h, h);
    auto A21 = view(A, h, 0, h);
    auto A22 = view(A, h, h, h);

    auto B11 = view(B, 0, 0, h);
    auto B12 = view(B, 0, h, h);
    auto B21 = view(B, h, 0, h);
    auto B22 = view(B, h, h, h);

    auto M1 = strassen(add(A11, A22), add(B11, B22));
    auto M2 = strassen(add(A21, A22), B11);
    auto M3 = strassen(A11, sub(B12, B22));
    auto M4 = strassen(A22, sub(B21, B11));
    auto M5 = strassen(add(A11, A12), B22);
    auto M6 = strassen(sub(A21, A11), add(B11, B12));
    auto M7 = strassen(sub(A12, A22), add(B21, B22));

    Matrix C(A.n);
    for (size_t i = 0; i < h; ++i)
        for (size_t j = 0; j < h; ++j) {
            C(i, j) = M1(i, j) + M4(i, j) - M5(i, j) + M7(i, j);
            C(i, j + h) = M3(i, j) + M5(i, j);
            C(i + h, j) = M2(i, j) + M4(i, j);
            C(i + h, j + h) = M1(i, j) - M2(i, j) + M3(i, j) + M6(i, j);
        }

    return C;
}


bool approx_equal(const Matrix& A, const Matrix& B) {
    constexpr double EPS = 1e-8;
    for (size_t i = 0; i < A.data.size(); ++i) {
        double diff = std::abs(A.data[i] - B.data[i]);
        double scale = std::max({1.0, std::abs(A.data[i]), std::abs(B.data[i])});
        if (diff > EPS * scale)
            return false;
    }
    return true;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: program <n> [test]\n";
        return 0;
    }

    size_t n = std::stoul(argv[1]);
    bool test = argc > 2 && std::string(argv[2]) == "test";

    auto A = random_matrix(n);
    auto B = random_matrix(n);

    auto [C1, t1] = measure([&]() { return standard(A, B); });
    auto [C2, t2] = measure([&]() { return dc_mul(A, B); });
    auto [C3, t3] = measure([&]() { return strassen(A, B); });

    if (test) {
        std::cout << t1 << " " << t2 << " " << t3 << "\n";
        return 0;
    }

    std::cout << "Correctness:\n";
    std::cout << "DC: " << (approx_equal(C1, C2) ? "OK" : "ERROR") << "\n";
    std::cout << "Strassen: " << (approx_equal(C1, C3) ? "OK" : "ERROR") << "\n\n";

    std::cout << "Times (ms):\n";
    std::cout << "Standard: " << t1 << "\n";
    std::cout << "DC: " << t2 << "\n";
    std::cout << "Strassen: " << t3 << "\n";
}