#include "MatrixMul.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: matrix_mul <N>  (square matrices NxN)\n";
        return 1;
    }
    int N = std::stoi(argv[1]);
    std::cout << "Matrix size: " << N << "x" << N << "\n";

    arma::arma_rng::set_seed_random();
    arma::mat A = arma::randu<arma::mat>(N,N);
    arma::mat B = arma::randu<arma::mat>(N,N);

    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    arma::mat Ccpu = MatrixMul::cpu_multiply(A,B);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU
    std::vector<double> Cgpu_flat;
    float gpu_ms = 0.0f;
    MatrixMul::gpu_multiply_rowmajor(A, B, Cgpu_flat, gpu_ms);
    arma::mat Cgpu = MatrixMul::rowmajor_to_arma(Cgpu_flat, N);

    // compare
    double frob_diff = arma::norm(Ccpu - Cgpu, "fro");
    double frob_cpu = arma::norm(Ccpu, "fro");
    double rel_err = frob_diff / (frob_cpu + 1e-16);

    std::cout << "CPU time (ms): " << cpu_ms << "\n";
    std::cout << "GPU kernel time (ms): " << gpu_ms << "\n";
    std::cout << "Frobenius norm of difference: " << frob_diff << "\n";
    std::cout << "Relative error: " << rel_err << "\n";

    return 0;
}
