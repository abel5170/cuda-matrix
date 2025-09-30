#include <gtest/gtest.h>
#include "MatrixMul.h"
#include <iostream>
#include <chrono>

TEST(MatrixMul, CPUvsGPU) {
    int N = 256;  // choose smaller N for unit test
    arma::arma_rng::set_seed(12345);
    arma::mat A = arma::randu<arma::mat>(N,N);
    arma::mat B = arma::randu<arma::mat>(N,N);

    auto t0 = std::chrono::high_resolution_clock::now();
    arma::mat Ccpu = MatrixMul::cpu_multiply(A,B);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1-t0).count();

    std::vector<double> Cgpu_flat;
    float gpu_ms = 0.0f;
    MatrixMul::gpu_multiply_rowmajor(A,B,Cgpu_flat,gpu_ms);
    arma::mat Cgpu = MatrixMul::rowmajor_to_arma(Cgpu_flat, N);

    double frob_diff = arma::norm(Ccpu - Cgpu, "fro");
    double frob_cpu = arma::norm(Ccpu, "fro");
    double rel_err = frob_diff / (frob_cpu + 1e-16);

    std::cout << "UNIT TEST - N="<<N<<": CPU(ms)="<<cpu_ms<<", GPU(ms)="<<gpu_ms<<", rel_err="<<rel_err << std::endl;

    // tolerances: for doubles, relative error should be tiny (1e-8..1e-6) for small N,
    // but numerical differences + naive kernel may produce slightly larger—choose safe tolerance
    EXPECT_LT(rel_err, 1e-6);
}
