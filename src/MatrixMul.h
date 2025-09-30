#pragma once
#include <armadillo>
#include <vector>
#include <cstddef>

struct MatrixResult {
    arma::mat cpu;                // CPU result
    std::vector<double> gpu_out;  // GPU result in row-major flat vector
};

class MatrixMul {
public:
    // Generate two random matrices A,B (double), run CPU multiply
    static arma::mat cpu_multiply(const arma::mat& A, const arma::mat& B);

    // Convert Armadillo matrices to row-major vectors and call CUDA wrapper
    //gpu_out will be row-major N*N vector
    static void gpu_multiply_rowmajor(const arma::mat& A, const arma::mat& B,
                                      std::vector<double>& gpu_out, float &gpu_ms);

    // helper: convert row-major vector back to arma::mat
    static arma::mat rowmajor_to_arma(const std::vector<double>& v, std::size_t N);
};
