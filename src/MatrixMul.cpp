#include "MatrixMul.h"
#include <chrono>
#include <stdexcept>

extern "C" void gpu_multiply_raw(const double* Arow, const double* Brow, double* Crow, int N, float &gpu_ms); // implemented in .cu

arma::mat MatrixMul::cpu_multiply(const arma::mat& A, const arma::mat& B) {
    return A * B;
}

void MatrixMul::gpu_multiply_rowmajor(const arma::mat& A, const arma::mat& B, std::vector<double>& gpu_out, float &gpu_ms) {
    int N = (int)A.n_rows;
    if ((int)A.n_cols != N || (int)B.n_rows != N || (int)B.n_cols != N) throw std::runtime_error("Matrices must be square same size");

    // Armadillo is column-major: convert to row-major for our naive kernel
    std::vector<double> Arow(N * N), Brow(N * N);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            Arow[i*N + j] = A(i,j);
            Brow[i*N + j] = B(i,j);
        }
    }
    gpu_out.assign(N*N, 0.0);
    gpu_multiply_raw(Arow.data(), Brow.data(), gpu_out.data(), N, gpu_ms);
}

arma::mat MatrixMul::rowmajor_to_arma(const std::vector<double>& v, std::size_t N) {
    arma::mat M(N, N);
    for (size_t i=0;i<N;i++)
        for (size_t j=0;j<N;j++)
            M(i,j) = v[i*N + j];
    return M;
}
