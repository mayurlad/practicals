#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

int main() {
    const size_t size = 1000; // 1000 elements for vectors, and size of the matrix

    // Initialize vectors A and B for addition
    vector<double> A(size, 1.0); // initialize all elements of A to 1.0
    vector<double> B(size, 2.0); // initialize all elements of B to 2.0
    vector<double> C(size, 0.0); // result vector for addition

    // Measure time for vector addition
    auto start = chrono::high_resolution_clock::now();

    // Parallel vector addition
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Time taken for vector addition: " << elapsed.count() << " seconds\n";
    cout << "Sample result of vector addition: C[0] = " << C[0] << ", C[" << size - 1 << "] = " << C[size - 1] << endl;

    // Now we will implement Matrix Multiplication

    const size_t N = 100;  // Size of the matrix (N x N)
    vector<vector<double>> matrixA(N, vector<double>(N, 1.0)); // initialize matrix A (NxN) with all elements = 1.0
    vector<vector<double>> matrixB(N, vector<double>(N, 2.0)); // initialize matrix B (NxN) with all elements = 2.0
    vector<vector<double>> matrixC(N, vector<double>(N, 0.0)); // result matrix C (NxN)

    // Measure time for matrix multiplication
    start = chrono::high_resolution_clock::now();

    // Parallel matrix multiplication
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    end = chrono::high_resolution_clock::now();
    elapsed = end - start;

    cout << "Time taken for matrix multiplication: " << elapsed.count() << " seconds\n";
    cout << "Sample result of matrix multiplication: matrixC[0][0] = " << matrixC[0][0] << ", matrixC[" << N-1 << "][" << N-1 << "] = " << matrixC[N-1][N-1] << endl;

    return 0;
}
