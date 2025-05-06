#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

int main() {
    const size_t size = 100000000; // 100 million elements

    vector<double> A(size, 1.5); // initialize all elements to 1.5
    vector<double> B(size, 2.5); // initialize all elements to 2.5
    vector<double> C(size, 0.0); // result vector

    auto start = chrono::high_resolution_clock::now();

    // Parallel element-wise vector multiplication
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] * B[i];
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Time taken for vector multiplication: " << elapsed.count() << " seconds\n";
    cout << "Sample result: C[0] = " << C[0] << ", C[" << size - 1 << "] = " << C[size - 1] << endl;

    return 0;
}


// g++ -fopenmp -O2 vector_multiplication.cpp -o vector_mul