#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;  // Avoid redundancy of 'std::'

int main() {
    const size_t size = 100000000; // 100 million elements

    vector<double> A(size, 1.0); // initialize all elements to 1.0
    vector<double> B(size, 2.0); // initialize all elements to 2.0
    vector<double> C(size, 0.0); // result vector

    auto start = chrono::high_resolution_clock::now();

    // Parallel vector addition
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Time taken for vector addition: " << elapsed.count() << " seconds\n";
    cout << "Sample result: C[0] = " << C[0] << ", C[" << size - 1 << "] = " << C[size - 1] << endl;

    return 0;
}


// To compile this code with OpenMP, use:

// bash
// Copy
// Edit
// g++ -fopenmp -O2 vector_addition.cpp -o vector_add
// 💡 Notes
// #pragma omp parallel for tells OpenMP to divide the loop iterations among threads.

// You can control the number of threads using OMP_NUM_THREADS:

// bash
// Copy
// Edit
// export OMP_NUM_THREADS=4
// ./vector_add