/*
Madhur Jaripatke
Roll No. 50
BE A Computer
RMDSSOE, Warje, Pune

Problem Statement: Write a CUDA Program for:
1. Addition of two large vectors
2. Matrix Multiplication using CUDA C
*/

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

// Function to initialize vectors with random values
void initialize_vector(vector<double>& vec, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        vec[i] = rand() % 100;  // Random values between 0 and 99
    }
}

// Sequential vector addition
void sequential_add(const vector<double>& a, const vector<double>& b, vector<double>& result, int size) {
    for(int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

// Parallel vector addition using OpenMP
void parallel_add(const vector<double>& a, const vector<double>& b, vector<double>& result, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

// Function to verify results
bool verify_results(const vector<double>& seq_result, const vector<double>& parallel_result, int size) {
    for(int i = 0; i < size; i++) {
        if(seq_result[i] != parallel_result[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    // Vector sizes to test
    const vector<int> sizes = {1000000, 10000000, 50000000};
    
    // Print header
    cout << "\nVector Addition using OpenMP" << endl;
    cout << "============================" << endl;
    cout << setw(12) << "Size" << setw(15) << "Sequential" 
         << setw(15) << "Parallel" << setw(15) << "Speedup" << endl;
    cout << string(57, '-') << endl;

    // Test different vector sizes
    for(int size : sizes) {
        // Create vectors
        vector<double> a(size), b(size);
        vector<double> seq_result(size), parallel_result(size);

        // Initialize vectors with random values
        initialize_vector(a, size);
        initialize_vector(b, size);

        // Sequential Addition
        auto start = high_resolution_clock::now();
        sequential_add(a, b, seq_result, size);
        auto stop = high_resolution_clock::now();
        auto seq_duration = duration_cast<milliseconds>(stop - start);

        // Parallel Addition
        start = high_resolution_clock::now();
        parallel_add(a, b, parallel_result, size);
        stop = high_resolution_clock::now();
        auto parallel_duration = duration_cast<milliseconds>(stop - start);

        // Calculate speedup
        double speedup = static_cast<double>(seq_duration.count()) / parallel_duration.count();

        // Verify results
        bool correct = verify_results(seq_result, parallel_result, size);

        // Print results
        cout << setw(12) << size 
             << setw(15) << seq_duration.count() 
             << setw(15) << parallel_duration.count()
             << setw(15) << fixed << setprecision(2) << speedup;
        
        if(!correct) {
            cout << "  [ERROR: Results don't match!]";
        }
        cout << endl;
    }

    return 0;
}