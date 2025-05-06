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

// Function to initialize matrix with random values
void initialize_matrix(vector<vector<double>>& matrix, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 100;  // Random values between 0 and 99
        }
    }
}

// Sequential matrix multiplication
void sequential_multiply(const vector<vector<double>>& a, 
                       const vector<vector<double>>& b,
                       vector<vector<double>>& result, 
                       int m, int n, int p) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0;
            for(int k = 0; k < n; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP
void parallel_multiply(const vector<vector<double>>& a, 
                      const vector<vector<double>>& b,
                      vector<vector<double>>& result, 
                      int m, int n, int p) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            result[i][j] = 0;
            for(int k = 0; k < n; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Function to verify results
bool verify_results(const vector<vector<double>>& seq_result, 
                   const vector<vector<double>>& parallel_result, 
                   int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(abs(seq_result[i][j] - parallel_result[i][j]) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Matrix dimensions to test
    vector<tuple<int, int, int>> dimensions = {
        {100, 100, 100},    // Small matrices
        {500, 500, 500},    // Medium matrices
        {1000, 1000, 1000}  // Large matrices
    };
    
    // Print header
    cout << "\nMatrix Multiplication using OpenMP" << endl;
    cout << "=================================" << endl;
    cout << setw(25) << "Dimensions" 
         << setw(15) << "Sequential" 
         << setw(15) << "Parallel" 
         << setw(15) << "Speedup" << endl;
    cout << string(70, '-') << endl;

    // Test different matrix sizes
    for(const auto& [m, n, p] : dimensions) {
        // Create matrices
        vector<vector<double>> a(m, vector<double>(n));
        vector<vector<double>> b(n, vector<double>(p));
        vector<vector<double>> seq_result(m, vector<double>(p));
        vector<vector<double>> parallel_result(m, vector<double>(p));

        // Initialize matrices with random values
        initialize_matrix(a, m, n);
        initialize_matrix(b, n, p);

        // Sequential Multiplication
        auto start = high_resolution_clock::now();
        sequential_multiply(a, b, seq_result, m, n, p);
        auto stop = high_resolution_clock::now();
        auto seq_duration = duration_cast<milliseconds>(stop - start);

        // Parallel Multiplication
        start = high_resolution_clock::now();
        parallel_multiply(a, b, parallel_result, m, n, p);
        stop = high_resolution_clock::now();
        auto parallel_duration = duration_cast<milliseconds>(stop - start);

        // Calculate speedup
        double speedup = static_cast<double>(seq_duration.count()) / parallel_duration.count();

        // Verify results
        bool correct = verify_results(seq_result, parallel_result, m, p);

        // Print results
        cout << setw(8) << m << "x" << n << "x" << p
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