#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

int main() {
    vector<int> data = {9, 5, 2, 7, 1, 8, 3, 6, 4, 0};

    // Split data into two halves and sort in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        sort(data.begin(), data.begin() + data.size() / 2);

        #pragma omp section
        sort(data.begin() + data.size() / 2, data.end());
    }

    // Merge the two sorted halves
    inplace_merge(data.begin(), data.begin() + data.size() / 2, data.end());

    // Print result
    for (int val : data)
        cout << val << " ";
    cout << endl;

    return 0;
}

// g++ -fopenmp -O2 small_parallel_sort.cpp -o small_sort
