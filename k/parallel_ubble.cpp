#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    vector<int> arr = {9, 5, 2, 7, 1, 8, 3, 6, 4, 0};
    int n = arr.size();

    for (int phase = 0; phase < n; ++phase) {
        #pragma omp parallel for
        for (int i = (phase % 2); i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1])
                swap(arr[i], arr[i + 1]);
        }
    }

    // Print result
    for (int val : arr)
        cout << val << " ";
    cout << endl;

    return 0;
}


//g++ -fopenmp -O2 parallel_bubble_sort.cpp -o bubble_sort
