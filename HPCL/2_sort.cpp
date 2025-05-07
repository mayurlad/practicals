#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

int main() {
    vector<int> arr = {9, 5, 2, 7, 1, 8, 3, 6, 4, 0};
    int n = arr.size();

    // First sorting technique: Odd-Even Sort (parallelized)
    for (int phase = 0; phase < n; ++phase) {
        #pragma omp parallel for
        for (int i = (phase % 2); i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1])
                swap(arr[i], arr[i + 1]);
        }
    }

    // Print result after Odd-Even Sort
    cout << "After Odd-Even Sort: ";
    for (int val : arr)
        cout << val << " ";
    cout << endl;

    // Second sorting technique: Merge Sort (parallelized)
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

    // Print result after Merge Sort
    cout << "After Merge Sort: ";
    for (int val : data)
        cout << val << " ";
    cout << endl;

    return 0;
}
