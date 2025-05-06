#include <iostream>
#include <omp.h>
#include <vector>
#include <limits>

using namespace std;

int main() {
    // Example array
    vector<int> data = {12, 7, 9, 3, 15, 21, 5, 18};

    // Variables for Min, Max, Sum
    int min_val = numeric_limits<int>::max();
    int max_val = numeric_limits<int>::min();
    int sum_val = 0;

    // Parallel reduction with OpenMP
    #pragma omp parallel
    {
        #pragma omp for reduction(min:min_val) reduction(max:max_val) reduction(+:sum_val)
        for (int i = 0; i < data.size(); i++) {
            min_val = min(min_val, data[i]);
            max_val = max(max_val, data[i]);
            sum_val += data[i];
        }
    }

    // Calculate average
    double average = static_cast<double>(sum_val) / data.size();

    // Output the results
    cout << "Min: " << min_val << endl;
    cout << "Max: " << max_val << endl;
    cout << "Sum: " << sum_val << endl;
    cout << "Average: " << average << endl;

    return 0;
}
