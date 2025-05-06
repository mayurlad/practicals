// preorder dfs

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void dfs(const vector<vector<int>>& tree, vector<bool>& visited, int node) {
    visited[node] = true;
    cout << node << " ";

    #pragma omp taskgroup
    for (int i = 0; i < tree[node].size(); ++i) {
        int child = tree[node][i];
        if (!visited[child]) {
            #pragma omp task
            dfs(tree, visited, child);
        }
    }
}

int main() {
    // Define a tree structure
    vector<vector<int>> tree = {{1, 2}, {3, 4}, {5, 6}, {}, {}, {}, {}};
    vector<bool> visited(tree.size(), false);

    #pragma omp parallel
    {
        #pragma omp single
        dfs(tree, visited, 0);
    }

    return 0;
}