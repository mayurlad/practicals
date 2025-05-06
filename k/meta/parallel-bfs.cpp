#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

void bfs(const vector<vector<int>>& tree) {
    vector<bool> visited(tree.size(), false);
    queue<int> q;
    q.push(0);
    visited[0] = true;

    while (!q.empty()) {
        int levelSize = q.size();

        #pragma omp parallel for
        for (int i = 0; i < levelSize; ++i) {
            int node = q.front();
            q.pop();
            cout << node << " ";

            for (int j = 0; j < tree[node].size(); ++j) {
                int child = tree[node][j];
                if (!visited[child]) {
                    #pragma omp critical
                    {
                        q.push(child);
                        visited[child] = true;
                    }
                }
            }
        }
    }
}

int main() {
    vector<vector<int>> tree = {{1, 2}, {3, 4}, {5, 6}, {}, {}, {}, {}};
    bfs(tree);
    return 0;
}