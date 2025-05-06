/*
Madhur Jaripatke
Roll No. 50
BE A Computer
RMDSSOE, Warje, Pune

Problem Statement: Design and implement Parallel Breadth First Search and Depth First Search based on existing 
algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS.
*/

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <unordered_set>

class Graph {
private:
    int V;  // number of vertices
    std::vector<std::vector<int>> adj;

public:
    Graph(int vertices) : V(vertices) {
        adj.resize(vertices);
    }

    void addEdge(int v1, int v2) {
        adj[v1].push_back(v2);
        adj[v2].push_back(v1);  // undirected graph
    }

    void sequentialBFS(int start) {
        std::vector<bool> visited(V, false);
        std::queue<int> q;

        visited[start] = true;
        q.push(start);

        std::cout << "Sequential BFS starting from vertex " << start << ": ";
        
        while (!q.empty()) {
            int vertex = q.front();
            q.pop();
            std::cout << vertex << " ";

            for (int neighbor : adj[vertex]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        std::cout << std::endl;
    }

    void parallelBFS(int start) {
        std::vector<bool> visited(V, false);
        std::queue<int> q;
        
        visited[start] = true;
        q.push(start);

        std::cout << "Parallel BFS starting from vertex " << start << ": ";

        while (!q.empty()) {
            int levelSize = q.size();
            std::vector<int> currentLevel;

            for (int i = 0; i < levelSize; i++) {
                currentLevel.push_back(q.front());
                q.pop();
            }

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < currentLevel.size(); i++) {
                int vertex = currentLevel[i];
                #pragma omp critical
                {
                    std::cout << vertex << " ";
                }

                for (int neighbor : adj[vertex]) {
                    bool wasNotVisited = false;
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            wasNotVisited = true;
                        }
                    }
                    if (wasNotVisited) {
                        #pragma omp critical
                        {
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
        std::cout << std::endl;
    }

    void sequentialDFS(int start) {
        std::vector<bool> visited(V, false);
        std::stack<int> s;

        s.push(start);
        std::cout << "Sequential DFS starting from vertex " << start << ": ";

        while (!s.empty()) {
            int vertex = s.top();
            s.pop();

            if (!visited[vertex]) {
                visited[vertex] = true;
                std::cout << vertex << " ";

                for (int i = adj[vertex].size() - 1; i >= 0; i--) {
                    int neighbor = adj[vertex][i];
                    if (!visited[neighbor]) {
                        s.push(neighbor);
                    }
                }
            }
        }
        std::cout << std::endl;
    }

    void parallelDFS(int start) {
        std::vector<bool> visited(V, false);
        std::stack<int> s;
        
        s.push(start);
        std::cout << "Parallel DFS starting from vertex " << start << ": ";

        while (!s.empty()) {
            std::vector<int> currentStack;
            #pragma omp critical
            {
                while (!s.empty() && currentStack.size() < omp_get_max_threads()) {
                    int vertex = s.top();
                    s.pop();
                    if (!visited[vertex]) {
                        currentStack.push_back(vertex);
                    }
                }
            }

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < currentStack.size(); i++) {
                int vertex = currentStack[i];
                
                #pragma omp critical
                {
                    if (!visited[vertex]) {
                        visited[vertex] = true;
                        std::cout << vertex << " ";
                    }
                }

                for (int j = adj[vertex].size() - 1; j >= 0; j--) {
                    int neighbor = adj[vertex][j];
                    bool wasNotVisited = false;
                    
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            wasNotVisited = true;
                        }
                    }

                    if (wasNotVisited) {
                        #pragma omp critical
                        {
                            s.push(neighbor);
                        }
                    }
                }
            }
        }
        std::cout << std::endl;
    }
};

int main() {
    Graph g(8);  // Create a graph with 8 vertices

    // Add edges to create a sample graph
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    g.addEdge(6, 7);

    // Set number of threads
    omp_set_num_threads(4);

    // Test sequential and parallel BFS
    g.sequentialBFS(0);
    g.parallelBFS(0);

    // Test sequential and parallel DFS
    g.sequentialDFS(0);
    g.parallelDFS(0);

    return 0;
}
