#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;

struct Node {
    int val;
    Node *left, *right;
    Node(int v) : val(v), left(nullptr), right(nullptr) {}
};

void parallelBFS(Node* root) {
    queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        int n = q.size();
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Node* curr;
            #pragma omp critical
            {
                curr = q.front(); q.pop();
            }
            cout << curr->val << " ";
            if (curr->left) {
                #pragma omp critical
                q.push(curr->left);
            }
            if (curr->right) {
                #pragma omp critical
                q.push(curr->right);
            }
        }
    }
}

void parallelDFS(Node* root) {
    if (!root) return;
    #pragma omp parallel sections
    {
        #pragma omp section
        parallelDFS(root->left);
        #pragma omp section
        parallelDFS(root->right);
    }
    #pragma omp critical
    cout << root->val << " ";
}

int main() {
    Node* root = new Node(1);
    root->left = new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);
    root->right->left = new Node(6);
    root->right->right = new Node(7);

    cout << "Parallel BFS: ";
    parallelBFS(root);
    cout << "\nParallel DFS: ";
    parallelDFS(root);
    cout << endl;
}
