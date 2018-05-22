#include <bits/stdc++.h>

int visit_time;
std::vector< std::vector<int> > G;
std::vector< std::pair<int, int> > bridges;

struct node {
    int preorder, low;
    bool visited;

    node(int visit_time = 0) :
            preorder(visit_time),
            low(preorder),
            visited(visit_time > 0)
    {}
};

std::vector<node> nodes;

void dfs(int start, int parent) {
    nodes[start] = node(visit_time++);

    for (auto x : G[start]) {

        if (x == parent) continue;
        if (!nodes[x].visited) {
            dfs(x, start);
            
            nodes[start].low = std::min(nodes[start].low, nodes[x].low);
        }
        else {
            nodes[start].low = std::min(nodes[start].low, nodes[x].preorder);  // non dfs-tree edge
        }
    }
    
    if (nodes[start].low == nodes[start].preorder && parent != -1) {
        bridges.push_back(std::make_pair(std::min(start, parent), std::max(start, parent)));
    }
    return;
}

void prepare(int const & n) {
    G.resize(n+1);
    nodes.resize(n+1);
    return;
}

int main() {
    std::ios_base::sync_with_stdio(false);

    int n, m;
    std::cin >> n >> m;
    
    prepare(n);

    for (int i = 0; i < m; ++i) {
        int x, y;
        std::cin >> x >> y;

        G[x].push_back(y);
        G[y].push_back(x);
    }
    
    visit_time = 1;
    dfs(1, -1);

    std::sort(bridges.begin(), bridges.end());

    std::cout << "Found " << bridges.size() << " bridge(s):" << std::endl;
    for (auto const & b : bridges) {
        std::cout << b.first << " " << b.second << std::endl;
    }
    return 0;
}
