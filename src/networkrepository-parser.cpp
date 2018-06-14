#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
using namespace std;

typedef pair<int, int> pii;
std::vector<std::vector<int>> G;
std::vector<bool> visited;
std::vector<int> label;

void dfs(int start, int parent) {
    visited[start] = true;

    for (auto x : G[start]) {
        if (x == parent) continue;
        if (!visited[x]) {
            dfs(x, start);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " IN OUT" << endl;
        exit(1);
    }

    vector<pii> in_edges;

    ifstream in(argv[1]);
    assert(in.is_open());

    string buf;
    getline(in, buf);
    while (buf[0] == '%') {
        getline(in, buf);
    }

    int n, m;
    sscanf(buf.c_str(), "%d %d %d", &n, &n, &m);

    G.resize(n + 1);
    visited.resize(n + 1);
    label.resize(n + 1);

    for (int i = 0; i < m; ++i) {
        getline(in, buf);
        istringstream parser(buf);

        int a, b;
        parser >> a >> b;
        if (a == b) continue;
        G[a].push_back(b);
        G[b].push_back(a);
        in_edges.push_back(make_pair(min(a, b), max(a, b)));
    }

    dfs(1, -1);

    int id = 1;
    for (int i = 1; i <= n; ++i) {
        if (visited[i]) {
            label[i] = id++;
        }
    }

    set<pii> edges;
    for (auto &e : in_edges) {
        if (visited[e.first]) {
          edges.insert(make_pair(label[e.first], label[e.second]));
        }
    }

    vector<pii> out_edges(edges.begin(), edges.end());
    n = id-1;
    m = out_edges.size();

    // cout << n << " " << m << endl;
    // for (auto & e : out_edges) {
    //   cout << e.first << " " << e.second << endl;
    // }

    ofstream out(argv[2], ios::binary);
    assert(out.is_open());

    out.write((char *)&n, sizeof(int));
    out.write((char *)&m, sizeof(int));
    out.write((char *)out_edges.data(), out_edges.size() * sizeof(pii));
}