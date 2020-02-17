#include "gpu-bridges-bfs.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map> 
#include <map>

#include "graph.hpp"
#include "timer.hpp"

using namespace std;

// Misc
const std::string _usage = "USAGE: ./runner BFS_INTUT GUNROCK_BFS_INPUT\n\n";

void magic2() {
    void *tmpArray;
    cudaMalloc((void **)&(tmpArray), sizeof(int) * 1);
    return; 
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    if (argc < 2) {
        std::cerr << _usage << std::endl;
        exit(1);
    }

    Graph const input_graph = Graph::read_from_file(argv[1]);
    Graph const input_graph_gunrock = Graph::read_from_file(argv[2]);

    cerr << "Running our bfs..." << endl;
    parallel_bfs_naive( input_graph );
    cerr << "Running our bfs second time..." << endl;
    parallel_bfs_naive( input_graph );
    cerr << "Running magic...\n";
    magic2();
    cerr << "After running magic, our bfs speeds up" << endl;
    parallel_bfs_naive( input_graph );

    return 0;
}