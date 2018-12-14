#include "cpu-bridges-dfs.hpp"
#include "gpu-bridges-bfs.cuh"
#include "gpu-bridges-bfs-gunrock.cuh"
#include "gpu-bridges-cc.cuh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <map>

#include "graph.hpp"
#include "test-result.hpp"
#include "timer.hpp"

// Misc
const std::string _usage =
    "USAGE: ./runner TESTFILE [VARIANTS]\n\n"
    "Available variants:\n"
    "cpu - simple sequential DFS implementation\n"
    "gpu - simple parallel BFS implementation (naive)\n\n"
    "If no variants are given, all possible are run by default.\n";

// Configuration of implementations to choose
typedef TestResult (*BridgesFunction)(Graph const &);

std::map<std::string, BridgesFunction> algorithms = {
    {"cpu-dfs", &sequential_dfs}, {"gpu-bfs", &parallel_bfs_naive},
    {"gpu-bfs-gunrock", &parallel_bfs_gunrock},
    {"gpu-bfs2", &parallel_bfs_naive},
    {"gpu-cc", &parallel_cc}};

// Main func
int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    if (argc < 2) {
        std::cerr << _usage << std::endl;
        exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    if (argc == 3 ) {
        // Take all algorithms if [VARIANTS] list is empty
        args.resize(2 + algorithms.size());
        int it = 2;
        for (auto &alg : algorithms) {
            args[it++] = alg.first;
        }
    }

    Graph const input_graph = Graph::read_from_file(argv[1]);
    Graph const input_graph2 = Graph::read_from_file(argv[2]);
    // Graph input_graph = Graph::read_from_stdin();
    // Graph input_graph = Graph::read_from_stdin();
    std::cout << "%%% File: " << args[1] << std::endl;
    std::cout << "%%% N: " << input_graph.get_N() << std::endl;
    std::cout << "%%% M: " << input_graph.get_M() << std::endl; 

    std::cout << "%%% File: " << args[2] << std::endl;
    std::cout << "%%% N: " << input_graph2.get_N() << std::endl;
    std::cout << "%%% M: " << input_graph2.get_M() << std::endl; 
    // auto xd = input_graph.get_Edges(); 
    // for (auto e : xd) {
    //     std::cout << e.first << " " << e.second << std::endl;
    // }
    // parallel_cc(input_graph);
    // return 0;

    TestResult previous_result(input_graph.get_M());

    for (int i = 2; i < args.size(); ++i) {
        // Prepare
        std::string alg_name = args[i];
        // std::cerr << "=== " << alg_name << " ===" << std::endl;

        TestResult current(1);
        if(alg_name == "gpu-bfs-gunrock") {
            current = (*algorithms[alg_name])(input_graph2);
        }

        // Execute
        else {
            current = (*algorithms[alg_name])(input_graph);
        }
        
        // Validate
        // current.write_to_stdout();

        if (i >= 3) {
            //assert(std::equal(previous_result.data(),
                              //previous_result.data() + input_graph.get_M(),
                              //current.data()));
        }
        previous_result = current;
    }
    std::vector<short> result = previous_result.get_isBridge();
    std::cout << "%%% # bridges: " << std::count(result.begin(), result.end(), 1) << std::endl;
    return 0;
}