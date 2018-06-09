#include "cpu-bridges-dfs.hpp"
#include "gpu-bridges-bfs.cuh"

#include <cassert>
#include <iostream>
#include <unordered_map>

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

std::unordered_map<std::string, BridgesFunction> algorithms = {
    {"cpu", &sequential_dfs}, {"gpu", &parallel_bfs_naive}};

// Main func
int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    if (argc < 2) {
        std::cerr << _usage << std::endl;
        exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    if (argc == 2) {
        // Take all algorithms if [VARIANTS] list is empty
        args.resize(2 + algorithms.size());
        int it = 2;
        for (auto &alg : algorithms) {
            args[it++] = alg.first;
        }
    }

    Graph const input_graph = Graph::read_from_file(argv[1]);
    // Graph input_graph = Graph::read_from_stdin();
    // std::cout << input_graph.get_N() << " " << input_graph.get_M() <<
    // std::endl; auto xd = input_graph.get_Edges(); for (auto e : xd) {
    //     std::cout << e.first << " " << e.second << std::endl;
    // }

    TestResult previous_result(input_graph.get_M());

    for (int i = 2; i < args.size(); ++i) {
        // Prepare
        std::string alg_name = args[i];
        std::cerr << "=== " << alg_name << " ===" << std::endl;

        Timer timer(alg_name);

        // Execute
        timer.start();
        TestResult current = (*algorithms[alg_name])(input_graph);
        timer.stop();

        // Validate
        timer.print_info();

        // current.write_to_stdout();

        if (i >= 3) {
            assert(std::equal(previous_result.data(),
                              previous_result.data() + input_graph.get_M(),
                              current.data()));
        }
        previous_result = current;
    }
    return 0;
}
