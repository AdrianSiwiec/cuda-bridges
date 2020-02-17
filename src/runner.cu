#include <algorithm>
#include <iostream>
using namespace std;

#include <moderngpu/kernel_mergesort.hxx>
using namespace mgpu;

#include "bfs-mgpu.cuh"
#include "gputils.cuh"
#include "graph.hpp"
#include "timer.hpp"


// Misc
const std::string _usage = "USAGE: ./runner BFS_INTUT\n\n";

void magic()
{
    void *tmpPtr;
    cudaMalloc((void **)(&tmpPtr), sizeof(int));
}

void parallel_bfs_test(Graph const &graph)
{
    standard_context_t context(false);

    // Preprocessing
    int const n = graph.get_N();
    int const undirected_m = graph.get_M();
    int const directed_m = graph.get_M() * 2;

    mem_t<edge> dev_edges = to_mem(graph.get_Edges(), context);

    Timer timer("gpu-bfs");

    mem_t<int> dev_directed_edge_from(directed_m, context);
    mem_t<int> dev_directed_edge_to(directed_m, context);
    mem_t<int> dev_nodes(n + 1, context);
    mem_t<int> dev_distance1 = mgpu::fill<int>(-1, n, context);
    mem_t<int> dev_distance2 = mgpu::fill<int>(-1, n, context);
    mem_t<int> dev_distance3 = mgpu::fill<int>(-1, n, context);
    mem_t<short> dev_final = mgpu::fill<short>(0, undirected_m, context);

    edge *dev_edges_data = dev_edges.data();
    int *dev_directed_edge_from_data = dev_directed_edge_from.data();
    int *dev_directed_edge_to_data = dev_directed_edge_to.data();
    int *dev_nodes_data = dev_nodes.data();

    transform(
        [=] MGPU_DEVICE(int index) {
            int from = dev_edges_data[index].first - 1;
            int to = dev_edges_data[index].second - 1;

            dev_directed_edge_from_data[index] = from;
            dev_directed_edge_from_data[index + undirected_m] = to;

            dev_directed_edge_to_data[index] = to;
            dev_directed_edge_to_data[index + undirected_m] = from;
        },
        undirected_m, context);

    mergesort(dev_directed_edge_from_data, dev_directed_edge_to_data,
              directed_m, mgpu::less_t<int>(), context);

    transform(
        [=] MGPU_DEVICE(int index) {
            int my_num = dev_directed_edge_from_data[index];
            if (index == directed_m - 1)
            {
                dev_nodes_data[my_num + 1] = index + 1;
                return;
            }
            int next_num = dev_directed_edge_from_data[index + 1];
            if (my_num != next_num)
            {
                dev_nodes_data[my_num + 1] = index + 1;
            }
            if (index == 0)
            {
                dev_nodes_data[0] = 0;
            }
        },
        directed_m, context);

    context.synchronize();
    timer.print_and_restart("Preprocessing");

    // Proper part
    // First - run BFS two times, to see there are no speed ups
    bfs_mgpu::ParallelBFS(n, directed_m, dev_nodes, dev_directed_edge_to, 0, dev_distance1,
                          context);
    context.synchronize();
    timer.print_and_restart("BFS first");

    bfs_mgpu::ParallelBFS(n, directed_m, dev_nodes, dev_directed_edge_to, 0, dev_distance2,
                          context);
    context.synchronize();
    timer.print_and_restart("BFS second");

    cerr << "Running magic" << endl;
    magic();

    // After dummy alloc, BFS speeds up.
    bfs_mgpu::ParallelBFS(n, directed_m, dev_nodes, dev_directed_edge_to, 0, dev_distance3,
                          context);
    context.synchronize();
    timer.print_and_restart("BFS speeds up");
}

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);

    if (argc < 1)
    {
        std::cerr << _usage << std::endl;
        exit(1);
    }

    Graph const input_graph = Graph::read_from_file(argv[1]);
    parallel_bfs_test(input_graph);

    return 0;
}
