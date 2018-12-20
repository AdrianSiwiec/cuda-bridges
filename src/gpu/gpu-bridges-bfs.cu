#include <moderngpu/context.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <iostream>
#include <vector>
using namespace std;

#include "bfs-mgpu.cuh"
#include "gpu-bridges-bfs.cuh"
#include "gputils.cuh"
#include "graph.hpp"
#include "timer.hpp"

void parallel_bfs_naive(Graph const& graph) {
    standard_context_t context(false);
    
    // Prepare memory
    int const n = graph.get_N();
    int const undirected_m = graph.get_M();
    int const directed_m = graph.get_M() * 2;

    mem_t<edge> dev_edges = to_mem(graph.get_Edges(), context);
    
    Timer timer("gpu-bfs");
    
    mem_t<int> dev_directed_edge_from(directed_m, context);
    mem_t<int> dev_directed_edge_to(directed_m, context);
    mem_t<int> dev_nodes(n + 1, context);
    mem_t<int> dev_distance = mgpu::fill<int>(-1, n, context);
    mem_t<short> dev_final = mgpu::fill<short>(0, undirected_m, context);

    edge* dev_edges_data = dev_edges.data();
    int* dev_directed_edge_from_data = dev_directed_edge_from.data();
    int* dev_directed_edge_to_data = dev_directed_edge_to.data();
    int* dev_nodes_data = dev_nodes.data();

    // if (detailed_time) {
    //     context.synchronize();
    //     timer.print_and_restart("init memory");
    // }

    // Fill _directed_ arrays
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

    // Sort them for BFS purposes
    mergesort(dev_directed_edge_from_data, dev_directed_edge_to_data,
              directed_m, mgpu::less_t<int>(), context);

    // Fill nodes array for BFS
    transform(
        [=] MGPU_DEVICE(int index) {
            int my_num = dev_directed_edge_from_data[index];
            if (index == directed_m - 1) {
                dev_nodes_data[my_num + 1] = index + 1;
                return;
            }
            int next_num = dev_directed_edge_from_data[index + 1];
            if (my_num != next_num) {
                dev_nodes_data[my_num + 1] = index + 1;
            }
            if (index == 0) {
                dev_nodes_data[0] = 0;
            }
        },
        directed_m, context);
    
    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("Preprocessing");
    }

    // Proper part
    // Execute BFS to compute distances (needed for determine parents)
    bfs_mgpu::ParallelBFS(n, directed_m, dev_nodes, dev_directed_edge_to, 0, dev_distance,
                context);

    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("BFS");
    }
}
