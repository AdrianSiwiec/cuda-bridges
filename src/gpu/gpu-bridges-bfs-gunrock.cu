#include <moderngpu/context.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <gunrock/gunrock.h>

#include <iostream>
#include <vector>
using namespace std;

#include "bfs-mgpu.cuh"
#include "gpu-bridges-bfs.cuh"
#include "gputils.cuh"
#include "graph.hpp"
#include "test-result.hpp"
#include "timer.hpp"

// Proper
void BridgesGunrock(int n, int m, mem_t<int>& nodes, mem_t<int>& edges_from,
             mem_t<int>& edges_to, mem_t<edge>& edges_undirected,
             mem_t<int>& distance, mem_t<short>& result, context_t& context) {
    // Prepare memory
    int* nodes_data = nodes.data();
    int* edges_from_data = edges_from.data();
    int* edges_to_data = edges_to.data();
    edge* edges_undirected_data = edges_undirected.data();
    int* distance_data = distance.data();
    short* result_data = result.data();

    mem_t<int> node_parent = mgpu::fill<int>(-1, n, context);
    mem_t<int> node_is_marked = mgpu::fill<int>(0, n, context);

    int* node_parent_data = node_parent.data();
    int* node_is_marked_data = node_is_marked.data();

    // Determine parents
    transform(
        [=] MGPU_DEVICE(int index) {
            int from = edges_from_data[index];
            int to = edges_to_data[index];

            if (distance_data[from] == distance_data[to] - 1) {
                node_parent_data[to] = from;
            }
        },
        m, context);

    // Mark nodes visited during traversal
    transform(
        [=] MGPU_DEVICE(int index) {
            int from = edges_from_data[index];
            int to = edges_to_data[index];

            // Check if its tree edge
            if (node_parent_data[to] == from || node_parent_data[from] == to) {
                return;
            }

            int higher = distance_data[to] < distance_data[from] ? to : from;
            int lower = higher == to ? from : to;
            int diff = distance_data[lower] - distance_data[higher];

            // Equalize heights
            while (diff--) {
                node_is_marked_data[lower] = 1;
                lower = node_parent_data[lower];
            }

            // Mark till LCA is found
            while (lower != higher) {
                node_is_marked_data[lower] = 1;
                lower = node_parent_data[lower];

                node_is_marked_data[higher] = 1;
                higher = node_parent_data[higher];
            }
        },
        m, context);

    // Fill result array
    transform(
        [=] MGPU_DEVICE(int index) {
            int to = edges_undirected_data[index].first - 1;
            int from = edges_undirected_data[index].second - 1;

            if (node_parent_data[to] == from && node_is_marked_data[to] == 0) {
                result_data[index] = 1;
            }
            if (node_parent_data[from] == to &&
                node_is_marked_data[from] == 0) {
                result_data[index] = 1;
            }
        },
        edges_undirected.size(), context);
}

TestResult parallel_bfs_gunrock(Graph const& graph) {
    standard_context_t context(false);   //true prints some additional info, for debug

    // Prepare memory
    int const n = graph.get_N();
    int const undirected_m = graph.get_M();
    int const directed_m = graph.get_M() * 2;

    mem_t<edge> dev_edges = to_mem(graph.get_Edges(), context);
    
    Timer timer("gpu-bfs-gunrock");
    
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
    // bfs_mgpu::ParallelBFS(n, directed_m, dev_nodes, dev_directed_edge_to, 0, dev_distance,
                // context);
    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graphi->num_nodes = n;
    graphi->num_edges = directed_m;

    auto fromDev_nodes = from_mem( dev_nodes);
    auto fromDev_directed_edge_to = from_mem( dev_directed_edge_to );
    graphi->row_offsets = fromDev_nodes.data();
    graphi->col_indices = fromDev_directed_edge_to.data();

    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_INT;         // attributes type
    int srcs[3] = {0,1,2};
    struct GRSetup *config = InitSetup(3, srcs); // gunrock configurations

    int* device = new int[1];
    //*device = 2;
    //config->device_list = device;
    //config->num_devices = 1;
    config->quiet = 1;
    config->num_iters = 1;
    config->enable_idempotence = true;

    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("Gunrock Mem Szacher-Macher");
    }

    gunrock_bfs( grapho, graphi, config, data_t );


    //dispatch_bfs( grapho, graphi, config, data_t, 0, 0 );

    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("GunrockProper BFS");
    }

    int *labels = (int*)malloc(sizeof(int) * graphi->num_nodes);
    labels = (int*)grapho->node_value1;

    vector<int> labelsVect( labels, labels + n );
    dev_distance = to_mem( labelsVect, context);


    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("Gunrock copy Result");
    }

    // Find bridges
    BridgesGunrock(n, directed_m, dev_nodes, dev_directed_edge_from,
            dev_directed_edge_to, dev_edges, dev_distance, dev_final, context);

    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("Find Bridges");
        timer.print_overall();
    }

    if (detailed_time) {
        mem_t<int> maxd(1, context);
        reduce(dev_distance.data(), dev_distance.size(), maxd.data(), mgpu::maximum_t<int>(), context);
        vector<int> maxd_host = from_mem(maxd);

        context.synchronize();
        timer.print_and_restart("Max distance: " + to_string(maxd_host.front()));
    }

    // Copy result to device and return
    return TestResult(from_mem(dev_final));
}
