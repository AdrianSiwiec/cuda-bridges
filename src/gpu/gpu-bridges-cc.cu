#include <moderngpu/context.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <iostream>
#include <utility>
using namespace std;

#include "cudaWeiJaJaListRank.h"

#include "gpu-bridges-cc.cuh"
#include "cc.cuh"

#include "graph.hpp"
#include "test-result.hpp"

typedef pair<int, int> edge;
typedef long long int ll;

mem_t<int> find_backward(mem_t<ll>& directed, context_t& context) {
    // TODO: sth better
    ll * directed_data = directed.data();

    mem_t<int> mine_edge_idx = mgpu::fill_function<int>(
        [=] MGPU_DEVICE(int index) {
            return index;
        },
        directed.size(), context
    );
    int * mine_edge_idx_data = mine_edge_idx.data();
    
    mem_t<ll> mine_edge_each_pair_sorted(directed.size(), context);
    ll * mine_edge_each_pair_sorted_data = mine_edge_each_pair_sorted.data();

    transform(
        [=] MGPU_DEVICE(int index) {
            ll packed = directed_data[index];
            
            int from = static_cast<int>(packed >> 32);
            int to = static_cast<int>(packed) & 0xFFFFFFFF;
            
            if (from < to) {
                mine_edge_each_pair_sorted_data[index] = packed;
                return;
            }

            ll new_packed = 0;
            new_packed = static_cast<ll>(to);
            new_packed <<= 32;
            new_packed += static_cast<ll>(from);

            mine_edge_each_pair_sorted_data[index] = new_packed;
        },
        directed.size(), context);

    mergesort(mine_edge_each_pair_sorted_data, mine_edge_idx_data, 
        directed.size(), mgpu::less_t<ll>(), context);
    
    mem_t<int> back_edge_idx = mem_t<int>(directed.size(), context);
    int * back_edge_idx_data = back_edge_idx.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index & 1) return;
            int mine_idx = mine_edge_idx_data[index];
            int pair_idx = mine_edge_idx_data[index+1];

            back_edge_idx_data[mine_idx] = pair_idx;
            back_edge_idx_data[pair_idx] = mine_idx;
        },
        directed.size(), context);
    return back_edge_idx;
}

mem_t<ll> make_directed(mem_t<ll>& undirected, context_t& context) {
    mem_t<ll> directed(undirected.size()*2, context);

    ll * directed_data = directed.data();
    ll * undirected_data = undirected.data();
    int undirected_m = undirected.size();
    transform(
        [=] MGPU_DEVICE(int index) {
            ll packed = undirected_data[index];
            
            int from = static_cast<int>(packed >> 32);
            int to = static_cast<int>(packed) & 0xFFFFFFFF;
            
            ll new_packed = 0;
            new_packed = static_cast<ll>(to);
            new_packed <<= 32;
            new_packed += static_cast<ll>(from);

            directed_data[index] = packed;
            directed_data[index+undirected_m] = new_packed;
        },
        undirected_m, context);

    mergesort(directed_data, directed.size(), mgpu::less_t<ll>(), context);

    return directed;
}

mem_t<int> count_succ(int const n, mem_t<ll>& directed, mem_t<int>& directed_backidx, context_t& context) {
    mem_t<int> first = mgpu::fill<int>(-1, n, context);
    mem_t<int> next = mgpu::fill<int>(-1, directed.size(), context);

    ll * directed_data = directed.data();
    int * first_data = first.data();
    int * next_data = next.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            ll edge = directed_data[index];
            int edge_u = static_cast<int>(edge >> 32);
            
            if (index == 0) {
                first_data[edge_u] = index;
                return;
            }

            ll prev = directed_data[index-1];
            int prev_x = static_cast<int>(prev >> 32);

            if (prev_x == edge_u) {
                next_data[index-1] = index;
            } else {
                first_data[edge_u] = index;
            }
        },
        directed.size(), context);

    mem_t<int> succ(directed.size(), context);
    int * succ_data = succ.data();
    int * directed_backidx_data = directed_backidx.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            int back_next = next_data[directed_backidx_data[index]];
            if (back_next != -1) {
                succ_data[index] = back_next;
            } else {
                int to = static_cast<int>(directed_data[index]) & 0xFFFFFFFF;
                succ_data[index] = first_data[to];
            }
        },
        directed.size(), context);    
    return succ;
}

#define dbg 1
template<typename T>
void print_device_mem(mem_t<T>& device_mem) {
    if (!dbg) return;
    cout << "= print <T>..." << endl;
    vector<T> tmp = from_mem(device_mem);
    for (auto x : tmp) {
        cout << x << endl;
    }
}

void print_device_mem(mem_t<ll>& device_mem) {
    if (!dbg) return;
    cout << "= print edge coded as ll..." << endl;
    vector<ll> tmp = from_mem(device_mem);
    for (auto xd : tmp) {
        ll t = xd;
        int x = (int)t & 0xFFFFFFFF;
        int y = (int)(t >> 32);
        cout << y << " " << x << endl;
    }
}

TestResult parallel_cc(Graph const& graph) {
    standard_context_t context(false);

    // Prepare memory
    int const n = graph.get_N();
    int const undirected_m = graph.get_M();
    int const directed_m = graph.get_M() * 2;

    mem_t<edge> device_edges = to_mem(graph.get_Edges(), context);
    mem_t<cc::edge> device_cc_graph(undirected_m, context);

    edge * device_edges_data = device_edges.data();
    cc::edge * device_cc_graph_data = device_cc_graph.data();

    // Store specific graph representation for cc algorithm
    transform(
        [=] MGPU_DEVICE(int index) {
            int from = device_edges_data[index].first - 1;
            int to = device_edges_data[index].second - 1;

            ll packed = 0;
            packed = static_cast<ll>(from);
            packed <<= 32;
            packed += static_cast<ll>(to);

            device_cc_graph_data[index].x = packed;
            device_cc_graph_data[index].tree = false;
        },
        undirected_m, context);
    
    // Use CC algorithm to find spanning tree
    mem_t<ll> device_tree_edges = cc_main(n, undirected_m, device_cc_graph_data, context);
    print_device_mem(device_tree_edges);

    // Create directed-spanning-tree edge list
    mem_t<int> device_tree_directed_edges_backidx;
    mem_t<ll> device_tree_directed_edges = make_directed(device_tree_edges, context);
    device_tree_directed_edges_backidx = find_backward(device_tree_directed_edges, context);
    print_device_mem(device_tree_directed_edges);
    print_device_mem(device_tree_directed_edges_backidx);

    // Count succ array
    mem_t<int> succ = count_succ(n, device_tree_directed_edges, device_tree_directed_edges_backidx, context);
    print_device_mem(succ);

    // List rank
    mem_t<int> rank(succ.size(), context);
    int * rank_data = rank.data();
    int * succ_data = succ.data();
    int root = 0;
    int head;
    int at_root = -1;
    dtoh(&head, succ_data + root, 1);
    htod(succ_data + root, &at_root, 1);

    cudaWeiJaJaListRank(rank.data(), succ.size(), head, succ_data, context);
    print_device_mem(rank);

    mem_t<ll> rank_ordered_edges(device_tree_directed_edges.size(), context);
    ll * rank_ordered_edges_data = rank_ordered_edges.data();
    ll * device_tree_directed_edges_data = device_tree_directed_edges.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            rank_ordered_edges_data[rank_data[index]] = device_tree_directed_edges_data[index];
        },
        succ.size(), context);
    print_device_mem(rank_ordered_edges);

    mem_t<int> rank_ordered_edges_backward = find_backward(rank_ordered_edges, context);
    print_device_mem(rank_ordered_edges_backward);
    
    // Count preorder
    mem_t<int> scan_params(succ.size(), context);
    int * scan_params_data = scan_params.data();
    int * rank_ordered_edges_backward_data = rank_ordered_edges_backward.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index < rank_ordered_edges_backward_data[index]) {
                scan_params_data[index] = 1;
            } else {
                scan_params_data[index] = 0;
            }
        },
        succ.size(), context);

    scan<scan_type_inc>(scan_params.data(), scan_params.size(), scan_params.data(), context);
    print_device_mem(scan_params);

    mem_t<int> preorder = mgpu::fill<int>(0, n, context);
    int * preorder_data = preorder.data();
    rank_ordered_edges_data = rank_ordered_edges.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index < rank_ordered_edges_backward_data[index]) {
                ll packed = rank_ordered_edges_data[index];
            
                int to = static_cast<int>(packed) & 0xFFFFFFFF;
                preorder_data[to] = scan_params_data[index];
            }
        },
        succ.size(), context);
    print_device_mem(preorder);

    // Count subtree size
    mem_t<int> subtree = mgpu::fill<int>(n, n, context);
    int * subtree_data = subtree.data();

    transform(
        [=] MGPU_DEVICE(int index) {
            if (index < rank_ordered_edges_backward_data[index]) {
                ll packed = rank_ordered_edges_data[index];
            
                int from = static_cast<int>(packed >> 32);
                int to = static_cast<int>(packed) & 0xFFFFFFFF;

                // parent[to] = from
                // czyli jestem w parze (parent[to], to)

                subtree_data[to] = (rank_ordered_edges_backward_data[index] - 1 - index) / 2 + 1;
            }
        },
        succ.size(), context);
    print_device_mem(subtree);

    // Change original vertex numeration
    mem_t<ll> final_edge_list(directed_m, context);
    ll * final_edge_list_data = final_edge_list.data();

    transform(
        [=] MGPU_DEVICE(int index) {
            ll packed = device_cc_graph_data[index].x;

            int from = static_cast<int>(packed >> 32);
            int to = static_cast<int>(packed) & 0xFFFFFFFF;

            from = preorder_data[from];
            to = preorder_data[to];

            packed = 0;
            packed = static_cast<ll>(from);
            packed <<= 32;
            packed += static_cast<ll>(to);

            final_edge_list_data[index] = packed;

            packed = 0;
            packed = static_cast<ll>(to);
            packed <<= 32;
            packed += static_cast<ll>(from);

            final_edge_list_data[index + undirected_m] = packed;
        },
        undirected_m, context);

    mergesort(final_edge_list.data(), final_edge_list.size(), mgpu::less_t<ll>(), context);
    print_device_mem(final_edge_list);

    // Find local min/max from outgoing edges for every vertex
    // Construct the compaction state with transform_compact.
    auto compact = transform_compact(final_edge_list.size(), context);

    // The upsweep determines which items to compact i.e. which edges belong to tree
    int stream_count = compact.upsweep([=]MGPU_DEVICE(int index) {
        if (index == 0) return true;
        ll packed = final_edge_list_data[index];
        ll prev_packed = final_edge_list_data[index-1];

        int from = static_cast<int>(packed >> 32);
        int prev_from = static_cast<int>(prev_packed >> 32);

        return from != prev_from;
    });

    // Compact the results into this buffer.
    mem_t<int> segments(stream_count, context);
    int* segments_data = segments.data();
    compact.downsweep([=]MGPU_DEVICE(int dest_index, int source_index) {
        segments_data[dest_index] = source_index;
    });
    print_device_mem(segments);

    // Reduce segments to achieve min/max
    mem_t<int> minima(n, context);
    int * minima_data = minima.data();
    
    transform_segreduce(
        [=] MGPU_DEVICE(int index) {
            ll packed = final_edge_list_data[index];
            return static_cast<int>(packed) & 0xFFFFFFFF;
        }, 
        final_edge_list.size(), segments.data(), 
        segments.size(), minima.data(), mgpu::minimum_t<int>(), n+1, 
        context);

    print_device_mem(minima);

    mem_t<int> maxima(n, context);
    int * maxima_data = maxima.data();
    
    transform_segreduce(
        [=] MGPU_DEVICE(int index) {
            ll packed = final_edge_list_data[index];
            return static_cast<int>(packed) & 0xFFFFFFFF;
        }, 
        final_edge_list.size(), segments.data(), 
        segments.size(), maxima.data(), mgpu::maximum_t<int>(), -1, 
        context);

    print_device_mem(maxima);

    // I N T E R V A L  T R E E to find min/max for each subtree

    return TestResult(0);
}