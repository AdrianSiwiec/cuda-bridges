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

#include "cc.cuh"
#include "gpu-bridges-cc.cuh"

#include "graph.hpp"
#include "test-result.hpp"

typedef pair<int, int> edge;
typedef long long int ll;

mem_t<int> find_backward(mem_t<ll>& directed, context_t& context) {
    // TODO: sth better
    ll* directed_data = directed.data();

    mem_t<int> mine_edge_idx = mgpu::fill_function<int>(
        [=] MGPU_DEVICE(int index) { return index; }, directed.size(), context);
    int* mine_edge_idx_data = mine_edge_idx.data();

    mem_t<ll> mine_edge_each_pair_sorted(directed.size(), context);
    ll* mine_edge_each_pair_sorted_data = mine_edge_each_pair_sorted.data();

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

    // mergesort(mine_edge_idx_data,
    //           directed.size(), [=] MGPU_DEVICE( int a, int b ) { return mine_edge_each_pair_sorted_data[a] < mine_edge_each_pair_sorted_data[b]; }, context);

    mem_t<int> back_edge_idx = mem_t<int>(directed.size(), context);
    int* back_edge_idx_data = back_edge_idx.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index & 1) return;
            int mine_idx = mine_edge_idx_data[index];
            int pair_idx = mine_edge_idx_data[index + 1];

            back_edge_idx_data[mine_idx] = pair_idx;
            back_edge_idx_data[pair_idx] = mine_idx;
        },
        directed.size(), context);
    return back_edge_idx;
}

mem_t<ll> make_directed(mem_t<ll>& undirected, context_t& context) {
    mem_t<ll> directed(undirected.size() * 2, context);

    ll* directed_data = directed.data();
    ll* undirected_data = undirected.data();
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
            directed_data[index + undirected_m] = new_packed;
        },
        undirected_m, context);

    mergesort(directed_data, directed.size(), mgpu::less_t<ll>(), context);

    return directed;
}

mem_t<int> count_succ(int const n, mem_t<ll>& directed, context_t& context) {
    mem_t<int> directed_backidx = find_backward(directed, context);

    mem_t<int> first = mgpu::fill<int>(-1, n, context);
    mem_t<int> next = mgpu::fill<int>(-1, directed.size(), context);

    ll* directed_data = directed.data();
    int* first_data = first.data();
    int* next_data = next.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            ll edge = directed_data[index];
            int edge_u = static_cast<int>(edge >> 32);

            if (index == 0) {
                first_data[edge_u] = index;
                return;
            }

            ll prev = directed_data[index - 1];
            int prev_x = static_cast<int>(prev >> 32);

            if (prev_x == edge_u) {
                next_data[index - 1] = index;
            } else {
                first_data[edge_u] = index;
            }
        },
        directed.size(), context);

    mem_t<int> succ(directed.size(), context);
    int* succ_data = succ.data();
    int* directed_backidx_data = directed_backidx.data();
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

template <typename op_t>
mem_t<int> segment_tree(mem_t<int>& init, op_t op, int init_leaf,
                        context_t& context) {
    int const n = init.size();
    // cout << "siz " << n << endl;
    int M = 1;
    while (M < init.size()) {
        M <<= 1;
    }
    mem_t<int> segtree(2 * M, context);
    dtod(segtree.data() + M, init.data(), init.size());

    int* segtree_data = segtree.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index >= M + n) {
                segtree_data[index] = init_leaf;
            }
        },
        2 * M, context);
    // cudaMemset(segtree.data() + M + init.size(), n, (M - init.size()) *
    // sizeof(int));

    int begin = M / 2;
    while (begin >= 1) {
        transform(
            [=] MGPU_DEVICE(int index) {
                if (index >= begin) {
                    segtree_data[index] = op(segtree_data[2 * index],
                                             segtree_data[2 * index + 1]);
                }
            },
            2 * begin, context);
        begin >>= 1;
    }

    return segtree;
}

#define dbg 0
template <typename T>
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

mem_t<ll> spanning_tree(int const n, mem_t<edge>& device_edges,
                        context_t& context) {
    mem_t<cc::edge> device_cc_graph(device_edges.size(), context);

    edge* device_edges_data = device_edges.data();
    cc::edge* device_cc_graph_data = device_cc_graph.data();

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
            // device_cc_graph_data[index].tree = false;
        },
        device_edges.size(), context);

    // Use CC algorithm to find spanning tree
    mem_t<ll> device_tree_edges =
        cc_main(n, device_edges.size(), device_cc_graph_data, context);
    print_device_mem(device_tree_edges);

    // Direct edges & return
    return make_directed(device_tree_edges, context);
}

mem_t<int> list_rank(int const n, mem_t<ll>& tree_edges_directed,
                     standard_context_t& context) {
    // Count succ array
    mem_t<int> succ = count_succ(n, tree_edges_directed, context);
    print_device_mem(succ);

    mem_t<int> rank(succ.size(), context);
    int* rank_data = rank.data();
    int* succ_data = succ.data();
    int root = 0;
    int head;
    int at_root = -1;
    dtoh(&head, succ_data + root, 1);
    htod(succ_data + root, &at_root, 1);

    cudaWeiJaJaListRank(rank.data(), succ.size(), head, succ_data, context);

    return rank;
}

mem_t<ll> order_by_rank(mem_t<ll>& device_tree_directed_edges, mem_t<int>& rank,
                        context_t& context) {
    int* rank_data = rank.data();
    mem_t<ll> rank_ordered_edges(device_tree_directed_edges.size(), context);
    ll* rank_ordered_edges_data = rank_ordered_edges.data();
    ll* device_tree_directed_edges_data = device_tree_directed_edges.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            rank_ordered_edges_data[rank_data[index]] =
                device_tree_directed_edges_data[index];
        },
        device_tree_directed_edges.size(), context);
    print_device_mem(rank_ordered_edges);

    return rank_ordered_edges;
}

mem_t<int> count_preorder(int const n, mem_t<ll>& rank_ordered_edges,
                          mem_t<int>& rank_ordered_edges_backward,
                          context_t& context) {
    ll* rank_ordered_edges_data = rank_ordered_edges.data();
    int* rank_ordered_edges_backward_data = rank_ordered_edges_backward.data();

    mem_t<int> scan_params(rank_ordered_edges.size(), context);
    int* scan_params_data = scan_params.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index < rank_ordered_edges_backward_data[index]) {
                scan_params_data[index] = 1;
            } else {
                scan_params_data[index] = 0;
            }
        },
        rank_ordered_edges.size(), context);

    scan<scan_type_inc>(scan_params.data(), scan_params.size(),
                        scan_params.data(), context);
    print_device_mem(scan_params);

    mem_t<int> preorder = mgpu::fill<int>(0, n, context);
    int* preorder_data = preorder.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index < rank_ordered_edges_backward_data[index]) {
                ll packed = rank_ordered_edges_data[index];

                int to = static_cast<int>(packed) & 0xFFFFFFFF;
                preorder_data[to] = scan_params_data[index];
            }
        },
        rank_ordered_edges.size(), context);
    print_device_mem(preorder);

    return preorder;
}

mem_t<ll> relabel_and_direct(mem_t<edge>& device_edges, mem_t<int>& preorder,
                             context_t& context) {
    int undirected_m = device_edges.size();

    edge* device_edges_data = device_edges.data();
    int* preorder_data = preorder.data();

    mem_t<ll> final_edge_list(2 * undirected_m, context);
    ll* final_edge_list_data = final_edge_list.data();

    transform(
        [=] MGPU_DEVICE(int index) {
            int from = device_edges_data[index].first - 1;
            int to = device_edges_data[index].second - 1;

            ll packed;
            // int from = static_cast<int>(packed >> 32);
            // int to = static_cast<int>(packed) & 0xFFFFFFFF;

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

    mergesort(final_edge_list.data(), final_edge_list.size(),
              mgpu::less_t<ll>(), context);

    return final_edge_list;
}

mem_t<int> count_segments(mem_t<ll>& final_edge_list, context_t& context) {
    ll* final_edge_list_data = final_edge_list.data();

    // Construct the compaction state with transform_compact.
    auto compact = transform_compact(final_edge_list.size(), context);

    // The upsweep determines which items to compact i.e. which edges belong to
    // tree
    int stream_count = compact.upsweep([=] MGPU_DEVICE(int index) {
        if (index == 0) return true;
        ll packed = final_edge_list_data[index];
        ll prev_packed = final_edge_list_data[index - 1];

        int from = static_cast<int>(packed >> 32);
        int prev_from = static_cast<int>(prev_packed >> 32);

        return from != prev_from;
    });

    // Compact the results into this buffer.
    mem_t<int> segments(stream_count, context);
    int* segments_data = segments.data();
    compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
        segments_data[dest_index] = source_index;
    });

    return segments;
}

template <typename op_t>
void reduce_outgoing(mem_t<int>& dest, mem_t<ll>& final_edge_list,
                     mem_t<int>& parent, mem_t<int>& segments, op_t op,
                     int op_init, context_t& context) {
    ll* final_edge_list_data = final_edge_list.data();
    int* parent_data = parent.data();
    transform_segreduce(
        [=] MGPU_DEVICE(int index) {
            ll packed = final_edge_list_data[index];

            int from = static_cast<int>(packed >> 32);
            int to = static_cast<int>(packed) & 0xFFFFFFFF;
            if (to == parent_data[from]) return (int)(op_init);
            return (int)(static_cast<int>(packed) & 0xFFFFFFFF);
        },
        final_edge_list.size(), segments.data(), segments.size(), dest.data(),
        op, op_init, context);
}

mem_t<int> execute_segtree_queries(int const n, mem_t<int>& segtree_min,
                                   mem_t<int>& segtree_max,
                                   mem_t<int>& preorder, mem_t<int>& subtree,
                                   context_t& context) {
    int const M = segtree_min.size() / 2;

    mem_t<int> is_bridge_end = mgpu::fill<int>(0, n, context);
    int* is_bridge_end_data = is_bridge_end.data();

    int* segtree_min_data = segtree_min.data();
    int* segtree_max_data = segtree_max.data();
    int* preorder_data = preorder.data();
    int* subtree_data = subtree.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            // index - vertex id in org graph
            // preorder[index] - preorder id in spanning tree
            // subtree[index] - size of the subtree of index

            int va = M + preorder_data[index],
                vb = M + preorder_data[index] + subtree_data[index] - 1;
            /* Skrajne przedziały do rozkładu. */
            int mini = segtree_min_data[va];
            mini = min(mini, segtree_min_data[vb]);
            // if (va != vb) wyn += w[vb];
            /* Spacer aż do momentu spotkania. */
            while (va / 2 != vb / 2) {
                if (va % 2 == 0)
                    mini =
                        min(mini, segtree_min_data[va + 1]); /* prawa bombka na
                                                                lewej ścieżce */
                if (vb % 2 == 1)
                    mini = min(mini,
                               segtree_min_data[vb - 1]); /* lewa bombka na
                                                             prawej ścieżce */
                va /= 2;
                vb /= 2;
            }

            va = M + preorder_data[index],
            vb = M + preorder_data[index] + subtree_data[index] - 1;
            /* Skrajne przedziały do rozkładu. */
            int maxi = segtree_max_data[va];
            maxi = max(maxi, segtree_max_data[vb]);
            // if (va != vb) wyn += w[vb];
            /* Spacer aż do momentu spotkania. */
            while (va / 2 != vb / 2) {
                if (va % 2 == 0)
                    maxi =
                        max(maxi, segtree_max_data[va + 1]); /* prawa bombka na
                                                                lewej ścieżce */
                if (vb % 2 == 1)
                    maxi = max(maxi,
                               segtree_max_data[vb - 1]); /* lewa bombka na
                                                             prawej ścieżce */
                va /= 2;
                vb /= 2;
            }
            bool wyskok = false;
            if (mini < preorder_data[index] ||
                maxi > preorder_data[index] + subtree_data[index] - 1) {
                wyskok = true;
            }
            if (!wyskok) {
                is_bridge_end_data[preorder_data[index]] = 1;
            }
            // printf("pre id: %d last id : %d minimum is : %d maximum is : %d
            // wyskok : %d\n",
            //     preorder_data[index], preorder_data[index] +
            //     subtree_data[index] - 1, mini, maxi, wyskok);
        },
        preorder.size(), context);

    return is_bridge_end;
}

mem_t<short> count_result(mem_t<edge>& device_edges, mem_t<int>& preorder,
                          mem_t<int>& parent, mem_t<int>& is_bridge_end,
                          context_t& context) {
    edge* device_edges_data = device_edges.data();
    int* preorder_data = preorder.data();
    int* parent_data = parent.data();
    int* is_bridge_end_data = is_bridge_end.data();

    mem_t<short> result = mgpu::fill<short>(0, device_edges.size(), context);
    short* result_data = result.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            int from = device_edges_data[index].first - 1;
            int to = device_edges_data[index].second - 1;

            from = preorder_data[from];
            to = preorder_data[to];

            if (is_bridge_end_data[to] && parent_data[to] == from) {
                result_data[index] = 1;
            }
            if (is_bridge_end_data[from] && parent_data[from] == to) {
                result_data[index] = 1;
            }
        },
        device_edges.size(), context);
    return result;
}

void fill_subtree_size_and_parent(mem_t<int>& subtree, mem_t<int>& parent,
                                  mem_t<int>& preorder,
                                  mem_t<ll>& tree_edges_directed,
                                  mem_t<int>& rank_ordered_edges_backward,
                                  context_t& context) {
    int* subtree_data = subtree.data();
    int* parent_data = parent.data();
    int* preorder_data = preorder.data();
    int* rank_ordered_edges_backward_data = rank_ordered_edges_backward.data();
    ll* rank_ordered_edges_data = tree_edges_directed.data();
    transform(
        [=] MGPU_DEVICE(int index) {
            if (index < rank_ordered_edges_backward_data[index]) {
                ll packed = rank_ordered_edges_data[index];

                int from = static_cast<int>(packed >> 32);
                int to = static_cast<int>(packed) & 0xFFFFFFFF;

                // parent[to] = from
                // czyli jestem w parze (parent[to], to)
                parent_data[preorder_data[to]] = preorder_data[from];

                subtree_data[to] =
                    (rank_ordered_edges_backward_data[index] - 1 - index) / 2 +
                    1;
            }
        },
        tree_edges_directed.size(), context);
}

TestResult parallel_cc(Graph const& graph) {
    standard_context_t context(false);

    // Prepare constants
    int const n = graph.get_N();
    int const undirected_m = graph.get_M();
    int const directed_m = graph.get_M() * 2;

    // Copy input graph to device mem
    mem_t<edge> all_edges_undirected = to_mem(graph.get_Edges(), context);

    // Find spanning tree & direct edges
    mem_t<ll> tree_edges_directed =
        spanning_tree(n, all_edges_undirected, context);
    print_device_mem(tree_edges_directed);

    // List rank
    mem_t<int> rank = list_rank(n, tree_edges_directed, context);
    print_device_mem(rank);

    // Rearrange tree edges using counted ranks
    tree_edges_directed = order_by_rank(tree_edges_directed, rank, context);
    print_device_mem(tree_edges_directed);

    // Find reverse-edge
    mem_t<int> tree_edges_backlink =
        find_backward(tree_edges_directed, context);
    print_device_mem(tree_edges_backlink);

    // Count preorder
    mem_t<int> preorder =
        count_preorder(n, tree_edges_directed, tree_edges_backlink, context);
    print_device_mem(preorder);

    // Count subtree size & parent
    mem_t<int> subtree = mgpu::fill<int>(n, n, context);
    mem_t<int> parent = mgpu::fill<int>(-1, n, context);
    fill_subtree_size_and_parent(subtree, parent, preorder, tree_edges_directed,
                                 tree_edges_backlink, context);
    print_device_mem(subtree);
    print_device_mem(parent);

    // Change original vertex numeration & direct edges
    mem_t<ll> all_edges_directed =
        relabel_and_direct(all_edges_undirected, preorder, context);
    print_device_mem(all_edges_directed);

    // Find local min/max from outgoing edges for every vertex
    mem_t<int> segments = count_segments(all_edges_directed, context);
    print_device_mem(segments);

    // Reduce segments to achieve min/max for each
    mem_t<int> minima(n, context);
    mem_t<int> maxima(n, context);
    reduce_outgoing(minima, all_edges_directed, parent, segments,
                    mgpu::minimum_t<int>(), n + 1, context);
    reduce_outgoing(maxima, all_edges_directed, parent, segments,
                    mgpu::maximum_t<int>(), -1, context);
    print_device_mem(minima);
    print_device_mem(maxima);

    // I N T E R V A L  T R E E to find min/max for each subtree
    mem_t<int> segtree_min = segment_tree(minima, mgpu::minimum_t<int>(),
                                          minima.size() + 1, context);
    print_device_mem(segtree_min);

    mem_t<int> segtree_max =
        segment_tree(maxima, mgpu::maximum_t<int>(), -1, context);
    print_device_mem(segtree_max);

    // sooo... time to mark bridges ends!
    mem_t<int> is_bridge_end = execute_segtree_queries(
        n, segtree_min, segtree_max, preorder, subtree, context);
    print_device_mem(is_bridge_end);

    // unbelievable, time to result!
    mem_t<short> result = count_result(all_edges_undirected, preorder, parent,
                                       is_bridge_end, context);
    print_device_mem(result);

    return TestResult(from_mem(result));
}
