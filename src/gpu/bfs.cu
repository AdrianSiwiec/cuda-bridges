#include <moderngpu/context.hxx>
#include <moderngpu/kernel_intervalmove.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
using namespace std;

#include "bfs.cuh"

__global__ void UpdateDistanceAndVisitedKernel(const int* __restrict__ frontier,
                                               int frontier_size, int d,
                                               int* distance, int* visited) {
    int from = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = from; i < frontier_size; i += step) {
        distance[frontier[i]] = d;
        atomicOr(visited + (frontier[i] >> 5), 1 << (frontier[i] & 31));
    }
}

__global__ void CalculateFrontierStartsAndDegreesKernel(
    const int* __restrict__ nodes, const int* __restrict__ frontier, int n,
    int* node_frontier_starts, int* node_frontier_degrees) {
    int from = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = from; i < n; i += step) {
        node_frontier_starts[i] = nodes[frontier[i]];
        node_frontier_degrees[i] = nodes[frontier[i] + 1] - nodes[frontier[i]];
    }
}

__global__ void AdvanceFrontierPhase1Kernel(
    const int* __restrict__ edge_frontier, int edge_frontier_size,
    const int* __restrict__ visited, int* parent, int* edge_frontier_success) {
    int from = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = from; i < edge_frontier_size; i += step) {
        int v = edge_frontier[i];
        int success =
            (((visited[v >> 5] >> (v & 31)) & 1) == 0 && parent[v] == -1) ? 1
                                                                          : 0;
        if (success) parent[edge_frontier[i]] = i;
        edge_frontier_success[i] = success;
    }
}

__global__ void AdvanceFrontierPhase2Kernel(
    const int* __restrict__ edge_frontier, int edge_frontier_size,
    const int* __restrict__ parent, int* edge_frontier_success) {
    int from = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = from; i < edge_frontier_size; i += step)
        if (edge_frontier_success[i] && parent[edge_frontier[i]] != i)
            edge_frontier_success[i] = 0;
}

void getMemInfo() {
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    cout << fr / 1e6 << " / " << tot / 1e6 << endl;
}

void ParallelBFS(int n, int m, mem_t<int>& nodes, mem_t<int>& edges, int source,
                 mem_t<int>& distance, context_t& context) {
    mem_t<int> visited = mgpu::fill<int>(0, (n + 31) / 32, context);
    mem_t<int> parent = mgpu::fill<int>(-1, n, context);
    mem_t<int> node_frontier(n, context);
    mem_t<int> node_frontier_starts(n, context);
    mem_t<int> node_frontier_degrees(n, context);
    mem_t<int> edge_frontier(m, context);
    mem_t<int> edge_frontier_success(m, context);

    htod(node_frontier.data(), &source, 1);

    vector<int> tmp_subarray;

    // getMemInfo();

    int node_frontier_size = 1;
    int edge_frontier_size = 0;
    for (int d = 0; node_frontier_size > 0; ++d) {
        UpdateDistanceAndVisitedKernel<<<128, 128, 0, context.stream()>>>(
            node_frontier.data(), node_frontier_size, d, distance.data(),
            visited.data());
        CalculateFrontierStartsAndDegreesKernel<<<128, 128, 0,
                                                  context.stream()>>>(
            nodes.data(), node_frontier.data(), node_frontier_size,
            node_frontier_starts.data(), node_frontier_degrees.data());

        // hacking a bit
        dtoh(tmp_subarray,
             node_frontier_degrees.data() + node_frontier_size - 1, 1);
        edge_frontier_size = tmp_subarray.front();

        scan<scan_type_exc>(node_frontier_degrees.data(), node_frontier_size,
                            node_frontier_degrees.data(), context);

        dtoh(tmp_subarray,
             node_frontier_degrees.data() + node_frontier_size - 1, 1);
        edge_frontier_size += tmp_subarray.front();

        interval_gather(edges.data(), edge_frontier_size,
                        node_frontier_degrees.data(), node_frontier_size,
                        node_frontier_starts.data(), edge_frontier.data(),
                        context);
        AdvanceFrontierPhase1Kernel<<<128, 128, 0, context.stream()>>>(
            edge_frontier.data(), edge_frontier_size, visited.data(),
            parent.data(), edge_frontier_success.data());
        AdvanceFrontierPhase2Kernel<<<128, 128, 0, context.stream()>>>(
            edge_frontier.data(), edge_frontier_size, parent.data(),
            edge_frontier_success.data());

        // hacking again
        dtoh(tmp_subarray,
             edge_frontier_success.data() + edge_frontier_size - 1, 1);
        node_frontier_size = tmp_subarray.front();

        scan<scan_type_exc>(edge_frontier_success.data(), edge_frontier_size,
                            edge_frontier_success.data(), context);

        dtoh(tmp_subarray,
             edge_frontier_success.data() + edge_frontier_size - 1, 1);
        node_frontier_size += tmp_subarray.front();

        interval_expand(edge_frontier.data(), node_frontier_size,
                        edge_frontier_success.data(), edge_frontier_size,
                        node_frontier.data(), context);
    }
}
