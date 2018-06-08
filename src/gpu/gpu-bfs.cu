#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_intervalmove.hxx>
using namespace mgpu;

#include <queue>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <set>
using namespace std;

#include "gpu-bfs.hpp"
#include "graph.hpp"
#include "test_result.hpp"

__global__ void UpdateDistanceAndVisitedKernel(
    const int* __restrict__ frontier, int frontier_size, int d,
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
      const int* __restrict__ visited,
      int* parent, int* edge_frontier_success) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < edge_frontier_size; i += step) {
    int v = edge_frontier[i];
    int success = (((visited[v >> 5] >> (v & 31)) & 1) == 0 && parent[v] == -1) ? 1 : 0;
    if (success)
      parent[edge_frontier[i]] = i;
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

#define MGPU_MEM(int) mem_t<int>

void getMemInfo() {
  size_t fr, tot;
  cudaMemGetInfo(&fr, &tot);
  cout << fr/1e6 << " / " << tot/1e6 << endl;
}

void ParallelBFS(
    int n, int m, MGPU_MEM(int) & nodes, MGPU_MEM(int) & edges, int source,
    MGPU_MEM(int) & distance, context_t& context) {
  MGPU_MEM(int) visited = mgpu::fill<int>(0, (n + 31) / 32, context);
  MGPU_MEM(int) parent = mgpu::fill<int>(-1, n, context);
  MGPU_MEM(int) node_frontier(n, context);
  MGPU_MEM(int) node_frontier_starts(n, context);
  MGPU_MEM(int) node_frontier_degrees(n, context);
  MGPU_MEM(int) edge_frontier(m, context);
  MGPU_MEM(int) edge_frontier_success(m, context);

  htod(node_frontier.data(), &source, 1);

  vector<int> tmp_subarray;
  
  // getMemInfo();
  
  int node_frontier_size = 1;
  int edge_frontier_size = 0;
  for (int d = 0; node_frontier_size > 0; ++d) {
    UpdateDistanceAndVisitedKernel<<<128, 128, 0, context.stream()>>>(
        node_frontier.data(), node_frontier_size, d,
        distance.data(), visited.data());
    CalculateFrontierStartsAndDegreesKernel<<<128, 128, 0, context.stream()>>>(
        nodes.data(), node_frontier.data(), node_frontier_size,
        node_frontier_starts.data(), node_frontier_degrees.data());
    
    // hacking a bit
    dtoh(tmp_subarray, node_frontier_degrees.data() + node_frontier_size-1, 1);
    edge_frontier_size = tmp_subarray.front();    
    
    scan<scan_type_exc>(
      node_frontier_degrees.data(), node_frontier_size,
      node_frontier_degrees.data(), context);

    dtoh(tmp_subarray, node_frontier_degrees.data() + node_frontier_size-1, 1);
    edge_frontier_size += tmp_subarray.front();        
    
    interval_gather(
        edges.data(),
        edge_frontier_size, 
        node_frontier_degrees.data(),
        node_frontier_size, 
        node_frontier_starts.data(),
        edge_frontier.data(), 
        context
    );
    AdvanceFrontierPhase1Kernel<<<128, 128, 0, context.stream()>>>(
        edge_frontier.data(), edge_frontier_size, visited.data(),
        parent.data(), edge_frontier_success.data());
    AdvanceFrontierPhase2Kernel<<<128, 128, 0, context.stream()>>>(
        edge_frontier.data(), edge_frontier_size,
        parent.data(), edge_frontier_success.data());

    // hacking again
    dtoh(tmp_subarray, edge_frontier_success.data() + edge_frontier_size-1, 1);
    node_frontier_size = tmp_subarray.front();
    
    scan<scan_type_exc>(
      edge_frontier_success.data(), edge_frontier_size,
      edge_frontier_success.data(), context);

    dtoh(tmp_subarray, edge_frontier_success.data() + edge_frontier_size-1, 1);
    node_frontier_size += tmp_subarray.front();

    interval_expand(
        edge_frontier.data(),
        node_frontier_size, 
        edge_frontier_success.data(),
        edge_frontier_size,
        node_frontier.data(), 
        context);
  }

  // vector<int> h_parent = from_mem(parent);
  // vector<int> h_dist = from_mem(distance);
  // for (int i = 0; i < h_parent.size(); ++i) {
  //   cout << "i=" << i << " parent[i]=" << h_parent[i] << " " << " distance[i]=" << h_dist[i] << " "; 
  //   // if (h_parent[i] != -1) cout << " parent[parent[i]]=" << h_parent[h_parent[i]]; 
  //   cout << endl;
  // }
  // cout << "n: " << n << " m: " << m << " s: " << source << endl;
}

// typedef unsigned long long uint64_t;

uint64_t CalculateChecksum(const vector<int>& distance) {
  uint64_t checksum = 0;
  for (int i = 0; i < distance.size(); ++i)
    if (distance[i] != -1)
      checksum += (uint64_t)i * (uint64_t)distance[i];
  return checksum;
}

uint64_t Time() {
  timespec tp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
  return (tp.tv_nsec + (uint64_t)1000000000 * tp.tv_sec) / 1000000;
}

template<typename T>
void cpyAndPrint(mem_t<T> & gpu_mem) {
  return;
  vector<T> dupa = from_mem(gpu_mem);
  for (T x : dupa) {
    cout << x << " ";
  }
  cout << endl;
}

void Bridges(
  int n, int m, MGPU_MEM(int) & nodes, MGPU_MEM(int) & edges, int source,
  MGPU_MEM(int) & distance, mem_t<int> & is_bridge, context_t& context) {

  MGPU_MEM(int) edge_parent = mgpu::fill<int>(0, m+1, context);

  cpyAndPrint(edge_parent);

  int* nd = nodes.data();
  int* ed = edges.data();
  int* epd = edge_parent.data();
  int* dd = distance.data();
  transform([=]MGPU_DEVICE(int index) {
    if (index == 0) return;
    epd[nd[index]] = 1;
  }, nodes.size(), context);

  cpyAndPrint(edge_parent);

  scan<scan_type_inc>(edge_parent.data(), edge_parent.size(), edge_parent.data(), context);


  cpyAndPrint(edge_parent);
  cpyAndPrint(edges);

  MGPU_MEM(int) node_parent_edge_up(n, context);
  MGPU_MEM(int) node_parent_edge_down(n, context);

  int* npeud = node_parent_edge_up.data();
  int* npedd = node_parent_edge_down.data();

  transform([=]MGPU_DEVICE(int index) {
    int mv = ed[index];
    int pv = epd[index];

    if (dd[pv] == dd[mv] - 1) {
      // parent candidate
      npeud[mv] = index;
    }
  }, edges.size(), context);

  cpyAndPrint(node_parent_edge_up);

  transform([=]MGPU_DEVICE(int index) {
    int mv = ed[index];
    int pv = epd[index];
    if (pv == source) return;

    int tree_pv_up = npeud[pv];

    if (epd[tree_pv_up] == mv) {
      // tree edge
      npedd[pv] = index;
    }
  }, edges.size(), context);

  cpyAndPrint(node_parent_edge_down);

  mem_t<int> is_tree_edge = mgpu::fill<int>(0, m, context);

  int* ited = is_tree_edge.data();
  transform([=]MGPU_DEVICE(int index) {
    if (index == source) return;
    ited[npeud[index]] = 1;
    ited[npedd[index]] = 1;
  }, n, context);
  
  cpyAndPrint(is_tree_edge);

  // mem_t<int> is_bridge = mgpu::fill<int>(1, m, context);

  int* ibd = is_bridge.data();
  transform([=]MGPU_DEVICE(int index) {
    if (ited[index]) return;
    int mv = ed[index];
    int pv = epd[index];

    // iter to root and mark
    int higher = dd[mv] < dd[pv] ? mv : pv;
    int lower = higher == mv ? pv : mv;

    int diff = dd[lower] - dd[higher];

    // printf("sprawdzam dla %d: m %d p %d h %d l %d d %d\n", index, mv, pv, higher, lower, diff);
    ibd[index] = 0;

    while (diff--) {
      int edge_up_id = npeud[lower];
      int edge_down_id = npedd[lower];
      ibd[edge_up_id] = 0;
      ibd[edge_down_id] = 0;
      lower = epd[edge_up_id];
    } 

    while (lower != higher) {
      int edge_up_id = npeud[lower];
      int edge_down_id = npedd[lower];
      ibd[edge_up_id] = 0;
      ibd[edge_down_id] = 0;
      lower = epd[edge_up_id];
      
      edge_up_id = npeud[higher];
      edge_down_id = npedd[higher];
      ibd[edge_up_id] = 0;
      ibd[edge_down_id] = 0;
      higher = epd[edge_up_id];
    }

  }, edges.size(), context);
  
  cpyAndPrint(is_bridge);


}

uint64_t ParallelBFS(
    const vector<int>& nodes, const vector<int>& edges, int source, TestResult & dest) {
  standard_context_t context(false);
  // cout << "source " << source << endl;
  MGPU_MEM(int) dev_nodes = to_mem(nodes, context);
  MGPU_MEM(int) dev_edges = to_mem(edges, context);
  MGPU_MEM(int) dev_distance = mgpu::fill<int>(-1, nodes.size() - 1, context);
  mem_t<int> dev_is_bridge = mgpu::fill<int>(1, edges.size(), context);
  // cout << "JEST OK" << endl;
  uint64_t t = Time();
  ParallelBFS(
      nodes.size() - 1, edges.size(), dev_nodes, dev_edges, source,
      dev_distance, context);
  // cout << "JEST OK" << endl;
  // MOSTY
  dev_edges = to_mem(edges, context);
  Bridges(
    nodes.size() - 1, edges.size(), dev_nodes, dev_edges, source,
    dev_distance, dev_is_bridge, context);
  // cout << "JEST OK" << endl;
  t = Time() - t;
  cerr << "GPU: " << t << " ms" << endl;
    
  vector<int> is_bridge = from_mem(dev_is_bridge);
  vector<short> dupa(is_bridge.begin(), is_bridge.end());
  dest = TestResult(dupa);
  // int bnum = 0;
  // for (int x : is_bridge) {
  //   if (x == 1) bnum++;
  // }
  // assert(!(bnum&1));
  // cout << "Source: " << source << " Bridges: " << bnum/2 << endl;

  vector<int> distance;
  dtoh(distance, dev_distance.data(), nodes.size() - 1);

  return CalculateChecksum(distance);
}

uint64_t SequentialBFS(
    const vector<int>& nodes, const vector<int>& edges, int source) {
  vector<int> distance(nodes.size() - 1, -1);
  uint64_t t = Time();
  distance[source] = 0;
  queue<int> q;
  q.push(source);
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int i = nodes[u]; i < nodes[u + 1]; ++i) {
      int v = edges[i];
      if (distance[v] == -1) {
        distance[v] = distance[u] + 1;
        q.push(v);
      }
    }
  }
  t = Time() - t;
  cerr << "CPU: " << t << " ms" << endl;
  return CalculateChecksum(distance);
}

std::set<std::pair<int, int>> directed_edges;

TestResult parallel_bfs_naive(Graph const& graph) {
  
  int n = graph.get_N(), m = graph.get_M() * 2;
  vector<int> nodes(n + 1, 0), edges(m);
  
  directed_edges.clear();
  auto const graph_edges = graph.get_Edges();

  for (int i = 0; i < m/2; ++i) {
    directed_edges.insert(std::make_pair(graph_edges[i].first-1, graph_edges[i].second-1));
    directed_edges.insert(std::make_pair(graph_edges[i].second-1, graph_edges[i].first-1));
  }
  assert(directed_edges.size() == m);

  int prev = -1;
  int curr = 0;
  int ite = 0;
  for (auto de : directed_edges) {
    if (de.first == prev) {
      nodes[curr]++;
    } else {
      curr++;
      nodes[curr] = nodes[curr-1];
      nodes[curr]++;
    }
    prev = de.first;
    edges[ite++] = de.second;
  }

  // cout << "TMP " << endl;
  // for (auto xd : nodes) {
  //   cout << xd << " ";

  // }
  // cout << endl;
  // for (auto xd : edges) {
  //   cout << xd << " ";

  // }
  // cout << endl;

  TestResult result(0);
  for (int i = 0; i < 1; ++i) {
    int source = 0;//rand() % n;
    uint64_t seqsum = SequentialBFS(nodes, edges, source);
    uint64_t parsum = ParallelBFS(nodes, edges, source, result);
    // cout << seqsum << " " << parsum << endl;
    assert(seqsum == parsum);
  }
  
  vector<short> final_res;
  int asd = 0;
  for (auto de : directed_edges) {
    if (de.first < de.second) {
      final_res.push_back(result[asd] == 1 ? 1 : 0);
    }
    asd++;
  }

  return TestResult(final_res);
}

// int main(int argc, char* argv[]) {
//   if (argc != 2) {
//     cerr << "Usage: " << argv[0] << " GRAPH" << endl;
//     exit(1);
//   }

//   ifstream in(argv[1], ios::binary);  
//   assert(in.is_open());
//   int n, m;
//   in.read((char*)&n, sizeof(int));
//   in.read((char*)&m, sizeof(int));
//   vector<int> nodes(n + 1), edges(m);
//   in.read((char*)nodes.data(), nodes.size() * sizeof(int));
//   in.read((char*)edges.data(), edges.size() * sizeof(int));

//   for (int i = 0; i < 5; ++i) {
//     int source = rand() % n;
//     uint64_t seqsum = SequentialBFS(nodes, edges, source);
//     uint64_t parsum = ParallelBFS(nodes, edges, source);
//     assert(seqsum == parsum);
//   }
// }
