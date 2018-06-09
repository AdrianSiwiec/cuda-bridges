#ifndef BFS_CUH
#define BFS_CUH

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

void ParallelBFS(int n, int m, mem_t<int>& nodes, mem_t<int>& edges, int source,
                 mem_t<int>& distance, context_t& context);

#endif  // BFS_CUH
