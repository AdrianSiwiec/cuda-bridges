#ifndef GPU_BRIDGES_BFS_GUNROCK_CUH
#define GPU_BRIDGES_BFS_GUNROCK_CUH

class TestResult;
class Graph;

TestResult parallel_bfs_gunrock(Graph const&);

#endif  // GPU_BRIDGES_BFS_GUNROCK_CUH
