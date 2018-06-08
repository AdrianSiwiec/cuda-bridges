#ifndef GPU_BFS_HPP
#define GPU_BFS_HPP

class TestResult;
class Graph;

TestResult parallel_bfs_naive(Graph const &);

#endif  // GPU_BFS_HPP