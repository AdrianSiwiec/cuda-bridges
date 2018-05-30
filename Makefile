CXX=g++
CXXFLAGS=-O2 -std=c++11 
LDFLAGS=

NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-arch sm_30 -O3 -std=c++11 --expt-extended-lambda -w

MGPU=3rdparty/moderngpu
MGPUFLAGS=-I $(MGPU)/src

all: cpu-bridges networkrepository-parser gpu-bfs-bridges dimacs-parser

cpu-bridges: src/cpu/cpu-bridges.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

networkrepository-parser: networkrepository-parser.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

gpu-bfs-bridges: src/gpu/gpu-bfs-bridges.cu
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $< -o $@

dimacs-parser: dimacs-parser.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf cpu-bridges networkrepository-parser gpu-bfs-bridges dimacs-parser