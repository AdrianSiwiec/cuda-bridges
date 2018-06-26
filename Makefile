CXX=g++
CXXFLAGS=-O2 -std=c++11 -fno-stack-protector
CXXINC=-I ./include/

NVCC=/usr/local/cuda/bin/nvcc
NVCC_SM=sm_50
NVCCFLAGS=-arch $(NVCC_SM) -O2 -std=c++11 --expt-extended-lambda -w
NVCCINC=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I./3rdparty/cudaweijajalistrank/ -I./3rdparty/GpuConnectedComponents/ -I./3rdparty/cudabfs/

LDFLAGS=-L/usr/local/cuda/lib64 -lcudart

MGPU=3rdparty/moderngpu
MGPUFLAGS=-I $(MGPU)/src

OBJDIR=obj
SRCDIR=src

RUNNER_OBJ=$(patsubst src/%.cpp,obj/%.o,$(wildcard src/runner.cpp src/util/*.cpp src/cpu/*.cpp)) \
	$(patsubst src/%.cu,obj/%.o,$(wildcard src/gpu/*.cu)) \
	$(patsubst %.cu,obj/%.o,$(wildcard 3rdparty/cudaweijajalistrank/*.cu)) \
	$(patsubst %.cu,obj/%.o,$(wildcard 3rdparty/GpuConnectedComponents/*.cu)) \
	$(patsubst %.cu,obj/%.o,$(wildcard 3rdparty/cudabfs/bfs-mgpu.cu))


all: runner.e networkrepository-parser.e

remake: clean all

%.e: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

runner.e: $(RUNNER_OBJ)
	$(CXX) $(CXXFLAGS) $(CXXINC) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CXXINC) -c $< -o $@

$(OBJDIR)/util/%.o: $(SRCDIR)/util/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CXXINC) -c $< -o $@

$(OBJDIR)/cpu/%.o: $(SRCDIR)/cpu/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CXXINC) -c $< -o $@

$(OBJDIR)/gpu/%.o: $(SRCDIR)/gpu/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $(CXXINC) $(NVCCINC) -c $< -o $@
	
$(OBJDIR)/3rdparty/cudaweijajalistrank/%.o: 3rdparty/cudaweijajalistrank/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $(CXXINC) $(NVCCINC) -c $< -o $@

$(OBJDIR)/3rdparty/GpuConnectedComponents/%.o: 3rdparty/GpuConnectedComponents/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $(CXXINC) $(NVCCINC) -c $< -o $@

$(OBJDIR)/3rdparty/cudabfs/%.o: 3rdparty/cudabfs/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $(CXXINC) $(NVCCINC) -c $< -o $@

.PHONY: all clean

clean:
	rm -rf obj *.e
