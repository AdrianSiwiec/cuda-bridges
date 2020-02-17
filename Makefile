NVCC=/usr/local/cuda-10.1/bin/nvcc
NVCC_SM=sm_50
NVCCFLAGS=-arch $(NVCC_SM) -std=c++11 --expt-extended-lambda -w
NVCCINC=-I ./include/ -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I./3rdparty/cudabfs/ 

LDFLAGS=-L/usr/local/cuda-10.1/lib64 -lcudart 
MGPU=3rdparty/moderngpu
MGPUFLAGS=-I $(MGPU)/src

OBJDIR=obj
SRCDIR=src

ALLINC=$(NVCCINC) $(MGPUFLAGS)

RUNNER_OBJ=$(patsubst src/%.cpp,obj/%.o,$(wildcard src/runner.cu src/util/*.cpp src/cpu/*.cpp)) \
	$(patsubst src/%.cu,obj/%.o,$(wildcard src/gpu/*.cu)) \
	$(patsubst %.cu,obj/%.o,$(wildcard 3rdparty/cudabfs/bfs-mgpu.cu))


all: prepare-tests run-tests

prepare-tests: networkrepository-parser.e
	python3 test/test-downloader.py
	ulimit -s unlimited; python3 test/test-parser.py

run-tests: runner.e
	./runner.e test/networkrepository/road-asia-osm.mtx.bin

remake: clean all

%.e: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(ALLINC) $< -o $@

runner.e: $(RUNNER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(ALLINC) $(CXXINC) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(ALLINC) $(CXXINC) -c $< -o $@

$(OBJDIR)/util/%.o: $(SRCDIR)/util/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(ALLINC) $(CXXINC) -c $< -o $@

$(OBJDIR)/3rdparty/cudabfs/%.o: 3rdparty/cudabfs/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $(ALLINC) $(CXXINC) $(NVCCINC) -c $< -o $@

.PHONY: all clean prepare-tests run-tests

clean:
	rm -rf obj *.e
