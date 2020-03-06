NVCC=/usr/local/cuda-10.1/bin/nvcc
NVCC_SM=sm_52
NVCCFLAGS=-arch $(NVCC_SM) -std=c++11 --expt-extended-lambda -w -O2
INC=-I/usr/local/cuda/include -I moderngpu/src
LDFLAGS=-L/usr/local/cuda-10.1/lib64 -lcudart 

all: run-tests

run-tests: runner.e
	./runner.e

remake: clean all

runner.e: runner.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

.PHONY: all clean prepare-tests run-tests

clean:
	rm -rf obj *.e
