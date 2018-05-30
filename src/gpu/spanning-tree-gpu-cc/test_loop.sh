#!/bin/bash
make clean; make;
export CUDA_PATH=/usr/local/cuda
make -C ../../../3rdparty/GpuConnectedComponents;

for i in {1..50}
do
echo "---- test ---"
./exec $1 $2
act_pwd=$(pwd)
cd ../../../3rdparty/GpuConnectedComponents/
./exec
cd $act_pwd
done

make -C ../../../3rdparty/GpuConnectedComponents clean;
rm -rf ../../../3rdparty/GpuConnectedComponents/parameters.txt ../../../3rdparty/GpuConnectedComponents/edge.txt
make clean;
