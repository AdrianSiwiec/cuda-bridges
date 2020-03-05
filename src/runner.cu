#include <algorithm>
#include <iostream>
using namespace std;

#include <moderngpu/kernel_mergesort.hxx>
using namespace mgpu;

#include <moderngpu/kernel_intervalmove.hxx>
#include <chrono>
#include <string>

class Timer {
   private:
    std::clock_t c_start, c_end;
    std::string slug;
    long double overall;

   public:
    Timer(std::string);

    void start();
    void stop();
    long double get_ms();
    void print_info(std::string);
    void print_and_restart(std::string);
    void print_overall();
};

Timer::Timer(std::string slug) : slug(slug), c_start(std::clock()), overall(0){}

void Timer::start() { c_start = std::clock(); }

void Timer::stop() { c_end = std::clock(); }

long double Timer::get_ms() { return 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC; }

void Timer::print_info(std::string desc) {
    std::cout.precision(3);
    std::cout << std::fixed << slug << ": " << desc << ": " << get_ms() << " ms." << std::endl;
}

void Timer::print_and_restart(std::string desc) {
    stop();
    overall += get_ms();
    print_info(desc);
    start();
}

void Timer::print_overall() {
    std::cout.precision(3);
    std::cout << std::fixed << slug << ": " << "Overall" << ": " << overall << " ms." << std::endl;
}


void fun(bool magic, context_t &context)
{
    int n = 10000000, m = 20000000;
    mem_t<int> emptyMem0(n, context);
    mem_t<int> emptyMem1(n, context);
    mem_t<int> emptyMem2(m, context);
    mem_t<int> emptyMem3(m, context);

    void *tmpPtr;
    cudaMalloc((void **)(&tmpPtr), sizeof(int) * 8);
    if (!magic)
        cudaFree(tmpPtr);

    // printf("n: %d, m: %d, edge_frontier_size: %d, node_frontier_size: %d\n", n, m, edge_frontier_size, node_frontier_size);

    for (int aa = 0; aa < 10000; aa++)
        interval_gather(emptyMem2.data(), 2,
                        emptyMem0.data(), 1,
                        emptyMem1.data(), emptyMem3.data(),
                        context);

    if (magic)
        cudaFree(tmpPtr);
}

int main(int argc, char *argv[])
{
    standard_context_t context(false);
    Timer timer("gpu-bfs");

    fun(true, context);
    context.synchronize();
    timer.print_and_restart("1");

    fun(false, context);
    context.synchronize();
    timer.print_and_restart("2");

    fun(true, context);
    context.synchronize();
    timer.print_and_restart("3");

    return 0;
}
