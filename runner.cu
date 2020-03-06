#include <iostream>
#include <chrono>
#include <string>

#include <moderngpu/kernel_intervalmove.hxx>

void print_ms(std::clock_t start, std::clock_t end, std::string desc)
{
    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    std::cout << desc << ": " << ms << " ms." << std::endl;
}

void fun(bool magic, mgpu::context_t &context)
{
    int n = 10000000;
    mgpu::mem_t<int> m0(n, context);
    mgpu::mem_t<int> m1(n, context);
    mgpu::mem_t<int> m2(n, context);
    mgpu::mem_t<int> m3(n, context);

    void *tmpPtr;
    cudaMalloc((void **)(&tmpPtr), sizeof(int));
    if (!magic)
        cudaFree(tmpPtr);

    for (int aa = 0; aa < 10000; aa++)
        mgpu::interval_gather(m0.data(), 1,
                              m1.data(), 1,
                              m2.data(), 
                              m3.data(),
                              context);

    if (magic)
        cudaFree(tmpPtr);
}

int main(int argc, char *argv[])
{
    mgpu::standard_context_t context(false);
    std::clock_t c0 = std::clock();

    fun(true, context);
    context.synchronize();

    std::clock_t c1 = std::clock();
    print_ms(c0, c1, "1st");

    fun(false, context);
    context.synchronize();

    std::clock_t c2 = std::clock();
    print_ms(c1, c2, "2nd");

    fun(true, context);
    context.synchronize();

    std::clock_t c3 = std::clock();
    print_ms(c2, c3, "3rd");

    return 0;
}
