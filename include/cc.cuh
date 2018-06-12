#ifndef CC_CUH
#define CC_CUH

namespace cc{
struct ed{
    long long int x;
};

typedef struct ed edge;

struct grp{
    int num_e,num_n;
    int**neigh,*deg;
};

typedef struct grp my_graph;
}

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

mem_t<long long int> cc_main(int const num_n, int const num_e, cc::edge * d_ed_list, context_t& context);

#endif  // CC_CUH
