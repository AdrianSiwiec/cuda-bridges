/****************************************************************************************
 *       CONNECTED COMPONENTS ON THE GPU
 *       ==============================
 *
 *
 *
 *       Copyright (c) 2010 International Institute of Information Technology,
 *       Hyderabad.
 *       All rights reserved.
 *
 *       Permission to use, copy, modify and distribute this software and its
 *       documentation for research purpose is hereby granted without fee,
 *       provided that the above copyright notice and this permission notice
 *appear in all copies of this software and that you do not sell the software.
 *
 *       THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,
 *       EXPRESS, IMPLIED OR OTHERWISE.
 *
 *       Please report any issues to Jyothish Soman
 *(jyothish@students.iiit.ac.in)
 *
 *       Please cite following paper, if you use this software for research
 *purpose
 *
 *       "Fast GPU Algorithms for Graph Connectivity, Jyothish Soman, K.
 *Kothapalli, and P. J. Narayanan, in Proc. of Large Scale Parallel Processing,
 *       IPDPS Workshops, 2010.
 *
 *
 *
 *
 *       Created by Jyothish Soman
 *
 ****************************************************************************************/
// includes, system
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_intervalmove.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "cc.cuh"
#include "graph.hpp"
#include "test-result.hpp"


/*
 * Function to speedup the selection process in the first iteration
 * The ancestor tree is initialized to the add the edge from larger edge to
 * its smaller neighbour in this method. The process is random and each edge
 * performs this task independently. select_winner_init
 */
__global__ void select_winner_init(int *an, cc::edge *ed_list, int num_e,
                                   int num_n, int *flag, char *mark,
                                   int *ISSPAN) {
    int a, b, x, y, mx;
    // int mn;
    long long int t;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;
    if (a < num_e) {
        t = ed_list[a].x;
        x = (int)t & 0xFFFFFFFF;
        y = (int)(t >> 32);

        mx = x > y ? x : y;
        // mn=x+y-mx;
        // edge is a candidate
        ISSPAN[mx] = a;
        // an[mx]=mn;
    }
    return;
}

/*
   Function to hook from higher valued tree to lower valued tree. For details,
   read the PPL Paper or LSPP paper or my master's thesis. Following greener's
   algorithm, there are two iterations, one from lower valued edges to higher
   values edges and the second iteration goes vice versa. The performance of
   this is largely related to the input.
 */
__global__ void select_winner2(int *an, cc::edge *ed_list, int num_e, int num_n,
                               int *flag, char *mark, int *ISSPAN) {
    int a, b, x, y, a_x, a_y, mn, mx;
    long long int t;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    __shared__ int s_flag;
    a = a * 512 + b;
    if (b == 1) s_flag = 0;
    __syncthreads();
    if (a < num_e) {
        if (mark[a] == 0) {
            t = ed_list[a].x;
            x = (int)t & 0xFFFFFFFF;
            y = (int)(t >> 32);

            a_x = an[x];
            a_y = an[y];
            mx = a_x > a_y ? a_x : a_y;
            mn = a_x + a_y - mx;
            if (mn == mx) {
                mark[a] = -1;
            } else {
                // edge is a candidate
                ISSPAN[mn] = a;
                // an[mn]=mx;
                s_flag = 1;
            }
        }
    }
    __syncthreads();
    if (b == 1) {
        if (s_flag == 1) {
            *flag = 1;
        }
    }
    return;
}

__global__ void select_tree_edges_and_merge2(int *an, cc::edge *ed_list,
                                             int num_e, int num_n, int *flag,
                                             char *mark, int *ISSPAN, bool *is_tree) {
    int a, b, x, y, a_x, a_y, mn, mx;
    long long int t;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;

    if (a < num_n) {
        if (ISSPAN[a] != -1) {
            is_tree[ISSPAN[a]] = true;

            t = ed_list[ISSPAN[a]].x;
            x = (int)t & 0xFFFFFFFF;
            y = (int)(t >> 32);

            a_x = an[x];
            a_y = an[y];
            mx = a_x > a_y ? a_x : a_y;
            mn = a_x + a_y - mx;

            an[mn] = mx;
        }
    }
}

/*
   Function to hook from lower valued to higher valued trees.
 */
__global__ void select_winner(int *an, cc::edge *ed_list, int num_e, int num_n,
                              int *flag, char *mark, int *ISSPAN) {
    int a, b, x, y, a_x, a_y, mn, mx;
    long long int t;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    __shared__ int s_flag;
    a = a * 512 + b;
    if (b == 1) s_flag = 0;
    __syncthreads();
    if (a < num_e) {
        if (mark[a] == 0) {
            t = ed_list[a].x;
            x = (int)t & 0xFFFFFFFF;
            y = (int)(t >> 32);

            a_x = an[x];
            a_y = an[y];
            mx = a_x > a_y ? a_x : a_y;
            mn = a_x + a_y - mx;
            if (mn == mx) {
                mark[a] = -1;
            } else {
                // edge is a candidate
                ISSPAN[mx] = a;
                // an[mx]=mn;
                s_flag = 1;
            }
        }
    }
    __syncthreads();
    if (b == 1) {
        if (s_flag == 1) {
            *flag = 1;
        }
    }
    return;
}

__global__ void select_tree_edges_and_merge(int *an, cc::edge *ed_list,
                                            int num_e, int num_n, int *flag,
                                            char *mark, int *ISSPAN, bool *is_tree) {
    int a, b, x, y, a_x, a_y, mn, mx;
    long long int t;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;

    if (a < num_n) {
        if (ISSPAN[a] != -1) {
            is_tree[ISSPAN[a]] = true;

            t = ed_list[ISSPAN[a]].x;
            x = (int)t & 0xFFFFFFFF;
            y = (int)(t >> 32);

            a_x = an[x];
            a_y = an[y];
            mx = a_x > a_y ? a_x : a_y;
            mn = a_x + a_y - mx;

            an[mx] = mn;
        }
    }
}

__global__ void p_jump(int num_n, int *an, int *flag) {
    int a, b, x, y;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;
    __shared__ int s_f;
    if (a >= num_n) return;
    if (b == 1) {
        s_f = 0;
    }
    __syncthreads();
    if (a < num_n) {
        y = an[a];
        x = an[y];
        if (x != y) {
            s_f = 1;
            an[a] = x;
        }
    }
    if (b == 1) {
        if (s_f == 1) {
            *flag = 1;
        }
    }
}

/*
   Function to do a masked jump
   Nodes are either root nodes or leaf nodes. Leaf nodes are directly connected
   to the root nodes, hence do not need to jump itertively. Once root nodes have
   reascertained the new root nodes, the leaf nodes can just jump once
 */
__global__ void p_jump_masked(int num_n, int *an, int *flag, char *mask) {
    int a, b, x, y;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;
    __shared__ int s_f;
    if (a >= num_n) return;
    if (b == 1) {
        s_f = 0;
    }

    __syncthreads();
    if (mask[a] == 0) {
        y = an[a];
        x = an[y];
        if (x != y) {
            s_f = 1;
            an[a] = x;
        } else {
            mask[a] = -1;
        }
    }
    if (b == 1) {
        if (s_f == 1) {
            *flag = 1;
        }
    }
}

/*
   Function for pointer jumping in the tree, the tree height is shortened by
   this method. Here the assumption is that all the nodes are root nodes, or not
   known whether they are leaf nodes. Works well in the early iterations
 */
__global__ void p_jump_unmasked(int num_n, int *an, char *mask) {
    int a, b, x, y;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;
    if (a >= num_n) return;
    __syncthreads();
    if (mask[a] == 1) {
        y = an[a];
        x = an[y];
        an[a] = x;
    }
}

/*
   Function to create self pointing tree.
 */
__global__ void update_an(int *an, int num_n) {
    int a, b;
    a = blockIdx.y * gridDim.x + blockIdx.x;
    b = threadIdx.x;
    a = a * 512 + b;
    if (a >= num_n) return;
    an[a] = a;

    return;
}

/*
   Function to initialize each edge as a clean copy.
 */
__global__ void update_mark(char *mark, int num_e) {
    int j;
    j = blockIdx.y * gridDim.x + blockIdx.x;
    j = j * 512 + threadIdx.x;
    if (j >= num_e) return;
    mark[j] = 0;
}

/*
   Function to check if each node is the parent of itself or not and to update
   it as a leaf or root node
 */
__global__ void update_mask(char *mask, int n, int *an) {
    int j;
    j = blockIdx.y * gridDim.x + blockIdx.x;
    j = j * 512 + threadIdx.x;
    if (j >= n) return;
    mask[j] = an[j] == j ? 0 : 1;
    return;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
mem_t<long long int> cc_main(int const num_n, int const num_e, cc::edge *d_ed_list,
             context_t &context) {
    // Prepare memory
    cc::edge *ed_list = (cc::edge *)calloc(num_e, sizeof(cc::edge));
    
    int nnx, nny, nex, ney;
    int flag, *d_winner, *d_an;
    int *d_flag, *an;
    char *d_mark, *mark;
    char *mask;

    int *d_hook_edge;
    int *hook_edge;
    hook_edge = (int *)calloc(num_n, sizeof(int));

    int num_threads, num_blocks_n, num_blocks_e;
    num_threads = 512;
    num_blocks_n = (num_n / 512) + 1;
    num_blocks_e = (num_e / 512) + 1;
    nny = (num_blocks_n / 1000) + 1;
    nnx = 1000;
    nex = (num_blocks_e / 1000) + 1;
    ney = 1000;
    dim3 grid_n(nnx, nny);
    dim3 grid_e(nex, ney);
    dim3 threads(num_threads, 1);

    an = (int *)calloc(num_n, sizeof(int));

    mem_t<char> MEM_d_mark(num_e, context);
    mem_t<char> MEM_d_mask(num_e, context);
    mem_t<int> MEM_d_winner(num_n, context);
    mem_t<int> MEM_d_an(num_n, context);
    mem_t<int> MEM_d_flag(1, context);
    mem_t<int> MEM_d_hook_edge = mgpu::fill<int>(-1, num_n, context);
    mem_t<bool> MEM_d_is_tree = mgpu::fill<bool>(false, num_e, context);
    d_mark = MEM_d_mark.data();
    mask = MEM_d_mask.data();
    d_winner = MEM_d_winner.data();
    d_an = MEM_d_an.data();
    d_flag = MEM_d_flag.data();
    d_hook_edge = MEM_d_hook_edge.data();
    bool * d_is_tree = MEM_d_is_tree.data();

    //   Finished intializing space for the program, ideally timing should be
    //   from here.
    clock_t t = clock();

    update_mark<<<grid_e, threads>>>(d_mark, num_e);
    update_an<<<grid_n, threads>>>(d_an, num_n);
    cudaThreadSynchronize();

    // First round of select winner

    //     select_winner_init<<<
    //     grid_e,threads>>>(d_an,d_ed_list,num_e,num_n,d_flag,d_mark,
    //     d_hook_edge); cudaThreadSynchronize();

    //     select_tree_edges_and_merge_init<<<
    //     grid_n,threads>>>(d_an,d_ed_list,num_e,num_n,d_flag,d_mark,
    //     d_hook_edge); cudaThreadSynchronize();

    // //    CUT_CHECK_ERROR("Kernel execution failed");

    //     do{
    //         flag=0;
    //         checkCudaErrors(cudaMemcpy(d_flag,&flag,sizeof(int),cudaMemcpyHostToDevice));
    //         p_jump<<< grid_n,threads>>>(num_n,d_an,d_flag);
    //         cudaThreadSynchronize();

    // //        CUT_CHECK_ERROR("Kernel execution failed");
    //         checkCudaErrors(cudaMemcpy(&flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost));
    //     }while(flag);

    // main code starts
    //
    update_mask<<<grid_n, threads>>>(mask, num_n, d_an);
    cudaThreadSynchronize();

    int lpc = 1;
    do {
        checkCudaErrors(cudaMemset(d_hook_edge, -1, num_n * sizeof(int)));
        
        flag = 0;
        checkCudaErrors(htod(d_flag, &flag, 1));
        
        if (lpc != 0) {
            select_winner<<<grid_e, threads>>>(d_an, d_ed_list, num_e, num_n,
                                               d_flag, d_mark, d_hook_edge);
            cudaThreadSynchronize();

            select_tree_edges_and_merge<<<grid_n, threads>>>(
                d_an, d_ed_list, num_e, num_n, d_flag, d_mark, d_hook_edge, d_is_tree);

            lpc++;
            lpc = lpc % 4;
        } else {
            select_winner2<<<grid_e, threads>>>(d_an, d_ed_list, num_e, num_n,
                                                d_flag, d_mark, d_hook_edge);
            cudaThreadSynchronize();

            select_tree_edges_and_merge2<<<grid_n, threads>>>(
                d_an, d_ed_list, num_e, num_n, d_flag, d_mark, d_hook_edge, d_is_tree);

            lpc = 0;
        }
        cudaThreadSynchronize();

        checkCudaErrors(
            cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (flag == 0) {
            break;
        }

        int flg;
        do {
            flg = 0;
            checkCudaErrors(
                cudaMemcpy(d_flag, &flg, sizeof(int), cudaMemcpyHostToDevice));

            p_jump_masked<<<grid_n, threads>>>(num_n, d_an, d_flag, mask);
            cudaThreadSynchronize();

            checkCudaErrors(
                cudaMemcpy(&flg, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        } while (flg);

        p_jump_unmasked<<<grid_n, threads>>>(num_n, d_an, mask);
        cudaThreadSynchronize();

        update_mask<<<grid_n, threads>>>(mask, num_n, d_an);
        cudaThreadSynchronize();
    } while (flag);

    t = clock() - t;
    // printf(
    //     "Time required for computing connected components on the graph is: %f "
    //     "seconds.\n",
    //     ((float)t) / CLOCKS_PER_SEC);

    mark = (char *)calloc(num_e, sizeof(char));
    // end of main loop
    // checkCudaErrors(
    //     cudaMemcpy(an, d_an, num_n * sizeof(int), cudaMemcpyDeviceToHost));
    // int j, cnt = 0;
    // for (j = 0; j < num_n; j++) {
    //     if (an[j] == j) {
    //         cnt++;
    //     }
    // }

    // checkCudaErrors(cudaMemcpy(ed_list, d_ed_list, num_e * sizeof(cc::edge),
    //                            cudaMemcpyDeviceToHost));

    // int tree_e_num = 0;
    // for (j = 0; j < num_e; ++j) {
    //     if (ed_list[j].tree) {
    //         tree_e_num++;

    //         long long int t = ed_list[j].x;
    //         int x = (int)t & 0xFFFFFFFF;
    //         int y = (int)(t >> 32);
    //         // cout << x + 1 << " " << y + 1 << "\n";
    //     }
    // }

    // printf("The number of components=%d\n", cnt);
    // printf("The number of tree edges=%d vertexes=%d isok=%d\n", tree_e_num,
    //        num_n, num_n - 1 == tree_e_num);

    // assert(num_n - 1 == tree_e_num);

    // Construct the compaction state with transform_compact.
    auto compact = transform_compact(num_e, context);

    // The upsweep determines which items to compact i.e. which edges belong to tree
    int stream_count = compact.upsweep([=]MGPU_DEVICE(int index) {
        return d_is_tree[index];
    });

    // Compact the results into this buffer.
    mem_t<long long int> tree_edges(stream_count, context);
    long long int* tree_edges_data = tree_edges.data();
    compact.downsweep([=]MGPU_DEVICE(int dest_index, int source_index) {
        tree_edges_data[dest_index] = d_ed_list[source_index].x;
    });

    // printf("res size : %d, n : %d\n", tree_edges.size(), num_n);
    assert(num_n - 1 == tree_edges.size());

    free(an);
    free(mark);
    free(ed_list);
    free(hook_edge);
    
    return tree_edges;
}
