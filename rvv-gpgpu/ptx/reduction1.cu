#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Two steps reduction
__global__ void reduction1_kernel(int *out, const int *in, size_t N)
{
    // length = threads (BlockDim.x)
    extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x;
    for (size_t i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    unsigned int floorPow2 = blockDim.x;
    if (floorPow2 & (floorPow2 - 1))
    {
        while(floorPow2 & (floorPow2 - 1))
        {
            floorPow2 &= (floorPow2 - 1);
        }
        if (tid >= floorPow2)
        {
            sPartials[tid - floorPow2] += sPartials[tid];
        }
        __syncthreads();
    }

    for (int activeTrheads = floorPow2 / 2; activeTrheads > 32; activeTrheads /= 2)
    {
        if (tid < activeTrheads)
        {
            sPartials[tid] += sPartials[tid + activeTrheads];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int *wsSum = sPartials;
        if (floorPow2 > 32)
        {
            wsSum[tid] += wsSum[tid + 32];
        }

        if (floorPow2 > 16) wsSum[tid] += wsSum[tid + 16];
        if (floorPow2 > 8) wsSum[tid] += wsSum[tid + 8];
        if (floorPow2 > 4) wsSum[tid] += wsSum[tid + 4];
        if (floorPow2 > 2) wsSum[tid] += wsSum[tid + 2];
        if (floorPow2 > 1) wsSum[tid] += wsSum[tid + 1];

        if (tid == 0)
        {
            volatile int *wsSum = sPartials;
            out[blockIdx.x] = wsSum[0];
        }
    }
}

void reduction1(int *answer, int *partial, const int *in, const size_t N, const int numBlocks, int numThreads)
{
    unsigned int sharedSize = numThreads * sizeof(int);

    // kernel execution
    reduction1_kernel<<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
    reduction1_kernel<<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}
