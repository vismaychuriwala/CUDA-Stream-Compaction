#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix_sort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer t;
            return t;
        }

        __global__ void uniformAdd(int n,
            int* odata,
            const int num) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] += num;
        }

        __global__ void negate_bools(int n,
            int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] &= 1;
        }

        __global__ void radix_to_bools(int n,
            int* b, const int *idata, size_t pos) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int bit_pos = 1 << pos;
            b[index] = ((idata[index] & bit_pos) > 0);
        }

        __global__ void assign_indexes(int n,
            int* odata, const int* idata, const int* b, const int* t, const int* f) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int is_true = b[index];

            if (is_true > 0) {
                odata[index] = idata[t[index]];
            }
            else {
                odata[index] = idata[f[index]];
            }
        }

        void onestep(int n, int* odata, const int* idata, int* t, int* f, int* b, size_t pos) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            RadixSort::radix_to_bools << <fullBlocksPerGrid, blockSize >> > (n, b, idata, pos);
            cudaDeviceSynchronize();
            Efficient::recursiveScan(n, f, b);
            cudaDeviceSynchronize();
            int total_falses;
            cudaMemcpy(&total_falses, f + n, sizeof(int), cudaMemcpyDeviceToHost);
            RadixSort::negate_bools<<<fullBlocksPerGrid, blockSize>>> (n, b);
            checkCUDAErrorFn("Negate bools failed!");
            cudaDeviceSynchronize();
            Efficient::recursiveScan(n, t, b);
            cudaDeviceSynchronize();
            RadixSort::uniformAdd << <fullBlocksPerGrid, blockSize >> > (n, t, total_falses);
            cudaDeviceSynchronize();
            // Assign idata at indexes (t and f) to odata
            RadixSort::assign_indexes << <fullBlocksPerGrid, blockSize >> > (n, odata, idata, b, t, f);
            cudaDeviceSynchronize();
        }


        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) return;

            // TODO: Alloc and Copy data then call onestep starting from pos = 0 .. number of bits - 1 in int
            timer().startGpuTimer();
            cudaMemcpy(odata, idata, n * sizeof(int), cudaMemcpyHostToHost);
            cudaDeviceSynchronize();
            timer().endGpuTimer();
        }
    }
}

