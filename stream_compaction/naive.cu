#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void onestep(int n, int* odata, const int* idata,int d) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int two_pow_d_minus_one = 1 << (d - 1);
            if (index >= two_pow_d_minus_one) {
                odata[index] = idata[index - two_pow_d_minus_one] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void shift_right(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n - 1) {
                return;
            }
            odata[index + 1] = idata[index];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }
            if (n == 1) {          // handle trivial case without GPU work
                odata[0] = 0;
                return;
            }
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_bufA = nullptr;
            int* dev_bufB = nullptr;

            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_buf_A failed!");

            cudaMalloc((void**)&dev_bufB, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_buf_B failed!");

            cudaMemcpy(dev_bufA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("MemCpy dev_buf_A failed!");

            timer().startGpuTimer();
            int num_iter = ilog2ceil(n);

            for (int d = 1; d <= num_iter; d++) {
                onestep << <fullBlocksPerGrid, blockSize >> > (n, dev_bufB, dev_bufA, d);
                checkCUDAErrorFn("onestep Naive failed!");
                cudaDeviceSynchronize();
                std::swap(dev_bufA, dev_bufB);  // Output in dev_buf_A
            }
            // Inclusive Scan to Exclusive
            dim3 fullBlocksPerGrid_nm1((n - 1 + blockSize - 1) / blockSize);
            shift_right << <fullBlocksPerGrid_nm1, blockSize >> > (n, dev_bufB, dev_bufA);  // Shifted Output in dev_buf_B
            checkCUDAErrorFn("Shift_right failed!");
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_bufB, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("MemCpy dev_buf_B failed!");
            cudaFree(dev_bufA);
            checkCUDAErrorFn("CudaFree dev_buf_A failed!");
            cudaFree(dev_bufB);
            checkCUDAErrorFn("CudaFree dev_buf_B failed!");
            odata[0] = 0;
        }
    }
}
