#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void prescan(int n, int* g_odata, int* g_idata)
        {
            int thid = threadIdx.x;
            int offset = 1;
            extern __shared__ int temp[];
            temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
             temp[2*thid+1] = g_idata[2*thid+1];
            for (int d = n >> 1; d > 0; d >>= 1)
                // build sum in place up the tree
            {
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }
            if (thid == 0)
            {
                temp[n - 1] = 0;
            } // clear the last element
            for (int d = 1; d < n; d <<= 1) // traverse down tree & build scan
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();
            g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
            g_odata[2 * thid + 1] = temp[2 * thid + 1];
        }

        __global__ void make_exclusive(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index == 0) {
                odata[index] = 0;
                return;
            }
            odata[index] = idata[index - 1];
        }

        //__global__ void pad_with_zeroes(int n, int m, int* g_data) {
        //    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        //    if (index >= n || index < m) {
        //        return;
        //    }
        //    g_data[index] = 0;
        //}

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
            int m = n;
            bool is_pow_two = (n & (n - 1)) == 0;
            if (!is_pow_two) {
                n = 1 << ilog2ceil(n);
            }

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int sharedMemBytes = 2 * blockSize * sizeof(int);
            int* dev_buf_i = nullptr;
            int* dev_buf_o = nullptr;

            cudaMalloc((void**)&dev_buf_i, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_buf_i failed!");
            cudaMalloc((void**)&dev_buf_o, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_buf_o failed!");\

            cudaMemcpy(dev_buf_i, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("MemCpy dev_buf_i failed!");

            if (!is_pow_two) {
                cudaMemset(dev_buf_i + m, 0, (n - m) * sizeof(int));
            }
            timer().startGpuTimer();
            Efficient::prescan << <fullBlocksPerGrid, blockSize, sharedMemBytes >>> (n, dev_buf_o, dev_buf_i);
            checkCUDAErrorFn("prescan failed!");
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buf_o, m * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("MemCpy dev_buf_o failed!");
            cudaFree(dev_buf_o);
            checkCUDAErrorFn("CudaFree dev_buf_o failed!");
            cudaFree(dev_buf_i);
            checkCUDAErrorFn("CudaFree dev_buf_i failed!");
        }

        __global__ void scatter(int n, int* bools, int* odata, const int* idata) {
            //odata already has indices
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (bools[index] == 1) {
                odata[odata[index]] = idata[index];
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            //Check trivial cases:
            if (n <= 0) {
                return 0;
            }
            if (n == 1) {
                if (idata[0] != 0)
                {
                    odata[0] = idata[0];
                    return 1;
                }
                return 0;
            }
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int m = n;
            n = 1 << ilog2ceil(n);
            timer().startGpuTimer();
                // Allocate and assign input
                int* dev_buf_i = nullptr;
                cudaMalloc((void**)&dev_buf_i, n * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev_buf_i failed!");
                cudaMemcpy(dev_buf_i, idata, m * sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAErrorFn("MemCpy dev_buf_i failed!");

         
                // Allocate bools buffer
                int* dev_buf_bools = nullptr;
                cudaMalloc((void**)&dev_buf_bools, n * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev_buf_bools failed!");

                // Fill bools buffer
                
                Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >> > (m, dev_buf_bools, dev_buf_i);

                // Scan bools to output

                dim3 fullBlocksPerGrid_scan((n + blockSize - 1) / blockSize);
                int sharedMemBytes = 2 * blockSize * sizeof(int);
                int* dev_buf_indices = nullptr;

                cudaMalloc((void**)&dev_buf_indices, n * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev_buf_indices failed!");

                cudaMemset(dev_buf_bools + m, 0, (n - m) * sizeof(int));
                checkCUDAErrorFn("CudaMemset zeroes failed!");

                Efficient::prescan<<<fullBlocksPerGrid_scan, blockSize, sharedMemBytes>>> (n, dev_buf_indices, dev_buf_bools);
                checkCUDAErrorFn("prescan failed!");
                cudaDeviceSynchronize();

                //std::swap(m, n);

                // Shorten now

                int* dev_buf_o = nullptr;
                cudaMalloc((void**)&dev_buf_o, m * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev_buf_i failed!");

                Common::kernScatter << < fullBlocksPerGrid, blockSize>>> (m, dev_buf_o, dev_buf_i, dev_buf_bools, dev_buf_indices);
                checkCUDAErrorFn("efficient scatter failed!");
                cudaDeviceSynchronize();

                int last_index;
                cudaMemcpy(&last_index, dev_buf_indices + m - 1, sizeof(int), cudaMemcpyDeviceToHost);

                int last_bool;
                cudaMemcpy(&last_bool, dev_buf_bools + m - 1, sizeof(int), cudaMemcpyDeviceToHost);

                int count = (last_index + last_bool);
                // Copy to CPU and Free data
                cudaMemcpy(odata, dev_buf_o, count * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAErrorFn("MemCpy dev_buf_o failed!");

                cudaFree(dev_buf_indices);
                checkCUDAErrorFn("CudaFree dev_buf_indices failed!");
                cudaFree(dev_buf_i);
                checkCUDAErrorFn("CudaFree dev_buf_i failed!");
                cudaFree(dev_buf_bools);
                checkCUDAErrorFn("CudaFree dev_buf_bools failed!");
                cudaFree(dev_buf_o);
                checkCUDAErrorFn("CudaFree dev_buf_o failed!");

                timer().endGpuTimer();
                return count;
        }
    }
}
