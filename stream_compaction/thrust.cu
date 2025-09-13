#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            //thrust::device_ptr<const int> in_begin(idata);
            //thrust::device_ptr<const int> in_end(idata + n);
            //thrust::device_ptr<int> out_begin(odata);

            //// Perform exclusive scan on device
            //thrust::exclusive_scan(thrust::device, in_begin, in_end, out_begin);

            timer().endGpuTimer();
        }
    }
}
