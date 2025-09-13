#include <cstdio>
#include "cpu.h"
#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            if (n == 0) {
                timer().endCpuTimer();
                return;
            }
            odata[0] = 0;
            for (int k = 1; k < n; k++) {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            int k_i = 0;
            for (int k = 0; k < n; k++) {
                int idat = idata[k];
                if (idat == 0) {
                    continue;
                }
                odata[k_i] = idata[k];
                k_i++;
            }
            timer().endCpuTimer();
            return k_i;
        }

        // Untimed Scan function (timed CPU::scan starts and stops the object timer)
        void untimed_scan(int n, int* odata, const int* idata) {
            if (n == 0) {
                return;
            }
            odata[0] = 0;
            for (int k = 1; k < n; k++) {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            if (n == 0) {
                timer().endCpuTimer();
                return 0;
            }
            int* bool_arr = new int[n];
            for (int k = 0; k < n; k++) {
                int idat = idata[k];
                if (idat == 0) {
                    bool_arr[k] = 0;
                }
                else {
                    bool_arr[k] = 1;
                }
            }

            CPU::untimed_scan(n, odata, bool_arr);  // Using odata as index array to save space
            for (int k = 0; k < n; k++) {
                int bool_ = bool_arr[k];
                if (bool_ == 0) {
                    continue;
                }
                else {
                    int idat = idata[k];
                    int index = odata[k];
                    odata[index] = idat;
                }
            }
            int count = odata[n - 1] + bool_arr[n - 1];
            delete[] bool_arr;
            timer().endCpuTimer();
            return count;
        }
    }
}
