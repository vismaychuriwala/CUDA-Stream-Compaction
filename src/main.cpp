/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#define SHOW 1;
int flush = 0;
#if SHOW

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix_sort.h>
#include "testing_helpers.hpp"
#include <algorithm>

const int SIZE = 1 << 20; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];
int* d = new int[NPOT];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);



    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    genArray(SIZE - 1, a, INT_MAX);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu sort, power-of-two");
    StreamCompaction::CPU::sort(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(NPOT, d);
    printDesc("cpu sort, non-power-of-two");
    StreamCompaction::CPU::sort(NPOT, d, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, d, true);
    //printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("Radix Sort, power-of-two");
    StreamCompaction::RadixSort::sort(SIZE, c, a);
    printElapsedTime(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    

    zeroArray(SIZE, c);
    printDesc("Radix Sort, non-power-of-two");
    StreamCompaction::RadixSort::sort(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, d, c);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
}

#else

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"
#include <thread>
#include <iostream>

const int TWO_POW = 25;
//Testing
int main(int argc, char* argv[]) {
    if (flush == 1) {
        FILE* f = freopen("blockSize_timings.txt", "a", stdout);
    }
        const int SIZE = 1 << TWO_POW; // feel free to change the size of array
        const int NPOT = SIZE - 3; // Non-Power-Of-Two
        int* a = new int[SIZE];
        int* b = new int[SIZE];
        int* c = new int[SIZE];
        // Scan tests

        //printf("\n");
        //printf("****************\n");
        printf("** SCAN TESTS **\n");
        //printf("****************\n");

        std::cout << "SIZE= " << SIZE;
         //std::cout << "blockSize= " << blockSize;
        //for (int i = 1; i < 10; i++) {
            printf("\n");
            //std::cout << "RUN#" << i << std::endl;
   
            genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
            a[SIZE - 1] = 0;
            //printArray(SIZE, a, true);

            // initialize b using StreamCompaction::CPU::scan you implement
            // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
            // At first all cases passed because b && c are all zeroes.
            zeroArray(SIZE, b);
            //printDesc("cpu scan, power-of-two");
            StreamCompaction::CPU::scan(SIZE, b, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            //printArray(SIZE, b, true);

            zeroArray(SIZE, c);
            //printDesc("cpu scan, non-power-of-two");
            StreamCompaction::CPU::scan(NPOT, c, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            //printArray(NPOT, c, true);
            printCmpResult(NPOT, b, c);

            zeroArray(SIZE, c);
            //printDesc("naive scan, power-of-two");
            StreamCompaction::Naive::scan(SIZE, c, a);
            printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(SIZE, c, true);
            printCmpResult(SIZE, b, c);

            /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
            onesArray(SIZE, c);
            printDesc("1s array for finding bugs");
            StreamCompaction::Naive::scan(SIZE, c, a);
            printArray(SIZE, c, true); */

            zeroArray(SIZE, c);
            //printDesc("naive scan, non-power-of-two");
            StreamCompaction::Naive::scan(NPOT, c, a);
            printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(SIZE, c, true);
            printCmpResult(NPOT, b, c);

            zeroArray(SIZE, c);
            //printDesc("work-efficient scan, power-of-two");
            StreamCompaction::Efficient::scan(SIZE, c, a);
            printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(SIZE, c, true);
            printCmpResult(SIZE, b, c);

            zeroArray(SIZE, c);
            //printDesc("work-efficient scan, non-power-of-two");
            StreamCompaction::Efficient::scan(NPOT, c, a);
            printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(NPOT, c, true);
            printCmpResult(NPOT, b, c);

            zeroArray(SIZE, c);
            //printDesc("thrust scan, power-of-two");
            StreamCompaction::Thrust::scan(SIZE, c, a);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(SIZE, c, true);
            printCmpResult(SIZE, b, c);

            zeroArray(SIZE, c);
            //printDesc("thrust scan, non-power-of-two");
            StreamCompaction::Thrust::scan(NPOT, c, a);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(NPOT, c, true);
            printCmpResult(NPOT, b, c);

            printf("\n");
            //printf("*****************************\n");
            //printf("** STREAM COMPACTION TESTS **\n");
            //printf("*****************************\n");

            // Compaction tests

            genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
            a[SIZE - 1] = 0;
            //printArray(SIZE, a, true);

            int count, expectedCount, expectedNPOT;

            // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
            // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
            zeroArray(SIZE, b);
            //printDesc("cpu compact without scan, power-of-two");
            count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            expectedCount = count;
            //printArray(count, b, true);
            printCmpLenResult(count, expectedCount, b, b);

            zeroArray(SIZE, c);
            //printDesc("cpu compact without scan, non-power-of-two");
            count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            expectedNPOT = count;
            //printArray(count, c, true);
            printCmpLenResult(count, expectedNPOT, b, c);

            zeroArray(SIZE, c);
            //printDesc("cpu compact with scan");
            count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            //printArray(count, c, true);
            printCmpLenResult(count, expectedCount, b, c);

            zeroArray(SIZE, c);
            //printDesc("work-efficient compact, power-of-two");
            count = StreamCompaction::Efficient::compact(SIZE, c, a);
            printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(count, c, true);
            //printCmpLenResult(count, expectedCount, b, c);

            zeroArray(SIZE, c);
            //printDesc("work-efficient compact, non-power-of-two");
            count = StreamCompaction::Efficient::compact(NPOT, c, a);
            printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            //printArray(count, c, true);
            printCmpLenResult(count, expectedNPOT, b, c);

            //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        delete[] a;
        delete[] b;
        delete[] c;
    //system("pause"); // stop Win32 console from closing on exit
}

#endif