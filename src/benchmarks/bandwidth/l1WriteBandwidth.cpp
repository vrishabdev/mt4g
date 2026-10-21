#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <tuple>
#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>
#include <limits>

static constexpr auto WARMUP_REPS = 128;
static constexpr auto MS_PER_SECOND = 1000.0; // ms


__global__ void l1WriteBandwidthKernel(uint32v4* __restrict__ dst, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps) 
{
    const uint32_t tid = threadIdx.x;
    uint32v4* base = dst + tid * elementsPerThread;

    #ifdef __HIP_PLATFORM_AMD__
    const uint64_t addr0 = reinterpret_cast<uint64_t>(base);
    #endif

    uint32v4 dummy = {tid, tid + 1, tid + 2, tid + 3}; 

    // Warm up L1
    for (size_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "flat_store_dwordx4 %0, %1\n\t"
                :
                : "v"(addr0 + i * sizeof(uint32v4)),
                "v"(dummy)
                : "memory"
            );
            #endif

#ifdef __HIP_PLATFORM_NVIDIA__
using vec4 = uint4;
#else
using vec4 = uint32v4;
#endif

__global__ void l1WriteBandwidthKernel(vec4* __restrict__ dst, size_t totalElements, size_t reps)
{
    const uint32_t gtid = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;

    const vec4 dummy = {gtid, gtid + 1, gtid + 2, gtid + 3};

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "flat_store_dwordx4 %0, %1\n\t"
                :
                : "v"(dst + i),
                  "v"(dummy)
                : "memory"
            );
            #else
            dst[i] = dummy;
            #endif
        }
    }
}

static std::tuple<double, double> l1WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    const size_t totalElements = arraySizeBytes / sizeof(vec4);

    vec4 *d_dstArr = util::allocateGPUMemory<vec4>(totalElements);

    //warm up
    l1WriteBandwidthKernel<<<NUM_BLOCKS, numThreads, 0, stream>>>(d_dstArr, totalElements, WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    l1WriteBandwidthKernel<<<NUM_BLOCKS, numThreads, 0, stream>>>(d_dstArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


__global__ void amdL1WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t totalElements, size_t reps)
{
    const uint32_t gtid = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;

    const uint32v4 dummy = {gtid, gtid + 1, gtid + 2, gtid + 3};

    #ifdef __HIP_PLATFORM_AMD__
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(dst);
    #endif

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "flat_store_dwordx4 %0, %1\n\t"
                :
                : "v"(baseAddr + i * sizeof(uint32v4)),
                  "v"(dummy)
                : "memory"
            );
            #endif
        }
    }
}

static std::tuple<double, double> amdL1WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    const size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    const size_t totalThreads = static_cast<size_t>(numBlocks) * numThreads;

    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(totalThreads);
    
    //warm up
    amdL1WriteBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, totalElements, WARMUP_REPS);
    
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    amdL1WriteBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark 
{
    CacheBandwidthResult measureL1WriteBandwidthSweep(size_t arraySizeBytes) 
    {
        auto stream = util::createStreamForCU(0);

        uint32_t minThreads = util::getWarpSize();
        uint32_t maxThreads = util::getMaxThreadsPerBlock();

        size_t minReps = MIN_REPS;
        size_t maxReps = MAX_REPS;

        CacheBandwidthResult result{};
        result.measuredBandwidth = 0.0;
        result.dataBytes = arraySizeBytes;
        result.cycles = 0;
        result.time = 0.0;
        result.numThreads = 0;
        result.numBlocks = NUM_BLOCKS;
        result.numReps = 0;

        // Precompute full thread/rep axes for CSV alignment.
        for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
        {
            result.threadsTested.push_back(numThreads);
        }
        for (size_t reps = minReps; reps <= maxReps; reps *= 2)
        {
            result.repsTested.push_back(reps);
        }

        const size_t numThreadSteps = result.threadsTested.size();
        const size_t numRepSteps = result.repsTested.size();
        std::vector<std::vector<double>> grid(numThreadSteps, std::vector<double>(numRepSteps, 0.0));

        // Measure every thread count and repetition count.
        for (size_t ti = 0; ti < numThreadSteps; ++ti)
        {
            const uint32_t numThreads = result.threadsTested[ti];

            for (size_t ri = 0; ri < numRepSteps; ++ri)
            {
                const size_t reps = result.repsTested[ri];

                auto [timeS, bandwidth] = l1WriteBandwidthLauncher(arraySizeBytes, numThreads, reps, stream);

                grid[ti][ri] = bandwidth;

                if (bandwidth > result.measuredBandwidth)
                {
                    result.measuredBandwidth = bandwidth;
                    result.time = timeS;
                    result.numThreads = numThreads;
                    result.numReps = reps;
                }
            }
        }

        util::hipCheck(hipStreamDestroy(stream));

        #ifdef __HIP_PLATFORM_AMD__
        result.blocksTested.push_back(NUM_BLOCKS);
        result.bandwidth3D.push_back(grid);
        #else
        result.bandwidthGridGiBs = grid;
        #endif

        return result;
    }

    namespace amd
    {
        CacheBandwidthResult measureL1WriteBandwidthSweep(size_t arraySizeBytes)
        {
            auto stream = util::createStreamForCU(0);

            const uint32_t numBlocks = 1;
            uint32_t minThreads = util::getWarpSize();
            uint32_t maxThreads = util::getMaxThreadsPerBlock();

            size_t minReps = MIN_REPS;
            size_t maxReps = MAX_REPS;

            CacheBandwidthResult result{};
            result.measuredBandwidth = 0.0;
            result.dataBytes = arraySizeBytes;
            result.cycles = 0;
            result.time = 0.0;
            result.numThreads = 0;
            result.numBlocks = numBlocks;
            result.numReps = 0;

            // Precompute full thread/rep axes for CSV alignment.
            result.blocksTested.push_back(numBlocks);
            for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
            {
                result.threadsTested.push_back(numThreads);
            }
            for (size_t reps = minReps; reps <= maxReps; reps *= 2)
            {
                result.repsTested.push_back(reps);
            }

            const size_t numThreadSteps = result.threadsTested.size();
            const size_t numRepSteps = result.repsTested.size();
            const double UNMEASURED = std::numeric_limits<double>::quiet_NaN();

            result.bandwidth3D.assign(1, std::vector<std::vector<double>>(
                numThreadSteps, std::vector<double>(numRepSteps, UNMEASURED)));

            // Search thread counts from highest to lowest.
            for (size_t ti = numThreadSteps; ti-- > 0; )
            {
                const uint32_t numThreads = result.threadsTested[ti];

                for (size_t ri = 0; ri < numRepSteps; ++ri)
                {
                    const size_t reps = result.repsTested[ri];

                    auto [timeS, bandwidth] = amdL1WriteBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, stream);

                    result.bandwidth3D[0][ti][ri] = bandwidth;

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numReps = reps;
                    }
                }
            }

            util::hipCheck(hipStreamDestroy(stream));

            return result;
        }
    }
}
