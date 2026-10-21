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


__global__ void l1ReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps) 
{
    const uint32_t tid = threadIdx.x;
    const uint32v4* base = src + tid * elementsPerThread;

    #ifdef __HIP_PLATFORM_AMD__
    const uint64_t addr0 = reinterpret_cast<uint64_t>(base);
    #endif

    uint32v4 dummy {0, 0, 0, 0};

    // Warm up L1
    for (size_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            uint32v4 loaded;
            #ifdef __HIP_PLATFORM_AMD__
            asm volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr0 + i * sizeof(uint32v4))
                : "memory"
            );
            #endif

#ifdef __HIP_PLATFORM_NVIDIA__
using vec4 = uint4;
#else
using vec4 = uint32v4;
#endif

__global__ void l1ReadBandwidthKernel(vec4* dst, const vec4* src, size_t totalElements, size_t reps, size_t groupLoadStride)
{
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    vec4 dummy {0, 0, 0, 0};

    for (size_t rep = 0; rep < reps; rep += LOADS_PER_GROUP)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            #pragma unroll
            for (size_t k = 0; k < LOADS_PER_GROUP; ++k)
            {
                const vec4 loaded = src[i + k * groupLoadStride];

                dummy.x ^= loaded.x;
                dummy.y ^= loaded.y;
                dummy.z ^= loaded.z;
                dummy.w ^= loaded.w;
            }
        }
    }

    dst[gtid] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> l1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    const size_t totalElements = arraySizeBytes / sizeof(vec4);
    const size_t totalThreads = static_cast<size_t>(NUM_BLOCKS) * numThreads;

    // Allocate device arrays
    vec4 *d_srcArr = util::allocateGPUMemory<vec4>(totalElements);
    vec4 *d_dstArr = util::allocateGPUMemory<vec4>(totalThreads);

    // Warm up
    l1ReadBandwidthKernel<<<NUM_BLOCKS, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, WARMUP_REPS, GROUP_LOAD_STRIDE);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    l1ReadBandwidthKernel<<<NUM_BLOCKS, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, reps, GROUP_LOAD_STRIDE);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


__global__ void amdL1ReadBandwidthKernel(uint32v4* __restrict__ dst, const uint32v4* __restrict__ src, size_t totalElements, size_t reps) 
{
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint32v4 dummy {0, 0, 0, 0};

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            uint32v4 loaded;

            #ifdef __HIP_PLATFORM_AMD__
            const uint64_t addr = reinterpret_cast<uint64_t>(src + i);

            __asm__ volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr)
                : "memory"
            );
            #endif

            dummy.x ^= loaded.x;
        }
    }

    dst[gtid] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> amdL1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, hipStream_t stream) 
{
    const size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    const size_t totalThreads = static_cast<size_t>(numBlocks) * numThreads;

    // Allocate device arrays
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(totalElements);
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(totalThreads);

    // Warm up
    amdL1ReadBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    amdL1ReadBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark 
{
    CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes) 
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

                auto [timeS, bandwidth] = l1ReadBandwidthLauncher(arraySizeBytes, numThreads, reps, stream);

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
        CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes)
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

                    auto [timeS, bandwidth] = amdL1ReadBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, stream);

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
