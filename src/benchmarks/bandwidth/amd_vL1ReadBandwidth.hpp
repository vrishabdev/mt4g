#pragma once

#include <cstddef>

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure vL1 read bandwidth of a single AMD CU (one block pinned to CU 0), sweeping threads and repetitions.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measurevL1ReadBandwidth(size_t arraySizeBytes);
    }
}
