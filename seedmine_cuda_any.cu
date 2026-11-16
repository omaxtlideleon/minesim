// CUDA Minesweeper simulator for 2D boards of different sizes.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>

using clk = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

// Allow much larger boards
// You can raise MAX_CELLS if needed (watch per-thread local memory).
#define MAX_CELLS 8192
#define MAX_SEGS ((MAX_CELLS + 63) / 64)

static std::string format_rate_long(double tiles_per_sec) {
    const char* suf[] = { "","K","M","B","T","Q" };
    int si = 0;
    while (tiles_per_sec >= 1000.0 && si + 1 < (int)(sizeof(suf)/sizeof(suf[0]))) {
        tiles_per_sec /= 1000.0;
        ++si;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << tiles_per_sec << " " << suf[si] << "tiles/s";
    return ss.str();
}
static std::string adaptive_time_short(ns duration) {
    auto n = duration.count();
    if (n < 1000) {
        return std::to_string(n) + " ns";
    } else if (n < 1000LL * 1000) {
        double val = n / 1000.0;
        std::ostringstream ss; ss << std::fixed << std::setprecision(1) << val << " µs"; return ss.str();
    } else if (n < 1000LL * 1000 * 1000) {
        double val = n / 1000.0 / 1000.0;
        std::ostringstream ss; ss << std::fixed << std::setprecision(1) << val << " ms"; return ss.str();
    } else {
        long long sec = n / 1000000000LL;
        long long h = sec / 3600;
        long long m = (sec % 3600) / 60;
        long long s = sec % 60;
        std::ostringstream ss;
        ss << std::setw(2) << std::setfill('0') << h << ":";
        ss << std::setw(2) << std::setfill('0') << m << ":";
        ss << std::setw(2) << std::setfill('0') << s;
        return ss.str();
    }
}
static std::string adaptive_time(ns duration) {
    auto n = duration.count();
    if (n < 1000) {
        return std::to_string(n) + " ns";
    } else if (n < 1000LL * 1000) {
        double val = n / 1000.0;
        std::ostringstream ss; ss << std::fixed << std::setprecision(3) << val << " µs"; return ss.str();
    } else if (n < 1000LL * 1000 * 1000) {
        double val = n / 1000.0 / 1000.0;
        std::ostringstream ss; ss << std::fixed << std::setprecision(3) << val << " ms"; return ss.str();
    } else {
        double val = n / 1e9;
        std::ostringstream ss; ss << std::fixed << std::setprecision(3) << val << " s"; return ss.str();
    }
}

__device__ __forceinline__ uint32_t lcg_next_dev(uint32_t &s) {
    s = uint32_t(1664525u * s + 1013904223u);
    return s;
}
__device__ __forceinline__ void bit_set(unsigned long long* a, int i) {
    a[i >> 6] |= (1ULL << (i & 63));
}
__device__ __forceinline__ bool bit_get(const unsigned long long* a, int i) {
    return ((a[i >> 6] >> (i & 63)) & 1ULL) != 0ULL;
}

extern "C" __global__
void simulate_boards_kernel_any(
    int L, int H, int W, int N,
    unsigned long long boards_offset, // global board index for seed
    unsigned long long boards_to_do,  // number of boards by htis kernel launch
    uint32_t* out_counts32,           // accumulators length 9 (32-bit)
    uint32_t* out_found32             // ^^
) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= boards_to_do) return;

    uint32_t seed = (uint32_t)((boards_offset + tid) ^ 0x9e3779b9u);

    //sized by L (up to MAX_CELLS)
    int S = (L + 63) >> 6;
    unsigned long long mines[MAX_SEGS];
    for (int i = 0; i < S; ++i) mines[i] = 0ULL;
    int placed = 0;
    while (placed < N) {
        uint32_t r = lcg_next_dev(seed);
        int pos = (int)(r % (uint32_t)L);
        int seg = pos >> 6;
        unsigned long long mask = (1ULL << (pos & 63));
        if ((mines[seg] & mask) == 0ULL) {
            mines[seg] |= mask;
            ++placed;
        }
    }
    uint32_t localCounts[9];
    uint32_t localFound[9];
    #pragma unroll
    for (int k = 0; k <= 8; ++k) { localCounts[k] = 0u; localFound[k] = 0u; }

    // Count neighbor values
    for (int pos = 0; pos < L; ++pos) {
        if (bit_get(mines, pos)) continue; // skip mines
        int r = pos / W;
        int c = pos % W;
        uint32_t cnt = 0u;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
                    int npos = nr * W + nc;
                    if (bit_get(mines, npos)) ++cnt;
                }
            }
        }
        if (cnt > 8u) cnt = 8u;
        localCounts[(int)cnt] += 1u;
        localFound[(int)cnt] = 1u;
    }
    extern __shared__ uint32_t shmem32[]; // size >= 9 + 9
    uint32_t* sh_counts = shmem32;       // 0..8
    uint32_t* sh_found  = shmem32 + 9;   // 9..17
    for (int i = threadIdx.x; i < 9; i += blockDim.x) {
        sh_counts[i] = 0u;
        sh_found[i]  = 0u;
    }
    __syncthreads();
 
    for (int k = 0; k <= 8; ++k) {
        uint32_t c = localCounts[k];
        uint32_t f = localFound[k];
        if (c) atomicAdd(&sh_counts[k], c);
        if (f) atomicAdd(&sh_found[k], f);
    }
    __syncthreads();

    // thread 0 writes block result
    if (threadIdx.x == 0) {
        for (int k = 0; k <= 8; ++k) {
            uint32_t bc = sh_counts[k];
            uint32_t bf = sh_found[k];
            if (bc) atomicAdd(&out_counts32[k], bc);
            if (bf) atomicAdd(&out_found32[k], bf);
        }
    }
}

static void checkCuda(cudaError_t e, const char* where = "") {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    int H = 8, W = 8, N = 10;
    unsigned long long SIMULATIONS = 100000000ULL;
    int threadsPerBlock = 256;
    unsigned long long chunkBoards = 16ull * 1024ull * 1024ull; // 16M default
    bool saveToFile = false;
    std::string filename;

    std::cout << "Board height (default 8): "; if (!(std::cin >> H)) return 0;
    std::cout << "Board width  (default 8): "; if (!(std::cin >> W)) return 0;
    std::cout << "Num mines   (default 10): "; if (!(std::cin >> N)) return 0;
    std::cout << "Simulations (default 100000000): "; if (!(std::cin >> SIMULATIONS)) return 0;
    std::cout << "Threads per block (default 256): "; if (!(std::cin >> threadsPerBlock)) return 0;
    std::cout << "Per-launch chunk boards (default 16777216): "; if (!(std::cin >> chunkBoards)) return 0;
    std::cout << "Save results to file? (0/1): "; if (!(std::cin >> saveToFile)) return 0;
    if (saveToFile) { std::cout << "Output filename: "; std::cin >> filename; }

    if (H <= 0 || W <= 0 || N < 0 || N > H*W) {
        std::cerr << "Invalid board parameters.\n";
        return 1;
    }
    int L = H * W;
    if (L > MAX_CELLS) {
        std::cerr << "Maximum supported cells is " << MAX_CELLS << ". Reduce board size or increase MAX_CELLS and recompile.\n";
        return 1;
    }

    uint32_t* d_out_counts32 = nullptr;
    uint32_t* d_out_found32  = nullptr;
    checkCuda(cudaMalloc((void**)&d_out_counts32, sizeof(uint32_t) * 9), "cudaMalloc out_counts32");
    checkCuda(cudaMalloc((void**)&d_out_found32,  sizeof(uint32_t) * 9), "cudaMalloc out_found32");
    checkCuda(cudaMemset(d_out_counts32, 0, sizeof(uint32_t) * 9), "cudaMemset out_counts32");
    checkCuda(cudaMemset(d_out_found32,  0, sizeof(uint32_t) * 9), "cudaMemset out_found32");
    uint64_t host_counts64[9] = {0};
    uint64_t host_found64[9]  = {0};

    uint32_t host_counts32[9];
    uint32_t host_found32[9];

    unsigned long long remaining = SIMULATIONS;
    unsigned long long processedTotal = 0;
    auto tstart = clk::now();

    while (remaining > 0) {
        unsigned long long doBoards = std::min(remaining, chunkBoards);

        unsigned long long threadsNeeded = doBoards;
        unsigned int blockSize = (unsigned int)threadsPerBlock;
        unsigned long long gridSize = (threadsNeeded + blockSize - 1) / blockSize;
        if (gridSize == 0) gridSize = 1;

        // shared memory: 9 + 9 uint32_t
        size_t sharedBytes = sizeof(uint32_t) * 18;

        simulate_boards_kernel_any<<<(unsigned int)gridSize, blockSize, (unsigned int)sharedBytes>>>(
            L, H, W, N,
            processedTotal,
            doBoards,
            d_out_counts32,
            d_out_found32
        );
        cudaError_t kernErr = cudaGetLastError();
        if (kernErr != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernErr) << "\n";
            return 1;
        }

        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

        checkCuda(cudaMemcpy(host_counts32, d_out_counts32, sizeof(uint32_t) * 9, cudaMemcpyDeviceToHost), "cudaMemcpy counts32->host");
        checkCuda(cudaMemcpy(host_found32,  d_out_found32,  sizeof(uint32_t) * 9, cudaMemcpyDeviceToHost), "cudaMemcpy found32->host");

        for (int k = 0; k <= 8; ++k) {
            host_counts64[k] += (uint64_t)host_counts32[k];
            host_found64[k]  += (uint64_t)host_found32[k];
        }

        // reset accumulators for next chunk
        checkCuda(cudaMemset(d_out_counts32, 0, sizeof(uint32_t) * 9), "cudaMemset out_counts32 reset");
        checkCuda(cudaMemset(d_out_found32,  0, sizeof(uint32_t) * 9), "cudaMemset out_found32 reset");

        processedTotal += doBoards;
        remaining -= doBoards;

        // progress
        auto elapsed = clk::now() - tstart;
        double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
        double pct = (double) processedTotal * 100.0 / (double) SIMULATIONS;
        double boardsPerSec = (elapsedSec > 0.0) ? (processedTotal / elapsedSec) : 0.0;
        double tilesPerSec = boardsPerSec * (double)(L - N);

        unsigned long long nowGlobal = processedTotal;
        ns etaNs(0);
        if (boardsPerSec > 0.0 && nowGlobal < SIMULATIONS) {
            double remainingSec = (double)(SIMULATIONS - nowGlobal) / boardsPerSec;
            etaNs = ns((long long)(remainingSec * 1e9));
        }
        double timePerBoardSec = (boardsPerSec > 0.0) ? (1.0 / boardsPerSec) : 0.0;
        ns timePerBoardNs = ns((long long)(timePerBoardSec * 1e9));

        std::ostringstream ss;
        ss << "\rProgress: " << std::fixed << std::setprecision(3) << pct << "%, ";
        ss << nowGlobal << "/" << SIMULATIONS << " boards, ";
        ss << format_rate_long(tilesPerSec) << ", ";
        ss << "avg: " << adaptive_time(timePerBoardNs) << "/board, ";
        ss << "elapsed " << adaptive_time(std::chrono::duration_cast<ns>(elapsed)) << ", ";
        if (etaNs.count() > 0) ss << "ETA " << adaptive_time_short(etaNs);
        std::cout << ss.str() << std::flush;
    }

    auto tend = clk::now();
    ns elapsed = std::chrono::duration_cast<ns>(tend - tstart);

    std::cout << "\n\nSimulation Results:\n";
    unsigned long long cellsPerBoard = (unsigned long long)(L - N);
    long double totalCells = (long double) processedTotal * (long double) cellsPerBoard;

    for (int k = 0; k <= 8; ++k) {
        long double tot = (long double) host_counts64[k];
        long double pct = (totalCells > 0.0L) ? (tot / totalCells * 100.0L) : 0.0L;
        std::cout << k << ": " << std::fixed << std::setprecision(0) << (long double)host_counts64[k]
                  << " appearances (" << std::setprecision(9) << pct << "%)\n";
    }

    for (int k = 1; k <= 8; ++k) {
        long double boardsWithK = (long double) host_found64[k];
        long double avg = (processedTotal > 0) ? (boardsWithK / (long double) processedTotal) : 0.0L;
        long double pct = avg * 100.0L;
        long double stddev = std::sqrt((double)(avg * (1.0L - avg) / (processedTotal > 0 ? processedTotal : 1)));
        long double stdPct = stddev * 100.0L;
        std::cout << k << " average occurrence per board: " << std::fixed << std::setprecision(9) << pct << "% ± " << std::setprecision(9) << stdPct << "%\n";
    }

    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
    double boardsPerSec = (elapsedSec > 0.0) ? (processedTotal / elapsedSec) : 0.0;
    double tilesPerSec = boardsPerSec * (double)cellsPerBoard;
    double timePerBoardSec = (boardsPerSec > 0.0) ? (1.0 / boardsPerSec) : 0.0;
    ns timePerBoardNs = ns((long long)(timePerBoardSec * 1e9));
    std::cout << "Time taken: " << std::fixed << std::setprecision(3) << elapsedSec << "s\n";
    std::cout << "Speed: ~" << (uint64_t)std::llround(boardsPerSec) << " boards/s\n";
    std::cout << "Speed (tiles): " << format_rate_long(tilesPerSec) << "\n";
    std::cout << "Average time per board: " << adaptive_time(timePerBoardNs) << "\n";

    if (saveToFile) {
        std::ofstream ofs(filename);
        if (ofs) {
            ofs << "H " << H << " W " << W << " N " << N << "\n";
            ofs << "Simulations requested: " << SIMULATIONS << " processed: " << processedTotal << "\n";
            for (int k = 0; k <= 8; ++k) {
                long double tot = (long double) host_counts64[k];
                long double pct = (totalCells > 0.0L) ? (tot / totalCells * 100.0L) : 0.0L;
                ofs << k << ": " << (unsigned long long) host_counts64[k] << " appearances (" << std::setprecision(9) << pct << "%)\n";
            }
            for (int k = 1; k <= 8; ++k) {
                long double boardsWithK = (long double) host_found64[k];
                long double avg = (processedTotal > 0) ? (boardsWithK / (long double) processedTotal) : 0.0L;
                long double pct = avg * 100.0L;
                long double stddev = std::sqrt((double)(avg * (1.0L - avg) / (processedTotal > 0 ? processedTotal : 1)));
                long double stdPct = stddev * 100.0L;
                ofs << k << " average occurrence per board: " << std::fixed << std::setprecision(9) << pct << "% ± " << std::setprecision(9) << stdPct << "%\n";
            }
            ofs << "Time taken: " << std::fixed << std::setprecision(3) << elapsedSec << "s\n";
            ofs << "Speed: ~" << (uint64_t)std::llround(boardsPerSec) << " boards/s\n";
            ofs << "Speed (tiles): ~" << format_rate_long(tilesPerSec) << "\n";
            ofs << "Average time per board: " << adaptive_time(timePerBoardNs) << "\n";
            ofs.close();
            std::cout << "Results written to " << filename << "\n";
        } else {
            std::cerr << "Failed to open output file.\n";
        }
    }

    cudaFree(d_out_counts32);
    cudaFree(d_out_found32);

    std::cout << "Press Enter to exit... ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return 0;
}
