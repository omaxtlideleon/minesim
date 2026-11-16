#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh> 

const long long TOTAL_SIMULATIONS = 100000000000LL; // 100 Billion
const unsigned int NUM_MINES = 10;
const unsigned int WORKGROUP_SIZE = 256; 
const unsigned int FULL_MASK = 0xffffffff;

__device__ __forceinline__ uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t xorshift64(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

__device__ __forceinline__ void FA(uint64_t a, uint64_t b, uint64_t c, uint64_t &s, uint64_t &cout) {
    s = a ^ b ^ c;
    cout = (a & b) | (b & c) | (c & a);
}

__device__ __forceinline__ void HA(uint64_t a, uint64_t b, uint64_t &s, uint64_t &cout) {
    s = a ^ b;
    cout = a & b;
}

__global__ void __launch_bounds__(256) fastSimulationKernel(unsigned long long* results) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x * 2ULL;

    uint64_t rng = idx + 1ULL;
    rng = splitmix64(rng);
    if (rng == 0ULL) rng = 0xDEADBEEFCAFE1234ULL;
    
    uint64_t c[9] = {0};

    const uint64_t NOT_COL0 = 0xFEFEFEFEFEFEFEFEULL;
    const uint64_t NOT_COL7 = 0x7F7F7F7F7F7F7F7FULL;

    for (unsigned long long i = idx * 2ULL; i < TOTAL_SIMULATIONS; i += stride) {
        uint64_t boardA = 0ULL;
        uint64_t boardB = 0ULL;
        uint64_t rA = xorshift64(rng);
        uint64_t rB = xorshift64(rng);

        #pragma unroll
        for (int k = 0; k < 10; ++k) {
            boardA |= (1ULL << (rA & 63u));
            boardB |= (1ULL << (rB & 63u));
            rA >>= 6;
            rB >>= 6;
        }
        while (__popcll(boardA) < NUM_MINES) {
            uint64_t rA_fill = xorshift64(rng);
            #pragma unroll
            for (int k = 0; k < 10 && __popcll(boardA) < NUM_MINES; ++k) {
                boardA |= (1ULL << (rA_fill & 63u));
                rA_fill >>= 6;
            }
        }
        while (__popcll(boardB) < NUM_MINES) {
            uint64_t rB_fill = xorshift64(rng);
            #pragma unroll
            for (int k = 0; k < 10 && __popcll(boardB) < NUM_MINES; ++k) {
                boardB |= (1ULL << (rB_fill & 63u));
                rB_fill >>= 6;
            }
        }
        uint64_t wA  = (boardA & NOT_COL0) >> 1;  uint64_t wB  = (boardB & NOT_COL0) >> 1;
        uint64_t eA  = (boardA & NOT_COL7) << 1;  uint64_t eB  = (boardB & NOT_COL7) << 1;
        uint64_t nA  = boardA >> 8;               uint64_t nB  = boardB >> 8;
        uint64_t sA_ = boardA << 8;               uint64_t sB_ = boardB << 8;
        
        uint64_t nwA = (nA & NOT_COL0) >> 1;      uint64_t nwB = (nB & NOT_COL0) >> 1;
        uint64_t neA = (nA & NOT_COL7) << 1;      uint64_t neB = (nB & NOT_COL7) << 1;
        uint64_t swA = (sA_ & NOT_COL0) >> 1;     uint64_t swB = (sB_ & NOT_COL0) >> 1;
        uint64_t seA = (sA_ & NOT_COL7) << 1;     uint64_t seB = (sB_ & NOT_COL7) << 1;

        uint64_t s1A, c1A, s2A, c2A, s3A, c3A, c4A, s4A, c5A, c6A, c7A, b0A, b1A, b2A, b3A;
        uint64_t s1B, c1B, s2B, c2B, s3B, c3B, c4B, s4B, c5B, c6B, c7B, b0B, b1B, b2B, b3B;

        FA(wA, eA, nA, s1A, c1A);        FA(sA_, nwA, neA, s2A, c2A);
        FA(s1A, s2A, swA, s3A, c3A);      HA(s3A, seA, b0A, c4A);
        FA(c1A, c2A, c3A, s4A, c5A);      HA(s4A, c4A, b1A, c6A);
        HA(c5A, c6A, b2A, c7A);          b3A = c7A;

        FA(wB, eB, nB, s1B, c1B);        FA(sB_, nwB, neB, s2B, c2B);
        FA(s1B, s2B, swB, s3B, c3B);      HA(s3B, seB, b0B, c4B);
        FA(c1B, c2B, c3B, s4B, c5B);      HA(s4B, c4B, b1B, c6B);
        HA(c5B, c6B, b2B, c7B);          b3B = c7B;

        uint64_t emptyA = ~boardA;
        uint64_t emptyB = ~boardB;

        // c[0] is intentionally not computed here! it's derived on
        // the host afterward from the fact that every board has exactly
        // 54 non-mine cells (64 - NUM_MINES), that's a constant.
        // sum(c[0..8]) == 54 * totalBoards always, so c[0] = 54*totalBoards - sum(c[1..8]).
        c[1] += __popcll(emptyA & ~b3A & ~b2A & ~b1A &  b0A) + __popcll(emptyB & ~b3B & ~b2B & ~b1B &  b0B);
        c[2] += __popcll(emptyA & ~b3A & ~b2A &  b1A & ~b0A) + __popcll(emptyB & ~b3B & ~b2B &  b1B & ~b0B);
        c[3] += __popcll(emptyA & ~b3A & ~b2A &  b1A &  b0A) + __popcll(emptyB & ~b3B & ~b2B &  b1B &  b0B);
        c[4] += __popcll(emptyA & ~b3A &  b2A & ~b1A & ~b0A) + __popcll(emptyB & ~b3B &  b2B & ~b1B & ~b0B);
        c[5] += __popcll(emptyA & ~b3A &  b2A & ~b1A &  b0A) + __popcll(emptyB & ~b3B &  b2B & ~b1B &  b0B);
        c[6] += __popcll(emptyA & ~b3A &  b2A &  b1A & ~b0A) + __popcll(emptyB & ~b3B &  b2B &  b1B & ~b0B);
        c[7] += __popcll(emptyA & ~b3A &  b2A &  b1A &  b0A) + __popcll(emptyB & ~b3B &  b2B &  b1B &  b0B);
        c[8] += __popcll(emptyA &  b3A)                       + __popcll(emptyB &  b3B);
    }

    // Allocate shared memoryy
    typedef cub::BlockReduce<uint64_t, WORKGROUP_SIZE> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    #pragma unroll
    for (int i = 0; i < 9; i++) {
        uint64_t block_sum = BlockReduceT(temp_storage).Sum(c[i]);
        __syncthreads(); 

        // only thread 0 runs the atomic write
        if (threadIdx.x == 0 && block_sum > 0) {
            atomicAdd(&results[i], (unsigned long long)block_sum);
        }
        __syncthreads();
    }
}

double calculateStandardError(unsigned long long count, unsigned long long total, double stD) {
    double p = static_cast<double>(count) / static_cast<double>(total);
    double standardError = std::sqrt(p * (1.0 - p) / static_cast<double>(total));
    return stD * standardError;
}

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    int numSMs = props.multiProcessorCount;
    unsigned int blocks = numSMs * 32; 

    std::cout << "Using " << props.name << "\n";
    std::cout << "    Generating boards: " << TOTAL_SIMULATIONS << "\n\n";

    std::vector<unsigned long long> h_results(9, 0);
    unsigned long long* d_results = nullptr;
    cudaMalloc(&d_results, 9 * sizeof(unsigned long long));
    cudaMemset(d_results, 0, 9 * sizeof(unsigned long long));

    auto startTime = std::chrono::high_resolution_clock::now();

    fastSimulationKernel<<<blocks, WORKGROUP_SIZE>>>(d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results.data(), d_results, 9 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    auto endTime = std::chrono::high_resolution_clock::now();

    // c[0] was never computed on the GPU (see kernel comment), we derive it
    // exactly here. Every board has exactly (64 - NUM_MINES) = 54 non-mine
    // cells, always
    {
        unsigned long long sumRest = 0;
        for (int i = 1; i < 9; i++) sumRest += h_results[i];
        h_results[0] = 54ULL * (unsigned long long)TOTAL_SIMULATIONS - sumRest;
    }
    double timeTaken = std::chrono::duration<double>(endTime - startTime).count();

    unsigned long long totalBoards = TOTAL_SIMULATIONS;
    unsigned long long boardWidth = 8;
    unsigned long long boardHeight = 8;
    unsigned long long totalCellsSimulated = totalBoards * boardWidth * boardHeight;
    double stD = 2.576; 

    std::cout << "Simulation Results:" << std::endl;
    for (int i = 0; i < 9; i++) {
        double percentage = (static_cast<double>(h_results[i]) / totalCellsSimulated) * 100.0;
        double confidenceInterval = calculateStandardError(h_results[i], totalCellsSimulated, stD);
        double errorPercentage = confidenceInterval * 100.0;

        std::cout << "    " << i << ": " << h_results[i] << " appearances (";
        if (h_results[i] == 0 || h_results[i] == totalCellsSimulated || confidenceInterval == 0) {
            std::cout << std::fixed << std::setprecision(9) << percentage << "%)" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(9) << percentage << "% ± " << errorPercentage << "%)" << std::endl;
        }
    }

    double boardSpeed = static_cast<double>(totalBoards) / timeTaken;
    std::cout << "\nExtra:" << std::endl;
    std::cout << "    Time taken: " << std::fixed << std::setprecision(3) << timeTaken << "s" << std::endl;
    std::cout << "    Speed: ~" << std::fixed << std::setprecision(0) << boardSpeed << " boards/s" << std::endl;
    std::cout << "    Speed (tiles): ~" << std::fixed << std::setprecision(1) << (boardSpeed * (boardWidth * boardHeight) / 1000000.0) << " Mtiles/s" << std::endl;

    cudaFree(d_results);
    return 0;
}
