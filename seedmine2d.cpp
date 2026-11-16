// Multithreaded CPU Minesweeper simulator for 2D boards.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <cassert>
#define NOMINMAX
#ifdef _WIN32
#include <windows.h> // For SetConsoleOutputCP and CP_UTF8
#endif
using clk = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

// Portable popcount64 (Kernighan's algorithm)
#include <intrin.h>

// Hardware-accelerated popcount64
static inline unsigned popcount64(uint64_t x) {
#ifdef _MSC_VER
    return (unsigned)__popcnt64(x);
#else
    return (unsigned)__builtin_popcountll(x);
#endif
}
struct BitVec {
    std::vector<uint64_t> words;
    int L;
    BitVec(int bits=0) : L(bits) {
        words.resize((L+63)/64, 0ULL);
    }
    void clear() { std::fill(words.begin(), words.end(), 0ULL); }
    void set(int idx) { words[idx>>6] |= (1ULL << (idx & 63)); }
    bool test(int idx) const { return (words[idx>>6] >> (idx & 63)) & 1ULL; }
};

static inline uint32_t lcg_next(uint32_t &s) {
    s = uint32_t(1664525u * s + 1013904223u);
    return s;
}

struct AdaptiveRate {
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
            // Only show up to Hours (H:MM:SS) for short time.
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
    
    // Use the long version for final output
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
};

struct BitboardGenerator {
    int H, W, N, L;
    std::vector<std::vector<int>> neighbors; // neighbor indices for each cell

    BitboardGenerator(int h=8, int w=8, int n=10) : H(h), W(w), N(n) {
        L = H*W;
        neighbors.resize(L);
        for (int r=0;r<H;++r) {
            for (int c=0;c<W;++c) {
                int pos = r*W+c;
                for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
                    if (dr==0 && dc==0) continue;
                    int nr=r+dr, nc=c+dc;
                    if (nr>=0 && nr<H && nc>=0 && nc<W) {
                        neighbors[pos].push_back(nr*W+nc);
                    }
                }
            }
        }
    }

    // Generate mines and count
    void generate_and_count(uint32_t &seed,
                            BitVec &out_mines,
                            std::array<uint64_t,9> &out_counts,
                            std::array<uint64_t,9> &out_found) const {
        out_mines = BitVec(L);
        out_counts.fill(0);
        out_found.fill(0);

        // Place N mines with rejection sampling
        int placed=0;
        while (placed<N) {
            uint32_t r = lcg_next(seed);
            int pos = r % L;
            if (!out_mines.test(pos)) {
                out_mines.set(pos);
                ++placed;
            }
        }

        // Count neighbors
        for (int pos=0; pos<L; ++pos) {
            if (out_mines.test(pos)) continue;
            unsigned cnt=0;
            for (int npos: neighbors[pos]) {
                if (out_mines.test(npos)) ++cnt;
            }
            if (cnt>8) cnt=8;
            out_counts[cnt]++;   // increment count of tiles with this value
            out_found[cnt]=1;    // mark that this value exists
        }
    }
};


struct ThreadResult {
    std::array<long double,9> totalCounts;
    std::array<uint64_t,9> foundCounts;
    uint64_t boardsProcessed;
    ThreadResult() : totalCounts{}, foundCounts{}, boardsProcessed(0) {}
};

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    int H = 8, W = 8, N = 10;
    uint64_t SIMULATIONS = 10000000ULL;
    int THREADS = std::max(1u, std::thread::hardware_concurrency());
    double perThreadSeconds = 0.0;
    double globalSeconds = 0.0;
    bool saveToFile = false;
    std::string filename;

    std::cout << "Board height (default 8): "; if (!(std::cin >> H)) return 0;
    std::cout << "Board width  (default 8): "; if (!(std::cin >> W)) return 0;
    std::cout << "Num mines   (default 10): "; if (!(std::cin >> N)) return 0;
    std::cout << "Simulations (default 10000000): "; if (!(std::cin >> SIMULATIONS)) return 0;
    std::cout << "Threads     (default hardware): "; if (!(std::cin >> THREADS)) return 0;
    std::cout << "Per-thread time limit seconds (0 = none): "; if (!(std::cin >> perThreadSeconds)) return 0;
    std::cout << "Global time limit seconds (0 = none): "; if (!(std::cin >> globalSeconds)) return 0;
    std::cout << "Save results to file? (0/1): "; if (!(std::cin >> saveToFile)) return 0;
    if (saveToFile) { std::cout << "Output filename: "; std::cin >> filename; }

    if (H <= 0 || W <= 0 || N < 0 || N > H*W) {
        std::cerr << "Invalid board parameters.\n";
        return 1;
    }

    BitboardGenerator gen(H, W, N);

    uint64_t sims = SIMULATIONS;
    uint64_t baseSimPerThread = sims / THREADS;
    uint64_t extra = sims % THREADS;

    std::vector<ThreadResult> results(THREADS);
    std::atomic<uint64_t> globalBoardsProcessed{0};
    std::array<long double,9> globalTotalCounts;
    std::array<uint64_t,9> globalFoundCounts;
    for (int i = 0; i < 9; ++i) { globalTotalCounts[i] = 0.0L; globalFoundCounts[i] = 0; }

    std::mutex coutMutex;
    std::mutex aggMutex;
    auto startTime = clk::now();

    uint64_t progressInterval = std::max<uint64_t>(1, sims / 100);
    std::atomic<uint64_t> nextProgressAt{progressInterval};

    std::vector<std::thread> pool;
    pool.reserve(THREADS);

    for (int ti = 0; ti < THREADS; ++ti) {
        uint64_t threadSims = baseSimPerThread + (ti < (int)extra ? 1 : 0);
        pool.emplace_back([&, ti, threadSims]() {
            ThreadResult local;
            uint32_t seed = static_cast<uint32_t>(
                static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ (uint64_t)(ti * 0x9e3779b9u)
            );
            uint64_t processed = 0;
            auto threadStart = clk::now();
            ns threadLimitNs = ns(0);
            if (perThreadSeconds > 0.0) threadLimitNs = ns((long long)(perThreadSeconds * 1e9));
            constexpr uint64_t batchSize = 10000; // tune this (1000–10000 is probably typical)

            for (uint64_t s = 0; s < threadSims; ++s) {
                if (threadLimitNs.count() > 0) {
                    if (clk::now() - threadStart >= threadLimitNs) break;
                }
                if (globalSeconds > 0.0) {
                    if (clk::now() - startTime >= ns((long long)(globalSeconds * 1e9))) break;
                }

                BitVec mines(gen.H*gen.W);
                std::array<uint64_t,9> counts;
                std::array<uint64_t,9> found;
                gen.generate_and_count(seed, mines, counts, found);

                for (int k=0;k<=8;++k)
                    local.totalCounts[k] += counts[k];
                for (int k=1;k<=8;++k)
                    local.foundCounts[k] += found[k];

                ++processed;

                // batch updates
                if ((processed % batchSize) == 0) {
                    uint64_t nowGlobal = globalBoardsProcessed.fetch_add(batchSize) + batchSize;
                  
                    if (nowGlobal >= nextProgressAt.load(std::memory_order_relaxed)) {
                        uint64_t prev = nextProgressAt.fetch_add(progressInterval, std::memory_order_acq_rel);
                        if (nowGlobal >= prev) {
                            auto elapsed = clk::now() - startTime;
                            double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
                            double pct = (double) nowGlobal * 100.0 / (double) sims;
                            double boardsPerSec = (elapsedSec > 0.0) ? (nowGlobal / elapsedSec) : 0.0;
                            double tilesPerSec = boardsPerSec * (double)(H * W - N);
                            ns etaNs(0);
                          
                            if (boardsPerSec > 0.0 && nowGlobal < sims) {
                                double remainingSec = (double)(sims - nowGlobal) / boardsPerSec;
                                etaNs = ns((long long)(remainingSec * 1e9));
                            }

                            double timePerBoardSec = (boardsPerSec > 0.0) ? (1.0 / boardsPerSec) : 0.0;
                            ns timePerBoardNs = ns((long long)(timePerBoardSec * 1e9));
                          
                            std::ostringstream ss;
                            ss << "\rProgress: " << std::fixed << std::setprecision(3) << pct << "%, ";
                            ss << nowGlobal << "/" << sims << " boards, ";
                            ss << AdaptiveRate::format_rate_long(tilesPerSec) << ", ";
                            ss << "avg: " << AdaptiveRate::adaptive_time(timePerBoardNs) << "/board, ";
                            ss << "elapsed " << AdaptiveRate::adaptive_time(std::chrono::duration_cast<ns>(elapsed)) << ", ";
                            if (etaNs.count() > 0) {
                                ss << "ETA " << AdaptiveRate::adaptive_time_short(etaNs);
                            }
                            std::lock_guard<std::mutex> lk(coutMutex);
                            std::cout << ss.str() << std::flush;
                        }
                    }
                }
            }
            if (processed % batchSize != 0) {
                globalBoardsProcessed.fetch_add(processed % batchSize);
            }
            {
                std::lock_guard<std::mutex> lk(aggMutex);
                for (int k = 0; k <= 8; ++k) globalTotalCounts[k] += local.totalCounts[k];
                for (int k = 0; k <= 8; ++k) globalFoundCounts[k] += local.foundCounts[k];
            }
            results[ti] = local;
            results[ti].boardsProcessed = processed;
            
        });
    }

    for (auto &t : pool) if (t.joinable()) t.join();

    uint64_t totalProcessed = globalBoardsProcessed.load();

    auto endTime = clk::now();
    ns elapsed = std::chrono::duration_cast<ns>(endTime - startTime);

    std::cout << "\n\nSimulation Results:\n";
    uint64_t cellsPerBoard = H * W - N;
    long double totalCells = (long double) totalProcessed * (long double) cellsPerBoard;

    for (int k = 0; k <= 8; ++k) {
        long double tot = globalTotalCounts[k];
        long double pct = (totalCells > 0.0L) ? (tot / totalCells * 100.0L) : 0.0L;
        std::cout << k << ": " << std::fixed << std::setprecision(0) << (long double)tot
                  << " appearances (" << std::setprecision(9) << pct << "%)\n";
    }

    for (int k = 1; k <= 8; ++k) {
        long double boardsWithK = (long double) globalFoundCounts[k];
        long double avg = (totalProcessed > 0) ? (boardsWithK / (long double) totalProcessed) : 0.0L;
        long double pct = avg * 100.0L;
        long double stddev = std::sqrt((double)(avg * (1.0L - avg) / (totalProcessed > 0 ? totalProcessed : 1)));
        long double stdPct = stddev * 100.0L;
        std::cout << k << " average occurrence per board: " << std::fixed << std::setprecision(9) << pct << "% ± " << std::setprecision(9) << stdPct << "%\n";
    }

    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
    double boardsPerSec = (elapsedSec > 0.0) ? (totalProcessed / elapsedSec) : 0.0;
    double tilesPerSec = boardsPerSec * (double)cellsPerBoard;
    double timePerBoardSec = (boardsPerSec > 0.0) ? (1.0 / boardsPerSec) : 0.0;
    ns timePerBoardNs = ns((long long)(timePerBoardSec * 1e9));
    std::cout << "Time taken: " << std::fixed << std::setprecision(3) << elapsedSec << "s\n";
    std::cout << "Speed: ~" << (uint64_t)std::llround(boardsPerSec) << " boards/s\n";
    std::cout << "Speed (tiles): " << AdaptiveRate::format_rate_long(tilesPerSec) << "\n";
    std::cout << "Average time per board: " << AdaptiveRate::adaptive_time(timePerBoardNs) << "\n";

    if (saveToFile) {
        std::ofstream ofs(filename);
        if (ofs) {
            double timePerBoardSec = (boardsPerSec > 0.0) ? (1.0 / boardsPerSec) : 0.0;
            ns timePerBoardNs = ns((long long)(timePerBoardSec * 1e9));
            ofs << "H " << H << " W " << W << " N " << N << "\n";
            ofs << "Simulations requested: " << SIMULATIONS << " processed: " << totalProcessed << "\n";
            for (int k = 0; k <= 8; ++k) {
                long double tot = globalTotalCounts[k];
                long double pct = (totalCells > 0.0L) ? (tot / totalCells * 100.0L) : 0.0L;
                ofs << k << ": " << (uint64_t) tot << " appearances (" << std::setprecision(9) << pct << "%)\n";
            }
            for (int k = 1; k <= 8; ++k) {
                long double boardsWithK = (long double) globalFoundCounts[k];
                long double avg = (totalProcessed > 0) ? (boardsWithK / (long double) totalProcessed) : 0.0L;
                long double pct = avg * 100.0L;
                long double stddev = std::sqrt((double)(avg * (1.0L - avg) / (totalProcessed > 0 ? totalProcessed : 1)));
                long double stdPct = stddev * 100.0L;
                ofs << k << " average occurrence per board: " << std::fixed << std::setprecision(9) << pct << "% ± " << std::setprecision(9) << stdPct << "%\n";
            }
            ofs << "Time taken: " << std::fixed << std::setprecision(3) << elapsedSec << "s\n";
            ofs << "Speed: ~" << (uint64_t)std::llround(boardsPerSec) << " boards/s\n";
            ofs << "Speed (tiles): ~" << AdaptiveRate::format_rate_long(tilesPerSec) << "\n";
            ofs << "Average time per board: " << AdaptiveRate::adaptive_time(timePerBoardNs) << "\n";
            ofs.close();
            std::cout << "Results written to " << filename << "\n";
        } else {
            std::cerr << "Failed to open output file.\n";
        }
    }

    std::cout << "Press Enter to exit... ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return 0;
}
