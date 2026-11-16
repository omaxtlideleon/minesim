# Minesim

Minesim builds random Minesweeper boards and counts how many neighboring mines each safe tile has. This repo has one CPU program and two CUDA programs: an any-size sampler and a faster 8x8 bitboard sampler.

## Build and run

Build from a Visual Studio Developer PowerShell with the NVIDIA CUDA Toolkit available:

```powershell
./build.ps1 all
```

Choose a program to run:

```powershell
./build/seedmine2d.exe
./build/seedmine_cuda_any.exe
./build/minesweep_derived_bucket.exe
```

The CPU and any-size CUDA versions ask for board dimensions, mine count, and run settings. The CPU version also asks how many threads to use. The derived-bucket CUDA version is fixed to 8x8 boards with 10 mines and runs 100 billion boards.

Build one program at a time with `./build.ps1 cpu`, `./build.ps1 cuda-any`, or `./build.ps1 cuda-derived`. All build output stays in `build/` (git ignores this)

## How they work

The CPU program and any-size CUDA program place mines on each board, then count the neighboring mines for every safe tile. The derived-bucket program packs an 8x8 board into a 64-bit value and counts neighboring mines with bit operations. It combines results in blocks and derives the zero-neighbor count from the total number of safe tiles.

## Results and limits

On the RTX 4060, the any-size CUDA program averaged about 406 million boards per second across five 1-billion-board runs. The derived-bucket source in this repo ran 100 billion boards in 4.528 seconds, about 22.09 billion boards per second. See [RESULTS.md](RESULTS.md) for the run details.

The CPU and any-size CUDA samplers use the low six bits of an LCG to choose cells on a 64-cell board. (Those bits repeat every 64 steps, limiting those runs to at most 64 mine layouts.)

## Files

- `seedmine2d.cpp` is the multithreaded CPU program.
- `seedmine_cuda_any.cu` is the configurable CUDA sampler.
- `minesweep_derived_bucket.cu` is the fixed-size CUDA bitboard sampler.
- `build.ps1` builds the three programs.
- `RESULTS.md` records the benchmark runs and checks.
