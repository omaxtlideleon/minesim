# Benchmark results

These runs use 8x8 boards with 10 mines on an NVIDIA GeForce RTX 4060. The CPU was an Intel Core i5-12400F (with 12 logical processors.)

## CPU and any-size CUDA baseline

The CPU was built with `/O2 /EHsc /std:c++17` and run with 12 worker threads. The any-size CUDA program was built with `-O3 -std=c++20`, 256 threads per block, and a 16,777,216 chunk size.

| Test | CPU boards | CPU time | CPU boards/s | CUDA boards | CUDA time | CUDA boards/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100,000,000 | 7.590 s | 13,175,908 | 1,000,000,000 | 2.532 s | 394,868,822 |
| 2 | 100,000,000 | 7.302 s | 13,694,497 | 1,000,000,000 | 2.467 s | 405,294,820 |
| 3 | 100,000,000 | 7.801 s | 12,818,724 | 1,000,000,000 | 2.430 s | 411,563,638 |
| 4 | 100,000,000 | 7.372 s | 13,564,101 | 1,000,000,000 | 2.435 s | 410,648,527 |
| 5 | 100,000,000 | 7.247 s | 13,799,489 | 1,000,000,000 | 2.449 s | 408,304,222 |
| Average | 100,000,000 | 7.462 s | 13,410,544 | 1,000,000,000 | 2.463 s | 406,136,006 |

## Derived-bucket CUDA

This is one run of the `minesweep_derived_bucket.cu` source included in this repo. It was built with `nvcc -O3 -std=c++20`.

| Boards | Time | Boards/s |
| ---: | ---: | ---: |
| 100,000,000,000 | 4.528 s | 22,085,103,178 |
