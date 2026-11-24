#define FMT_UNICODE 0

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <print>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/universal_vector.h>

#include "Timer.hpp"

using uint = unsigned;

template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max, unsigned seed);

template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError = 1e-4);

template<std::size_t N, typename F>
void
unroll (F&& f);

template<typename T>
void
parallelQuickSort (std::vector<T>& arr, int l, int r, int depth = 0);

template<typename T>
void
sleepSort (std::vector<T>& arr);

void
oddEvenSortCPU (std::vector<int>& arr);

void
oddEvenSortGPU (int* d_arr, int n);

__global__ void
oddEvenKernel (int* arr, int n, int phase);

void
bitonicSortCPUParallel (std::vector<int>& arr);

void
bitonicSortGPU (thrust::universal_vector<int>& vec);
__global__ void
bitonicKernel (int* arr, int j, int k, int n);

int
main (int argc, char* argv[])
{
  cudaError_t err = cudaFree (0);

  if (err != cudaSuccess)
  {
    printf ("CUDA init failed: %d (%s)\n", err, cudaGetErrorString (err));
    return 1;
  }

  if (argc == 1)
    std::print ("N ==> ");
  else if (argc > 2)
  {
    std::print ("Too Many Arguments!\nOnly 0 and 1 arguments allowed!");
    return 1;
  }
  uint n;
  if (argc == 1)
    std::cin >> n;
  else if (argc == 2)
    n = std::stoi (argv[1]);
  Timer timer;

  thrust::universal_vector<int> vec (n);

  fillRandom (std::span<int> (vec.data ().get (), vec.size ()), 0, 5000, 1);
  if (argc == 1)
    std::print ("\n--- Sorting {} elements ---\n\n", n);
  else if (argc == 2)
    std::print ("--- Sorting {} elements ---\n\n", n);

  std::vector<int> base (vec.begin (), vec.end ());

  // std sort
  auto v = base;

  timer.start ();
  std::sort (v.begin (), v.end ());
  timer.stop ();

  bool stdOk = std::is_sorted (v.begin (), v.end ());
  double stdTime = timer.getElapsedMs ();

  // thrust std sort
  thrust::universal_vector<int> v2 = vec;

  timer.start ();
  thrust::sort (v2.begin (), v2.end ());
  cudaDeviceSynchronize ();
  timer.stop ();

  bool thrustSTDok = thrust::is_sorted (v2.begin (), v2.end ());
  double thrustSTDTime = timer.getElapsedMs ();

  // parallel quick sort (CPU)
  auto v3 = base;

  timer.start ();
  parallelQuickSort (v3, 0, static_cast<int> (v3.size ()) - 1);
  timer.stop ();

  bool quickCPUok = std::is_sorted (v3.begin (), v3.end ());
  double quickCPUTime = timer.getElapsedMs ();

  // odd-even sort (CPU)
  bool oddEvenCPUok;
  double oddEvenCPUTime;
  if (n <= 100000)
  {
    auto v4 = base;

    timer.start ();
    oddEvenSortCPU (v4);
    timer.stop ();
    oddEvenCPUok = std::is_sorted (v4.begin (), v4.end ());
    oddEvenCPUTime = timer.getElapsedMs ();
  }

  // odd-even sort (GPU)
  bool oddEvenGPUok;
  double oddEvenGPUTime;
  if (n <= 100000)
  {
    thrust::universal_vector<int> v5 = vec;

    timer.start ();
    oddEvenSortGPU (v5.data ().get (), v5.size ());
    timer.stop ();

    oddEvenGPUok = thrust::is_sorted (v5.begin (), v5.end ());
    oddEvenGPUTime = timer.getElapsedMs ();
  }

  // bitonic sort (CPU)
  auto v6 = base;

  timer.start ();
  bitonicSortCPUParallel (v6);
  timer.stop ();

  bool bitonicCPUok = std::is_sorted (v6.begin (), v6.end ());
  double bitonicCPUTime = timer.getElapsedMs ();

  // bitonic sort (GPU)
  thrust::universal_vector<int> v7 = vec;

  timer.start ();
  bitonicSortGPU (v7);
  timer.stop ();

  bool bitonicGPUok = thrust::is_sorted (v7.begin (), v7.end ());
  double bitonicGPUTime = timer.getElapsedMs ();

  // Print results
  std::print ("{:<18} {:>12}   {}\n", "Algorithm", "Time (ms)", "Status");
  std::print ("{:-<18} {:-<12}   {:-<6}\n", "", "", "");

  std::print (
    "{:<18} {:>12.3f}   {}\n", "std::sort", stdTime, stdOk ? "OK" : "FAIL");
  std::print ("{:<18} {:>12.3f}   {}\n",
              "thrust::sort",
              thrustSTDTime,
              thrustSTDok ? "OK" : "FAIL");
  std::print ("{:<18} {:>12.3f}   {}\n",
              "(CPU) quick",
              quickCPUTime,
              quickCPUok ? "OK" : "FAIL");
  // if (n <= 100000)
  // {
    std::print ("{:<18} {:>12.3f}   {}\n",
                "(CPU) odd-even",
                oddEvenCPUTime,
                oddEvenCPUok ? "OK" : "FAIL");
    std::print ("{:<18} {:>12.3f}   {}\n",
                "(GPU) odd-even",
                oddEvenGPUTime,
                oddEvenGPUok ? "OK" : "FAIL");
  // }
  // else
  // {
    // std::print ("{:<18} {:>12}   {}\n", "(CPU) odd-even", "N/A", "Skipped");
    // std::print ("{:<18} {:>12}   {}\n", "(GPU) odd-even", "N/A", "Skipped");
  // }
  std::print ("{:<18} {:>12.3f}   {}\n",
              "(CPU) bitonic",
              bitonicCPUTime,
              bitonicCPUok ? "OK" : "FAIL");

  std::print ("{:<18} {:>12.3f}   {}\n\n\n",
              "(GPU) bitonic",
              bitonicGPUTime,
              bitonicGPUok ? "OK" : "FAIL");
  return 0;
}

template<typename T>
void
parallelQuickSort (std::vector<T>& arr, int l, int r, int depth)
{
  if (l >= r)
    return;

  T pivot = arr[(l + r) / 2];
  int i = l, j = r;

  while (i <= j)
  {
    while (arr[i] < pivot)
      ++i;
    while (arr[j] > pivot)
      --j;

    if (i <= j)
    {
      std::swap (arr[i], arr[j]);
      ++i;
      --j;
    }
  }

  if (depth < 4)
  {
    std::thread t1 (parallelQuickSort<T>, std::ref (arr), l, j, depth + 1);
    std::thread t2 (parallelQuickSort<T>, std::ref (arr), i, r, depth + 1);
    t1.join ();
    t2.join ();
  }
  else
  {
    parallelQuickSort (arr, l, j, depth + 1);
    parallelQuickSort (arr, i, r, depth + 1);
  }
}

template<typename T>
void
sleepSort (std::vector<T>& arr)
{
  std::vector<T> result;
  std::mutex mtx;
  std::vector<std::thread> threads;

  for (T v : arr)
  {
    threads.emplace_back (
      [v, &result, &mtx] ()
      {
        std::this_thread::sleep_for (std::chrono::microseconds (v * 1000));

        std::lock_guard lock (mtx);
        result.push_back (v);
      });
  }

  for (auto& t : threads)
    t.join ();

  arr = result;
}

template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max, unsigned seed)
{
  static std::minstd_rand gen (seed);

  if constexpr (std::is_integral_v<T>)
  {
    std::uniform_int_distribution<T> dist (min, max);
    std::ranges::generate (seq, [&] () { return dist (gen); });
  }
  else
  {
    std::uniform_real_distribution<T> dist (min, max);
    std::ranges::generate (seq, [&] () { return dist (gen); });
  }
}

void
oddEvenSortCPU (std::vector<int>& arr)
{
  int n = arr.size ();

  for (int phase = 0; phase < n; ++phase)
  {
    int start = phase % 2;

#pragma omp parallel for
    for (int i = start; i < n - 1; i += 2)
    {
      if (arr[i] > arr[i + 1])
      {
        std::swap (arr[i], arr[i + 1]);
      }
    }
  }
}

__global__ void
oddEvenKernel (int* arr, int n, int phase)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int i = tid * 2 + (phase & 1);

  if (i + 1 < n)
  {
    int a = arr[i];
    int b = arr[i + 1];

    if (a > b)
    {
      arr[i] = b;
      arr[i + 1] = a;
    }
  }
}

void
oddEvenSortGPU (int* d_arr, int n)
{
  int threads = 256;
  int pairs = (n + 1) / 2;
  int blocks = (pairs + threads - 1) / threads;

  for (int phase = 0; phase < n; ++phase)
  {
    oddEvenKernel<<<blocks, threads>>> (d_arr, n, phase);
    cudaDeviceSynchronize ();
  }
}

void
bitonicSortCPUParallel (std::vector<int>& vec)
{
  int n = 1 << static_cast<int> (std::ceil (std::log2 (vec.size ())));
  vec.resize (1 << static_cast<int> (std::ceil (std::log2 (n))),
              std::numeric_limits<int>::max ());

  for (int k = 2; k <= n; k <<= 1)
  {
    for (int j = k >> 1; j > 0; j >>= 1)
    {

#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i)
      {
        int ixj = i ^ j;

        if (ixj > i)
        {
          bool ascending = ((i & k) == 0);

          int a = vec[i];
          int b = vec[ixj];

          if ((ascending && a > b) || (!ascending && a < b))
          {
            vec[i] = b;
            vec[ixj] = a;
          }
        }
      }
    }
  }
}

void
bitonicSortGPU (thrust::universal_vector<int>& vec)
{
  int n = 1 << static_cast<int> (std::ceil (std::log2 (vec.size ())));
  vec.resize (1 << static_cast<int> (std::ceil (std::log2 (n))),
              std::numeric_limits<int>::max ());

  int* d_arr = thrust::raw_pointer_cast (vec.data ());
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  for (int k = 2; k <= n; k <<= 1)
  {
    for (int j = k >> 1; j > 0; j >>= 1)
    {
      bitonicKernel<<<blocks, threads>>> (d_arr, j, k, n);
      cudaDeviceSynchronize ();
    }
  }
}

__global__ void
bitonicKernel (int* arr, int j, int k, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int ixj = i ^ j;

  if (ixj > i && ixj < n)
  {
    bool ascending = ((i & k) == 0);

    int a = arr[i];
    int b = arr[ixj];

    if ((ascending && a > b) || (!ascending && a < b))
    {
      arr[i] = b;
      arr[ixj] = a;
    }
  }
}
