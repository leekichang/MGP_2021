#include <stdlib.h>
// #include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>

void filter_parallel(float *array_out_parallel, int start_idx, int end_idx)
{
  int i;
  for (i = start_idx; i < end_idx; ++i){
    array_out_parallel[i] = 0.25f * i + 0.5f * (i + 1) + 0.25f * (i + 2);
  }
}

int main(int argc, char **argv)
{

  if (argc < 2)
    std::cout << "Usage : ./filter num_items" << std::endl;
  int N = atoi(argv[1]);
  int NT = 32; //Default value. change it as you like.

  //0. Initialize

  const int FILTER_SIZE = 3;
  const float k[FILTER_SIZE] = {0.25, 0.5, 0.25};
  float *array_in = new float[N];
  float *array_out_serial = new float[N];
  float *array_out_parallel = new float[N];
  {
    std::chrono::duration<double> diff;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++)
    {
      array_in[i] = i;
    }
    auto end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "init took " << diff.count() << " sec" << std::endl;
  }

  {
    //1. Serial
    std::chrono::duration<double> diff;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < N - 2; i++)
    {
      for (int j = 0; j < FILTER_SIZE; j++)
      {
        array_out_serial[i] += array_in[i + j] * k[j];
      }
    }
    auto end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "serial 1D filter took " << diff.count() << " sec" << std::endl;
  }

  {
    //2. parallel 1D filter
    std::chrono::duration<double> diff;
    auto start = std::chrono::steady_clock::now();
    /* TODO: put your own parallelized 1D filter here */
    /****************/
    std::vector <std::thread> vecOfThreads;
    int i;
    int j = (N - 2) / NT; 
    for (i = 0; i < NT; ++i ){
      if (i != NT - 1)
        vecOfThreads.push_back(std::thread(filter_parallel, array_out_parallel, j * i, j * (i + 1)));
      else
        vecOfThreads.push_back(std::thread(filter_parallel, array_out_parallel, j * i, N - 2));
    }
    for (auto &thread: vecOfThreads)
       thread.join();

    /****************/
    /* TODO: put your own parallelized 1D filter here */
    auto end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "parallel 1D filter took " << diff.count() << " sec" << std::endl;

    int error_counts = 0;
    const float epsilon = 0.01;
    for (int i = 0; i < N; i++)
    {
      float err = std::abs(array_out_serial[i] - array_out_parallel[i]);
      if (err > epsilon)
      {
        error_counts++;
        if (error_counts < 5)
        {
          std::cout << "ERROR at " << i << ": Serial[" << i << "] = " << array_out_serial[i] << " Parallel[" << i << "] = " << array_out_parallel[i] << std::endl;
          std::cout << "err: " << err << std::endl;
        }
      }
    }

    if (error_counts == 0)
    {
      std::cout << "PASS" << std::endl;
    }
    else
    {
      std::cout << "There are " << error_counts << " errors" << std::endl;
    }
  }
  return 0;
}
