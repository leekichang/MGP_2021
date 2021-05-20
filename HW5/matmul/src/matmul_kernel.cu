#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include "matmul.h"
#define TILE_WIDTH 32
using namespace std;

void allocateDeviceMemory(void** M, int size)
{
  cudaError_t err = cudaMalloc(M, size);
  assert(err==cudaSuccess);
}

void deallocateDeviceMemory(void* M)
{
  cudaError_t err = cudaFree(M);
  assert(err==cudaSuccess);
}

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

__global__ void MatMulKernel(int* d_A, int* d_B, int* d_C, int n){
  __shared__ int subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ int subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  int value = 0;
  if (Col < n && Row < n) {
    for (int m = 0; m < n/TILE_WIDTH; ++m){
      subTileA[ty][tx] = d_A[Row*n + m*TILE_WIDTH+tx];
      subTileB[ty][tx] = d_B[(m*TILE_WIDTH+ty)*n + Col];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; ++k){
        value += subTileA[ty][k] * subTileB[k][tx];
      }
      __syncthreads();
    }
    d_C[Row*n + Col] = value;
  }
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int* const d_A, const int* const d_B,  int* const d_C, const int n) {
  // TODO: Implement your CUDA code
  int size = n*n*sizeof(int);

  allocateDeviceMemory((void**)&d_A, size);
  allocateDeviceMemory((void**)&d_B, size);
  allocateDeviceMemory((void**)&d_C, size);

  dim3 dimGrid(ceil(n/TILE_WIDTH), ceil(n/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  cudaMemcpy((int *)d_A, matrixA, size, cudaMemcpyHostToDevice);
  cudaMemcpy((int *)d_B, matrixB, size, cudaMemcpyHostToDevice);

  MatMulKernel<<<dimGrid, dimBlock>>>((int *)d_A, (int *)d_B, (int *)d_C, n);

  cudaMemcpy(matrixC, (int *)d_C, size, cudaMemcpyDeviceToHost);
}

