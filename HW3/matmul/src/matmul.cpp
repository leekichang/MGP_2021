#include "matmul.h"
#include <x86intrin.h>

void matmul_ref(const int *const matrixA, const int *const matrixB,
                int *const matrixC, const int n)
{
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

void transpose(const int *const matrix_origin, int *const matrix_transpose, int n)
{
  int i;
  for (i = 0; i < n * n; ++i)
  {
    matrix_transpose[(i / n) * n + i % n] = matrix_origin[(i % n) * n + i / n];
  }
}

void matmul_optimized(const int *const matrixA, const int *const matrixB, int *const matrixC, const int n)
{ 
  if (n == 4096 or n == 2048)
  {
    int _mSize = 32;
    #pragma omp parallel for
    for (int T1 = 0; T1 < n; T1 += 256)
    {
      for (int T2 = 0; T2 < n; T2 += 256)
      {
        for (int T3 = 0; T3 < n; T3 += 64)
        {
          for (int i = 0; i < 256; ++i)
          {
            int row = i + T1;
            int col = T2;
            __m256i tileC[_mSize];

            for (int i = 0; i < _mSize; ++i)
            {
              tileC[i] = _mm256_loadu_si256((__m256i *)(matrixC + row * n + col + 8 * i));
            } 

            for (int i = T3; i < 64 + T3; ++i)
            {
              for (int j = 0; j < _mSize; ++j)
                tileC[j] = _mm256_add_epi32(tileC[j], _mm256_mullo_epi32(_mm256_loadu_si256((__m256i *)(matrixB + i * n + col + 8 * j)), _mm256_set1_epi32(matrixA[i + row * n])));
            }

            for (int i = 0; i < _mSize; ++i)
            {
              _mm256_storeu_si256((__m256i *)(matrixC + row * n + col + 8 * i), tileC[i]);
            }
          }
        }
      }
    }
  }
  else{
    int* matrixBT = (int*)malloc(sizeof(int)*n*n);
    transpose(matrixB, matrixBT, n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
          matrixC[i * n + j] += matrixA[i * n + k] * matrixBT[j * n + k];
  }
}
