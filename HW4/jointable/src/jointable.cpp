#include "jointable.h"

void jointable_ref(const long long* const tableA, const long long* const tableB,
                   std::vector<long long>* const solution, const int R, const int S) {
  
  for (long long i = 0; i < R; i++)
    for (long long j = 0; j < S; j++)
      if (tableA[i] == tableB[j]) {
        solution->push_back(tableA[i]);
        break;
      }

      
}

void jointable_optimized(const long long* const tableA, const long long* const tableB,
                         std::vector<long long>* const solution, const int R,
                         const int S) {
  // TODO: Implement your code!
}
