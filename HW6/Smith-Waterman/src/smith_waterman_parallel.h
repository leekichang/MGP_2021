#ifndef _SMITH_WATERMAN_PARALLEL_H_
#define _SMITH_WATERMAN_PARALLEL_H_

#include <climits>
#include "similarity_algorithm_parallel.h"

namespace Algorithms
{
  class SmithWatermanParallel:public SimilarityAlgorithmParallel
  {
    public:
    SmithWatermanParallel(int maxLen, int match, int mismatch, int gapOp, int gapEx);
    ~SmithWatermanParallel();
    protected:
    virtual void FillMatrices();
    virtual void FillCell(int i, int j);
    virtual void BackwardMoving();
    virtual int matchMissmatchScore(char, char);
    int maxX;
    int maxY;
    int maxVal;
    int matchScore;
    int missmatchScore;
    int gapOp;  // gap open
    int gapEx;  // gap extension
  };
}

#endif
