#ifndef _SIMILARITY_ALGORITHM_CPU_PARALLEL_H_
#define _SIMILARITY_ALGORITHM_CPU_PARALLEL_H_

#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <utility>


namespace Algorithms
{


  enum BackDirection
  {
    stop, //if H == 0 in (Smith Waterman)
    up,
    left,
    crosswise
  };

  struct BackUpStruct
  {
    BackDirection backDirection;
    bool continueUp;
    bool continueLeft;
  };


  class SimilarityAlgorithmParallel
  {
    public:
      virtual void Run();
      virtual void PrintResults(std::string fileName);
      virtual void DeallocateMemoryForSingleRun();
      SimilarityAlgorithmParallel() {/*empty*/}
      virtual ~SimilarityAlgorithmParallel();

      virtual void setSeq1(const char* seq, int len) {this->seq1 = seq; this->seq1Length = len;}
      virtual void setSeq2(const char* seq, int len) {this->seq2 = seq; this->seq2Length = len;}
    protected:
      virtual void FillMatrices() = 0;
      virtual void BackwardMoving() = 0;

      //INPUT DATA
      // seq1 and seq2 are 1-based array of sequence characters
      const char* seq1;
      const char* seq2;
      int seq1Length;
      int seq2Length;

      //MATRICES
      int **A;
      int **E; //left matrix
      int **F; //up matrix
      BackUpStruct **B;

      //backtrack results
      std::vector<std::pair<int,int> > path;
  };
}

#endif
