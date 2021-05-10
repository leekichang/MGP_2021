#include "smith_waterman_parallel.h"
//#include <algorithm>
#include "utils.h"
#include <stdio.h>

using namespace Algorithms;

SmithWatermanParallel::SmithWatermanParallel(int seq1Length, int seq2Length, char* seq1, char* seq2, int gapOp, int gapEx):SimilarityAlgorithmParallel(seq1Length, seq2Length), gapOp(gapOp), gapEx(gapEx)
{
  A = new int*[seq1Length + 1];
  E = new int*[seq1Length + 1]; //left matrix
  F = new int*[seq1Length + 1]; //up matrix
  B = new BackUpStruct*[seq1Length + 1];

  A[0] = new int[(seq1Length + 1) * (seq2Length + 1)];
  E[0] = new int[(seq1Length + 1) * (seq2Length + 1)];
  F[0] = new int[(seq1Length + 1) * (seq2Length + 1)];
  B[0] = new BackUpStruct[(seq1Length + 1) * (seq2Length + 1)];

  for (int i = 1; i < seq1Length + 1; i++)
  {
    A[i] = A[0] + (seq2Length + 1)*i;
    E[i] = E[0] + (seq2Length + 1)*i;
    F[i] = F[0] + (seq2Length + 1)*i;
    B[i] = B[0] + (seq2Length + 1)*i;
  }

  setSeq1(seq1, seq1Length);
  setSeq2(seq2, seq2Length);
}

int SmithWatermanParallel::matchMissmatchScore(char a, char b) {
  if (a == b)
    return matchScore;
  else
    return missmatchScore;
}  /* End of matchMissmatchScore */


void SmithWatermanParallel::FillCell(int i, int j)
{

      //printf("at %d, %d = %c %c\n", i, j, seq1[i-1], seq2[j-1]);
      E[i][j] = MAX(E[i][j - 1] - gapEx, A[i][j - 1] - gapOp);
      B[i][j - 1].continueLeft = (E[i][j] == E[i][j - 1] - gapEx);
      F[i][j] = MAX(F[i - 1][j] - gapEx, A[i - 1][j] - gapOp);
      B[i - 1][j].continueUp = (F[i][j] == F[i - 1][j] - gapEx);

      A[i][j] = MAX3(E[i][j], F[i][j], A[i - 1][j - 1] + matchMissmatchScore(seq1[i-1], seq2[j-1]));
      A[i][j] = MAX(A[i][j], 0);


      if (A[i][j] == 0)
        B[i][j].backDirection = stop; //SPECYFIC FOR SMITH WATERMAN
      else if(A[i][j] == (A[i - 1][j - 1] + matchMissmatchScore(seq1[i-1], seq2[j-1])))
        B[i][j].backDirection = crosswise;
      else if(A[i][j] == E[i][j])
        B[i][j].backDirection = left;
      else //if(A[i][j] == F[i][j])
        B[i][j].backDirection = up;


      if(A[i][j] > maxVal)
      {
        maxX = j;
        maxY = i;
        maxVal = A[i][j];
      }

    }
void SmithWatermanParallel::FillMatrices()
{
  /*
   *   s e q 2
   * s
   * e
   * q
   * 1
   */
  //E - responsible for left direction
  //F - responsible for up   direction

  maxVal = INT_MIN;
  // for (int i = 1; i <= seq1Length; i++)
  // {
  //   for (int j = 1; j <= seq2Length; j++)
  //   {
  //     FillCell(i, j);
  //   }
    
  // }
  
  /*
  for(int m = 0; m < block_num; m++){
     for(int n = 0; n < block_num; n++){
     	
     }
  } 
  for (int i = 1; i <= seq1Length; i++)
  {
    #pragma omp parallel for
    for (int j = 1; j <= i; j++)
    {
      FillCell(i-(j-1), j);
    }
  }
  for (int i = 2; i <= seq1Length; i++){
    #pragma omp parallel for
    for (int j = i; j <= seq2Length; j++){
      FillCell(seq1Length-(j-i), j);
    }
  }*/
  
  if(seq1Length == 10000 && seq2Length == 10000){
    int blockSize = 80;
    int blockNum = seq1Length / blockSize;
    for(int m = 0; m < blockNum; ++m){
	  #pragma omp parallel for
      for(int n = 0; n <= m; ++n){
	    for(int i = 1; i <= blockSize; ++i){
	      for(int j = 1; j <= blockSize; ++j){
		    FillCell(blockSize * (m-n) + i, blockSize * n + j);
		  }
	    }
      }
    }  

    for(int m = 1; m < blockNum; ++m){
      #pragma omp parallel for
	  for(int n = m; n < blockNum; ++n){
	    for(int i = 1; i <= blockSize; ++i){
	      for(int j = 1; j <= blockSize; ++j){
	        FillCell(blockSize * (blockNum-(n-m+1)) + i, blockSize * n + j);
	      }
        }
  	  }
    }
  }
  else{
	for (int i = 1; i <= seq1Length; i++){
	  for (int j = 1; j <= seq2Length; j++){
	    FillCell(i, j);
	  }
    }
  }

  /*
  for(int m = 0; m < blockNum; ++m){
    #pragma omp parallel for
	for(int n = 0; n <= m; ++n){
	  for(int i = 1; i <= blockSize; ++i){
        #pragma omp parallel for  
		for(int j = 1; j <= i; ++j){
		  FillCell(blockSize*(m-n)+i-(j-1), blockSize*n+j);
		}
	  }
	  for(int i = 2; i <= blockSize; ++i){
        #pragma omp parallel for
		for(int j = i; j <= blockSize; ++j){
		  FillCell(blockSize*(m-n+1)-(j-i), blockSize*n+j);
		}
	  }
	}
  }

  for(int m = 1; m < blockNum; ++m){
    #pragma omp parallel for
	for(int n = m; n < blockNum; ++n){
      for(int i = 1; i <= blockSize; ++i){
        #pragma omp parallel for
		for(int j = 1; j <= i; j++){
			FillCell(blockSize*(blockNum-(n-m+1))+i-(j-1), blockSize*n+j);
		}
	  }
	  for(int i = 2; i <= blockSize; ++i){
        #pragma omp parallel for
		for(int j = i; j <= blockSize; j++){
          FillCell(blockSize*(blockNum-(n-m))-(j-i), blockSize*n+j);
        }
      }
	}
  }
  */
  
  printf("maxY %d maxX %d maxVal %d\n", maxY, maxX, maxVal);
}

void SmithWatermanParallel::BackwardMoving()
{
  //BACKWARD MOVING
  int carret = 0;

  int y = maxY;
  int x = maxX;

  BackDirection prev = crosswise;
  while(B[y][x].backDirection != stop)
  {
    path.push_back(std::make_pair(y, x));
    if (prev == up && B[y][x].continueUp) //CONTINUE GOING UP
    {                                          //GAP EXTENSION
      carret++;
      y--;
    }
    else if (prev == left && B[y][x].continueLeft) //CONTINUE GOING LEFT
    {                                         //GAP EXTENSION
      carret++;
      x--;
    }
    else
    {
      prev = B[y][x].backDirection;
      if(prev == up)
      {
        carret++;
        y--;
      }
      else if(prev == left)
      {
        carret++;
        x--;
      }
      else //prev == crosswise
      {
        carret++;
        x--;
        y--;
      }
    }
  }
}
