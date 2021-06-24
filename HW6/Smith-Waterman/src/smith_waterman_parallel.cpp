#include "smith_waterman_parallel.h"
//#include <algorithm>
#include "utils.h"
#include <stdio.h>

using namespace Algorithms;

SmithWatermanParallel::SmithWatermanParallel(int maxLen, int match, int mismatch, int gapOp, int gapEx):SimilarityAlgorithmParallel(), matchScore(match), missmatchScore(-1*mismatch), gapOp(gapOp), gapEx(gapEx)
{
  A = new int*[maxLen + 1];
  E = new int*[maxLen + 1]; //left matrix
  F = new int*[maxLen + 1]; //up matrix
  B = new BackUpStruct*[maxLen + 1];

  A[0] = new int[(maxLen + 1) * (maxLen + 1)];
  E[0] = new int[(maxLen + 1) * (maxLen + 1)];
  F[0] = new int[(maxLen + 1) * (maxLen + 1)];
  B[0] = new BackUpStruct[(maxLen + 1) * (maxLen + 1)];

  for (int i = 1; i < maxLen + 1; i++)
  {
    A[i] = A[0] + (maxLen + 1)*i;
    E[i] = E[0] + (maxLen + 1)*i;
    F[i] = F[0] + (maxLen + 1)*i;
    B[i] = B[0] + (maxLen + 1)*i;
  }
  for(int i=0;i<maxLen+1;i++) {
    for(int j=0;j<maxLen+1;j++) {
      A[i][j]=0;
      E[i][j]=0;
      F[i][j]=0;
      B[i][j].backDirection=stop;
      B[i][j].continueUp=false;
      B[i][j].continueLeft=false;
    }
  }

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

  for (int i = 1; i <= seq1Length; i++)
  {
    for (int j = 1; j <= seq2Length; j++)
    {
      FillCell(i, j);
    }
    
  }
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
  //printf("Y:%d X:%d\n", y, x);
}
SmithWatermanParallel::~SmithWatermanParallel()
{
    DeallocateMemoryForSingleRun();
}
