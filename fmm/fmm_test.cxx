#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <assert.h>
 
#include "fmm.h"
 
void test_fmm(int const		seed, 
	      int const		n, 
	      int const		m, 
	      int const		k, 
	      char const	trans_A,
	      char const	trans_B, 
	      int const		lda_A, 
	      int const		lda_B, 
	      int const		lda_C){
  REAL *A,*B,*C;
  int sz_A, sz_B, sz_C;

  if (trans_A == 'N') sz_A = lda_A*k;
  else sz_A = lda_A*m;
  if (trans_B == 'N') sz_B = lda_B*n;
  else sz_B = lda_B*k;
  sz_C = lda_C*n;

  A = (REAL*)malloc(sizeof(REAL)*sz_A);
  B = (REAL*)malloc(sizeof(REAL)*sz_B);
  C = (REAL*)malloc(sizeof(REAL)*sz_C);

  std::fill(A, A+sz_A, 1.0);
  std::fill(B, B+sz_B, 1.0);
  std::fill(C, C+sz_C, 100.0);

  fmm_naive(trans_A, trans_B,
	    m, n, k,
	    A, lda_A,
	    B, lda_B,
	    C, lda_C);

}

char* getCmdOption(char ** begin, 
		   char ** end, 
		   const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int seed, n, m, k;
  char trans_A, trans_B;
  int lda_A, lda_B, lda_C;
  
  int const in_num = argc;
  char ** input_str = argv;

 /* MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);*/

  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n <= 0) n = 64;
  } else n = 64;
  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m <= 0) m = 64;
  } else m = 64;
  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k <= 0) k = 64;
  } else k = 64;
  if (getCmdOption(input_str, input_str+in_num, "-trans_A")){
    trans_A = *getCmdOption(input_str, input_str+in_num, "-trans_A");
    if (trans_A != 'N' || trans_A != 'T') trans_A = 'N';
  } else trans_A = 'N';
  if (getCmdOption(input_str, input_str+in_num, "-trans_B")){
    trans_B = *getCmdOption(input_str, input_str+in_num, "-trans_B");
    if (trans_B != 'N' || trans_B != 'T') trans_B = 'N';
  } else trans_B = 'N';
  if (getCmdOption(input_str, input_str+in_num, "-lda_A")){
    lda_A = atoi(getCmdOption(input_str, input_str+in_num, "-lda_A"));
    if (trans_A == 'N') lda_A = MAX(lda_A, m);
    else lda_A = MAX(lda_A, k);
  } else {
    if (trans_A == 'N') lda_A = m;
    else lda_A = k;
  }
  if (getCmdOption(input_str, input_str+in_num, "-lda_B")){
    lda_B = atoi(getCmdOption(input_str, input_str+in_num, "-lda_B"));
    if (trans_B == 'N') lda_B = MAX(lda_B, k);
    else lda_B = MAX(lda_B, n);
  } else {
    if (trans_B == 'N') lda_B = k;
    else lda_B = n;
  }
  if (getCmdOption(input_str, input_str+in_num, "-lda_C")){
    lda_C = atoi(getCmdOption(input_str, input_str+in_num, "-lda_C"));
    lda_C = MAX(lda_C, m);
  } else {
    lda_C = m;
  }

  printf("Testing funny matrix multiply of size %d-by-%d A and %d-by-%d B\n", 
	 m, k, k, n);
  printf("seed = %d trans_A = %c trans_B = %c lda_A = %d lda_B = %d lda_C = %d\n",
	  seed, trans_A, trans_B, lda_A, lda_B, lda_C);
  
  test_fmm(seed, n, m, k, trans_A, trans_B, lda_A, lda_B, lda_C);

//  MPI_Finalize();
  return 0;
}
