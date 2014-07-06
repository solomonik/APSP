#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <assert.h>
 
#include "fmm.h"

#ifdef BGP
#include "mpi.h"
extern "C"{
void HPM_Init();
void HPM_Print();
void HPM_Start(char *);
void HPM_Stop(char *);
}
#endif

void bench_fmm(int const                seed, 
               int const                n, 
               int const                m, 
               int const                k, 
               char const               trans_A,
               char const               trans_B, 
               int const                lda_A, 
               int const                lda_B, 
               int const                lda_C,
               int const                nwarm, 
               int const                niter){
  REAL *A,*B,*C;
  int sz_A, sz_B, sz_C, i, j;
  double tstart, tend;

  if (trans_A == 'N') sz_A = lda_A*k;
  else sz_A = lda_A*m;
  if (trans_B == 'N') sz_B = lda_B*n;
  else sz_B = lda_B*k;
  sz_C = lda_C*n;

  A = (REAL*)malloc(sizeof(REAL)*sz_A);
  B = (REAL*)malloc(sizeof(REAL)*sz_B);
  C = (REAL*)malloc(sizeof(REAL)*sz_C);

  srand48(seed);
  for (i=0; i<sz_C; i++){
    C[i] = drand48();
  }
  for (i=0; i<sz_A; i++){
    A[i] = drand48();
  }
  for (i=0; i<sz_B; i++){
    B[i] = drand48();
  }

  for (i=0; i<nwarm; i++){
    fmm_opt  (trans_A, trans_B,
              m, n, k,
              A, lda_A,
              B, lda_B,
              C, lda_C);
  }
#ifdef BGP
  HPM_Init();
  HPM_Start("fmm_opt");
#endif
  tstart = TIME_SEC();
  for (i=0; i<niter; i++){
    fmm_opt  (trans_A, trans_B,
              m, n, k,
              A, lda_A,
              B, lda_B,
              C, lda_C);
  }
  tend = TIME_SEC();
  printf("benchmark complete.\n");
  printf("Performed %d iterations at %lf sec/iter, achieving a flop/flops rate of %lf GF.\n",
          niter, (tend-tstart)/niter, (2.*n*m*k*niter*1.E-9)/(tend-tstart));
#ifdef BGP
  HPM_Stop("fmm_opt");
  HPM_Print();
#endif
}
 
void test_fmm(int const         seed, 
              int const         n, 
              int const         m, 
              int const         k, 
              char const        trans_A,
              char const        trans_B, 
              int const         lda_A, 
              int const         lda_B, 
              int const         lda_C){
  REAL *A,*B,*C,*ans_C;
  int sz_A, sz_B, sz_C, i, j, pass;

  if (trans_A == 'N') sz_A = lda_A*k;
  else sz_A = lda_A*m;
  if (trans_B == 'N') sz_B = lda_B*n;
  else sz_B = lda_B*k;
  sz_C = lda_C*n;

  A = (REAL*)malloc(sizeof(REAL)*sz_A);
  B = (REAL*)malloc(sizeof(REAL)*sz_B);
  C = (REAL*)malloc(sizeof(REAL)*sz_C);
  ans_C = (REAL*)malloc(sizeof(REAL)*sz_C);

  srand48(seed);
  for (i=0; i<sz_C; i++){
    C[i] = drand48();
  }
  for (i=0; i<sz_A; i++){
    A[i] = drand48();
  }
  for (i=0; i<sz_B; i++){
    B[i] = drand48();
  }
  srand48(seed);
  for (i=0; i<sz_C; i++){
    ans_C[i] = drand48();
  }
  /*std::fill(A, A+sz_A, 1.0);
  std::fill(B, B+sz_B, 1.0);
  std::fill(C, C+sz_C, 100.0);
  std::fill(ans_C, ans_C+sz_C, 100.0);*/

  fmm_naive(trans_A, trans_B,
            m, n, k,
            A, lda_A,
            B, lda_B,
            ans_C, lda_C);
  
  fmm_opt  (trans_A, trans_B,
            m, n, k,
            A, lda_A,
            B, lda_B,
            C, lda_C);

  pass = 1;
  for (i=0; i<n; i++){
    for (j=0; j<m; j++){
      if (fabs(C[i*lda_C+j] - ans_C[i*lda_C+j]) > 1.E-6){
#if (PRECISION==1)
        printf("opt_C[%d,%d] = %f, naive_C[%d,%d] = %f\n",
                i,j,C[i*lda_C+j],
                i,j,ans_C[i*lda_C+j]);
#else
        printf("opt_C[%d,%d] = %lf, naive_C[%d,%d] = %lf\n",
                i,j,C[i*lda_C+j],
                i,j,ans_C[i*lda_C+j]);
#endif
        pass = 0;
        printf("Optimized fmm incorrect!\n");
        return;
      }
    }
  }
  if (pass)
   printf("Optimized fmm correct.\n");
  return;

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
  int lda_A, lda_B, lda_C, nwarm, niter;
  
  int const in_num = argc;
  char ** input_str = argv;
#ifdef BGP
  MPI_Init(&argc, &argv);
#endif

 /* MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);*/

  if (getCmdOption(input_str, input_str+in_num, "-nwarm")){
    nwarm = atoi(getCmdOption(input_str, input_str+in_num, "-nwarm"));
    if (nwarm < 0) seed = 1;
  } else nwarm = 1;
  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) seed = 10;
  } else niter = 10;
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
    if (trans_A != 'N' && trans_A != 'T') trans_A = 'N';
  } else trans_A = 'N';
  if (getCmdOption(input_str, input_str+in_num, "-trans_B")){
    trans_B = *getCmdOption(input_str, input_str+in_num, "-trans_B");
    if (trans_B != 'N' && trans_B != 'T') trans_B = 'N';
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
  
#ifndef NOBENCH
  bench_fmm(seed, n, m, k, trans_A, trans_B, lda_A, lda_B, lda_C, nwarm, niter);
#endif
#ifndef NOTEST
  test_fmm(seed, n, m, k, trans_A, trans_B, lda_A, lda_B, lda_C);
#endif

#ifdef BGP
  MPI_Finalize();
#endif
  return 0;
}
