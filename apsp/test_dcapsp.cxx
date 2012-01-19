#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include "dcapsp.h"
#include "../shared/util.h"


void read_matrix_block(	int const	n,
			int const	irow,
			int const	nrow,
			int const	icol,
			int const	ncol,
			REAL const *	A,
			REAL *		sub_A){
  int i, j;
  for (i=0; i<n/nrow; i++){
    for (j=0; j<n/ncol; j++){
      sub_A[i*n/ncol+j] = A[(irow*n/nrow+i)*n+(icol*n/ncol+j)];
    }
  }
}

void test_dcapsp( int const	n,
		  int const	b,
		  int const	seed,
		  int const	rank,
		  int const	np,
		  int const	pdim){
  int i, pass, allpass;
  topology_t topo;
  topo.world 	= MPI_COMM_WORLD;
  topo.layer	= MPI_COMM_WORLD;
  topo.nrow 	= pdim;
  topo.ncol 	= pdim;
  topo.irow 	= rank/pdim;
  topo.icol 	= rank%pdim;

  MPI_Comm_split(MPI_COMM_WORLD, topo.icol, topo.irow, &topo.row);
  MPI_Comm_split(MPI_COMM_WORLD, topo.irow, topo.icol, &topo.col);

  REAL * A = (REAL*)malloc(n*n*sizeof(REAL));
  REAL * sub_A = (REAL*)malloc(n*n*sizeof(REAL)/np);
  REAL * ans_A = (REAL*)malloc(n*n*sizeof(REAL)/np);

  srand48(seed);
  for (i=0; i<n*n; i++){
    A[i] = drand48();
  }

  read_matrix_block(n, topo.irow, topo.nrow, topo.icol, topo.ncol, A, sub_A);

  if (rank == 0)
    printf("Testing dcapsp.\n");
  dcapsp(&topo, n, sub_A, 0, b);

  if (rank == 0)
    printf("Completed dcapsp.\n");

  floyd_warshall(A, n);
  read_matrix_block(n, topo.irow, topo.nrow, topo.icol, topo.ncol, A, ans_A);

  pass = 1;
  for (i=0; i<n/np; i++){
    if (fabs(sub_A[i] - ans_A[i]) > 1.E-6){
      printf("P[%d][%d] computed A[%d] = %lf, ans_A[%d] = %lf\n",
	      topo.irow, topo.icol, i, sub_A[i], i, ans_A[i]);
      pass = 0;
    }
  }
  MPI_Reduce(&pass, &allpass, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0 && allpass == np)
    printf("Test successful.\n");
}

/* Defines elsewhere deprecate */
static
char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int seed, rank, np, pdim, n, b;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-b")){
    b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
    if (b < 0) b = 32;
  } else b = 32;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 128;
  } else n = 128;
  if (getCmdOption(input_str, input_str+in_num, "-pdim")){
    pdim = atoi(getCmdOption(input_str, input_str+in_num, "-pdim"));
    if (pdim < 0) pdim = 1;
  } else pdim = 1;

  assert(pdim*pdim == np);

  test_dcapsp(n, b, seed, rank, np, pdim);

  MPI_Finalize();
}
