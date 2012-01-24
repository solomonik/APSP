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

void bench_dcapsp(int const	n,
		  int const	b1,
		  int const	b2,
		  int const	crep,
		  int const	seed,
		  int const	iter,
		  int const	rank,
		  int const	np,
		  int const	pdim){
  int i, it, trank;
  double t_st, t_end, t_all;
  MPI_Comm inlayer;
  
  trank 	= rank%(np/crep);

  topology_t topo;
  topo.world 	= MPI_COMM_WORLD;
  topo.nrow 	= pdim;
  topo.ncol 	= pdim;
  topo.nlayer	= crep;
  topo.ilayer	= rank/(np/crep);
  topo.irow 	= trank/pdim;
  topo.icol 	= trank%pdim;

  MPI_Comm_split(MPI_COMM_WORLD, trank, topo.ilayer, &topo.layer);
  MPI_Comm_split(MPI_COMM_WORLD, topo.ilayer, trank, &inlayer);
  MPI_Comm_split(inlayer, topo.icol, topo.irow, &topo.row);
  MPI_Comm_split(inlayer, topo.irow, topo.icol, &topo.col);

  REAL A[n*n*crep/np] __attribute__ ((aligned(16)));


  if (rank == 0)
    printf("Benchmarking dcapsp.\n");
  MPI_Barrier(MPI_COMM_WORLD);
  t_all = 0.0;
  for (it=0; it<iter; it++){
    srand48(seed);
    for (i=0; i<n*n*crep/np; i++){
      A[i] = drand48();
    }
    t_st = TIME_SEC();
    dcapsp(&topo, n, A, 0, b1, b2);
    t_end = TIME_SEC();
    t_all += t_end - t_st;
  }

  if (rank == 0){
    printf("Completed %d iterations of dcapsp.\n", iter);
    printf("%lf seconds per iteration, %lf Gigaflops (GF)\n", t_all, 2.E-9*n*n*n*iter/t_all);
  }
}


void test_dcapsp( int const	n,
		  int const	b1,
		  int const	b2,
		  int const	crep,
		  int const	seed,
		  int const	rank,
		  int const	np,
		  int const	pdim){
  int i, pass, allpass, trank;
  MPI_Comm inlayer;
  
  trank 	= rank%(np/crep);

  topology_t topo;
  topo.world 	= MPI_COMM_WORLD;
  topo.nrow 	= pdim;
  topo.ncol 	= pdim;
  topo.nlayer	= crep;
  topo.ilayer	= rank/(np/crep);
  topo.irow 	= trank/pdim;
  topo.icol 	= trank%pdim;

  MPI_Comm_split(MPI_COMM_WORLD, trank, topo.ilayer, &topo.layer);
  MPI_Comm_split(MPI_COMM_WORLD, topo.ilayer, trank, &inlayer);
  MPI_Comm_split(inlayer, topo.icol, topo.irow, &topo.row);
  MPI_Comm_split(inlayer, topo.irow, topo.icol, &topo.col);
//  printf("[%d] is [%d][%d][%d]\n", rank, topo.irow, topo.icol, topo.ilayer);

  REAL * A = (REAL*)malloc(n*n*sizeof(REAL));
  REAL * sub_A = (REAL*)malloc(n*n*sizeof(REAL)*crep/np);
  REAL * ans_A = (REAL*)malloc(n*n*sizeof(REAL)*crep/np);

  srand48(seed);
  for (i=0; i<n*n; i++){
    A[i] = drand48();
  }

  read_matrix_block(n, topo.irow, topo.nrow, topo.icol, topo.ncol, A, sub_A);

  if (rank == 0)
    printf("Testing dcapsp.\n");
  dcapsp(&topo, n, sub_A, 0, b1, b2);

  if (rank == 0)
    printf("Completed dcapsp.\n");

  if (topo.ilayer == 0){
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
    MPI_Reduce(&pass, &allpass, 1, MPI_INT, MPI_SUM, 0, inlayer);
    if (rank == 0 && allpass == np/crep)
      printf("Test successful.\n");
  }
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
  int seed, rank, np, pdim, n, b1, b2, crep, test, iter;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-iter")){
    iter = atoi(getCmdOption(input_str, input_str+in_num, "-iter"));
    if (iter < 0) iter = 3;
  } else iter = 3;
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test < 0 || test > 1) test = 1;
  } else test = 1;
  if (getCmdOption(input_str, input_str+in_num, "-b1")){
    b1 = atoi(getCmdOption(input_str, input_str+in_num, "-b1"));
    if (b1 < 0) b1 = 32;
  } else b1 = 32;
  if (getCmdOption(input_str, input_str+in_num, "-b2")){
    b2 = atoi(getCmdOption(input_str, input_str+in_num, "-b2"));
    if (b2 < 0) b2 = 32;
  } else b2 = 32;
  if (getCmdOption(input_str, input_str+in_num, "-crep")){
    crep = atoi(getCmdOption(input_str, input_str+in_num, "-crep"));
    if (crep < 0) crep = 1;
  } else crep = 1;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 128;
  } else n = 128;
  if (getCmdOption(input_str, input_str+in_num, "-pdim")){
    pdim = atoi(getCmdOption(input_str, input_str+in_num, "-pdim"));
    if (pdim < 0) pdim = 1;
  } else pdim = 1;

  if (rank == 0){
    printf("seed = %d, iter = %d, test = %d, b1 = %d, b2 =%d, crep = %d,\n",
	    seed, iter, test, b1, b2, crep);
    printf("n = %d, pdim = %d\n", n, pdim);
  }

  assert(pdim*pdim*crep == np);

#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(rank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif

  if (iter > 0)
    bench_dcapsp(n, b1, b2, crep, seed, iter, rank, np, pdim);
  if (test)
    test_dcapsp(n, b1, b2, crep, seed, rank, np, pdim);
  
  TAU_PROFILE_STOP(timer);
  MPI_Finalize();
}
