#ifndef __DCAPSP_H__
#define __DCAPSP_H__

#include "mpi.h"
#include "../fmm/fmm.h"

#define DASSERT 1
#if DASSERT
#include "assert.h"
#define ASRT assert
#else 
#define ASRT()
#endif

typedef struct topology {
  MPI_Comm world;
  MPI_Comm row;
  MPI_Comm col;
  MPI_Comm layer;
  int nrow, ncol, nlayer;
  int irow, icol, ilayer;
  struct topology * tsub;
} topology_t;

void split_topo(topology_t const * tparent,
		topology_t * tsub);

/**
 * \brief computes all-pairs-shortest-path via divide-and-conquer
 *
 * \param[in] topo the process topology information
 * \param[in] n is the matrix dimension (must be divisible by nrow, ncol)
 * \param[in,out] A is a n/nrow-by-n/ncol block of the adjacency matrix on 
 *		  input and a block of the distance matrix on output
 * \param[in,out] pred_A predecessors of A on input and output (ignored if NULL, 0)
 * \param[in] b is the blocking factor for when to switch from cyclic to blocked
 */
void dcapsp(topology_t * topo,
	    const int64_t n,
	    REAL * A,		
	    int * pred_A = 0,
	    const int64_t b1 = 32,
	    const int64_t b2 = 256);


void floyd_warshall(REAL * A, int64_t const n);
void floyd_warshall(REAL * A, int64_t const n, int64_t const lda);

#endif
