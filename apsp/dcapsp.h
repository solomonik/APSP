#ifndef __DCAPSP_H__
#define __DCAPSP_H__

#include "mpi.h"

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
  MPI_Comm column;
  MPI_Comm layer;
  int nrow, ncolumn, nlayer;
  int irow, icolumn, ilayer;
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
 */
void dcapsp( 	const topology_t * topo,
		const int n,
		REAL * A,		
		int * pred_A = 0);

#endif
