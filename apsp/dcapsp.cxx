#include <stdio.h>
#include "dcapsp.h"
#include "../fmm/fmm.h"

void split_topo(topology_t const * tparent,
		topology_t * tsub){
  ASRT(tparent->nrow%2 == 0);
  ASRT(tparent->ncol%2 == 0);

  tsub->ilayer 	= tparent->ilayer;  
  tsub->nlayer 	= tparent->nlayer;
  tsub->nrow	= tparent->nrow/2;
  tsub->irow	= tparent->irow/2;
  tsub->ncol	= tparent->ncol/2;
  tsub->icol	= tparent->icol/2;

  tsub->world 	= tparent->world;  
  MPI_Comm_split(tparent->row, tparent->irow%2, tsub->irow, tsub->row);
  MPI_Comm_split(tparent->col, tparent->icol%2, tsub->icol, tsub->col);
}


void blocked-dcapsp(const topology_t * topo,
		    const int n,
		    REAL * A,		
		    int * pred_A = 0){
  if (topo->nrow == 0 && topo->ncol == 0){
    fmm_naive('N', 'N', n, n, n, A, n, A, n, A, n);
  } else {
    topology_t * tsub;
    tsub = (topology_t*)malloc(sizeof(topology_t));

    split_topo(topo, tsub);
  }
}


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
		int * pred_A = 0){
  
  

}

