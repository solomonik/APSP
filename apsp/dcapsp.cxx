#include <stdio.h>
#include "dcapsp.h"
#include "../fmm/fmm.h"
#include "../ctr_comm/ctr_comm.h"

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
  tsub->layer 	= tparent->layer;  
  MPI_Comm_split(tparent->row, tparent->irow%2, tsub->irow, &tsub->row);
  MPI_Comm_split(tparent->col, tparent->icol%2, tsub->icol, &tsub->col);
}

ctr * construct_ctr(const topology_t * topo,
		    const int n){
  CommData_t cdt_x, cdt_y;
  ASRT(topo->nrow == topo->ncol);

  ctr_fmm * cfmm = new ctr_fmm();
  ctr_2d_sqr_bcast * csb = new ctr_2d_sqr_bcast();

  cfmm->n 	= n / topo->nrow;

  csb->rec_ctr 	= cfmm;
  csb->k 	= n;
  csb->sz_A	= (n/topo->nrow)*(n/topo->ncol);
  csb->sz_B	= (n/topo->nrow)*(n/topo->ncol);

  SET_COMM((topo->row), (topo->irow), (topo->nrow), (&cdt_x));
  SET_COMM((topo->col), (topo->icol), (topo->ncol), (&cdt_y));

  csb->cdt_x = &cdt_x;
  csb->cdt_y = &cdt_y;

  return csb;
}


void blocked_dcapsp(const topology_t * topo,
		    const int n,
		    REAL * A,		
		    int * pred_A = 0){
  if (topo->nrow == 0 && topo->ncol == 0){
    fmm_naive('N', 'N', n, n, n, A, n, A, n, A, n);
  } else {
    topology_t * tsub;
    ctr * myctr;
    tsub = (topology_t*)malloc(sizeof(topology_t));

    split_topo(topo, tsub);
    myctr = construct_ctr(tsub, n/2);

    if (topo->irow % 2 == 0 &&
	topo->icol % 2 == 0){
      blocked_dcapsp(tsub, n/2, A, pred_A);
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow+1, 0, topo->row);
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol+1, 1, topo->col);
      {
	MPI_Status stat12, stat21;
	REAL A12[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	REAL A21[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));

	MPI_Recv(A12, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow+1, 6, topo->row, &stat21);
	MPI_Recv(A21, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol+1, 7, topo->col, &stat12);
	myctr->A = A12;
	myctr->B = A21;
	myctr->C = A;
	myctr->num_lyr = 1;
	myctr->idx_lyr = 0;
	myctr->buffer = NULL;
	myctr->run();
      }
    } else if (topo->irow % 2 == 1 &&
	       topo->icol % 2 == 0){
      MPI_Status stat;
      {
	REAL A11[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	MPI_Recv(A11, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow-1, 0, topo->row, &stat);
	myctr->A = A11;
	myctr->B = A;
	myctr->C = A;
	myctr->num_lyr = 1;
	myctr->idx_lyr = 0;
	myctr->buffer = NULL;
	myctr->run();
      }
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol+1, 2, topo->col);
      {
	REAL A22[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	MPI_Recv(A22, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol+1, 5, topo->col, &stat);
	myctr->A = A;
	myctr->B = A22;
	myctr->C = A;
	myctr->num_lyr = 1;
	myctr->idx_lyr = 0;
	myctr->buffer = NULL;
	myctr->run();
      }
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow+1, 6, topo->row);
    } else if (topo->irow % 2 == 0 &&
	       topo->icol % 2 == 1){
      MPI_Status stat;
      {
	REAL A11[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	MPI_Recv(A11, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol-1, 1, topo->col, &stat);
	myctr->A = A;
	myctr->B = A11;
	myctr->C = A;
	myctr->num_lyr = 1;
	myctr->idx_lyr = 0;
	myctr->buffer = NULL;
	myctr->run();
      }
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow+1, 3, topo->row);
      {
	REAL A22[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	MPI_Recv(A22, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow+1, 4, topo->row, &stat);
	myctr->A = A22;
	myctr->B = A;
	myctr->C = A;
	myctr->num_lyr = 1;
	myctr->idx_lyr = 0;
	myctr->buffer = NULL;
	myctr->run();
      }
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol-1, 7, topo->col);
    } else if (topo->irow % 2 == 1 &&
	       topo->icol % 2 == 1){
      {
	MPI_Status stat12, stat21;
	REAL A12[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	REAL A21[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));

	MPI_Recv(A12, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol-1, 2, topo->col, &stat12);
	MPI_Recv(A21, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow-1, 3, topo->row, &stat21);
	myctr->A = A21;
	myctr->B = A12;
	myctr->C = A;
	myctr->num_lyr = 1;
	myctr->idx_lyr = 0;
	myctr->buffer = NULL;
	myctr->run();
      }
      blocked_dcapsp(tsub, n/2, A, pred_A);
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->irow-1, 4, topo->row);
      MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), MPI_DOUBLE, topo->icol-1, 5, topo->col);
    }
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
void dcapsp(const topology_t * topo,
	    const int n,
	    REAL * A,		
	    int * pred_A){

  blocked_dcapsp(topo, n, A, pred_A);

}

