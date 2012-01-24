#include <stdio.h>
#include "dcapsp.h"
#include "../fmm/fmm.h"
#include "../ctr_comm/ctr_comm.h"

void split_topo(topology_t const * tparent,
		topology_t * tsub){
  ASRT(tparent->nrow%2 == 0);
  ASRT(tparent->ncol%2 == 0);

  tsub->ilayer 	= tparent->ilayer/2;  
  tsub->nlayer 	= MAX(1,tparent->nlayer/2);
  tsub->nrow	= tparent->nrow/2;
  tsub->irow	= tparent->irow/2;
  tsub->ncol	= tparent->ncol/2;
  tsub->icol	= tparent->icol/2;

  tsub->world 	= tparent->world;  
  MPI_Comm_split(tparent->row, tparent->irow%2, tsub->irow, &tsub->row);
  MPI_Comm_split(tparent->col, tparent->icol%2, tsub->icol, &tsub->col);
  if (tparent->nlayer > 1)
    MPI_Comm_split(tparent->layer, tparent->ilayer%2, tsub->ilayer, &tsub->layer);
  else
    tsub->layer = tparent->layer;
}

ctr * construct_ctr(const topology_t * topo,
		    const int n){
  CommData_t * cdt_x, * cdt_y, * cdt_lyr;
  ASRT(topo->nrow == topo->ncol);

  cdt_x = (CommData_t*)malloc(sizeof(CommData_t));
  cdt_y = (CommData_t*)malloc(sizeof(CommData_t));

  ctr_fmm * cfmm = new ctr_fmm();
  ctr_2d_sqr_bcast * csb = new ctr_2d_sqr_bcast();

  cfmm->n 	= n / topo->nrow;

  csb->rec_ctr 	= cfmm;
  csb->k 	= n;
  csb->sz_A	= (n/topo->nrow)*(n/topo->ncol);
  csb->sz_B	= (n/topo->nrow)*(n/topo->ncol);

  SET_COMM((topo->row), (topo->irow), (topo->nrow), cdt_x);
  SET_COMM((topo->col), (topo->icol), (topo->ncol), cdt_y);

  csb->cdt_x = cdt_x;
  csb->cdt_y = cdt_y;
  
  if (topo->nlayer > 1){
    cdt_lyr = (CommData_t*)malloc(sizeof(CommData_t));
    SET_COMM((topo->layer), (topo->ilayer), (topo->nlayer), cdt_lyr);
    ctr_lyr * clyr 	= new ctr_lyr();
    clyr->k		= n;
    clyr->sz_A		= (n/topo->nrow)*(n/topo->ncol);
    clyr->sz_B		= (n/topo->nrow)*(n/topo->ncol);
    clyr->sz_C		= (n/topo->nrow)*(n/topo->ncol);
    clyr->rec_ctr 	= csb;
    clyr->cdt 		= cdt_lyr;
    clyr->idx_lyr	= topo->ilayer;
    clyr->num_lyr	= topo->nlayer;
    clyr->red_op	= MPI_MIN;
    return clyr;
  } else {
    csb->idx_lyr	= 0;
    csb->num_lyr	= 1;
    return csb;
  }
}

#define TAG0 10
#define TAG1 11
#define TAG2 12
#define TAG3 13
#define TAG4 14
#define TAG5 15
#define TAG6 16
#define TAG7 17

void seq_dcapsp(int const 	n,
		int const	lda,
		int const	b,
		REAL *		A,
		int *		pred_A = 0){
  if (n<=b){
    floyd_warshall(A, n, lda);
  } else {
    TAU_FSTART(seq_dcapsp);
    seq_dcapsp(n/2, lda, b, A, pred_A);
    fmm_opt('N', 'N', n/2, n/2, n/2, A+n/2, lda, A, lda, A+n/2, lda, pred_A, pred_A, lda);
    fmm_opt('N', 'N', n/2, n/2, n/2, A, lda, A+lda*n/2, lda, A+lda*n/2, lda, pred_A, pred_A, lda);
    fmm_opt('N', 'N', n/2, n/2, n/2, A+n/2, lda, A+lda*n/2, lda, A+(lda+1)*n/2, lda, pred_A, pred_A, lda);
    seq_dcapsp(n/2, lda, b, A+(lda+1)*n/2, pred_A);
    fmm_opt('N', 'N', n/2, n/2, n/2, A+(lda+1)*n/2, lda, A+n/2, lda, A+n/2, lda, pred_A, pred_A, lda);
    fmm_opt('N', 'N', n/2, n/2, n/2, A+lda*n/2, lda, A+(lda+1)*n/2, lda, A+lda*n/2, lda, pred_A, pred_A, lda);
    fmm_opt('N', 'N', n/2, n/2, n/2, A+lda*n/2, lda, A+n/2, lda, A, lda, pred_A, pred_A, lda);
    TAU_FSTOP(seq_dcapsp);
  }  
}

void cyclic_dcapsp(const topology_t * topo,
		   const int n,
		   const int b1,
		   const int b2,
		   REAL * A,		
		   int * pred_A = 0);

void blocked_dcapsp(const topology_t * topo,
		    const int n,
		    const int b,
		    REAL * A,		
		    int * pred_A = 0){
  if (topo->nrow == 1 && topo->ncol == 1){
//    floyd_warshall(A, n);
    seq_dcapsp(n, n, 8, A, pred_A);
  } else if (b != -1 && topo->nlayer == 1) {
    cyclic_dcapsp(topo, n, b, b, A, pred_A);
  } else {
    //printf("n=%d, nrow = %d, ncol = %d\n",n,topo->nrow, topo->ncol);
    topology_t * tsub;
    ctr * myctr;
    tsub = (topology_t*)malloc(sizeof(topology_t));

    split_topo(topo, tsub);
    myctr = construct_ctr(tsub, n/2);
    myctr->buffer = NULL;

    if (topo->ilayer%2 == 0){
      if (topo->irow % 2 == 0 &&
	  topo->icol % 2 == 0){
	blocked_dcapsp(tsub, n/2, b, A, pred_A);
	if (topo->ilayer == 0){
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow+1, TAG0, topo->row);
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol+1, TAG1, topo->col);
	}
	{
	  MPI_Status stat12, stat21;
	  REAL A12[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	  REAL A21[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));

	  if (topo->ilayer == 0){
	    MPI_Recv(A12, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow+1, TAG6, topo->row, &stat21);
	    MPI_Recv(A21, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol+1, TAG7, topo->col, &stat12);
	  }
	  myctr->A = A12;
	  myctr->B = A21;
	  myctr->C = A;
	  myctr->run();
	}
      } else if (topo->irow % 2 == 1 &&
		 topo->icol % 2 == 0){
	MPI_Status stat;
	{
	  REAL A11[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	  if (topo->ilayer == 0){
	    MPI_Recv(A11, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow-1, TAG0, topo->row, &stat);
	  }
	  myctr->A = A11;
	  myctr->B = A;
	  myctr->C = A;
	  myctr->run();
	}
	if (topo->ilayer == 0){
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol+1, TAG2, topo->col);
	}
	{
	  REAL A22[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	  if (topo->ilayer == 0){
	    MPI_Recv(A22, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol+1, TAG5, topo->col, &stat);
	  }
	  myctr->A = A;
	  myctr->B = A22;
	  myctr->C = A;
	  myctr->run();
	}
	if (topo->ilayer == 0){
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow-1, TAG6, topo->row);
	}
      } else if (topo->irow % 2 == 0 &&
		 topo->icol % 2 == 1){
	MPI_Status stat;
	{
	  REAL A11[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	  if (topo->ilayer == 0){
	    MPI_Recv(A11, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol-1, TAG1, topo->col, &stat);
	  }
	  myctr->A = A;
	  myctr->B = A11;
	  myctr->C = A;
	  myctr->run();
	}
	if (topo->ilayer == 0){
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow+1, TAG3, topo->row);
	}
	{
	  REAL A22[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	  if (topo->ilayer == 0){
	    MPI_Recv(A22, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow+1, TAG4, topo->row, &stat);
	  }
	  myctr->A = A22;
	  myctr->B = A;
	  myctr->C = A;
	  myctr->run();
	}
	if (topo->ilayer == 0){
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol-1, TAG7, topo->col);
	}
      } else if (topo->irow % 2 == 1 &&
		 topo->icol % 2 == 1){
	{
	  MPI_Status stat12, stat21;
	  REAL A12[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));
	  REAL A21[(n/topo->nrow)*(n/topo->ncol)] __attribute__ ((aligned(16)));

	  if (topo->ilayer == 0){
	    MPI_Recv(A12, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol-1, TAG2, topo->col, &stat12);
	    MPI_Recv(A21, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow-1, TAG3, topo->row, &stat21);
	  }
	  myctr->A = A21;
	  myctr->B = A12;
	  myctr->C = A;
	  myctr->run();
	}
	blocked_dcapsp(tsub, n/2, b, A, pred_A);
	if (topo->ilayer == 0){
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->irow-1, TAG4, topo->row);
	  MPI_Send(A, (n/topo->nrow)*(n/topo->ncol), REAL_MPI, topo->icol-1, TAG5, topo->col);
	}
      }
    }
  }
}

void cyclic_dcapsp(const topology_t * topo,
		   const int n,
		   const int b1,
		   const int b2,
		   REAL * A,		
		   int * pred_A){
  int nb = n/topo->nrow;
  if (nb <= b1){
    blocked_dcapsp(topo, n, -1, A, pred_A);
  } else if (nb <= b2){
    blocked_dcapsp(topo, n, b1, A, pred_A);
  } else {
    ctr * myctr = construct_ctr(topo, n/2);
    //myctr->num_lyr = topo->nlayer;
    //myctr->idx_lyr = topo->ilayer;
    myctr->buffer = NULL;

    REAL sub_A11[nb*nb/4] __attribute__ ((aligned(16)));   
    if (topo->ilayer == 0){
      lda_cpy(nb/2, nb/2, nb, nb/2, A, sub_A11);
    }
    cyclic_dcapsp(topo, n/2, b1, b2, sub_A11, pred_A);

    REAL sub_A21[nb*nb/4] __attribute__ ((aligned(16)));   
    REAL sub_A12[nb*nb/4] __attribute__ ((aligned(16)));   
    REAL sub_A22[nb*nb/4] __attribute__ ((aligned(16)));   
    if (topo->ilayer == 0){
      lda_cpy(nb/2, nb/2, nb, nb/2, A+nb/2, sub_A21);
      lda_cpy(nb/2, nb/2, nb, nb/2, A+nb*nb/2, sub_A12);
      lda_cpy(nb/2, nb/2, nb, nb/2, A+nb/2+nb*nb/2, sub_A22);
    }
	
    myctr->A = sub_A11;
    myctr->B = sub_A12;
    myctr->C = sub_A12;
    myctr->run();

    myctr->A = sub_A21;
    myctr->B = sub_A11;
    myctr->C = sub_A21;
    myctr->run();

    myctr->A = sub_A21;
    myctr->B = sub_A12;
    myctr->C = sub_A22;
    myctr->run();

    cyclic_dcapsp(topo, n/2, b1, b2, sub_A22, pred_A);
    
    myctr->A = sub_A12;
    myctr->B = sub_A22;
    myctr->C = sub_A12;
    myctr->run();

    myctr->A = sub_A22;
    myctr->B = sub_A21;
    myctr->C = sub_A21;
    myctr->run();

    myctr->A = sub_A12;
    myctr->B = sub_A21;
    myctr->C = sub_A11;
    myctr->run();

    if (topo->ilayer == 0){    
      lda_cpy(nb/2, nb/2, nb/2, nb, sub_A11, A);
      lda_cpy(nb/2, nb/2, nb/2, nb, sub_A21, A+nb/2);
      lda_cpy(nb/2, nb/2, nb/2, nb, sub_A12, A+nb*nb/2);
      lda_cpy(nb/2, nb/2, nb/2, nb, sub_A22, A+nb/2+nb*nb/2);
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
 * \param[in] b is the blocking factor for when to switch from cyclic to blocked
 */
void dcapsp(const topology_t * topo,
	    const int n,
	    REAL * A,		
	    int * pred_A,
	    const int b1,
	    const int b2){

//  blocked_dcapsp(topo, n, A, pred_A);
//  cyclic_dcapsp(topo, n, b, A, pred_A);
  cyclic_dcapsp(topo, n, b1, b2, A, pred_A);

}

void floyd_warshall(REAL * A, int const n){
  int i,j,k;
  TAU_FSTART(floyd_warshall);
  
  for (k=0; k<n; k++){
    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
	A[j*n+i] = MIN(A[j*n+i], (A[k*n+i] + A[j*n+k]));
      }
    }
  }
  TAU_FSTOP(floyd_warshall);
}


void floyd_warshall(REAL * A, int const n, int const lda){
  int i,j,k;
  TAU_FSTART(floyd_warshall);
  
  for (k=0; k<n; k++){
    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
	A[j*lda+i] = MIN(A[j*lda+i], (A[k*lda+i] + A[j*lda+k]));
      }
    }
  }
  TAU_FSTOP(floyd_warshall);
}

