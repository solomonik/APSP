/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include <stdio.h>
#include <stdint.h>
#include "string.h"
#include "assert.h"
#include "util.h"

#ifdef BGP
#define UTIL_DGEMM 	dgemm
#define UTIL_DAXPY 	daxpy
#define UTIL_DSCAL 	dscal
#define UTIL_DDOT 	ddot
#else
#define UTIL_DGEMM 	dgemm_
#define UTIL_DAXPY 	daxpy_
#define UTIL_DSCAL 	dscal_
#define UTIL_DDOT 	ddot_
#endif
extern "C"
void UTIL_DGEMM(const char *,	const char *,
		const int *,	const int *,
		const int *,	const double *,
		const double *,	const int *,
		const double *,	const int *,
		const double *,	double *,
				const int *);

extern "C"
void UTIL_DAXPY(const int * n,		double * dA,
		const double * dX,	const int * incX,
		double * dY,		const int * incY);

extern "C"
void UTIL_DSCAL(const int *n,		double *dA,
		const double * dX,	const int *incX);

extern "C"
double UTIL_DDOT(const int * n,		const double * dX,	
		 const int * incX,	const double * dY,	
		 const int * incY);

void cdgemm(const char transa,	const char transb,
	    const int m,	const int n,
	    const int k,	const double a,
	    const double * A,	const int lda,
	    const double * B,	const int ldb,
	    const double b,	double * C,
				const int ldc){
  UTIL_DGEMM(&transa, &transb, &m, &n, &k, &a, A,
	     &lda, B, &ldb, &b, C, &ldc);
}

void cdaxpy(const int n,	double dA,
	    const double * dX,	const int incX,
	    double * dY,	const int incY){
  UTIL_DAXPY(&n, &dA, dX, &incX, dY, &incY);
}

void cdscal(const int n,	double dA,
	    const double * dX,	const int incX){
  UTIL_DSCAL(&n, &dA, dX, &incX);
}

double cddot(const int n,	const double *dX,
	     const int incX,	const double *dY,
	     const int incY){
  return UTIL_DDOT(&n, dX, &incX, dY, &incY);
}

/**
 * \brief Copies submatrix to submatrix (column-major)
 * \param[in] nrow number of rows
 * \param[in] ncol number of columns
 * \param[in] lda_A lda along rows for A
 * \param[in] lda_B lda along rows for B
 * \param[in] A matrix to read from
 * \param[in,out] B matrix to write to
 */
void lda_cpy(const int nrow,  const int ncol,
	     const int lda_A, const int lda_B,
	     const double *A,  	    double *B){
  if (lda_A == nrow && lda_B == nrow){
    memcpy(B,A,nrow*ncol*sizeof(double));
  }
  int i;
  for (i=0; i<ncol; i++){
    memcpy(B+lda_B*i,A+lda_A*i,nrow*sizeof(double));
  }
}

/**
 * \brief prints matrix in 2D
 * \param[in] M matrix
 * \param[in] n number of rows
 * \param[in] m number of columns
 */
void print_matrix(double *M, int n, int m){
  int i,j;
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      printf("%lf ", M[i+j*n]);
    }
    printf("\n");
  }
}

/* abomination */
double util_dabs(double x){
  if (x >= 0.0) return x;
  return -x;
}

/** 
 * \brief we receive a contiguous buffer kb-by-n B and (k-kb)-by-n B_aux 
 * which is the block below.
 * To get a k-by-n buffer, we need to combine this buffer with our original
 * block. Since we are working with column-major ordering we need to interleave
 * the blocks. Thats what this function does.
 * \param[in,out] B the buffer to coalesce into
 * \param[in] B_aux the second buffer to coalesce from
 * \param[in] k the total number of rows
 * \param[in] n the number of columns
 * \param[in] kb the number of rows in a B originally
 */
void coalesce_bwd(double 	*B,	
		  double const	*B_aux,
		  int const	k,
		  int const	n,
		  int const	kb){
  int i;
  for (i=n-1; i>=0; i--){
    memcpy(B+i*k+kb, B_aux+i*(k-kb), (k-kb)*sizeof(double));
    memcpy(B+i*k, B+i*kb, kb*sizeof(double));
  }
}


/* Copies submatrix to submatrix */
void transp(const int size,  const int lda_i, const int lda_o,
	    const double *A, double *B){
  if (lda_i == 1){
    memcpy(B,A,size*sizeof(double));
  }
  int i,j,o;
  LIBT_ASSERT(size%lda_o == 0);
  LIBT_ASSERT(lda_o%lda_i == 0);
  for (o=0; o<size/lda_o; o++){
    for (j=0; j<lda_i; j++){
      for (i=0; i<lda_o/lda_i; i++){
	B[o*lda_o + j*lda_o/lda_i + i] = A[o*lda_o+j+i*lda_i];
      }
    }
  }
}


#ifdef COMM_TIME
/** 
 * \brief ugliest timer implementation on Earth
 * \param[in] end the type of operation this function should do (oh god why?)
 * \param[in] cdt the communicator
 * \param[in] p the number of processors
 * \param[in] iter the number of iterations if relevant
 * \param[in] myRank your rank if relevant
 */
void __CM(const int 	end, 
	  const	CommData *cdt, 
	  const int 	p, 
	  const int 	iter, 
	  const int 	myRank){
  static volatile double __commTime 	=0.0;
  static volatile double __commTimeDelta=0.0;
  static volatile double __idleTime	=0.0;
  static volatile double __idleTimeDelta=0.0;
  if (end == 0){
    __idleTimeDelta = TIME_SEC();	
    COMM_BARRIER(cdt); 
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  }
  else if (end == 1){
    __commTime += TIME_SEC() - __commTimeDelta;
  } else if (end == 2) {
    MPI_Reduce((void*)&__commTime, (void*)&__commTimeDelta, 1, COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt->cm);	
    __commTime = __commTimeDelta/p;
    if (myRank == 0)
      printf("%lf seconds spent doing communication on average per iteration\n", __commTime/iter); 

    MPI_Reduce((void*)&__idleTime, (void*)&__idleTimeDelta, 1,
	    COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt->cm);
    __idleTime = __idleTimeDelta/p;
    if (myRank == 0)
      printf("%lf seconds spent idle per iteration\n", __idleTime/iter); 
  } else if (end == 3){
    __commTime =0.0;
    __idleTime =0.0;
  } else if (end == 4){
    MPI_Irecv(NULL,0,MPI_CHAR,iter,myRank,cdt->cm,&(cdt->req[myRank]));
  } else if (end == 5){
    __idleTimeDelta =TIME_SEC();
    MPI_Send(NULL,0,MPI_CHAR,iter,myRank,cdt->cm);
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  } else if (end == 6){
    MPI_Status __stat;
    __idleTimeDelta =TIME_SEC();
    MPI_Wait(&(cdt->req[myRank]),&__stat);
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  }
} 

#endif


/**
 * \brief computes the size of a tensor in packed symmetric layout
 * \param[in] ndim tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] symm tensor symmetries
 * \param[in] symm_type tensor symmetry types
 * \return size of tensor in packed layout
 */
uint64_t packed_size(const int ndim, const int* len, const int* symm, 
		     const int* symm_type){

  int i, k, mp;
  uint64_t size, tmp;

  k = 1;
  tmp = 1;
  size = 1;
  mp = len[0];
  for (i = 0;i < ndim;i++){
    tmp = (tmp * mp) / k;
    k++;
    mp += symm_type[i];

    if (symm[i] == -1){
      size *= tmp;
      k = 1;
      tmp = 1;
      if (i < ndim - 1) mp = len[i + 1];
    }
  }

  return size;
}
