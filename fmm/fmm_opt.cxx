#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <xmmintrin.h>
#include <algorithm>
#include <math.h>

#include "fmm.h"

#ifndef CACHE_HOP_SIZE
#define CACHE_HOP_SIZE 256
#endif

#if !defined(BLOCK_SIZE_N)
#define BLOCK_SIZE_N 32
#endif
#if !defined(BLOCK_SIZE_K)
#define BLOCK_SIZE_K 32
#endif
#if !defined(BLOCK_SIZE_M)
#define BLOCK_SIZE_M 32
#endif

#ifndef L2M
#define L2M 128
#endif

#ifndef L2N
#define L2N 128
#endif

#ifndef L2K
#define L2K 128
#endif


#ifndef RBN
#define RBN 1
#endif

#ifndef RBK
#define RBK 2
#endif

#ifndef RBM
#define RBM 2
#endif

//#define min(a,b) (((a)<(b))?(a):(b))
//#define max(a,b) (((a)>(b))?(a):(b))


#if (RBK==2)
  #define INIT_ROW_A(ib) \
    __m128d A_##ib##_0; 
  #define INIT_ROW_B(jb) \
    __m128d B_##jb##_0; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = _mm_load_pd(A+k+0+(i+(ib))*ldk);
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = _mm_load_pd(B+k+0+(j+(jb))*ldk);
  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib,_mm_mul_pd(A_##ib##_0,B_##jb##_0)); 
#endif

#if (RBK==4)
  #define INIT_ROW_A(ib) \
    __m128d A_##ib##_0; \
    __m128d A_##ib##_1; 
  #define INIT_ROW_B(jb) \
    __m128d B_##jb##_0; \
    __m128d B_##jb##_1; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m128d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m128d *)(A+k+2+(i+(ib))*ldk))[0];
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m128d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m128d *)(B+k+2+(j+(jb))*ldk))[0];
  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_1,B_##jb##_1));
#endif    

#if (RBK==8)
  #define INIT_ROW_A(ib) \
    __m128d A_##ib##_0; __m128d A_##ib##_1; \
    __m128d A_##ib##_2; __m128d A_##ib##_3; 
  #define INIT_ROW_B(jb) \
    __m128d B_##jb##_0; __m128d B_##jb##_1; \
    __m128d B_##jb##_2; __m128d B_##jb##_3; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m128d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m128d *)(A+k+2+(i+(ib))*ldk))[0]; \
    A_##ib##_2 = ((__m128d *)(A+k+4+(i+(ib))*ldk))[0]; \
    A_##ib##_3 = ((__m128d *)(A+k+6+(i+(ib))*ldk))[0];
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m128d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m128d *)(B+k+2+(j+(jb))*ldk))[0]; \
    B_##jb##_2 = ((__m128d *)(B+k+4+(j+(jb))*ldk))[0]; \
    B_##jb##_3 = ((__m128d *)(B+k+6+(j+(jb))*ldk))[0];

  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_1,B_##jb##_1)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_2,B_##jb##_2)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_3,B_##jb##_3));
#endif    


#if (RBN==1)
  #define INIT_B \
    INIT_ROW_B(0)
  #define LOAD_B \
    LOAD_ROW_B(0)
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; \
    double C1_0_##ib; \
    double C2_0_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd();
       
  #define STORE_ROW_C(ib) \
    _mm_storeh_pd(&C1_0_##ib,C_0_##ib);\
    _mm_storel_pd(&C2_0_##ib,C_0_##ib);\
    C[j+0+(i+(ib))*ldn] += C1_0_##ib + C2_0_##ib;

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0)
#endif

#if (RBN==2)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) 
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; __m128d C_1_##ib; \
    __m128d C_swap_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd(); \
    C_1_##ib = _mm_setzero_pd();

  #define STORE_ROW_C(ib) \
    C_swap_##ib = _mm_unpacklo_pd(C_0_##ib, C_1_##ib); \
    C_0_##ib = _mm_unpackhi_pd(C_0_##ib, C_1_##ib); \
    C_1_##ib = _mm_load_pd(C+j+0+(i+(ib))*ldn); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_0_##ib); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_swap_##ib); \
    _mm_store_pd(C+j+0+(i+(ib))*ldn, C_1_##ib); 

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1)
#endif

#if (RBN==4)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1) \
    INIT_ROW_B(2) INIT_ROW_B(3)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) \
    LOAD_ROW_B(2) LOAD_ROW_B(3) 
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; __m128d C_1_##ib; \
    __m128d C_swap_0_##ib; \
    __m128d C_2_##ib; __m128d C_3_##ib; \
    __m128d C_swap_1_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd(); \
    C_1_##ib = _mm_setzero_pd(); \
    C_2_##ib = _mm_setzero_pd(); \
    C_3_##ib = _mm_setzero_pd();

  #define STORE_ROW_C(ib) \
    C_swap_0_##ib = _mm_unpacklo_pd(C_0_##ib, C_1_##ib); \
    C_0_##ib = _mm_unpackhi_pd(C_0_##ib, C_1_##ib); \
    C_1_##ib = _mm_load_pd(C+j+0+(i+(ib))*ldn); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_0_##ib); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_swap_0_##ib); \
    _mm_store_pd(C+j+0+(i+(ib))*ldn, C_1_##ib); \
    C_swap_1_##ib = _mm_unpacklo_pd(C_2_##ib, C_3_##ib); \
    C_2_##ib = _mm_unpackhi_pd(C_2_##ib, C_3_##ib); \
    C_3_##ib = _mm_load_pd(C+j+2+(i+(ib))*ldn); \
    C_3_##ib = _mm_add_pd(C_3_##ib,C_2_##ib); \
    C_3_##ib = _mm_add_pd(C_3_##ib,C_swap_1_##ib); \
    _mm_store_pd(C+j+2+(i+(ib))*ldn, C_3_##ib); 


  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1) \
    MUL_ROW_A_B(ib, 2) MUL_ROW_A_B(ib, 3)
#endif

#if (RBM==1)
  #define INIT_A \
    INIT_ROW_A(0) 
  #define LOAD_A \
    LOAD_ROW_A(0)
  #define INIT_C \
    INIT_ROW_C(0) 
  #define LOAD_C \
    LOAD_ROW_C(0)
  #define STORE_C \
    STORE_ROW_C(0)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) 
#endif

#if (RBM==2)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) 
#endif

#if (RBM==4)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) \
    INIT_ROW_A(2) INIT_ROW_A(3) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1) \
    LOAD_ROW_A(2) LOAD_ROW_A(3)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) \
    INIT_ROW_C(2) INIT_ROW_C(3) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1) \
    LOAD_ROW_C(2) LOAD_ROW_C(3)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1) \
    STORE_ROW_C(2) STORE_ROW_C(3)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) \
    MUL_SQUARE_A_B(2) MUL_SQUARE_A_B(3) 
#endif




inline static void blk_transp(REAL * block,
			      REAL * new_block,
			      int const nrow,
			      int const ncol){
  for (int i = 0; i < ncol; i++){
    for (int j = 0; j < nrow; j++){
      new_block[i*nrow+j]=block[j*ncol+i];
    }
  }
}

#pragma safeptr=all
inline static void do_block(int const ldm, 
			    int const ldn,
			    int const ldk,
			    int const M, 
			    int const N, 
			    int const K, 
			    double const *A, 
			    double const *B,
			    double *  C,
			    int const full_KN){

  int const mm = (M/RBM + (M%RBM > 0))*RBM;
  int const nn = (N/RBN + (N%RBN > 0))*RBN;
  int const kk = (K/RBK + (K%RBK > 0))*RBK;


  int i,j,k;//,ib,jb,kb;

  INIT_A
  INIT_B
  INIT_C

  if (full_KN == 3){
    for (i = 0; i < mm; i+= RBM){
      #pragma unroll BLOCK_SIZE_N/RBN
      for (j = 0; j < nn; j+= RBN){
	LOAD_C;
	#pragma unroll BLOCK_SIZE_K/RBK
	for (k = 0; k < BLOCK_SIZE_K; k+= RBK){
	  LOAD_A;
	  LOAD_B;
	  MUL_A_B;
	}
	STORE_C;
      }
    }
  } else if (full_KN == 2){
    for (i = 0; i < mm; i+= RBM){
      for (j = 0; j < nn; j+= RBN){
	LOAD_C;
	#pragma unroll BLOCK_SIZE_K/RBK
	for (k = 0; k < BLOCK_SIZE_K; k+= RBK){
	  LOAD_A;
	  LOAD_B;
	  MUL_A_B;
	}
	STORE_C;
      }
    }
  } else{
    for (i = 0; i < mm; i+= RBM){
      for (j = 0; j < nn; j+= RBN){
	LOAD_C;
	#pragma unroll
	for (k = 0; k < kk; k+= RBK){
	  LOAD_A;
	  LOAD_B;
	  MUL_A_B;
	}
	STORE_C;
      }
    }
  }
}

void fmm_opt( 	const char trans_A,	const char trans_B,
		const int m,		const int n,		const int k,		
		const REAL * A,		const int lda_A,
		const REAL * B,		const int lda_B,
		      REAL * C,		const int lda_C,
		const int * pred_A,	int * pred_C,		const int lda_P){
  int mpad, npad, kpad;
  mpad = m, npad = n, kpad = k;

  
  const REAL inf = std::numeric_limits<REAL>::max();

  if (mpad % RBM != 0)
    mpad = (mpad/RBM)*RBM + RBM;
  if (npad % RBN != 0)
    npad = (npad/RBN)*RBN + RBN;
  if (kpad % RBK != 0)
    kpad = (kpad/RBK)*RBK + RBK;
  if (kpad % CACHE_HOP_SIZE == 0)
    kpad = kpad + MAX(MAX(RBN,RBK),RBM);

  if(true) {
    REAL pad_A[mpad*kpad] __attribute__ ((aligned(16)));
    REAL pad_B[kpad*npad] __attribute__ ((aligned(16)));
    REAL pad_C[mpad*npad] __attribute__ ((aligned(16)));
    if (trans_A == 'n' || trans_A == 'N'){
      for (int i=0; i<k; i++){
	memcpy(pad_A+i*mpad, A+i*lda_A, m*sizeof(REAL));
	std::fill(pad_A+i*mpad+m, pad_A+(i+1)*mpad, inf);
      }
      for (int i=k; i<kpad; i++){
	std::fill(pad_A+i*mpad, pad_A+(i+1)*mpad, inf);
      }
    } else {
      for (int i=0; i<m; i++){
	memcpy(pad_A+i*kpad, A+i*lda_A, k*sizeof(REAL));
	std::fill(pad_A+i*kpad+k, pad_A+(i+1)*kpad, inf);
      }
      for (int i=m; i<mpad; i++){
	std::fill(pad_A+i*kpad, pad_A+(i+1)*kpad, inf);
      }
    }
    if (trans_B == 'n' || trans_B == 'N'){
      for (int i=0; i<n; i++){
	memcpy(pad_B+i*kpad, B+i*lda_B, k*sizeof(REAL));
	std::fill(pad_B+i*kpad+k, pad_B+(i+1)*kpad, inf);
      }
      for (int i=n; i<npad; i++){
	std::fill(pad_B+i*kpad, pad_B+(i+1)*kpad, inf);
      }
    } else {
      for (int i=0; i<k; i++){
	memcpy(pad_B+i*npad, B+i*lda_B, n*sizeof(REAL));
	std::fill(pad_B+i*npad+n, pad_B+(i+1)*npad, inf);
      }
      for (int i=k; i<kpad; i++){
	std::fill(pad_B+i*npad, pad_B+(i+1)*npad, inf);
      }
    }
    for (int i=0; i<n; i++){
      memcpy(pad_C+i*mpad, C+i*lda_C, m*sizeof(REAL));
      std::fill(pad_C+i*mpad+m, pad_C+(i+1)*mpad, inf);
    }
    for (int i=n; i<npad; i++){
      std::fill(pad_C+i*mpad, pad_C+(i+1)*mpad, inf);
    }
    {
      REAL A_swap[kpad*mpad] __attribute__ ((aligned(16)));
      REAL B_swap[npad*kpad] __attribute__ ((aligned(16)));
      
      if (trans_A == 'n' || trans_A == 'N') blk_transp(pad_A, A_swap, kpad, mpad);
      else memcpy(A_swap, pad_A, kpad*mpad*sizeof(REAL));

      if (trans_B == 't' || trans_B == 'T') blk_transp(pad_B, B_swap, kpad, npad);
      else memcpy(B_swap, pad_B, npad*kpad*sizeof(REAL));

      //  pad_A = A_swap;
      /*For each block combination*/
      for( int i2 = 0; i2 < mpad; i2 += L2M ) {
	for( int j2 = 0; j2 < npad; j2 += L2N ) {
	  for( int k2 = 0; k2 < kpad; k2 += L2K ) {
	    for( int i = i2; i < MIN(mpad,i2+L2M); i += BLOCK_SIZE_M ) {
	      for( int j = j2; j < MIN(npad,j2+L2N); j += BLOCK_SIZE_N ) {
		for( int k = k2; k < MIN(kpad,k2+L2K); k += BLOCK_SIZE_K ) {
		  /*This gets the correct block size (for fringe blocks also)*/
		  int M = MIN( BLOCK_SIZE_M, mpad-i );
		  int N = MIN( BLOCK_SIZE_N, npad-j );
		  int K = MIN( BLOCK_SIZE_K, kpad-k );

		  fmm_naive('T', 'N', M, N, K, 
			    A_swap+k+i*kpad, kpad,
			    B_swap+k+j*kpad, kpad,
			    pad_C+i+j*mpad, mpad);
		  
		/*  do_block(mpad, npad, kpad, M, N, K, 
			   A_swap+k+i*kpad, B_swap+k+j*kpad, pad_C+j+i*npad, 
			   2*(K==BLOCK_SIZE_K)+(N==BLOCK_SIZE_N));*/

		}
	      }
	    }
	  }
	}
      }
    }
    {
      //REAL A_swap[lda*lda];
#if 0
      REAL C_swap[npad*kpad] __attribute__ ((aligned(16)));
      blk_transp(pad_C, C_swap, mpad, npad);
      for (int i = 0; i<n; i++){
	memcpy(C+i*lda_C,C_swap+i*mpad, m*sizeof(REAL));
      }
#else
      for (int i=0; i<n; i++){
	memcpy(C+i*lda_C,pad_C+i*mpad, m*sizeof(REAL));
      }
#endif
    }
  } else {
#if 0
    REAL A_swap[lda*lda] __attribute__ ((aligned(16)));
    blk_transp(A, A_swap, lda, lda);
    /*For each block combination*/
    for( int i2 = 0; i2 < lda; i2 += L2M ) {
      for( int j2 = 0; j2 < lda; j2 += L2N ) {
	for( int k2 = 0; k2 < lda; k2 += L2K ) {
	  for( int i = i2; i < MIN(lda,i2+L2M); i += BLOCK_SIZE_M ) {
	    for( int j = j2; j < MIN(lda,j2+L2N); j += BLOCK_SIZE_N ) {
	      for( int k = k2; k < MIN(lda,k2+L2K); k += BLOCK_SIZE_K ) {
		/*This gets the correct block size (for fringe blocks also)*/
		int M = MIN( BLOCK_SIZE_M, lda-i );
		int N = MIN( BLOCK_SIZE_N, lda-j );
		int K = MIN( BLOCK_SIZE_K, lda-k );
		
		do_block(lda, lda, lda, M, N, K, 
			 A_swap+k+i*lda, B+k+j*lda, C+j+i*lda, 
			 2*(K==BLOCK_SIZE_K)+(N==BLOCK_SIZE_N));
	      }
	    }
	  }
	}
      }
    }
    //REAL A_swap[lda*lda];
    blk_transp(C, A_swap, lda, lda);
    for (int i = 0; i< lda; i++){
      memcpy(C+i*lda,A_swap+i*lda, lda*sizeof(REAL));
    }
#endif
  }
}	
