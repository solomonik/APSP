#include "string.h"
#include "stdio.h"
#include <xmmintrin.h>
#include <algorithm>

const char* dgemm_desc = "Simple blocked dgemm.";

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

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


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




inline static void blk_transp(double * block,
                              double * new_block,
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
/* To optimize this, think about loop unrolling and software
    pipelining.  Hint:  For the majority of the matmuls, you
    know exactly how many iterations there are (the block size)...  */

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

void square_dgemm(int lda, 
                  double *A, 
                  double *B, 
                  double *C )
{
  int ldapad = lda;
  if (ldapad % RBM != 0)
    ldapad = (ldapad/RBM)*RBM + RBM;
  if (ldapad % RBN != 0)
    ldapad = (ldapad/RBN)*RBN + RBN;
  if (ldapad % RBK != 0)
    ldapad = (ldapad/RBK)*RBK + RBK;
  if (ldapad % CACHE_HOP_SIZE == 0)
    ldapad = ldapad + max(max(RBN,RBK),RBM);

  if(true || lda != ldapad ) {
    double pad_A[ldapad*ldapad] __attribute__ ((aligned(16)));
    double pad_B[ldapad*ldapad] __attribute__ ((aligned(16)));
    double pad_C[ldapad*ldapad] __attribute__ ((aligned(16)));
    for (int i = 0; i< lda; i++){
      memcpy(pad_A+i*ldapad, A+i*lda, lda*sizeof(double));
      std::fill(pad_A+i*ldapad+lda, pad_A+(i+1)*ldapad, 0);
    }
    for (int i = lda; i< ldapad; i++){
      std::fill(pad_A+i*ldapad, pad_A+(i+1)*ldapad, 0);
    }
    for (int i = 0; i< lda; i++){
      memcpy(pad_B+i*ldapad, B+i*lda, lda*sizeof(double));
      std::fill(pad_B+i*ldapad+lda, pad_B+(i+1)*ldapad, 0);
    }
    for (int i = lda; i< ldapad; i++){
      std::fill(pad_B+i*ldapad, pad_B+(i+1)*ldapad, 0);
    }
    std::fill(pad_C, pad_C+ldapad*ldapad, 0);
    
    double A_swap[ldapad*ldapad] __attribute__ ((aligned(16)));
    blk_transp(pad_A, A_swap, ldapad, ldapad);
    //  pad_A = A_swap;
    /*For each block combination*/
    for( int i2 = 0; i2 < ldapad; i2 += L2M ) {
      for( int j2 = 0; j2 < ldapad; j2 += L2N ) {
        for( int k2 = 0; k2 < ldapad; k2 += L2K ) {
          for( int i = i2; i < min(ldapad,i2+L2M); i += BLOCK_SIZE_M ) {
            for( int j = j2; j < min(ldapad,j2+L2N); j += BLOCK_SIZE_N ) {
              for( int k = k2; k < min(ldapad,k2+L2K); k += BLOCK_SIZE_K ) {
                /*This gets the correct block size (for fringe blocks also)*/
                int M = min( BLOCK_SIZE_M, ldapad-i );
                int N = min( BLOCK_SIZE_N, ldapad-j );
                int K = min( BLOCK_SIZE_K, ldapad-k );
                
                do_block(ldapad, ldapad, ldapad, M, N, K, 
                         A_swap+k+i*ldapad, pad_B+k+j*ldapad, pad_C+j+i*ldapad, 
                         2*(K==BLOCK_SIZE_K)+(N==BLOCK_SIZE_N));
              }
            }
          }
        }
      }
    }
    //double A_swap[lda*lda];
    blk_transp(pad_C, A_swap, ldapad, ldapad);
    for (int i = 0; i< lda; i++){
      memcpy(C+i*lda,A_swap+i*ldapad, lda*sizeof(double));
    }
  } else {
    double A_swap[lda*lda] __attribute__ ((aligned(16)));
    blk_transp(A, A_swap, lda, lda);
    /*For each block combination*/
    for( int i2 = 0; i2 < lda; i2 += L2M ) {
      for( int j2 = 0; j2 < lda; j2 += L2N ) {
        for( int k2 = 0; k2 < lda; k2 += L2K ) {
          for( int i = i2; i < min(lda,i2+L2M); i += BLOCK_SIZE_M ) {
            for( int j = j2; j < min(lda,j2+L2N); j += BLOCK_SIZE_N ) {
              for( int k = k2; k < min(lda,k2+L2K); k += BLOCK_SIZE_K ) {
                /*This gets the correct block size (for fringe blocks also)*/
                int M = min( BLOCK_SIZE_M, lda-i );
                int N = min( BLOCK_SIZE_N, lda-j );
                int K = min( BLOCK_SIZE_K, lda-k );
                
                do_block(lda, lda, lda, M, N, K, 
                         A_swap+k+i*lda, B+k+j*lda, C+j+i*lda, 
                         2*(K==BLOCK_SIZE_K)+(N==BLOCK_SIZE_N));
              }
            }
          }
        }
      }
    }
    //double A_swap[lda*lda];
    blk_transp(C, A_swap, lda, lda);
    for (int i = 0; i< lda; i++){
      memcpy(C+i*lda,A_swap+i*lda, lda*sizeof(double));
    }
  }
}       
