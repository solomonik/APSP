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

#include "../shared/util.h"
#include "ctr_comm.h"
#include "../fmm/fmm.h"
#include <float.h>

/**
 * \brief deallocates generic ctr object
 */
ctr::~ctr(){
  if (buffer != NULL) free(buffer);
}

/**
 * \brief copies generic ctr object
 */
ctr::ctr(ctr * other){
  A = other->A;
  B = other->B;
  C = other->C;
  beta = other->beta;
  num_lyr = other->num_lyr;
  idx_lyr = other->idx_lyr;
  buffer = NULL;
}

/**
 * \brief deallocates ctr_fmm object
 */
ctr_fmm::~ctr_fmm() { }

/**
 * \brief copies ctr object
 */
ctr_fmm::ctr_fmm(ctr * other) : ctr(other) {
  ctr_fmm * o = (ctr_fmm*)other;
  n = o->n;
}
/**
 * \brief copies ctr object
 */
ctr * ctr_fmm::clone() {
  return new ctr_fmm(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int ctr_fmm::mem_fp(){
  return 0;
}

/**
 * \brief a wrapper for fmm
 */
void ctr_fmm::run(){
/*  const int lda_A = transp_A == 'n' ? m : k;
  const int lda_B = transp_B == 'n' ? k : n;
  const int lda_C = m;
  cfmm(transp_A,
	 transp_B,
	 m,
	 n,
	 k,
	 alpha,
	 A,
	 lda_A,
	 B,
	 lda_B,
	 beta,
	 C,
	 lda_C);*/
  TAU_FSTART(ctr_fmm__fmm_opt);
  fmm_opt('N', 'N', n, n, n, A, n, B, n, C, n);
  TAU_FSTOP(ctr_fmm__fmm_opt);

}


/**
 * \brief deallocates ctr_dgemm object
 */
ctr_dgemm::~ctr_dgemm() { }

/**
 * \brief copies ctr object
 */
ctr_dgemm::ctr_dgemm(ctr * other) : ctr(other) {
  ctr_dgemm * o = (ctr_dgemm*)other;
  n = o->n;
  m = o->m;
  k = o->k;
  alpha = o->alpha;
  transp_A = o->transp_A;
  transp_B = o->transp_B;
}
/**
 * \brief copies ctr object
 */
ctr * ctr_dgemm::clone() {
  return new ctr_dgemm(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int ctr_dgemm::mem_fp(){
  return 0;
}

/**
 * \brief a wrapper for dgemm
 */
void ctr_dgemm::run(){
  const int lda_A = transp_A == 'n' ? m : k;
  const int lda_B = transp_B == 'n' ? k : n;
  const int lda_C = m;
  cdgemm(transp_A,
	 transp_B,
	 m,
	 n,
	 k,
	 alpha,
	 A,
	 lda_A,
	 B,
	 lda_B,
	 beta,
	 C,
	 lda_C);

}

/**
 * \brief deallocates ctr_lyr object
 */
ctr_lyr::~ctr_lyr() {
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
ctr_lyr::ctr_lyr(ctr * other) : ctr(other) {
  ctr_lyr * o = (ctr_lyr*)other;
  rec_ctr = o->rec_ctr->clone();
  k = o->k;
  cdt = o->cdt;
  sz_C = o->sz_C;
}

/**
 * \brief copies ctr object
 */
ctr * ctr_lyr::clone() {
  return new ctr_lyr(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int ctr_lyr::mem_fp(){
  return sz_C*sizeof(double);
}

/**
 * \brief performs replication along a dimension, generates 2.5D algs
 */
void ctr_lyr::run(){
  int alloced, ret;

  if (buffer != NULL){	
    alloced = 0;
  } else {
    alloced = 1;
    ret = posix_memalign((void**)&buffer,
			 ALIGN_BYTES,
			 mem_fp());
    LIBT_ASSERT(ret==0);
  }
  rec_ctr->C = buffer;
  if (idx_lyr == 0)
    memcpy(rec_ctr->C, C, sz_C*sizeof(double));
  else
    std::fill(rec_ctr->C, rec_ctr->C+sz_C, DBL_MAX);
  
  rec_ctr->A 		= A;
  rec_ctr->B 		= B;
  rec_ctr->beta		= cdt->rank > 0 ? 0.0 : beta;
  rec_ctr->num_lyr 	= cdt->np;
  rec_ctr->idx_lyr 	= cdt->rank;

  BCAST(A, sz_A, MPI_DOUBLE, 0, cdt);
  BCAST(B, sz_B, MPI_DOUBLE, 0, cdt);
  
  rec_ctr->run();
  
  /* FIXME: unnecessary except for current DCMF wrapper */
  COMM_BARRIER(cdt);
  REDUCE(buffer, C, sz_C, MPI_DOUBLE, red_op, 0, cdt);

  if (alloced){
    free(buffer);
    buffer = NULL;
  }

}




