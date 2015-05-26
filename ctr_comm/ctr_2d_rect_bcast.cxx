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

#include "ctr_comm.h"
#include "../shared/util.h"

/**
 * \brief deallocates ctr_2d_rect_bcast object
 */
ctr_2d_rect_bcast::~ctr_2d_rect_bcast() {
   delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
ctr_2d_rect_bcast::ctr_2d_rect_bcast(ctr * other) : ctr(other) {
  ctr_2d_rect_bcast * o = (ctr_2d_rect_bcast*)other;
  rec_ctr = o->rec_ctr->clone();
  k = o->k;
  ctr_lda_A = o->ctr_lda_A;
  ctr_sub_lda_A = o->ctr_sub_lda_A;
  ctr_lda_B = o->ctr_lda_B;
  ctr_sub_lda_B = o->ctr_sub_lda_B;
  cdt_x = o->cdt_x;
  cdt_y = o->cdt_y;
}

/**
 * \brief copies ctr object
 */
ctr * ctr_2d_rect_bcast::clone() {
  return new ctr_2d_rect_bcast(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int ctr_2d_rect_bcast::mem_fp() {
  const int np_x 	= cdt_x->np;
  const int np_y 	= cdt_y->np;
  const int mb 		= ctr_lda_A*ctr_sub_lda_A;
  const int nb 		= ctr_lda_B*ctr_sub_lda_B;
  const int kb_A 	= k/np_x;
  const int kb_B 	= k/np_y;
  const int kb 		= MIN(kb_A,kb_B);

  return (mb+nb+MAX(nb,mb))*kb*sizeof(double);
}

/**
 * \brief Basically doing SUMMA, except assumes equal block size on
 *  each processor. Performs rank-kb updates 
 *  where kb is the smallest k blocking factor among A and B. 
 *  If blocking factor for B is smaller than for A, then this
 *  function uses an extra block buffer (3 buffers total) 
 */
void ctr_2d_rect_bcast::run() {
  int dk, owner_A, owner_B, ck_A, ck_B, alloced, ret;
  double * buf_A, * buf_B, * buf_aux; 
  
  TAU_FSTART(ctr_2d_rect_bcast);

  LIBT_ASSERT(cdt_x->nbcast >= 2);
  LIBT_ASSERT(cdt_y->nbcast >= 2);

  const int x_rank 	= cdt_x->rank;
  const int y_rank	= cdt_y->rank;
  const int np_x 	= cdt_x->np;
  const int np_y 	= cdt_y->np;

  LIBT_ASSERT(k%np_y == 0);
  LIBT_ASSERT(k%np_x == 0);

  const int mb 		= ctr_lda_A*ctr_sub_lda_A;
  const int nb 		= ctr_lda_B*ctr_sub_lda_B;

  const int kb_A 	= k/np_x;
  const int kb_B 	= k/np_y;

  /* Pick kb to be the smaller blocking factor, to bound by 2
     the number of processors bcasting each iteration. */
  const int kb = MIN(kb_A,kb_B);

  rec_ctr->beta 	= beta;
  rec_ctr->num_lyr 	= 1;
  rec_ctr->idx_lyr 	= 0;
  
  if (buffer != NULL){	
    alloced = 0;
  } else {
    alloced = 1;
    ret = posix_memalign((void**)&buffer,
			 ALIGN_BYTES,
			 mem_fp());
    LIBT_ASSERT(ret==0);
  }
  
  buf_A   	= buffer;
  buf_B   	= buf_A+mb*kb;
  /* FIXME: using extra buffer space here. If this is a problem, 
     we can write a new algorithm that works for arbitrary kb. */
  buf_aux 	= buf_B+nb*kb;

#ifndef USE_MPI
  int bid;
#endif

  for (dk=idx_lyr*kb; dk<k; dk+=kb*num_lyr){
    COMM_BARRIER(cdt_x); 
    COMM_BARRIER(cdt_y); 

    owner_A = dk / kb_A;
    owner_B = dk / kb_B;

    ck_A = MIN(((owner_A+1)*kb_A-dk), kb);
    ck_B = MIN(((owner_B+1)*kb_B-dk), kb);
    
    /* Since we picked kb to equal kb_A or kb_B, one of these must hold */
    LIBT_ASSERT(ck_A == kb || ck_B == kb);

    if (x_rank == owner_A){
      if (ck_A == kb_A){
	buf_A = A;
      } else {
	//memcpy(buf_A, A+(idk%kb_A)*mb, ck_A*mb*sizeof(double));
	lda_cpy(ctr_sub_lda_A*ck_A, ctr_lda_A,
		ctr_sub_lda_A*kb_A, ctr_sub_lda_A*ck_A, 
		A+(dk%kb_A)*ctr_sub_lda_A, buf_A);
      } 
    }
    if (y_rank == owner_B){
      //if (dk % kb_B == 0 && kb_B == ck_B){
      if (ck_B == kb_B){
	buf_B = B;
      } else {
	//lda_cpy(ck_B, nb, kb_B, ck_B, B+(dk%kb_B), buf_B);
	/* FIXME: Failing here */
	lda_cpy(ctr_sub_lda_B*ck_B, ctr_lda_B,
		ctr_sub_lda_B*kb_B, ctr_sub_lda_B*ck_B, 
		B+(dk%kb_B)*ctr_sub_lda_B, buf_B);
      } 
    }
    POST_BCAST(buf_A, ck_A*mb, COMM_DOUBLE_T, owner_A, cdt_x, 0);
    POST_BCAST(buf_B, ck_B*nb, COMM_DOUBLE_T, owner_B, cdt_y, 0);

    WAIT_BCAST(cdt_x, 1, &bid);
    WAIT_BCAST(cdt_y, 1, &bid);
    
    /* FIXME: unnecessary, but safer for DCMF for now */
    COMM_BARRIER(cdt_x); 
    COMM_BARRIER(cdt_y); 

    if (ck_A < kb){ /* If the required A block is cut between 2 procs */
      if (x_rank == owner_A+1)
	//memcpy(buf_A+ck_A*mb, A, mb*(kb-ck_A)*sizeof(double)); 
	lda_cpy(ctr_sub_lda_A*(kb-ck_A), ctr_lda_A,
		ctr_sub_lda_A*kb_A, ctr_sub_lda_A*(kb-ck_A), 
		A, buf_aux);
      POST_BCAST(buf_aux, mb*(kb-ck_A), COMM_DOUBLE_T, owner_A+1, cdt_x, 0);
      WAIT_BCAST(cdt_x, 1, &bid);
      coalesce_bwd(buf_A, 
		   buf_aux, 
		   ctr_sub_lda_A*kb, 
		   ctr_lda_A, 
		   ctr_sub_lda_A*ck_A);
      buf_A = buf_aux;
    } else if (ck_B < kb){ /* If the B block is cut between 2 procs */
      if (x_rank == owner_B+1)
	lda_cpy(ctr_sub_lda_B*(kb-ck_B), ctr_lda_B,
		ctr_sub_lda_B*kb_B, ctr_sub_lda_B*(kb-ck_B), 
		B, buf_aux);
      POST_BCAST(buf_aux, nb*(kb-ck_B), COMM_DOUBLE_T, owner_B+1, cdt_y, 0);
      WAIT_BCAST(cdt_y, 1, &bid);
      coalesce_bwd(buf_B, 
		   buf_aux, 
		   ctr_sub_lda_B*kb, 
		   ctr_lda_B, 
		   ctr_sub_lda_B*ck_B);
      buf_B = buf_aux;
    }

    DEBUG_PRINTF("[%d][%d] multiplying %lf by %lf\n",
		 cdt_x->rank, cdt_y->rank,
		 buf_A[0],  buf_B[0]);

    rec_ctr->A = buf_A;
    rec_ctr->B = buf_B;
    rec_ctr->C = C;

    rec_ctr->run();

    buf_A   = buffer;
    buf_B   = buf_A+mb*kb;
  
    rec_ctr->beta = 1.0;
  }
  /* FIXME: reuse that shit */
  if (alloced){
    free(buffer);
    buffer = NULL;
  }
  TAU_FSTOP(ctr_2d_rect_bcast);
}


