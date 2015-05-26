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

#ifndef __CTR_COMM_H__
#define __CTR_COMM_H__

#include "../shared/comm.h"
#include "../shared/util.h"

class ctr {
  public: 
    double * A; /* m by k */
    double * B; /* k by n */
    double * C; /* m by n */
    double beta;
    int num_lyr; /* number of copies of this matrix being computed on */
    int idx_lyr; /* the index of this copy */
    double * buffer;

    virtual void run() { printf("SHOULD NOTR\n"); };
    virtual int mem_fp() { return 0; };
    virtual ctr * clone() { return NULL; };
    
    virtual ~ctr();
  
    ctr(ctr * other);
    ctr(){ buffer = NULL; }
};

class ctr_2d_rect_bcast : public ctr {
  public: 
    int k;
    int ctr_lda_A; /* local lda_A of contraction dimension 'k' */
    int ctr_sub_lda_A; /* elements per local lda_A 
			  of contraction dimension 'k' */
    int ctr_lda_B; /* local lda_B of contraction dimension 'k' */
    int ctr_sub_lda_B; /* elements per local lda_B 
			  of contraction dimension 'k' */
    CommData_t * cdt_x;
    CommData_t * cdt_y;
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    
    void run();
    int mem_fp();
    ctr * clone();

    ctr_2d_rect_bcast(ctr * other);
    ~ctr_2d_rect_bcast();
    ctr_2d_rect_bcast(){}
};


class ctr_2d_sqr_bcast : public ctr {
  public: 
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    int k;
    int sz_A; /* number of elements in a block of A */
    int sz_B; /* number of elements in a block of A */
    CommData_t * cdt_x;
    CommData_t * cdt_y;
    
    void run();
    int mem_fp();
    ctr * clone();

    ctr_2d_sqr_bcast(ctr * other);
    ~ctr_2d_sqr_bcast();
    ctr_2d_sqr_bcast(){}
};

/* Assume LDA equal to dim */
class ctr_dgemm : public ctr {
  public: 
    char transp_A;
    char transp_B;
  /*  int lda_A;
    int lda_B;
    int lda_C;*/
    double alpha;
    int n;
    int m;
    int k;
    
    void run();
    int mem_fp();
    ctr * clone();

    ctr_dgemm(ctr * other);
    ~ctr_dgemm();
    ctr_dgemm(){}
};

class ctr_lyr : public ctr {
  public: 
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    int k;
    CommData_t * cdt;
    int sz_A, sz_B, sz_C;
    MPI_Op red_op;
    
    void run();
    int mem_fp();
    ctr * clone();

    ctr_lyr(ctr * other);
    ~ctr_lyr();
    ctr_lyr(){ }
};

class ctr_fmm : public ctr {
  public: 
  /*  char transp_A;
    char transp_B;
    int lda_A;
    int lda_B;
    int lda_C;*/
    int n;
    
    void run();
    int mem_fp();
    ctr * clone();

    ctr_fmm(ctr * other);
    ~ctr_fmm();
    ctr_fmm(){}
};


#endif // __CTR_COMM_H__
