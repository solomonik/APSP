#include "fmm.h"
#include <stdio.h>

void fmm_naive( const char trans_A,	const char trans_B,
		const int m,		const int n,		const int k,		
		const REAL * A,		const int lda_A,
		const REAL * B,		const int lda_B,
		      REAL * C,		const int lda_C,
		const int  * pred_A,	int * pred_C, 		const int lda_P){
  int i,j,l,vij;
  REAL c, nc;
  int pA, pB, pC;
  int p2A, p2B, p2C;
  int pPA, p2PA, pPC, p2PC;
  int row_str_A, row_str_B;
  int col_str_A, col_str_B;
  int col_str_P, row_str_P;

  if (trans_A == 'N'){
    row_str_A = 1;
    col_str_A = lda_A;
    row_str_P = 1;
    col_str_P = lda_P;
  } else {
    row_str_A = lda_A;
    col_str_A = 1;
    row_str_P = lda_P;
    col_str_P = 1;
  }
  if (trans_B == 'N'){
    row_str_B = 1;
    col_str_B = lda_B;
  } else {
    row_str_B = lda_B;
    col_str_B = 1;
  }

  /*printf("col_str_A = %d, row_str_A = %d, col_str_B = %d, row_str_B = %d\n",
	  col_str_A, row_str_A, col_str_B, row_str_B);*/

  pB = 0, pC = 0, pPC = 0;
  for (i=0; i<n; i++){
    pA = 0;
    pPA = 0;
    p2B = pB;
    p2C = pC;
    p2PC = pPC;
    for (j=0; j<m; j++){
      p2A = pA;
      p2PA = pPA;
      p2B = pB;

      c = C[p2C];
      if (pred_A && pred_C){
	vij = pred_C[p2PC];
	//printf("vij = %d\n",vij);
      }
      for (l=0; l<k; l++){
	if (pred_A && pred_C) {
	  nc = PLUS((A[p2A]), (B[p2B]));
	  if (nc < c){
	    c = nc;
	    vij = pred_A[p2PA];
	   // printf("new C val at %d %d is %f %d\n",i,j, nc, vij);
	  }
	} else 
	  c = MIN(c, PLUS((A[p2A]), (B[p2B])));
	
	p2A += col_str_A;
	p2PA += col_str_P;
	p2B += row_str_B;
      }
      C[p2C] = c;
      if (pred_A && pred_C) pred_C[p2PC] = vij;
      
      pA += row_str_A;
      pPA += row_str_P;
      p2PC++;
      p2C++;
    }
    pB += col_str_B;
    pC += lda_C;
    pPC += lda_P;
  }
}
