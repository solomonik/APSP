#include "fmm.h"
#include <stdio.h>

void fmm_naive( const char trans_A,	const char trans_B,
		const int m,		const int n,		const int k,		
		const REAL * A,		const int lda_A,
		const REAL * B,		const int lda_B,
		      REAL * C,		const int lda_C){
  int i,j,l;
  REAL c;
  REAL const * pA;
  REAL const * pB;
  REAL * pC;
  REAL const * p2A;
  REAL const * p2B;
  REAL * p2C;
  int row_str_A, row_str_B;
  int col_str_A, col_str_B;

  if (trans_A == 'N'){
    row_str_A = 1;
    col_str_A = lda_A;
  } else {
    row_str_A = lda_A;
    col_str_A = 1;
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

  pB = B, pC = C;
  for (i=0; i<n; i++){
    pA = A;
    p2B = pB;
    p2C = pC;
    for (j=0; j<m; j++){
      p2A = pA;
      p2B = pB;

      c = *p2C;
      for (l=0; l<k; l++){
	c = MIN(c, PLUS((*p2A), (*p2B)));
	
	p2A += col_str_A;
	p2B += row_str_B;
      }
      *p2C = c;
      
      pA += row_str_A;
      p2C++;
    }
    pB += col_str_B;
    pC += lda_C;
  }
}
