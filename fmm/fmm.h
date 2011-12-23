#ifndef __FMM_H__
#define __FMM_H__

#include <limits>

#ifndef REAL
typedef float REAL;
#endif

#ifndef MIN
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef PLUS
inline REAL __plus(const REAL& a, const REAL& b){
  static REAL inf = std::numeric_limits<REAL>::max();
  if (a == inf || b == inf) return inf;
  return a+b;
}
#define PLUS( a, b ) __plus((a), (b)) 
//#define PLUS( a, b ) ( (a) + (b) )
#endif


void fmm_naive( const char trans_A,	const char trans_B,
		const int m,		const int n,		const int k,		
		const REAL * A,		const int lda_A,
		const REAL * B,		const int lda_B,
		      REAL * C,		const int lda_C,
		const int * pred_A = 0,	int * pred_C = 0,	const int lda_P=1);

#endif
