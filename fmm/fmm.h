#ifndef __FMM_H__
#define __FMM_H__

#include <limits>
#include <stdlib.h>
#include <sys/time.h>

#ifndef PRECISION
#define PRECISION 2
#endif

#if (PRECISION==1)
typedef float REAL;
#define REAL_MPI MPI_FLOAT
#endif

#if (PRECISION==2)
typedef double REAL;
#define REAL_MPI MPI_DOUBLE
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

//#include "../shared/util.h"

#ifndef TIME_SEC
static double __timer(){
  static bool initialized = false;
  static struct timeval start;
  struct timeval end;
  if(!initialized){
    gettimeofday( &start, NULL );
    initialized = true;
  }
  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}
#define TIME_SEC() __timer()
#endif

#ifdef TAU
#include <Profile/Profiler.h>
#define TAU_FSTART(ARG)					\
    TAU_PROFILE_TIMER(timer##ARG, #ARG, "", TAU_USER);	\
    TAU_PROFILE_START(timer##ARG)

#define TAU_FSTOP(ARG)					\
    TAU_PROFILE_STOP(timer##ARG)

#else
#define TAU_PROFILE(NAME,ARG,USER)
#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)
#define TAU_PROFILE_STOP(ARG)
#define TAU_PROFILE_START(ARG)
#define TAU_FSTART(ARG) 
#define TAU_FSTOP(ARG) 
#endif


void fmm_naive( const char trans_A,	const char trans_B,
		const int m,		const int n,		const int k,		
		const REAL * A,		const int lda_A,
		const REAL * B,		const int lda_B,
		      REAL * C,		const int lda_C,
		const int * pred_A = 0,	int * pred_C = 0,	const int lda_P=1);

void fmm_opt( 	const char trans_A,	const char trans_B,
		const int m,		const int n,		const int k,		
		const REAL * A,		const int lda_A,
		const REAL * B,		const int lda_B,
		      REAL * C,		const int lda_C,
		const int * pred_A = 0,	int * pred_C = 0,	const int lda_P=1);


#endif
