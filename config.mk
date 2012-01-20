CC	= mpicxx
CXX	= mpicxx
#CFLAGS  = -g -O0 $(DFLAGS)
OPENMP = -fopenmp
CFLAGS = -O3 -msse4.2 -msse4 -msse2 -DUSE_OMP $(OPENMP) $(DFLAGS)
LIBS    = -lblas -llapack $(OPENMP)

ifneq (,$(findstring DTAU,$(DFLAGS)))
        include  $(TAUROOTDIR)/include/Makefile
        CFLAGS+=$(TAU_INCLUDE) $(TAU_DEFS) 
#       LIBS+=$(TAU_MPI_LIBS) $(TAU_LIBS) 
        LIBS+= $(TAU_LIBS)
endif

