CC	= mpicxx
CXX	= mpicxx
OPENMP = -fopenmp
CFLAGS = -O3 -DUSE_OMP -DHOPPER $(DFLAGS) $(OPENMP)
LIBS    = $(OPENMP)  -lblas

ifneq (,$(findstring DTAU,$(CFLAGS)))
        include  $(TAUROOTDIR)/include/Makefile
        CFLAGS+=$(TAU_INCLUDE) $(TAU_DEFS) 
#        LIBS+=$(TAU_MPI_LIBS) $(TAU_LIBS) 
        LIBS+= $(TAU_LIBS)
endif

