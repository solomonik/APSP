CC	= CC
CXX	= CC
#CFLAGS  = -g -O0 $(DFLAGS)
OPENMP = -fopenmp
CFLAGS = -O4 -DUSE_OMP -DHOPPER $(DFLAGS) $(OPENMP)
LIBS    = $(OPENMP) 

ifneq (,$(findstring DTAU,$(CFLAGS)))
        include  $(TAUROOTDIR)/include/Makefile
        CFLAGS+=$(TAU_INCLUDE) $(TAU_DEFS) 
#        LIBS+=$(TAU_MPI_LIBS) $(TAU_LIBS) 
        LIBS+= $(TAU_LIBS)
endif

