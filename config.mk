CC	= mpicxx
CXX	= mpicxx
#CFLAGS  = -g -O0 $(DFLAGS)
CFLAGS = -O3 $(DFLAGS) #-msse4.2 -msse4 -msse2 -march=native $(DFLAGS)
LIBS    = -lblas -llapack

