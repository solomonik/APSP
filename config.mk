CC	= g++
#CFLAGS  = -g -O0 $(DFLAGS)
CFLAGS = -O0 -g -msse4.2 -msse4 -msse2 -march=native $(DFLAGS)
LIBS    = -lblas -llapack

