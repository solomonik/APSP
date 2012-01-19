# Makefile for Cilk++ example.  See source code for details.
include ../config.mk

BOOST = /home/aydin/localinstall/boost_new
INCADD = -I/usr/local/include/cilk/ -I$(BOOST)/include/boost-1_35/
CILKPP	= cilk++
LIBARG	=
# uncomment the following line to build with Cilkscreen metadata.
# LIBARG	= -fcilkscreen -g
TARGET	= apsp_fmm
SRC	= $(addsuffix .cilk,$(TARGET))

GCCOPT = -g -O2
INTELOPT = -g -O2
DEB = -g -pg

all: test_dcapsp 


test_dcapsp: util.o dcapsp.o fmm_naive.o ctr_2d_sqr_bcast.o ctr_simple.o \
	     test_dcapsp.cxx
	$(CC) $(CFLAGS) util.o dcapsp.o fmm_naive.o ctr_2d_sqr_bcast.o ctr_simple.o \
			test_dcapsp.cxx -o test_dcapsp $(LIBS)

dcapsp.o: dcapsp.h dcapsp.cxx
	$(CC) $(CFLAGS) -c dcapsp.cxx;

fmm_naive.o: 
	$(CC) $(CFLAGS) -c ../fmm/fmm_naive.cxx; cp ../fmm/fmm_naive.o .

ctr_2d_sqr_bcast.o: ../ctr_comm/ctr_2d_sqr_bcast.cxx ../ctr_comm/ctr_comm.h
	cd ../ctr_comm/; make ctr_2d_sqr_bcast.o; cd ../apsp; cp ../ctr_comm/ctr_2d_sqr_bcast.o .

ctr_simple.o: ../ctr_comm/ctr_simple.cxx ../ctr_comm/ctr_comm.h
	cd ../ctr_comm/; make ctr_simple.o; cd ../apsp; cp ../ctr_comm/ctr_simple.o .

util.o: ../shared/util.cxx ../shared/util.h
	cd ../shared/; make util.o; cd ../apsp; cp ../shared/util.o .

serial: apsp_cpp.cpp
	g++ $(INCADD) $(GCCOPT) -o apspserial apsp_cpp.cpp -lpthread


#$(TARGET): $(SRC)
#	$(CILKPP) $(INCADD) $(SRC) $(LIBARG) -o $@

apsp_fmm: apsp_fmm.cpp fmm_naive.o
	g++ $(INCADD) $(GCCOPT) ../fmm/fmm_naive.o -o apspfmm apsp_fmm.cpp -lpthread


clean:
	rm -f $(TARGET)
	rm -f apspserial apspfmm
