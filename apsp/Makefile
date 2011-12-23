# Makefile for Cilk++ example.  See source code for details.

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

all: apsp_fmm 

serial: apsp_cpp.cpp
	g++ $(INCADD) $(GCCOPT) -o apspserial apsp_cpp.cpp -lpthread


#$(TARGET): $(SRC)
#	$(CILKPP) $(INCADD) $(SRC) $(LIBARG) -o $@

apsp_fmm: apsp_fmm.cpp fmm_naive.o
	g++ $(INCADD) $(GCCOPT) ../fmm/fmm_naive.o -o apspfmm apsp_fmm.cpp -lpthread

fmm_naive.o:
	cd ../fmm/; make fmm_naive.o; cd ../apsp/;

clean:
	rm -f $(TARGET)
	rm -f apspserial apspfmm
