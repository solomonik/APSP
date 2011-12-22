# Makefile for Cilk++ example.  See source code for details.

BOOST = /home/aydin/localinstall/boost_new
INCADD = -I/usr/local/include/cilk/ -I$(BOOST)/include/boost-1_35/
CILKPP	= cilk++
LIBARG	=
# uncomment the following line to build with Cilkscreen metadata.
# LIBARG	= -fcilkscreen -g
TARGET	= apsp
SRC	= $(addsuffix .cilk,$(TARGET))

GCCOPT = -O2 
INTELOPT = -O2
DEB = -g -pg

serial: apsp_cpp.cpp
	g++ $(INCADD) $(GCCOPT) -o apspserial apsp_cpp.cpp -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(CILKPP) $(INCADD) $(SRC) $(LIBARG) -o $@

clean:
	rm -f $(TARGET)
	rm -f apspserial
