include ../config.mk

all: test
test: fmm_test

fmm_test: fmm_naive.o fmm_test.cxx
	$(CC) $(CFLAGS) fmm_naive.o fmm_test.cxx -o fmm_test $(LIBS)

fmm_naive.o: fmm_naive.cxx fmm.h
	$(CC) $(CFLAGS) -c fmm_naive.cxx

clean: 
	rm -f *.o fmm_test

