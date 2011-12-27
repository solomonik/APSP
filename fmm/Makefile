include ../config.mk

all: test
test: fmm_test

fmm_test: fmm_naive.o fmm_opt.o fmm_test.cxx
	$(CC) $(CFLAGS) fmm_naive.o fmm_opt.o fmm_test.cxx -o fmm_test $(LIBS)

fmm_naive.o: fmm_naive.cxx fmm.h
	$(CC) $(CFLAGS) -c fmm_naive.cxx

fmm_opt.o: fmm_opt.cxx fmm.h
	$(CC) $(CFLAGS) -c fmm_opt.cxx

clean: 
	rm -f *.o fmm_test

