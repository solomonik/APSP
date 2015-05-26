include ../config.mk

CTR_COMM_LOC_OBJS = ctr_simple.o ctr_2d_sqr_bcast.o ctr_2d_rect_bcast.o 

all: objs

objs: $(CTR_COMM_LOC_OBJS)

unit_ctr_comm: unit_ctr_comm.cxx shared $(CTR_COMM_LOC_OBJS)
	$(CXX) $(SHARED_OBJS) $(CTR_COMM_LOC_OBJS)	\
	-o unit_ctr_comm unit_ctr_comm.cxx \
	$(UFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS) $(BLAS) $(LAPACK) $(BLAS) $(OPENMP) -lm

bench_ctr_comm: bench_ctr_comm.cxx shared $(CTR_COMM_LOC_OBJS)
	$(CXX) $(SHARED_OBJS) $(CTR_COMM_LOC_OBJS)	\
	-o bench_ctr_comm bench_ctr_comm.cxx \
	$(UFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS) $(BLAS) $(LAPACK) $(BLAS) $(OPENMP) -lm

ctr_simple.o: ctr_simple.cxx ctr_comm.h
	$(CXX) -c ctr_simple.cxx -o ctr_simple.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

ctr_1d_sqr_bcast.o: ctr_1d_sqr_bcast.cxx ctr_comm.h
	$(CXX) -c ctr_1d_sqr_bcast.cxx -o ctr_1d_sqr_bcast.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

ctr_2d_sqr_bcast.o: ctr_2d_sqr_bcast.cxx ctr_comm.h
	$(CXX) -c ctr_2d_sqr_bcast.cxx -o ctr_2d_sqr_bcast.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

ctr_2d_rect_bcast.o: ctr_2d_rect_bcast.cxx ctr_comm.h
	$(CXX) -c ctr_2d_rect_bcast.cxx -o ctr_2d_rect_bcast.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

ctr_tsr.o: ctr_tsr.cxx ctr_comm.h ctr_tsr.h
	$(CXX) -c ctr_tsr.cxx -o ctr_tsr.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS) $(OPENMP)

strp_tsr.o: strp_tsr.cxx strp_tsr.h
	$(CXX) -c strp_tsr.cxx -o strp_tsr.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

sum_tsr.o: sum_tsr.cxx sum_tsr.h
	$(CXX) -c sum_tsr.cxx -o sum_tsr.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

scale_tsr.o: scale_tsr.cxx scale_tsr.h
	$(CXX) -c scale_tsr.cxx -o scale_tsr.o \
	$(CFLAGS) $(AFLAGS) $(DFLAGS) $(UFLAGS)

comm_util.o: 
	cd ../shared; make comm_util.o;

comm.o: 
	cd ../shared; make comm.o;

shared:
	cd ../shared; make;

clean:
	rm -f *.o unit_ctr_comm 
