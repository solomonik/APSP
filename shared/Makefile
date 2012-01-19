include ../config.mk

all: util.o

unit_util.o: unit_util.h unit_util.cxx
	$(CXX) -c unit_util.cxx -o unit_util.o $(CFLAGS) $(DFLAGS) $(UFLAGS) $(AFLAGS)

util.o: util.h util.cxx
	$(CXX) -c util.cxx -o util.o $(CFLAGS) $(DFLAGS) $(UFLAGS) $(AFLAGS)

comm.o: comm.h comm.cxx
	$(CXX) -c comm.cxx -o comm.o $(CFLAGS) $(DFLAGS) $(UFLAGS) $(AFLAGS)

clean: 
	rm -f *.o


