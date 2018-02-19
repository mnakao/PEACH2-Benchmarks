CC=mpicc
CUDA=/opt/CUDA/7.5.18/cudatoolkit
ODAJIMA=../../odajima
FUJITA=../../fujita
CFLAGS=-Wall -O3 -std=c99 -I$(CUDA)/include -I$(ODAJIMA)/include -I$(FUJITA)/include -I../
CLINKS=-L$(FUJITA)/lib -ltca -L$(ODAJIMA)/lib -ltcacomm -L$(CUDA)/lib64 -lcudart

all: mpi_bcast.out
mpi_bcast.out: mpi_bcast.c ../common.h
	$(CC) $(CFLAGS) -o mpi_bcast.out mpi_bcast.c $(CLINKS)

clean:
	rm -f *.o *.out


