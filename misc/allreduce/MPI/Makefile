CC=mpicc
CUDA=/opt/CUDA/7.5.18/cudatoolkit
ODAJIMA=../../odajima
FUJITA=../../fujita
CFLAGS=-Wall -O3 -std=c99 -I$(CUDA)/include -I$(ODAJIMA)/include -I$(FUJITA)/include -I../
CLINKS=-L$(FUJITA)/lib -ltca -L$(ODAJIMA)/lib -ltcacomm -L$(CUDA)/lib64 -lcudart

all: mpi_allreduce.out
mpi_allreduce.out: mpi_allreduce.c ../common.h
	$(CC) $(CFLAGS) -o mpi_allreduce.out mpi_allreduce.c $(CLINKS)

clean:
	rm -f *.o *.out


