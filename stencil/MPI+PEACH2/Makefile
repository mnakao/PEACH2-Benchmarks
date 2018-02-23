CC=mpicc
CUDA=/opt/CUDA/7.5.18/cudatoolkit
ODAJIMA=../../misc/odajima
FUJITA=../../misc/fujita
CFLAGS=-Wall -O3 -std=c99 -I$(CUDA)/include -I$(ODAJIMA)/include -I$(FUJITA)/include -I../
CLINKS=-L$(FUJITA)/lib -ltca -L$(ODAJIMA)/lib -ltcacomm -L$(CUDA)/lib64 -lcudart

all: mpi-peach2-stencil.out
mpi-peach2-stencil.out: mpi-peach2-stencil.c ../common.h
	$(CC) $(CFLAGS) -o mpi-peach2-stencil.out mpi-peach2-stencil.c $(CLINKS)

clean:
	rm -f *.o *.out


