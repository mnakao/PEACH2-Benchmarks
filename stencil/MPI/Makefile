CC=mpicc
CUDA=/opt/CUDA/7.5.18/cudatoolkit
ODAJIMA=../../misc/odajima
FUJITA=../../misc/fujita
CFLAGS=-Wall -O3 -std=c99 -I$(CUDA)/include -I$(ODAJIMA)/include -I$(FUJITA)/include -I../
CLINKS=-L$(FUJITA)/lib -ltca -L$(ODAJIMA)/lib -ltcacomm -L$(CUDA)/lib64 -lcudart

all: mpi-stencil.out
mpi-stencil.out: pack_unpack.o mpi-stencil.o ../common.h
	$(CC) $(CFLAGS) -o mpi-stencil.out pack_unpack.o mpi-stencil.o $(CLINKS)

mpi-stencil.o: mpi-stencil.c ../common.h
	$(CC) $(CFLAGS) -c mpi-stencil.c

pack_unpack.o: pack_unpack.cu ../common.h
	nvcc -O3 pack_unpack.cu -o pack_unpack.o -c

clean:
	rm -f *.o *.out


