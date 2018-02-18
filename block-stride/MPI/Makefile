CC=mpicc
CUDA=/opt/CUDA/7.5.18/cudatoolkit
ODAJIMA=../../odajima
FUJITA=../../fujita
CFLAGS=-Wall -O3 -std=c99 -I$(CUDA)/include -I$(ODAJIMA)/include -I$(FUJITA)/include -I../
CLINKS=-L$(FUJITA)/lib -ltca -L$(ODAJIMA)/lib -ltcacomm -L$(CUDA)/lib64 -lcudart

all: mpi-block-stride.out
mpi-block-stride.out: pack_unpack.o mpi-block-stride.o
	$(CC) $(CFLAGS) -o mpi-block-stride.out pack_unpack.o mpi-block-stride.o $(CLINKS)

mpi-block-stride.o: mpi-block-stride.c
	$(CC) $(CFLAGS) -c mpi-block-stride.c
	
pack_unpack.o: pack_unpack.cu
	nvcc -O3 pack_unpack.cu -o pack_unpack.o -c

clean:
	rm -f *.o *.out


