CC=mpicc
CFLAGS=-Wall -O3 -std=c99 

all: ping.out
ping.out: mpi-pingpong.c ../common.h
	$(CC) $(CFLAGS) -o ping.out mpi-pingpong.c

clean:
	rm -f *.o ping.out


