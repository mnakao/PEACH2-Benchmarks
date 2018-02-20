CC=mpicc
CUDA=/opt/CUDA/7.5.18/cudatoolkit
ODAJIMA=../../odajima
FUJITA=../../fujita
CFLAGS=-Wall -O3 -std=c99 -I$(CUDA)/include -I$(ODAJIMA)/include -I$(FUJITA)/include -I../
CLINKS=-L$(FUJITA)/lib -ltca -L$(ODAJIMA)/lib -ltcacomm -L$(CUDA)/lib64 -lcudart

all: tca_allgather.out
tca_allgather.out: tca_allgather.c ../common.h
	$(CC) $(CFLAGS) -o tca_allgather.out tca_allgather.c $(CLINKS)

clean:
	rm -f *.o ping.out


