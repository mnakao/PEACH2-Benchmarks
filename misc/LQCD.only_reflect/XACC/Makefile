PACKAGE	= wilson_solver
SRCS	= main_reflect_ha_mpi.c
HEADS	= lattice_reflect_ha_mpi.h
OBJS	= $(SRCS:.c=.o) pack_unpack.o
FILES	= Makefile $(HEADS) $(SRCS)

CC=xmpcc 
CUDA=/opt/CUDA/7.5.18/cudatoolkit/
CCFLAGS=-xacc -O3 --device=Kepler -mcmodel=medium -I$(CUDA)/include -D_PROF
#CCFLAGS=-xacc -O3 --device=Kepler -mcmodel=medium -I$(CUDA)/include
LDFLAGS=$(CCFLAGS) -L$(CUDA)/lib64 -lcublas
MPI=/opt/MPI/MVAPICH2-GDR/2.2/Intel/CUDA/7.5/

all: $(PACKAGE)
$(PACKAGE): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(PROCS) $(LDLIBS)

$(OBJS): $(HEADS) Makefile

.c.o: $(HEADS)
	$(CC) $(CCFLAGS) $(CPPFLAGS) -c $< -o $@

pack_unpack.o: pack_unpack.cu $(HEADS)
	nvcc -arch=sm_35 -O3 -c pack_unpack.cu -I$(MPI)/include

clean:
	$(RM) $(PACKAGE) $(OBJS)
