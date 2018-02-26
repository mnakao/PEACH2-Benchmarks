CC      = nvcc
MPI     = /opt/MPI/MVAPICH2-GDR/2.2/Intel/CUDA/7.5/
CFLAGS  = -arch=sm_35 -O3 -I$(MPI)/include -D_PROF --compiler-options "-mcmodel=medium"
CLIB    = -L$(MPI)/lib64 -lmpich
HEADERS = lattice.h
OBJECTS = main.o solver_CG.o fopr_wilson_alt.o field.o

all: wilson_solver
wilson_solver: $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(PROCS) $(CLIB)

main.o: main.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $<

solver_CG.o: solver_CG.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $<

fopr_wilson_alt.o: fopr_wilson_alt.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $<

field.o: field.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o 1 2 4 8 16 32 64 wilson_solver

