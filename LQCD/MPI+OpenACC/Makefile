PACKAGE	= wilson_solver
SRCS	= main.c solver_CG.c fopr_wilson_alt.c field.c
HEADS	= lattice.h
OBJS	= $(SRCS:.c=.o)
FILES	= Makefile $(HEADS) $(SRCS)

CC=ompcc 
CCFLAGS=-acc -O3 --device=Kepler -mcmodel=medium -D_PROF
LDFLAGS=$(CCFLAGS)

all: $(PACKAGE)
$(PACKAGE): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(PROCS) $(LDLIBS)

$(OBJS): $(HEADS) Makefile

.c.o: lattice.h
	$(CC) $(CCFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(PACKAGE) $(OBJS)
