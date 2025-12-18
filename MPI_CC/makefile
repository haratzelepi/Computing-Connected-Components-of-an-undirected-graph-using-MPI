SRC =  colouringCC_MPI.c mmio.c converter.c
TARGET = cc_mpi

compile:
	mpicc -O3 -fopenmp colouringCC_MPI.c mmio.c converter.c -o cc_mpi

run: compile
	@echo "===== MPI ====="
	mpirun -np 4 ./cc_mpi  ~/Desktop/com-Orkut.mtx

clean:
	rm -f cc_mpi
	