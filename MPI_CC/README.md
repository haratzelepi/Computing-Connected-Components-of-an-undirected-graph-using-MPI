# Hybrid MPI + OpenMP Connected Components

This project implements a hybrid MPI + OpenMP version of the label-propagation algorithm for computing
the connected components of large undirected graphs. The implementation targets distributed-memory
systems and is designed to efficiently process large sparse graphs using a combination of MPI for
inter-node parallelism and OpenMP for intra-node parallelism.

## Algorithm Overview

The algorithm is based on an iterative label-propagation (coloring) approach. Initially, each vertex
is assigned a unique label equal to its index. During each iteration, vertices update their labels
by selecting the minimum label among themselves and their neighbors. This process is repeated until
no label changes occur, indicating convergence.

The graph is stored in Compressed Sparse Row (CSR) format to enable efficient adjacency traversal.
The vertex set is statically partitioned across MPI processes, and OpenMP threads are used within
each process to parallelize local label updates.

Global convergence is ensured using MPI collective operations:
- An `MPI_Allreduce` with a logical OR checks whether any label updates occurred.
- An `MPI_Allreduce` with a minimum operation guarantees a consistent global view of labels across
  all MPI processes.

## Parallelization Strategy

- **MPI** is used to distribute the vertex set across multiple processes.
- **OpenMP** is employed within each MPI process to parallelize the label updates over local vertices.
- The input graph is read only by MPI rank 0 and then broadcast to all processes to avoid parallel
  file I/O overhead.

## Input Format

Graphs are provided in Matrix Market (`.mtx`) format. The input matrices represent undirected graphs
and are converted internally from COO to CSR format. Only the upper triangular part of the matrix
is stored to reduce memory usage.

## Datasets Used

The implementation was evaluated using large real-world graphs, including:
- **MAWI** (network traffic graph, many connected components)
- **GenBank** (biological sequence graph)
- **Friendster** (large-scale social network graph, single giant component)

These datasets allow studying both scalability and the impact of graph structure on performance.

## Limitations

- Does not support directed graphs

## Compilation

## Building and Running the Code

The project supports two execution modes:
1. Local execution on a single machine
2. Distributed execution on the Aristotelis HPC system

---

## 1. Local Execution (Single machine)

For local testing and debugging, the code can be compiled and executed on a single machine using MPI
and OpenMP. A Makefile is provided to simplify the build process.

### Requirements
- MPI implementation (e.g. OpenMPI)
- GCC with OpenMP support

### Compilation

```bash
# To build all versions
make compile

# To run all versions automatically (each prints its method name):
make run

# To clean all executables:
make clean
```
## 2. Distributed execution on the Aristotelis 

```bash
#To create the .sh file 
\#SBATCH --job-name=mpi_cc
\#SBATCH --partition=rome
\#SBATCH --nodes=2
\#SBATCH --ntasks-per-node=4
\#SBATCH --cpus-per-task=4
\#SBATCH --time=00:20:00
\#SBATCH --mem=64G

module load gcc/13.2.0
module load openmpi/4.1.6

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd /path/to/MPI_CC

mpicc -O3 -fopenmp colouringCC_MPI_openmp.c converter.c mmio.c -o cc_mpi
srun ./cc_mpi input_graph.mtx

#To submit the job 
sbatch mpi_cc.sh
```