#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "mmio.h"
#include "converter.h"

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *str = argv[1];
    int *indexes = NULL;
    int *indices = NULL;
    int N;

    if (rank == 0) {
        N = cooReader(str, &indexes, &indices) - 1;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        indexes = malloc((N + 1) * sizeof(int));

    MPI_Bcast(indexes, N + 1, MPI_INT, 0, MPI_COMM_WORLD);

    int nnz = indexes[N];

    if (rank != 0)
        indices = malloc(nnz * sizeof(int));

    MPI_Bcast(indices, nnz, MPI_INT, 0, MPI_COMM_WORLD);

    int *colors = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        colors[i] = i;


    int local_start = (N * rank) / size;
    int local_end   = (N * (rank + 1)) / size;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int global_active = 1;

    while (global_active) {
        int local_active = 0;

        #pragma omp parallel for schedule(static) reduction(||:local_active)
        for (int i = local_start; i < local_end; i++) {
            for (int j = indexes[i]; j < indexes[i+1]; j++) {
                int v = indices[j];
                int cmin = (colors[i] < colors[v]) ? colors[i] : colors[v];

                if (colors[i] != cmin) {
                    colors[i] = cmin;
                    local_active = 1;
                }
                if (colors[v] != cmin) {
                    colors[v] = cmin;
                    local_active = 1;
                }
            }
        }

        MPI_Allreduce(&local_active, &global_active,
                      1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, colors,
                      N, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        int *flags = calloc(N, sizeof(int));
        int cc = 0;
        for (int i = 0; i < N; i++) {
            if (!flags[colors[i]]) {
                flags[colors[i]] = 1;
                cc++;
            }
        }
        printf("Connected Components = %d\n", cc);
        printf("Hybrid MPI+OMP time = %f sec\n", t1 - t0);
        free(flags);
    }

    free(colors);
    free(indexes);
    free(indices);

    MPI_Finalize();
    return 0;
}
