#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

/*
 * PARALLEL COMPUTING LAB - ASSIGNMENT II
 * Implementation of matrix operations using MPI.
 *
 * Requirements:
 * I.   C + D (Matrix Addition)
 * II.  C * B (Matrix * Column Vector)
 * III. A * B (Row Vector * Column Vector = Dot Product)
 * IV.  C * D (Matrix Multiplication using Ring Topology)
 */

// Helper function to print matrices (Rank 0 only)
void print_matrix(double *mat, int rows, int cols, const char *name) {
    printf("%s \n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Initialize matrices with random values
void init_matrix(double *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (double)(rand() % 10);
    }
}

int main(int argc, char *argv[]) {
    int p, my_rank;
    int N;
    int choice;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    // Pointers for the matrices
    double *A = NULL, *B = NULL, *C = NULL, *D = NULL;

    // Seed for random numbers
    srand(time(NULL) + my_rank);

    do {
        choice = 0;
        N = 0;

        if (my_rank == 0) {
            printf("\n========== MENU ==========\n");
            printf("1. Operation I:   C + D (Sum of two 2D matrices)\n");
            printf("2. Operation II:  C * B (Multiplication of 2D matrix with column-vector)\n");
            printf("3. Operation III: A * B (Multiplication of row-vector with column-vector)\n");
            printf("4. Operation IV:  C * D (Multiplication of 2D matrices using Ring algorithm)\n");
            printf("5. Exit\n");
            printf("Selection: ");
            scanf("%d", &choice);

            if (choice >= 1 && choice <= 4) {
                printf("Enter dimension N: ");
                scanf("%d", &N);

                // Constraints Check
                // For IV: N=p
                if (choice == 4 && N != p) {
                    printf("WARNING: For Operation IV, N = p is required.\n");
                    printf("N has been automatically set to %d.\n", p);
                    N = p;
                } 
                // For I, II, III: N%p=0
                else if (choice != 4 && N % p != 0) {
                    printf("WARNING: N is required to be a multiple of p.\n");
                    choice = 5; // Force exit
                }
            }
        }

        // Broadcast choice and N to all processes
        MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (choice == 5) break;

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Initialize matrices on rank 0
        if (my_rank == 0) {
            A = (double*) malloc(N * sizeof(double));      // 1 x N
            B = (double*) malloc(N * sizeof(double));      // N x 1
            C = (double*) malloc(N * N * sizeof(double)); // N x N
            D = (double*) malloc(N * N * sizeof(double)); // N x N

            // Initialize with random values
            init_matrix(A, N);
            init_matrix(B, N);
            init_matrix(C, N * N);
            init_matrix(D, N * N);

            printf("\nMatrices initialized with random values (0-9)\n");
        }

        if (choice == 1) {
            // I. C(NxN) + D(NxN) 
            int l_rows = N / p;
            int elements = l_rows * N;

            // Local matrices
            double *l_c = (double*) malloc(elements * sizeof(double));
            double *l_d = (double*) malloc(elements * sizeof(double));
            double *l_res = (double*) malloc(elements * sizeof(double));

            // Distribution (Scatter)
            MPI_Scatter(C, elements, MPI_DOUBLE, l_c, elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(D, elements, MPI_DOUBLE, l_d, elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Calculation
            for (int i = 0; i < elements; i++) {
                l_res[i] = l_c[i] + l_d[i];
            }

            double *Result = NULL;
            if (my_rank == 0)
                Result = (double*) malloc(N * N * sizeof(double));

            // Collection of results (Gather)
            MPI_Gather(l_res, elements, MPI_DOUBLE, Result, elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (my_rank == 0) {
                print_matrix(C, N, N, "C:");
                print_matrix(D, N, N, "D:");
                print_matrix(Result, N, N, "Result I (C + D):");
                free(Result);
            }
            
            free(l_c);
            free(l_d);
            free(l_res);

        } else if (choice == 2) {
            // II. C(NxN) * B(Nx1)
            int l_rows = N / p;

            double *l_c = (double*) malloc(l_rows * N * sizeof(double));
            double *l_b = (double*) malloc(N * sizeof(double));
            double *l_res = (double*) malloc(l_rows * sizeof(double));

            // Scatter rows of C
            MPI_Scatter(C, l_rows * N, MPI_DOUBLE, l_c, l_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Broadcast vector B to all processes
            if (my_rank == 0) {
                for (int i = 0; i < N; i++)
                    l_b[i] = B[i];
            }
            MPI_Bcast(l_b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Calculate local rows
            for (int i = 0; i < l_rows; i++) {
                l_res[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    l_res[i] += l_c[i * N + j] * l_b[j];
                }
            }
            
            double *Result = NULL;
            if (my_rank == 0) 
                Result = (double*) malloc(N * sizeof(double));

            // Collect results
            MPI_Gather(l_res, l_rows, MPI_DOUBLE, Result, l_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (my_rank == 0) {
                print_matrix(C, N, N, "C:");
                printf("B (column vector):\n");
                for (int i = 0; i < N; i++) 
                    printf("%.2f\n", B[i]);
                printf("\nResult II (C * B):\n");
                for (int i = 0; i < N; i++) 
                    printf("%.2f\n", Result[i]);
                printf("\n");
                free(Result);
            }

            free(l_c); 
            free(l_b);
            free(l_res);

        } else if (choice == 3) {
            // III. A(1xN) * B(Nx1) - Dot Product
            int size = N / p;

            double *l_a = (double*) malloc(size * sizeof(double));
            double *l_b = (double*) malloc(size * sizeof(double));
            
            // Scatter A and B
            MPI_Scatter(A, size, MPI_DOUBLE, l_a, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(B, size, MPI_DOUBLE, l_b, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Local calculation
            double l_dot = 0.0;
            for (int i = 0; i < size; i++) {
                l_dot += l_a[i] * l_b[i];
            }
            
            double dot = 0.0;
            
            // Sum all local products
            MPI_Reduce(&l_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (my_rank == 0) {
                printf("A (row vector): ");
                for (int i = 0; i < N; i++) 
                    printf("%.2f ", A[i]);
                printf("\nB (column vector): ");
                for (int i = 0; i < N; i++) 
                    printf("%.2f ", B[i]);
                printf("\n\nResult III (A * B): %.2f\n\n", dot);
            }

            free(l_a);
            free(l_b);

        } else if (choice == 4) {
            // IV. C(NxN) * D(NxN) with Ring Topology
            // N = p

            double *row_c = (double*) malloc(N * sizeof(double));
            double *row_d = (double*) malloc(N * sizeof(double));
            double *recv_d = (double*) malloc(N * sizeof(double));
            double *res = (double*) malloc(N * sizeof(double));

            // Initialize result
            for (int i = 0; i < N; i++)
                res[i] = 0.0;

            // Scatter rows
            MPI_Scatter(C, N, MPI_DOUBLE, row_c, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(D, N, MPI_DOUBLE, row_d, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Calculate next and previous in the ring
            int next = (my_rank + 1) % p;
            int prev = (my_rank - 1 + p) % p;

            // Which column of C we are using (initially my_rank)
            int current_col = my_rank;

            // p steps to pass all rows of D
            for (int step = 0; step < p; step++) {
                // Calculation: row_c[current_col] * row_d[:]
                double c_elem = row_c[current_col];
                for (int k = 0; k < N; k++) {
                    res[k] += c_elem * row_d[k];
                }

                // Send row_d to previous and receive from next
                MPI_Status status;
                MPI_Sendrecv(row_d, N, MPI_DOUBLE, prev, 0, 
                            recv_d, N, MPI_DOUBLE, next, 0, 
                            MPI_COMM_WORLD, &status);
                
                // Copy new row_d
                for (int i = 0; i < N; i++)
                    row_d[i] = recv_d[i];

                // Next column
                current_col = (current_col + 1) % p;
            }

            double *Result = NULL;
            if (my_rank == 0)
                Result = (double*) malloc(N * N * sizeof(double));

            // Collect results
            MPI_Gather(res, N, MPI_DOUBLE, Result, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (my_rank == 0) {
                print_matrix(C, N, N, "C:");
                print_matrix(D, N, N, "D:");
                print_matrix(Result, N, N, "Result IV (C * D):");
                free(Result);
            }

            free(row_c);
            free(row_d);
            free(recv_d);
            free(res);
        }

        // Free memory for input matrices
        if (my_rank == 0) {
            if (A) free(A);
            if (B) free(B);
            if (C) free(C);
            if (D) free(D);
            A = B = C = D = NULL;
        }

    } while (choice != 5);

    if (my_rank == 0) {
        printf("End of program.\n");
    }

    MPI_Finalize();
    return 0;
}
