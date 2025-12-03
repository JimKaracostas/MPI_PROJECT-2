# MPI Parallel Matrix Operations

A C-based parallel computing application using **MPI (Message Passing Interface)** to perform various linear algebra operations on matrices and vectors.

## 📌 Overview

This program implements a **Master/Worker** model where four distinct mathematical operations are performed in parallel across $p$ processors. The user interacts with a menu-driven interface controlled by the Master node (Rank 0) to select operations and define matrix dimensions.

## ⚙️ Features & Operations

The application supports the following operations on matrices $A(1 \times N)$, $B(N \times 1)$, $C(N \times N)$, and $D(N \times N)$:

1.  **Operation I: Matrix Addition ($C + D$)**
    * Calculates the sum of two 2D matrices.
    * **Constraint:** $N$ must be a multiple of $p$ ($N \% p = 0$).

2.  **Operation II: Matrix-Vector Multiplication ($C \cdot B$)**
    * Multiplies a 2D matrix by a column vector[cite: 8].
    * **Constraint:** $N$ must be a multiple of $p$ ($N \% p = 0$).

3.  **Operation III: Dot Product ($A \cdot B$)**
    * Calculates the inner product of a row vector and a column vector.
    * **Constraint:** $N$ must be a multiple of $p$ ($N \% p = 0$).

4.  **Operation IV: Matrix Multiplication (Ring Topology) ($C \cdot D$)**
    * Multiplies two 2D matrices using a **Ring Topology** algorithm.
    * Processors are virtually organized in a ring. Rows of matrix $D$ are shifted cyclically between processors in $p$ steps.
    * **Constraint:** $N$ must equal $p$ ($N = p$) for this specific implementation.

## 🛠️ Implementation Details

* **Communication:**
    * **Collective Communication:** Uses `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, and `MPI_Reduce` for data distribution and result collection.
    * **Point-to-Point Communication:** Uses `MPI_Sendrecv` specifically for the Ring Topology algorithm in Operation IV.
* **Memory Management:** * The Master node (Rank 0) allocates global matrices and handles I/O.
    * All processors manage their own local memory to prevent leaks, freeing resources at the end of each menu iteration.

## 🚀 How to Compile and Run

### Prerequisites
* GCC Compiler
* MPI implementation (e.g., MPICH or OpenMPI)

### Compilation
Compile the code using `mpicc`:

```bash
mpicc main.c -o mpi_matrix_ops
