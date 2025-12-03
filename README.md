# MPI Parallel Matrix Operations

A C-based parallel computing application using **MPI (Message Passing Interface)** to perform various linear algebra operations on matrices and vectors.

## 📌 Overview

[cite_start]This program implements a **Master/Worker** model where four distinct mathematical operations are performed in parallel across $p$ processors[cite: 6, 27]. [cite_start]The user interacts with a menu-driven interface controlled by the Master node (Rank 0) to select operations and define matrix dimensions[cite: 17, 18].

## ⚙️ Features & Operations

[cite_start]The application supports the following operations on matrices $A(1 \times N)$, $B(N \times 1)$, $C(N \times N)$, and $D(N \times N)$[cite: 6]:

1.  **Operation I: Matrix Addition ($C + D$)**
    * [cite_start]Calculates the sum of two 2D matrices[cite: 7].
    * [cite_start]**Constraint:** $N$ must be a multiple of $p$ ($N \% p = 0$)[cite: 19].

2.  **Operation II: Matrix-Vector Multiplication ($C \cdot B$)**
    * [cite_start]Multiplies a 2D matrix by a column vector[cite: 8].
    * [cite_start]**Constraint:** $N$ must be a multiple of $p$ ($N \% p = 0$)[cite: 19].

3.  **Operation III: Dot Product ($A \cdot B$)**
    * [cite_start]Calculates the inner product of a row vector and a column vector[cite: 9].
    * [cite_start]**Constraint:** $N$ must be a multiple of $p$ ($N \% p = 0$)[cite: 19].

4.  **Operation IV: Matrix Multiplication (Ring Topology) ($C \cdot D$)**
    * [cite_start]Multiplies two 2D matrices using a **Ring Topology** algorithm[cite: 10, 11].
    * Processors are virtually organized in a ring. [cite_start]Rows of matrix $D$ are shifted cyclically between processors in $p$ steps[cite: 14].
    * [cite_start]**Constraint:** $N$ must equal $p$ ($N = p$) for this specific implementation[cite: 19].

## 🛠️ Implementation Details

* **Communication:**
    * [cite_start]**Collective Communication:** Uses `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, and `MPI_Reduce` for data distribution and result collection[cite: 15].
    * [cite_start]**Point-to-Point Communication:** Uses `MPI_Sendrecv` specifically for the Ring Topology algorithm in Operation IV[cite: 14, 15].
* [cite_start]**Memory Management:** * The Master node (Rank 0) allocates global matrices and handles I/O[cite: 18].
    * [cite_start]All processors manage their own local memory to prevent leaks, freeing resources at the end of each menu iteration[cite: 189].

## 🚀 How to Compile and Run

### Prerequisites
* GCC Compiler
* MPI implementation (e.g., MPICH or OpenMPI)

### Compilation
Compile the code using `mpicc`:

```bash
mpicc main.c -o mpi_matrix_ops
