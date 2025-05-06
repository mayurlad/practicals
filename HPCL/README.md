# High Performance Computing Laboratory

This repository contains implementations of parallel computing algorithms using OpenMP, Python's multiprocessing, and CUDA. Each implementation demonstrates the performance benefits of parallel processing compared to sequential execution.

## Programs List and Instructions

### 1. Parallel Graph Search (BFS and DFS)

**Files**: 
- [`HPCL_Exp_01.cpp`](HPCL_Exp_01.cpp) (OpenMP implementation)
- [`HPCL_Exp_01.py`](HPCL_Exp_01.py) (Python multiprocessing implementation)

**Algorithms**:
- Breadth-First Search (BFS): Level-wise traversal of graph
- Depth-First Search (DFS): Branch-wise traversal of graph

**Parallelization Strategy**:
- BFS: Process nodes at each level in parallel
- DFS: Process independent branches in parallel
- Uses OpenMP parallel for and critical sections in C++
- Uses multiprocessing Pool in Python

**Compilation and Execution**:
```bash
# C++ version
g++ -fopenmp HPCL_Exp_01.cpp -o parallel_graph_search
./parallel_graph_search

# Python version
python HPCL_Exp_01.py
```

### 2. Parallel Sorting Algorithms

**Files**:
- [`HPCL_Exp_02.cpp`](HPCL_Exp_02.cpp) (OpenMP implementation)
- [`HPCL_Exp_02.py`](HPCL_Exp_02.py) (Python multiprocessing implementation)

**Algorithms**:
- Bubble Sort: O(n²) comparison-based sorting
- Merge Sort: O(n log n) divide-and-conquer sorting

**Parallelization Strategy**:
- Bubble Sort: Parallel comparison and swapping of adjacent elements
- Merge Sort: Parallel division and sorting of subarrays
- Performance improvement most noticeable for large arrays

**Compilation and Execution**:
```bash
# C++ version
g++ -fopenmp HPCL_Exp_02.cpp -o parallel_sorting
./parallel_sorting

# Python version
python HPCL_Exp_02.py
```

### 3. Parallel Reduction Operations

**Files**:
- [`HPCL_Exp_03.cpp`](HPCL_Exp_03.cpp) (OpenMP implementation)
- [`HPCL_Exp_03.py`](HPCL_Exp_03.py) (Python multiprocessing implementation)

**Operations**:
- Minimum value finding
- Maximum value finding
- Sum calculation
- Average calculation

**Parallelization Strategy**:
- Data divided into chunks processed in parallel
- Uses OpenMP reduction clauses in C++
- Uses multiprocessing Pool in Python
- Near-linear speedup with number of cores

**Compilation and Execution**:
```bash
# C++ version
g++ -fopenmp HPCL_Exp_03.cpp -o parallel_reduction
./parallel_reduction

# Python version
python HPCL_Exp_03.py
```

### 4. Parallel Vector and Matrix Operations

#### A. CUDA Implementation

##### Vector Addition
**File**: [`HPCL_Exp_04_vector_addition.cu`](HPCL_Exp_04_vector_addition.cu)

**Features**:
- Element-wise addition of two vectors
- Coalesced memory access pattern
- Block size optimization for GPU architecture
- Handles large vectors efficiently

**Compilation and Execution**:
```bash
nvcc HPCL_Exp_04_vector_addition.cu -o vector_addition_cuda
./vector_addition_cuda
```

##### Matrix Multiplication
**File**: [`HPCL_Exp_04_matrix_multiplication.cu`](HPCL_Exp_04_matrix_multiplication.cu)

**Features**:
- Dense matrix multiplication
- Optimized thread block size (32x32)
- Efficient memory access patterns
- Handles large matrices (default 1024x1024)

**Compilation and Execution**:
```bash
nvcc HPCL_Exp_04_matrix_multiplication.cu -o matrix_multiplication_cuda
./matrix_multiplication_cuda
```

#### B. OpenMP Implementation

##### Vector Addition
**File**: [`HPCL_Exp_04_vector_addition_omp.cpp`](HPCL_Exp_04_vector_addition_omp.cpp)

**Features**:
- Parallel vector addition using OpenMP
- Comparison of sequential vs parallel performance
- Memory-efficient implementation
- Automatic work distribution among threads
- Handles vectors of various sizes (1M to 50M elements)

**Compilation and Execution**:
```bash
g++ -fopenmp HPCL_Exp_04_vector_addition_omp.cpp -o vector_addition_omp
./vector_addition_omp
```

##### Matrix Multiplication
**File**: [`HPCL_Exp_04_matrix_multiplication.cpp`](HPCL_Exp_04_matrix_multiplication.cpp)

**Features**:
- Dense matrix multiplication using OpenMP
- Nested parallelism with collapse clause
- Efficient cache utilization
- Automatic workload distribution
- Supports various matrix sizes (100x100 to 1000x1000)

**Compilation and Execution**:
```bash
g++ -fopenmp HPCL_Exp_04_matrix_multiplication.cpp -o matrix_multiplication_omp
./matrix_multiplication_omp
```

## Requirements

### 1. For C++ Programs:
- GCC compiler 7.0 or later with OpenMP support
- C++11 or later
- OpenMP 4.5 or later
- Compilation flag: `-fopenmp`

### 2. For Python Programs:
- Python 3.6 or later
- NumPy 1.19 or later
- multiprocessing module (built into Python)
- Install dependencies:
  ```bash
  pip install numpy
  ```

### 3. For CUDA Programs:
- NVIDIA CUDA Toolkit 10.0 or later
- NVIDIA GPU with compute capability 3.0 or higher
- CUDA-capable driver (460.x or later recommended)
- Check CUDA installation:
  ```bash
  nvcc --version
  nvidia-smi
  ```

## Performance Notes

1. **OpenMP Programs**:
   - Default to 4 threads (can be modified via `OMP_NUM_THREADS`)
   - Best performance on multi-core CPUs
   - Thread affinity matters for optimal performance

2. **Python Parallel Programs**:
   - Utilize available CPU cores automatically
   - Limited by Global Interpreter Lock (GIL) for some operations
   - Best for CPU-bound tasks

3. **CUDA Programs**:
   - Optimized for GPUs with compute capability 3.0 or higher
   - Performance scales with GPU capabilities
   - Memory transfers can be a bottleneck

## Best Practices

1. **Data Size Considerations**:
   - Small datasets: Sequential might be faster due to overhead
   - Large datasets: Parallel versions show significant speedup
   - CUDA: Best for very large datasets (millions of elements)

2. **Memory Management**:
   - OpenMP: Be careful with shared vs private variables
   - CUDA: Optimize memory transfers between host and device
   - Python: Consider using numpy arrays for better performance

3. **Thread Safety**:
   - Use critical sections only when necessary
   - Minimize thread contention
   - Be aware of race conditions

## Troubleshooting

1. **OpenMP Issues**:
   - Check if OpenMP is enabled: `echo |cpp -fopenmp -dM |grep -i open`
   - Set number of threads: `export OMP_NUM_THREADS=4`

2. **CUDA Issues**:
   - Verify GPU availability: `nvidia-smi`
   - Check CUDA path: `echo $PATH | grep cuda`
   - Ensure correct compute capability

## Dataset Description

### Vector Operations Dataset
The vector operations are tested on randomly generated datasets with the following characteristics:
- Small vectors: 1 million elements (1M)
- Medium vectors: 10 million elements (10M)
- Large vectors: 50 million elements (50M)
- Data type: Double-precision floating-point numbers
- Value range: Random values between 0 and 99

### Matrix Operations Dataset
The matrix multiplication operations are evaluated using randomly generated square matrices:
- Small matrices: 100 × 100 elements
- Medium matrices: 500 × 500 elements
- Large matrices: 1000 × 1000 elements
- Data type: Double-precision floating-point numbers
- Value range: Random values between 0 and 99

### Performance Evaluation
- Each operation is performed multiple times to ensure consistent timing
- Both sequential and parallel implementations are tested on identical datasets
- Results include execution time and speedup metrics
- Correctness verification is performed by comparing sequential and parallel outputs
