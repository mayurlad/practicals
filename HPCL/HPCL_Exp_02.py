'''
Madhur Jaripatke
Roll No. 50
BE A Computer
RMDSSOE, Warje, Pune

Problem Statement: Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use 
existing algorithms and measure the performance of sequential and parallel algorithms.
'''

import random
import time
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np

def sequential_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def parallel_bubble_sort(arr):
    n = len(arr)
    swapped = True
    while swapped:
        swapped = False
        
        # Parallel processing for even indices
        with Pool() as pool:
            chunks = [(arr[i:i+2]) for i in range(0, n-1, 2)]
            results = pool.map(sort_pair, chunks)
            
            for i, pair in enumerate(results):
                if len(pair) == 2:
                    if arr[i*2:i*2+2] != pair:
                        arr[i*2:i*2+2] = pair
                        swapped = True
        
        # Parallel processing for odd indices
        with Pool() as pool:
            chunks = [(arr[i:i+2]) for i in range(1, n-1, 2)]
            results = pool.map(sort_pair, chunks)
            
            for i, pair in enumerate(results):
                if len(pair) == 2:
                    if arr[i*2+1:i*2+3] != pair:
                        arr[i*2+1:i*2+3] = pair
                        swapped = True
    
    return arr

def sort_pair(pair):
    if len(pair) == 2 and pair[0] > pair[1]:
        pair[0], pair[1] = pair[1], pair[0]
    return pair

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def sequential_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])
    
    return merge(left, right)

def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    
    with Pool(2) as pool:
        left, right = pool.map(parallel_merge_sort, [arr[:mid], arr[mid:]])
    
    return merge(left, right)

def main():
    SIZE = 50000
    arr = [random.randint(1, 10000) for _ in range(SIZE)]
    
    # Test Bubble Sort
    arr_copy = arr.copy()
    start = time.time()
    sequential_bubble_sort(arr_copy)
    end = time.time()
    print(f"Sequential Bubble Sort Time: {(end - start)*1000:.2f} ms")
    
    arr_copy = arr.copy()
    start = time.time()
    parallel_bubble_sort(arr_copy)
    end = time.time()
    print(f"Parallel Bubble Sort Time: {(end - start)*1000:.2f} ms")
    
    # Test Merge Sort
    arr_copy = arr.copy()
    start = time.time()
    sequential_merge_sort(arr_copy)
    end = time.time()
    print(f"Sequential Merge Sort Time: {(end - start)*1000:.2f} ms")
    
    arr_copy = arr.copy()
    start = time.time()
    parallel_merge_sort(arr_copy)
    end = time.time()
    print(f"Parallel Merge Sort Time: {(end - start)*1000:.2f} ms")

if __name__ == "__main__":
    main()
