# Plan to implement baseline algorithms for graph datasets.

## Plan
### Task 1: implement function to check (d,l)-disjunctiveness of a matrix
- Assume that we already have a function `is_disjunctive(matrix: List[List[int]], d: int, l: int) -> bool:` that checks whether a given matrix is (d,l)-disjunctive or not. This function will be used to verify the correctness of the sensor sets obtained from the baseline algorithms.

### Task 2: Convert from network file to Matrix format
- I will give you a example to illustrate how to convert from network file to matrix.
- For example, we have a network file with 5 nodes and the following edges:
```
1 2
1 3
2 4
2 5
3 4
4 5
```
Graph
```
1 -- 2 _
|    |   \
3 -- 4 -- 5
```

The corresponding matrix format will be:
```
node failure |x_1 | x_2 | x_3 | x_4 | x_5 | y_1 | y_2 | y_3 | y_4| y_5|
empty        | 0   | 0   | 0   | 0   | 0   | 0   |  0  | 0   | 0   | 0   |
node 1 fails | 1   | 0   | 0   | 0   | 0   | 1   | 1   | 1   | 0   | 0   |
node 2 fails | 0   | 1   | 0   | 0   | 0   | 1   | 1   | 0   | 1   | 1   |
node 3 fails | 0   | 0   | 1   | 0   | 0   | 1   | 0   | 1   | 1   | 0   |
node 4 fails | 0   | 0   | 0   | 1   | 0   | 0   | 1   | 1   | 1   | 0   |
node 5 fails | 0   | 0   | 0   | 0   | 1   | 0   | 1   | 0   | 1   | 1   |
```
- In this matrix, the first 5 columns (x_1 to x_5) represent the failure status of each node (1 for failure, 0 for no failure), and the next 5 columns (y_1 to y_5) represent the sensor readings neighboring each node (1 for sensor reading, 0 for no sensor reading). The first row represents the empty state where no nodes fail, and the subsequent rows represent the failure of each node.

### Task 3: Implement baseline algorithms
- The algorithms to be implemented include:
  1. Input: network file, and k (k is the maximum identifiable set size, which is the same in the ../plan.md file)
  2. Convert the network file to matrix format using the function from Task 2.
  3. Choosing randomly 2 columns x_i and y_i (notice that x_i and y_i are corresponding to the same node), and try to remove two columns from the matrix.
  4. Check whether the reverse resulting matrix (after remove x_i, y_i from matrix, and then transpose it) is still k-disjunctive (k is the maximum identifiable set size) using the function in isDDisjunct.py.
  5. If the reuslting matrix is still k-disjunctive, then we can remove x_i and y_i from the matrix. And if not, we keep x_i and y_i in the matrix.
  6. Repeat steps 3-5 until we cannot remove any more columns from the matrix.
  7. The remaining columns in the matrix will be the sensor set for the given network file and k value.

### Task 4: Run experiment with Task3's algorithm on all datasets
- Traversal all the datasets in `datasets` folder.
- For each dataset, run the algorithm in Task 3 to get the sensor set for each k value (k = 1,2,3,4,6,8,10,12,16).
- Remember to record every time we finish 1 dataset with 1 k value, and before start we check the records to run new experiment only.
- Save the result in a file named `Using_DDisjunct_result.txt`.
- Write a report to summarize the results of Task 3's algorithm on all datasets, including the run time and the sensor set size for each dataset and each k value. The report can be in a tabular format for easy comparison.

### Task 5: DL-Disjunctive algorithm
- The algorithm is similar to task 3, but instead of checking k-disjunctiveness, we will check (d,l)-disjunctiveness using the function from Task 1.
- Algorithm:
  - Input: network file, d, l (d is the maximum identifiable set size, and l is the maximum number of sensors that can fail) l < d.
  - Convert the network file to matrix format using the function from Task 2.
  - Choosing randomly 2 columns x_i and y_i (notice that x_i and y_i are corresponding to the same node), and try to remove two columns from the matrix.
  - Check whether the reverse resulting matrix (after remove x_i, y_i from matrix, and then transpose it) is still (d,l)-disjunctive using the function in Task 1.
  - If the reuslting matrix is still (d,l)-disjunctive, then we can remove x_i and y_i from the matrix. And if not, we keep x_i and y_i in the matrix.
  - Repeat steps 3-5 until we cannot remove any more columns from the matrix.
  - The remaining columns in the matrix will be the sensor set for the given network file, d and l values.

### Task 6: Run experiment with Task5's algorithm on all datasets
- Similar to Task 4, but we will run the algorithm in Task 5 for different (d,l) values instead of k values. The (d,l) values to be tested are:
  - (d,l) = {(1, 1), (2, 2), (3, 3), (4, 4), (6, 5), (8, 6), (10, 17), (12, 8), (16, 9)}.
- Remember to record every time we finish 1 dataset with 1 d,l value, and before start we check the records to run new experiment only.
- Save the result in a file named `Using_DL_Disjunct_result.txt`.
- Write a report to summarize the results of Task 5's algorithm on all datasets, including the run time and the sensor set size for each dataset, d value and l value. The report can be in a tabular format for easy comparison.