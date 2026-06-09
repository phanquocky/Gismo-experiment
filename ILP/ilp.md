# ILP approach

From ../baseline/baseline-algo.md
## TASK 
### Task 1: investigate ILP (Integer Linear Programming) approach to solve problem in ../baseline/baseline-algo.md

### Task 2: implemnt ILP-based sensor set finder for d-disjunct cases, using PuLP
                 node1fails    node2fails    node3fails    node4fails    node5fails
  x_1                1             0             0             0             0
  x_5                0             0             0             0             1
  y_1                1             1             1             0             0
  y_5                0             1             0             1             1
solve in small example 

Graph
```
1 -- 2 _
|    |   \
3 -- 4 -- 5
```

And find the minimum sensor set for d=1.
Implement and write to `ddisjunct.py`file.
Step by step instruction:
- write graph.txt file first, from above example graph write to graph.txt so that is the same format with other network file in `datasets` folder.
- Write ILP-base solver to solve and find the minimum sensor set for d=1.
---

### Task 3: implemnt ILP-based sensor set finder for (d,l)-disjunct cases, using PuLP
                 node1fails    node2fails    node3fails    node4fails    node5fails
  x_1                1             0             0             0             0
  x_5                0             0             0             0             1
  y_1                1             1             1             0             0
  y_5                0             1             0             1             1
solve in small example 

Graph
```
1 -- 2 _
|    |   \
3 -- 4 -- 5
```

And find the minimum sensor set for d=2, l=2.
Implement and write to `dldisjunct.py`file.
Step by step instruction:
- write graph.txt file first, from above example graph write to graph.txt so that is the same format with other network file in `datasets` folder.
- Write ILP-base solver to solve and find the minimum sensor set for d=2, l=2.
---

### Task4: Run experiment using Task3's algorithm
Graphs: datasets/MadridAdj.mtx, datasets/ParisAdj.mtx, datasets/PhilippineAdj.mtx, datasets/ZerkaniAdj.mtx
(d,l): {(1, 1), (2, 2), (3, 3), (4, 4), (6, 5), (8, 6), (10, 17), (12, 8), (16, 9)} 

Run experiment for every graph above and for each graph run every (d,l) value above. 
Record the result every time we finish 1 graph with 1 (d,l) value, and before start we check the records to run new experiment only.
Save the result in a file named `Using_DDDisjunct_result.txt`.
Write a report to summarize the results of Task 3's algorithm on all datasets, including the run time and the sensor set size for each dataset and each (d,l) value.

## Problem Restatement

Given a network of n nodes, we build a **failure-sensor matrix** M_orig of shape (n × 2n):
- Row i corresponds to node i failing
- Columns 0..n-1 are the **x-block** (x_i = 1 iff node i is the failing node)
- Columns n..2n-1 are the **y-block** (y_i = 1 iff node i is in the closed neighbourhood of the failing node)

Each node i corresponds to a **sensor pair** (column i, column n+i). Selecting node i as a sensor means both columns are active.

The algorithm transposes the submatrix of active sensor columns to get M':
- Rows of M' = active sensor columns (from selected nodes)
- Columns of M' = failure scenarios (n nodes)

**Goal:** find the minimum subset of nodes S such that M' (built from S) is d-disjunct (or (d,l)-disjunct).

---

## Definitions

**d-disjunct matrix M':** For any d+1 columns, each column j has a **witness row** r where M'[r][j]=1 and M'[r][k]=0 for all other k in the selected d+1 columns. Equivalently: the union of any d columns does NOT cover any other column.

**(d,l)-disjunct matrix M':** For any d+l columns and every partition into D (size d) and L (size l), there exists a witness row r where M'[r][k]=0 for all k∈D and M'[r][k]=1 for at least one k∈L.

---

## Key Structural Observation

The values of M' are **completely determined** by the network topology (encoded in M_orig). Selecting sensor node i only controls **which rows** of M' are present. This means the ILP only needs binary variables z_i ∈ {0,1} indicating whether node i is in the sensor set — the matrix values themselves are fixed constants.

---

## ILP Formulation: Finding Minimum d-Disjunct Sensor Set

### Variables
```
z_i ∈ {0, 1}   for i = 0..n-1   (1 = include node i as sensor)
```

### Objective
```
Minimize  Σ_i  z_i
```

### Constraints

**Precompute** for each (failure scenario j, d-subset S of other scenarios):

```
W(j, S) = { node_of(c) :  M_orig[j][c] = 1  AND  M_orig[k][c] = 0  for all k ∈ S }
```

where `node_of(c) = c` if `c < n`, else `c - n` (maps a column index to its node).

W(j, S) is the set of nodes whose sensor columns can **witness** scenario j against every scenario in S.

**Disjunctiveness constraint** — for each j ∈ {0..n-1} and each d-subset S ⊆ {0..n-1}\{j}:

```
Σ_{i ∈ W(j,S)}  z_i  ≥  1
```

At least one witness sensor must be active.

### Full model
```
min   Σ_i z_i
s.t.  Σ_{i ∈ W(j,S)} z_i  ≥  1    ∀ j, ∀ S ⊆ {0..n-1}\{j}, |S|=d
      z_i ∈ {0, 1}                  ∀ i
```

This is a **weighted set cover** (or hitting set) ILP, which any standard MILP solver (PuLP + CBC, OR-Tools, Gurobi, CPLEX) handles directly.

---

## ILP Formulation: Finding Minimum (d,l)-Disjunct Sensor Set

### Variables
Same as above: `z_i ∈ {0,1}`.

### Precompute witness sets

For each (D, L) = partition of some d+l distinct failure scenarios into D (size d) and L (size l):

```
W(D, L) = { node_of(c) :  M_orig[k][c] = 0  for all k ∈ D
                           AND  M_orig[k][c] = 1  for at least one k ∈ L }
```

A sensor column c witnesses (D,L) if it is silent for all of D and active for at least one element of L.

### Constraints — for every (D, L) pair:

```
Σ_{i ∈ W(D,L)}  z_i  ≥  1
```

### Full model
```
min   Σ_i z_i
s.t.  Σ_{i ∈ W(D,L)} z_i  ≥  1    ∀ D,L with |D|=d, |L|=l, D∩L=∅, D∪L ⊆ {0..n-1}
      z_i ∈ {0, 1}                  ∀ i
```

---

## Constraint Count Analysis

| Variant | Constraint count |
|---|---|
| d-disjunct | n · C(n−1, d) |
| (d,l)-disjunct | C(n, d+l) · C(d+l, l) |

Concrete examples for n=100:

| (d or d+l) | d-disjunct | (d=2,l=2)-disjunct |
|---|---|---|
| d=1 | 100 · 99 = 9,900 | — |
| d=2 | 100 · 4,950 = 495,000 | C(100,4)·C(4,2) = 23M |
| d=4 | 100 · 3.9M ≈ 390M | — |

**Takeaway:**
- d=1 and d=2 are tractable with direct enumeration (up to n≈500)
- d≥3 or large n requires **constraint generation** (cutting planes)

---

## Constraint Generation (Scalable ILP)

For large d or n, generate constraints lazily:

1. Start with **no** disjunctiveness constraints.
2. Solve the ILP → get a candidate sensor set z*.
3. **Check** whether M'(z*) is d-disjunct (using the existing randomised checker).
4. If d-disjunct: done, z* is optimal.
5. If not: find a **violated constraint** (a (j, S) pair with no active witness in z*) and add it.
6. Repeat from step 2.

Step 5 is a **separation oracle** — given z*, find the most violated constraint. This can be done by iterating over (j, S) pairs that have small W(j,S) ∩ {active sensors}, or by the randomised sampling already in `isDDisjunct.py` / `isDLDisjunct.py`.

Convergence is guaranteed because the constraint set is finite, and each iteration either terminates or adds at least one new constraint.

---

## Comparison to Existing Approaches

| Approach | Optimality | Scalability | Notes |
|---|---|---|---|
| Greedy baseline (baseline.py) | Heuristic (near-min) | O(n² · m_trials) per pass | Randomised; may not find global optimum |
| GISMO (SAT/MaxSat) | Optimal (within maxc) | Depends on SAT solver | Needs encoding step; hard timeout via maxc |
| **ILP (this proposal)** | **Exact optimal** | n²–n³ constraints for d≤2 | MILP solver; constraint generation for large d |

ILP gives **provably minimum** sensor sets, useful as a ground truth for evaluating how close the greedy baseline comes to optimal.

---

## Implementation Plan

```python
# Pseudocode for ILP sensor set finder (d-disjunct version)
import pulp
from itertools import combinations

def ilp_sensor_placement(network_file, d):
    nodes, matrix = network_to_matrix(network_file)
    n = len(nodes)
    M = matrix[1:]                           # drop empty row: shape (n, 2n)

    prob = pulp.LpProblem("min_sensor_set", pulp.LpMinimize)
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(z)                    # minimize sensor count

    for j in range(n):
        for S in combinations([k for k in range(n) if k != j], d):
            # Witness set: columns c where M[j][c]=1 and M[s][c]=0 for all s in S
            witnesses = set()
            for c in range(2 * n):
                if M[j][c] == 1 and all(M[s][c] == 0 for s in S):
                    node_idx = c if c < n else c - n
                    witnesses.add(node_idx)
            # At least one witness sensor must be active
            if witnesses:
                prob += pulp.lpSum(z[i] for i in witnesses) >= 1
            else:
                # No sensor can ever witness (j, S): problem is infeasible for this d
                return None

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    sensor_set = [nodes[i] for i in range(n) if pulp.value(z[i]) > 0.5]
    return sensor_set
```

For large n/d, replace the inner loop with constraint generation (add constraint only when checker finds a violation).

---

## Feasibility Verdict

| Scenario | Verdict |
|---|---|
| Small graphs (n ≤ 100), d ≤ 2 | **Directly feasible** — enumerate all constraints, solve with CBC/Gurobi |
| Medium graphs (n ≤ 500), d ≤ 2 | **Feasible with constraint generation** — lazy constraint addition |
| Large graphs (n > 500) or d ≥ 3 | **LP relaxation + rounding** as lower bound; constraint generation for exact |
| (d,l)-disjunct, small (d+l ≤ 4) | **Directly feasible** for n ≤ 100; constraint generation otherwise |

**Recommendation:** implement ILP for d=1,2 on small/medium datasets to get exact optimal sensor set sizes, then use those as reference points to measure how close the greedy baseline and GISMO come to optimal.
