"""
boolean_matrix_factorization_ilp.py

Integer Linear Programming formulation for
Boolean Matrix Factorization (BMF)

Require:
    pip install pulp
"""

from pulp import *


def boolean_matrix_factorization(C, k):
    """
    Solve Boolean Matrix Factorization

    Parameters
    ----------
    C : list[list[int]]
        Boolean matrix (n x m)

    k : int
        Factorization rank

    Returns
    -------
    S : list[list[int]]
    B : list[list[int]]
    Y : reconstructed matrix
    error : minimum Hamming distance
    """

    n = len(C)
    m = len(C[0])

    model = LpProblem("BooleanMatrixFactorization", LpMinimize)

    ########################################################
    # Variables
    ########################################################

    # S matrix
    s = {
        (i, t): LpVariable(f"s_{i}_{t}", cat="Binary")
        for i in range(n)
        for t in range(k)
    }

    # B matrix
    b = {
        (t, j): LpVariable(f"b_{t}_{j}", cat="Binary")
        for t in range(k)
        for j in range(m)
    }

    # z = s AND b
    z = {
        (i, t, j): LpVariable(f"z_{i}_{t}_{j}", cat="Binary")
        for i in range(n)
        for t in range(k)
        for j in range(m)
    }

    # reconstructed matrix
    y = {
        (i, j): LpVariable(f"y_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in range(m)
    }

    # error variables
    e = {
        (i, j): LpVariable(f"e_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in range(m)
    }

    ########################################################
    # Objective
    ########################################################

    model += lpSum(e[i, j] for i in range(n) for j in range(m))

    ########################################################
    # AND linearization
    ########################################################

    for i in range(n):
        for t in range(k):
            for j in range(m):

                model += z[i, t, j] <= s[i, t]

                model += z[i, t, j] <= b[t, j]

                model += (
                    z[i, t, j]
                    >= s[i, t] + b[t, j] - 1
                )

    ########################################################
    # OR linearization
    ########################################################

    for i in range(n):
        for j in range(m):

            for t in range(k):
                model += y[i, j] >= z[i, t, j]

            model += (
                y[i, j]
                <= lpSum(z[i, t, j] for t in range(k))
            )

    ########################################################
    # Reconstruction error
    ########################################################

    for i in range(n):
        for j in range(m):

            model += e[i, j] >= C[i][j] - y[i, j]

            model += e[i, j] >= y[i, j] - C[i][j]

    ########################################################
    # Solve
    ########################################################

    model.solve(PULP_CBC_CMD(msg=False))

    if LpStatus[model.status] != "Optimal":
        raise RuntimeError("No optimal solution found.")

    ########################################################
    # Extract solution
    ########################################################

    S = [
        [
            int(value(s[i, t]))
            for t in range(k)
        ]
        for i in range(n)
    ]

    B = [
        [
            int(value(b[t, j]))
            for j in range(m)
        ]
        for t in range(k)
    ]

    Y = [
        [
            int(value(y[i, j]))
            for j in range(m)
        ]
        for i in range(n)
    ]

    error = int(value(model.objective))

    return S, B, Y, error


############################################################
# Pretty print
############################################################

def print_matrix(name, M):
    print(name)
    for row in M:
        print(" ".join(map(str, row)))
    print()


############################################################
# Example
############################################################

if __name__ == "__main__":

    C = [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ]

    k = 2

    S, B, Y, err = boolean_matrix_factorization(C, k)

    print_matrix("Original C", C)

    print_matrix("Factor S", S)

    print_matrix("Factor B", B)

    print_matrix("Reconstructed Y", Y)

    print("Minimum Hamming Error =", err)