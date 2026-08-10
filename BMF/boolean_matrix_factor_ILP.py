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

    # C = [
    #     # Khối I_3
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],

    #     # Hàng [1, 1, 0] của Q_G, lặp 3 lần
    #     [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],

    #     # Hàng [1, 1, 1] của Q_G, lặp 3 lần
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

    #     # Hàng [0, 1, 1] của Q_G, lặp 3 lần
    #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],

    #     # Hàng [0, 0, 1] của Q_G, lặp 3 lần
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],

    #     # Hàng [1, 0, 1] của Q_G, lặp 3 lần
    #     [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    #     [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    #     [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],

    #     # Hàng [1, 0, 0] của Q_G, lặp 3 lần
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    # ]
    QG = [
        # Domination rows
        [1, 1, 0, 0, 1],  # (⊥, G)
        [1, 1, 1, 1, 0],  # (⊥, O)
        [0, 1, 1, 1, 0],  # (⊥, P)
        [0, 1, 1, 1, 1],  # (⊥, R)
        [1, 0, 0, 1, 1],  # (⊥, B)

        # Separation rows
        [0, 0, 1, 1, 1],  # (G, O)
        [1, 0, 1, 1, 1],  # (G, P)
        [1, 0, 1, 1, 0],  # (G, R)
        [0, 1, 0, 1, 0],  # (G, B)
        [1, 0, 0, 0, 0],  # (O, P)
        [1, 0, 0, 0, 1],  # (O, R)
        [0, 1, 1, 0, 1],  # (O, B)
        [0, 0, 0, 0, 1],  # (P, R)
        [1, 1, 1, 0, 1],  # (P, B)
        [1, 1, 1, 0, 0],  # (R, B)
    ]
    n = 5
    K = 3
    rho = K + 1  # rho = 3

    I_n = [
        [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ]

    # Lặp mỗi hàng của Q_G đúng rho lần
    Q_repeated = [
        row.copy()
        for row in QG
        for _ in range(rho)
    ]

    # A_G = [I_n; Q_G^(rho)]
    A_G = I_n + Q_repeated

    # C_G = [a1,...,a1, a2,...,a2, ..., an,...,an | y]
    # Mỗi anchor a_i xuất hiện rho lần.
    CG = []

    for row_index, row in enumerate(A_G):
        expanded_row = []

        for value in row:
            expanded_row.extend([value] * rho)

        # y = [0_n; 1_(rho*q)]
        y_value = 0 if row_index < n else 1
        expanded_row.append(y_value)

        CG.append(expanded_row)

    print(len(CG))       # 50 hàng
    print(len(CG[0]))    # 16 cột

    k = 5

    S, B, Y, err = boolean_matrix_factorization(CG, k)

    print_matrix("Original C", CG)

    print_matrix("Factor S", S)

    print_matrix("Factor B", B)

    print_matrix("Reconstructed Y", Y)

    print("Minimum Hamming Error =", err)