"""
identifying_code_ilp.py

Integer Linear Programming formulation for the Minimum Identifying Code problem.

Requirement:
    pip install pulp

Author: ChatGPT
"""

from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    value,
    PULP_CBC_CMD,
    LpStatus,
)


def closed_neighborhood(adj):
    """
    Compute closed 1-neighborhood for every vertex.

    Parameters
    ----------
    adj : list[list[int]]
        Adjacency matrix.

    Returns
    -------
    list[set]
        N[i] = closed neighborhood of vertex i.
    """

    n = len(adj)
    N = []

    for i in range(n):
        neigh = {i}

        for j in range(n):
            if adj[i][j] == 1:
                neigh.add(j)

        N.append(neigh)

    return N


def minimum_identifying_code(adj):
    """
    Solve Minimum Identifying Code using ILP.

    Parameters
    ----------
    adj : list[list[int]]

    Returns
    -------
    code : list[int]
        Selected vertices.

    obj : int
        Minimum code size.
    """

    n = len(adj)

    N = closed_neighborhood(adj)

    # -------------------------
    # Build ILP model
    # -------------------------

    model = LpProblem("Minimum_Identifying_Code", LpMinimize)

    x = [
        LpVariable(f"x_{i}", cat="Binary")
        for i in range(n)
    ]

    # -------------------------
    # Objective
    # -------------------------

    model += lpSum(x)

    # -------------------------
    # Coverage constraints
    # -------------------------

    for v in range(n):
        model += (
            lpSum(x[u] for u in N[v]) >= 1,
            f"Coverage_{v}",
        )

    # -------------------------
    # Identification constraints
    # -------------------------

    for u in range(n):
        for v in range(u + 1, n):

            diff = N[u] ^ N[v]

            if len(diff) == 0:
                raise ValueError(
                    f"Vertices {u} and {v} are twins. "
                    "Identifying code does not exist."
                )

            model += (
                lpSum(x[w] for w in diff) >= 1,
                f"Identify_{u}_{v}",
            )

    # -------------------------
    # Solve
    # -------------------------

    model.solve(PULP_CBC_CMD(msg=False))

    if LpStatus[model.status] != "Optimal":
        raise RuntimeError("No optimal solution found.")

    code = [
        i
        for i in range(n)
        if value(x[i]) > 0.5
    ]

    return code, int(value(model.objective))


def print_neighborhoods(adj, names=None):
    """
    Print closed neighborhoods.
    """

    n = len(adj)

    if names is None:
        names = list(range(n))

    N = closed_neighborhood(adj)

    print("Closed neighborhoods:\n")

    for i in range(n):
        vertices = [names[v] for v in sorted(N[i])]
        print(f"N[{names[i]}] = {vertices}")


def main():
    """
    Example graph.

        G ----- O
        |       | \
        |       |  \
        B ----- R---P
    """

    names = ["G", "O", "B", "R", "P"]

    adj = [

    # G O B R P

        [0,1,1,0,0],   # G
        [1,0,0,1,1],   # O
        [1,0,0,1,0],   # B
        [0,1,1,0,1],   # R
        [0,1,0,1,0],   # P
    ]

    print_neighborhoods(adj, names)

    code, obj = minimum_identifying_code(adj)

    print("\n==============================")
    print("Minimum Identifying Code")
    print("==============================")

    print(f"Minimum size = {obj}")

    print("Selected vertices:")

    for v in code:
        print(names[v])


if __name__ == "__main__":
    main()