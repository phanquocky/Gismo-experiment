import random
from itertools import combinations


def contains_identity(submatrix):
    """
    Check whether the selected (d+1)-column submatrix
    contains all rows of the identity matrix.

    submatrix: list of rows
               each row is a tuple/list of length k
    """

    k = len(submatrix[0])

    # Required identity rows:
    # (1,0,0,...), (0,1,0,...), ...
    required = set()

    for i in range(k):
        row = [0] * k
        row[i] = 1
        required.add(tuple(row))

    existing = set(tuple(row) for row in submatrix)

    return required.issubset(existing)


def randomized_d_disjunct_test(M, d, m=100_000):
    """
    Randomized heuristic checker for d-disjunct matrices.

    Parameters
    ----------
    M : list[list[int]]
        Boolean matrix of size t x n

    d : int
        disjunct parameter

    m : int
        number of randomized trials

    Returns
    -------
    bool
        False -> definitely NOT d-disjunct
        True  -> probably d-disjunct
    """

    t = len(M)
    n = len(M[0])

    k = d + 1

    if k > n:
        raise ValueError("d + 1 must be <= n")

    for trial in range(m):

        # Step 2.1:
        # randomly choose d+1 columns
        cols = random.sample(range(n), k)

        # Step 2.2:
        # build the selected submatrix
        submatrix = []

        for r in range(t):
            row = [M[r][c] for c in cols]
            submatrix.append(row)

        # Step 2.3:
        if not contains_identity(submatrix):
            print(f"Failed at trial {trial + 1}")
            print(f"Counterexample columns: {cols}")
            return False

    return True


if __name__ == "__main__":

    # Example matrix
    M = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]

    d = 2

    result = randomized_d_disjunct_test(M, d)

    print("Probably d-disjunct" if result else "Not d-disjunct")