import random
from itertools import combinations


def has_witness_row(M, D, L):
    """
    Check whether there exists a row such that:
        - all columns in D are 0
        - at least one column in L is 1
    """

    t = len(M)

    for r in range(t):

        # Every column in D must be 0
        d_ok = all(M[r][c] == 0 for c in D)

        if not d_ok:
            continue

        # At least one column in L must be 1
        l_ok = any(M[r][c] == 1 for c in L)

        if l_ok:
            return True

    return False


def randomized_dl_disjunct_test(M, d, l, m=10_000):
    """
    Randomized heuristic checker for (d,l)-disjunct matrices.

    Strategy:
        1. Randomly choose d+l columns
        2. Enumerate ALL partitions into:
              - D (size d)
              - L (size l)
        3. Check the witness condition

    Returns:
        False -> definitely NOT (d,l)-disjunct
        True  -> probably (d,l)-disjunct
    """

    t = len(M)
    n = len(M[0])

    if l > d:
        raise ValueError("Require l <= d")

    if d + l > n:
        raise ValueError("Require d + l <= n")

    for trial in range(m):

        # Step 1:
        # Randomly choose d+l distinct columns
        S = random.sample(range(n), d + l)

        # Step 2:
        # Enumerate every possible L of size l
        for L_tuple in combinations(S, l):

            L = set(L_tuple)

            # D = S \ L
            D = [c for c in S if c not in L]

            # Step 3:
            # Check witness row
            if not has_witness_row(M, D, L):

                print(f"Failed at trial {trial + 1}")
                print(f"S = {S}")
                print(f"D = {D}")
                print(f"L = {list(L)}")

                return False

    return True


if __name__ == "__main__":

    M = [
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0],
    ]

    d = 2
    l = 1

    result = randomized_dl_disjunct_test(M, d, l)

    if result:
        print("Probably (d,l)-disjunct")
    else:
        print("Not (d,l)-disjunct")