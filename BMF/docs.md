# Boolean Matrix Factorization

## Problem Definition

Let

\[
C \in \{0,1\}^{n\times m}
\]

be a Boolean matrix, where \(n\) is the number of rows (objects or observations) and \(m\) is the number of columns (features). Let \(k\) be a positive integer representing the target factorization rank.

The goal of the **Boolean Matrix Factorization (BMF)** problem is to find two Boolean matrices

\[
S \in \{0,1\}^{n\times k},
\qquad
B \in \{0,1\}^{k\times m},
\]

such that their Boolean product approximates the original matrix \(C\).

The Boolean matrix product is defined as

\[
(S \circ B)_{ij}
=
\bigvee_{t=1}^{k}
\left(
S_{it}\land B_{tj}
\right),
\]

where

- \(\land\) denotes the logical **AND** operation,
- \(\lor\) denotes the logical **OR** operation.

The objective is to minimize the reconstruction error between \(C\) and the Boolean product \(S\circ B\).

Formally,

\[
\min_{S,B}
\;
\left\|
C-(S\circ B)
\right\|_0,
\]

where

- \(\|\cdot\|_0\) denotes the number of mismatched entries (Hamming distance) between two Boolean matrices.

Equivalently,

\[
\min_{S,B}
\sum_{i=1}^{n}
\sum_{j=1}^{m}
\mathbf{1}
\left[
C_{ij}
\neq
(S\circ B)_{ij}
\right].
\]


# Identifying Code

## Problem Definition

Let \(G=(V,E)\) be a simple, undirected graph and let \(r \ge 1\) be an integer.

For each vertex \(v \in V\), define its **closed \(r\)-neighborhood** as

\[
N_r[v] = \{u \in V \mid d(u,v) \le r\},
\]

where \(d(u,v)\) is the shortest-path distance between vertices \(u\) and \(v\).

A subset of vertices \(C \subseteq V\) is called an **\(r\)-identifying code** if, for every vertex \(v \in V\),

1. **Coverage:** Every vertex is covered by at least one codeword:

   \[
   N_r[v] \cap C \neq \emptyset.
   \]

2. **Identification:** Every pair of distinct vertices has a unique identifying set:

   \[
   N_r[u] \cap C \neq N_r[v] \cap C,
   \qquad \forall\, u,v \in V,\; u \neq v.
   \]

The identifying set (or signature) of a vertex \(v\) is defined as

\[
I_r(v) = N_r[v] \cap C.
\]

The goal is to find an \(r\)-identifying code of **minimum cardinality**, i.e., solve

\[
\min_{C \subseteq V} |C|
\]

subject to the coverage and identification constraints above.
