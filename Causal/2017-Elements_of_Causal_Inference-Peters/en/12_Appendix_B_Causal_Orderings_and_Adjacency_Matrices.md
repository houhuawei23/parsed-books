# Appendix B Causal Orderings and Adjacency Matrices

Definition B.1 Given a DAG G, we call a permutation, that is, a bijective mapping,

$$
\pi : \{1, \dots , p \} \rightarrow \{1, \dots , p \},
$$

a causal ordering (sometimes one says topological ordering) if it satisfies

$$
\pi (i) <   \pi (j) \quad i f \quad j \in \mathbf {D E} _ {i} ^ {\mathcal {G}}.
$$

Because of the acyclic structure of the DAG, there is always a topological ordering (see Proposition B.2). But this order does not have to be unique. The node $\pi ^ { - 1 } ( 1 )$ does not have any parents and is therefore a source node, and $\pi ^ { - 1 } ( p )$ does not have any descendants and is thus a sink node.

Proposition B.2 For each DAG there is a topological ordering.

Proof. We proceed by induction. We need to show that in each DAG, there is a node without any ancestors. Start with any node and move to one of its parents (if there are any). You will never visit a parent that you have seen before (if you did there had been a directed cycle). After at most $p - 1$ steps you reach a node without any parent. 

Definition B.3 We can represent a directed graph $\mathcal { G } = ( V , \mathcal { E } )$ over d nodes with a binary $d \times d$ matrix A (taking values 0 or 1):

$$
A _ {i, j} = 1 \quad \Leftrightarrow \quad (i, j) \in \mathcal {E}.
$$

A is called the adjacency matrix of G.

This representation of DAGs is particularly useful for the efficient implementation of algorithms. There are a couple of useful results transforming adjacency matrices, some of which we report here.

Remark B.4 (i) Let A be the adjacency matrix for DAG G. The entry $( i , j )$ of the squared matrix $A ^ { 2 }$ equals the number of paths of length two from i to $j .$ . This is because

$$
A _ {i, j} ^ {2} = \sum_ {k} A _ {i k} A _ {k j}.
$$

(ii) In general, we have

$$
A _ {i j} ^ {k} = \# \text {   paths   of   length   } k \text {   from   } i \text {   to   } j.
$$

(iii) If indices increase on directed paths, that is, $j \in \mathbf { D } \mathbf { E } _ { i } ^ { \mathcal { G } }$ implies $j > i ,$ , then the identity is a causal ordering and the adjacency matrix is upper triangular, that is, only the upper-right half of the matrix contains non-zeros.

(iv) We may want to use sparse matrices when the graph is sparse to save space and/or computation time.

The number of DAGs with d nodes have been studied by Robinson [1970, 1973] and independently by Stanley [1973]. The number of such matrices (or DAGs) is growing very quickly in d (see Table B.1).

McKay [2004] proves the following equivalent description of DAGs which had been conjectured by Eric W. Weisstein.

Theorem B.5 The matrix A is an adjacency matrix of a DAG G if and only $i f A + { \mathrm { I d } }$ is a 0-1-matrix with all eigenvalues being real and strictly greater than zero.

**d Number of DAGs with d nodes Table B.1: The number of DAGs depending on the number d of nodes, taken from http: //oeis.org/A003024 [OEIS Foundation Inc., 2017]. The length of the numbers grows faster than any linear term.**

| 1 | 1 |
| --- | --- |
| 2 | 3 |
| 3 | 25 |
| 4 | 543 |
| 5 | 29281 |
| 6 | 3781503 |
| 7 | 1138779265 |
| 8 | 783702329343 |
| 9 | 1213442454842881 |
| 10 | 4175098976430598143 |
| 11 | 31603459396418917607425 |
| 12 | 521939651343829405020504063 |
| 13 | 18676600744432035186664816926721 |
| 14 | 1439428141044398334941790719839535103 |
| 15 | 237725265553410354992180218286376719253505 |
| 16 | 83756670773733320287699303047996412235223138303 |
| 17 | 62707921196923889899446452602494921906963551482675201 |
| 18 | 99421195322159515895228914592354524516555026878588305014783 |
| 19 | 332771901227107591736177573311261125883583076258421902583546773505 |

## C