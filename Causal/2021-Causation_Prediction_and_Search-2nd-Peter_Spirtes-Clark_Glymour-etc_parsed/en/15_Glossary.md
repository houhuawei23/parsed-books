# Glossary

A: In a graph G, Let $\mathbf { A } ( A , B )$ be the union of the ancestors of A or B.

Acceptable: Let a total order Ord of variables in a graph $G ^ { \prime }$ be acceptable for G if and only if whenever $A \ne B$ and there is a directed path from A to B in $G ^ { \prime } ,$ A precedes B in Ord.

After: In a graph G, vertex X is after vertex Y if and only if there is a directed path from Y to X in $G .$ .

Almost Pure: We say that a measurement model is almost pure if the only kind of impurities among the measured variables are common cause impurities. An almost pure latent variable graph is one in which the measurement model is almost pure.

Before: In a graph G, vertex X is before vertex Y if and only if there is a directed path from X to Y in G.

C.F: See constant factor.

Choke point: In a directed acyclic graph G, if for all $T ( K , L )$ in $\mathbf { T } ( K \mathcal { L } )$ and all $T ( I , J )$ in $\mathbf { T } ( I , J )$ , L(T(K,L)) and $J ( T ( I , J ) )$ intersect at a vertex $Q ,$ then Q is an $L J ( T ( I , J ) , T ( K , L ) )$ choke point. Similarly, if for all $T ( K , L )$ in $\mathbf { T } ( K , L )$ and all $T ( I , J )$ in $\mathbf { T } ( I , J )$ , L(T(K,L)) and all $J ( T ( I , J ) )$ intersect at a vertex $Q ,$ and for all $T ( I , L )$ in $\mathbf { T } ( I , L )$ and all $T ( J , K )$ in $\mathbf { T } ( J , K )$ , $L ( T ( I , L ) )$ and ${ \cal J } ( T ( J , K ) )$ also intersect at $Q ,$ then $Q$ is an $L J ( T ( I , J ) , T ( K , L ) , T ( I , L ) , T ( J , K ) )$ choke point. Also see the definition of trek.

Combined graph: See manipulation.

Constant factor: In an LCF or LCT T, if an expression is equal to $c e ,$ , where $c$ is a nonzero constant, and $e$ is a product of equation coefficients raised to positive integral powers, then c is the constant factor (c.f.) of ce.

Contains: In a directed acyclic graph, directed paths $R ( U , I )$ and $R ( U , J )$ contain trek $T$ iff $I ( T ( I , J ) )$ is a final segment of $R ( U , I )$ and ${ \cal J } ( T ( I , J ) )$ is a final segment of $R ( U , J )$ .

D: Given a directed acyclic graph G, $ { \mathbf Ḋ ( X } _ { i } , X _ { j } ) $ is the set of all directed paths from $X _ { i }$ to $X _ { j }$

D-connection: See D-separation.

Definite discriminating path: In a partially oriented inducing path graph , U is a definite discriminating path for B if and only if U is an undirected path between X and Y containing B, $B \neq X , B \neq Y ,$ every vertex on U except for B and the endpoints is a collider or a definite noncollider on U, and

- (i) if V and $V ^ { \prime }$ are adjacent on U, and V is between V and B on U, then $V ^ { * } {  } V ^ { \prime }$ on U,
- (ii) if V is between X and B on U and V is a collider on U then $V  Y$ in , else $V  { ^ { * } Y }$ in ,
- (iii) if V is between Y and B on U and V is a collider on U then $V  X$ in , else $V  { } ^ { * } X$ in ,
- (iv) X and Y are not adjacent in .

Definite noncollider: A vertex B is a definite noncollider on undirected path U if and only if either B is an endpoint of U, or there exist vertices A and C such that U contains one of the subpaths $A \left. B ^ { * } \ – ^ { * } C , A ^ { * } \ – ^ { * } B \right. C , \mathrm { o r } A ^ { * } \ – ^ { * } B ^ { * } \ – ^ { * } C .$

Definite nondescendant: If is the FCI partially oriented inducing path graph of $G$ over O, then X is in Definite-Nondescendants(Y) if and only if there is no semidirected path from any member of Y to X in .

Definite-SP: For a partially oriented inducing path graph over O and ordering Ord acceptable for , V is in Definite-SP(Ord,X) if and only if $V \neq X$ and there is an undirected path U in between V and X such that every vertex on $U$ except for X is a predecessor of X in Ord, and every vertex on U except for the endpoints is a collider on U.

Dependent: In an LCT or LCF S , a variable $X _ { i }$ is dependent iff $X _ { i }$ does not have zero indegree.

Det: Det(Z) is the set of variables determined by any subset of Z.

Determines: A set of variables Z determines the set of variables A, when every variable in A is a deterministic function of the variables in Z, and not every variable in A is a deterministic function of any proper subset of Z.

Det-connected: See Det-separation.

Det-separated: If G is a directed acyclic graph over V, Z is a subset of V that does not contain X or Y, and $X \neq Y ,$ , then X and Y are det-separated given Z and Deterministic(V) if and only if either X and Y are d-separated given $\mathbf { Z } \cup \mathbf { D e t } ( \mathbf { Z } )$ in some Mod(G) relative to Deterministic(V) and Z, or X or Y is in Det(Z); otherwise if $X \neq Y$ and X and Y are not in Z, then X and Y are det-connected given Z and Deterministic(V). If X, Y and Z are disjoint sets of variables in V, and X and Y are non-empty, then X and Y are detseparated given Z if and only if every member X of X and every member Y of Y are detseparated given Z; otherise if X, Y and Z are disjoint sets of variables in V, and X and Y are non-empty, then X and Y are det-connected given Z and Deterministic(V).

Discriminating path: In an inducing path graph G , U is a discriminating path for B if and only if U is an undirected path between X and Y containing B, $B \neq X , B \neq Y ,$ and

- (i) if V and V are adjacent on U, and V is between V and B on U, then $V ^ { * } {  } V ^ { \prime }$ on U,
- (ii) if V is between X and B on U and V is a collider on U then $V  Y \mathrm { i n } G ^ { \prime } ,$ else $V  { } ^ { * } Y$ in $G ^ { \prime } ,$
- (iii) if V is between Y and B on U and V is a collider on U then $V  X$ in $G ^ { \prime } ,$ else $V  { } ^ { * } X$ in $G ^ { \prime } ,$
- (iv) X and Y are not adjacent in $G ^ { \prime } .$

Distributed form: The distributed form of an expression or equation E is the result of carrying out every multiplication, but no additions, subtractions, or divisions in E. If there are no divisions in an equation then its distributed form is a sum of terms. For example, the distributed form of the equation $u = ( a + b ) ( c + d ) \nu { \mathrm { ~ i s ~ } } u = a c \nu + a d \nu + b c \nu + b d \nu .$ .

D-map: An acyclic graph G over V is a D-map of probability distribution P(V) iff for every X, Y, and Z that are disjoint sets of random variables in V, if X is not d-separated from Y given Z in G then X is not independent of Y given Z in P(V). However, when Dmap is applied to the graph in an LCT, the quantifiers in the definitions apply only to sets of non-error variables.

D-Sep: If $G ^ { \prime }$ is an inducing path graph over O and $A \neq B ,$ let $V \in { \bf \delta D - S E P } ( A , B )$ if and only if $A \neq V$ and there is an undirected path U between A and V such that every vertex on U is an ancestor of A or B, and (except for the endpoints) is a collider on U.

D-separated: If G is a directed acyclic graph with vertex set V, Z is a set of vertices not containing X or $Y , X \neq Y ,$ and X and Y are not in Z, then X and Y are D-separated given Z and Deterministic(V) if and only if there is no undirected path U in G between X and Y such that each collider on U has a descendant in Z, and no other vertex on U is in Det(Z); otherwise if $X \neq Y$ and X and Y are not in Z, then X and Y are D-connected given Z and Deterministic(V). Similarly, if X, Y, and Z are disjoint sets of variables, and X and Y are non-empty, then X and Y are D-separated given Z and Deterministic(V) if and only if each pair ${ < } X , Y { > }$ in the Cartesian product of X and Y are D-separated given Z and Deterministic(V); otherwise if X, Y, and Z are disjoint, and X and Y are non-empty, then X and Y are D-connected given Z and Deterministic(V).(Note that this is different from d-separation, which begins with a lowercase “d,” and d-connection, which also begins with a lowercase “d.”)e: In an LCF F, e(S) is equal to S if S is an independent variable, and it is equal to the error variable into S if S is not an independent variable.

E: If X is a random variable, E(X) is the expected value of X.

Equiv(G ): If $G ^ { \prime }$ is an inducing path graph over O, Equiv(G ) is the set of inducing path graphs over the same vertices with the same d-connections as G.

E.C.F: See equation coefficient factor.

Equation coefficient: See linear causal theory, linear causal form.

Equation coefficient factor: In an LCF or LCT T, if an expression is equal to ce, where c is a nonzero constant, and e is a product of equation coefficients raised to positive integral powers, then e is the equation coefficient factor(e.c.f.) of ce.

Equivalent to a polynomial: In an LCF, a quantity (e.g., a covariance) X is equivalent to a polynomial in the coefficients and variances of exogenous variables if and only if for each LCF $F = < < \mathbf { R , M , E } > , \mathbf { C } ,$ , V, EQ,L,Err> and in every LCT $S = < < \mathbf { R } ^ { \prime } , \mathbf { M } ^ { \prime } , \mathbf { E } ^ { \prime } > ,$ , $( \Omega , f , P )$ , EQ ,L ,Err > that is an instance of F, there is a polynomial in the variables in C and V such that X is equal to the result of substituting the linear coefficients of S in as values for the corresponding variables in C, and the variances of the exogenous variables in S as values for the corresponding variables in V.

Error variable: See linear causal theory, linear causal form.

Exogenous: If G is a directed acyclic graph over a set of variables $\mathbf { V } \cup \mathbf { W }$ , and $\mathbf { V } \cap \mathbf { W } =$ ∅, then W is exogenous with respect to V in G if and only if there is no directed edge from any member of V to any member of W.

Faithfully indistinguishable: We will say that two directed acyclic graphs, $G , G ^ { \prime }$ are faithfully indistinguishable (f.i.) if and only if every distribution faithful to G is faithful to $G ^ { \prime }$ and vice-versa.

F.I.: See faithfully indistinguishable.

Final segment: In a graph $G ,$ a path U of length n is a final segment of path V of length m iff $m \geq n ,$ , and for $1 \leq i \leq n + 1$ , the $i ^ { \mathrm { t h } }$ vertex of V equals the $( m { - } n { + } i ) ^ { \mathrm { t h } }$ vertex of U.

I-Map: An acyclic directed graph Gover V is an I-map of probability distribution P(V)G over V iff for every X, Y, and Z that are disjoint sets of random variables in V, if X is dseparated from Y given Z in G then X is independent of Y given Z in P(V). However, when I-map is applied to the graph in an LCT, the quantifiers in the definitions apply only to sets of non-error variables.

Ind: For a directed acyclic graph G, Ind is the set of independent variables in G.

$^ { I n d } a _ { I J } : \ ^ { I n d } a _ { I J }$ is the coefficient of J in the independent equational for I. See also independent equational.

Independent: In an LCT or LCF S , a variable $X _ { i }$ is independent iff $X _ { i }$ has zero indegree (i.e., there are no edges directed into it). Note that the property of independence is completely distinct from the relation of statistical independence. The context will make clear in which of these senses the term is used.

Independent equational: In an $\mathrm { L C F < < R , M , E > }$ , C, V, EQ,L,S> an equation is an independent equational for a dependent variable $X _ { j }$ if and only if it is implied by EQ and the variables in R which appear on the r.h.s. are independent and occur at most once.

Inducing path: If G is a directed acyclic graph over a set of variables V, O is a subset of V containing A and B, and $A \neq B ,$ , then an undirected path $U$ between A and B is an inducing path relative to O if and only if every member of O on $U$ except for the endpoints is a collider on U, and every collider on U is an ancestor of either A or B. We will sometimes refer to members of O as observed variables.

Inducing path graph: $G ^ { \prime }$ is an inducing path graph over O for directed acyclic graph G if and only if O is a subset of the vertices in $G ,$ there is an edge between variables A and B with an arrowhead at A if and only if A and B are in O, and there is an inducing path in G between A and B relative to O that is into A. (Using the notation of chapter 2, the set of marks in an inducing path graph is {>, EM}.)

Initial segment: In a graph $G ,$ a path U of length n is an initial segment of path V of length m iff $m \geq n ,$ , and for $1 \leq i \leq n + 1$ , the $i ^ { \mathrm { { t h } } }$ vertex of V equals the $i ^ { \mathrm { t h } }$ vertex of U.

Into: In a graph G, an edge between A and B is into A if and only if the mark at the A end of the edge is an $\mathit { \Omega } ^ { 6 6 } > . \mathit { \Omega } ^ { 5 9 }$ If an undirected path U between A and B contains an edge into A we will say that U is into A.

Invariant: If G is a directed acyclic graph over a set of variables $\mathbf { V } \cup \mathbf { W } .$ , W is exogenous with respect to V in G, Y and Z are disjoint subsets of V, $P ( \mathbf { V } \cup \mathbf { W } )$ is a distribution that satisfies the Markov condition for $G ,$ and Manipulated(W) = X, then $P ( \mathbf { Y } | \mathbf { Z } )$ is invariant under direct manipulation of X in G by changing W from $\mathbf { w _ { 1 } }$ to ${ \bf w } _ { 2 }$ if and only if $P ( \mathbf { Y } | \mathbf { Z } , \mathbf { W } = \mathbf { w _ { 1 } } ) = P ( \mathbf { Y } | \mathbf { Z } , \mathbf { W } = \mathbf { w } _ { 2 } )$ wherever they are both defined.

Instance: An LCT S is an instance of an LCF F if and only if the graph of S is isomorphic to the graph of F.

IP: In a directed acyclic graph G, if $\mathbf { Y } \cap \mathbf { Z } = \emptyset$ , W is in IP(Y,Z) (W has a parent that is an informative variable for Y given Z) if and only if W is a member of $\mathbf { Z } ,$ and W has a parent in $\mathbf { I V } ( \mathbf { Y } , \mathbf { Z } ) \cup \mathbf { Y }$ .

IV: In a directed acyclic graph G, if $\mathbf { Y } \cap \mathbf { Z } = \emptyset .$ , then V is in IV(Y,Z) (informative variables for Y given $\mathbf { Z } )$ if and only if V is d-connected to Y given Z, and V is not in ND(YZ). (This entails that V is not in $\mathbf { Y } \cup \mathbf { Z } . )$

Label: See linear causal theory, linear causal form.

Length: In a graph G, the length of a path equals the number of vertices in the path minus one.

Last point of intersection: In a directed acyclic graph G, the last point of intersection of directed path $R ( U , I )$ with directed path $R ( V , J )$ is the last vertex on $R ( U , I )$ that is also on $R ( V , J )$ . Note that if G is a directed acyclic graph, the last point of intersection of directed path $R ( U , I )$ with directed path $R ( V , J )$ equals the last point of intersection of $R ( V , J )$ with $R ( U , I )$ ; this is not true of directed cyclic paths.

LCF: See linear causal form.

LCT: See linear causal theory.

Linear causal form: A linear causal form is an unestimated LCT in which the linear coefficients and the variances of the exogenous variables are real variables instead of constants. This entails that an edge label in an LCF is a real variable instead of a constant (except that the label of an edge from an error variable is fixed at one.) More formally, let a linear causal form (LCF) be $< < \mathbf { R , M , E } > , \mathbf { C , V , E Q , L , E r r } > \mathrm { w h e r e }$

(i) $< \mathbf { R , M , E } >$ is a directed acyclic graph. Err is a subset of R called the error variables. Each error variable is of indegree 0 and outdegree 1. For every $X _ { i }$ in R of indegree $\neq 0$ there is exactly one error variable with an edge into $X _ { i } .$ .

- (ii) $c _ { i j }$ is a unique real variable associated with an edge from $X _ { j }$ to $X _ { i }$ , and C is the set of $c _ { i j } .$ . V is the set of variables $\boldsymbol { \sigma } _ { i } ^ { 2 }$ , where $X _ { i }$ is an exogenous variable in $< \mathbf { R , M , E } >$ and $\boldsymbol { \sigma } _ { i } ^ { 2 }$ is a variable that ranges over the positive real numbers.
- (iii) L is a function with domain E such that for each e in E, $L ( e ) = c _ { i j }$ iff $h e a d ( e ) = X _ { i }$ and $t a i l ( e ) = X _ { j } . ~ L ( e )$ will be called the label of e. By extension, the product of labels of edges in any acyclic undirected path U will be denoted by $L ( U )$ , and $L ( U )$ will be called the label of U. The label of an empty path is fixed at 1.
- (iv) EQ is a consistent set of independent homogeneous linear equationals in variables in R. For each $X _ { i }$ in R of positive indegree there is an equation in EQ of the form

$$
X _ {i} = \sum_ {X _ {j} \in \mathbf {P a r e n t s} (X _ {i})} c _ {i j} X _ {j}
$$

where each $c _ { i j }$ is a real variable in C and each $X _ { i }$ is in R. There are no other equations in EQ. $c _ { i j }$ is the equation coefficient of $X _ { j }$ in the equation for $X _ { i }$ .

Linear causal theory: Let a linear causal theory be (LCT) be $< < \mathbf { R , M , E } >$ , $( \Omega , f , P )$ , EQ,L,Err> where

- (i) $( \Omega , f , P )$ is a probability space, where is the sample space, f is a sigma-field over $\varOmega .$ , and $P$ is a probability distribution over f.
- (ii) $< \mathbf { R , M , E } >$ is a directed acyclic graph. R is a set of random variables over $( \Omega , f , P )$ .
- (iii) The variables in R have a joint distribution. Every variable in R has a nonzero variance. E is a set of directed edges between variables in R. (M is the set of marks that occur in a directed graph, that is, $\{ \mathrm { E M } , > \}$ .
- (iv) EQ is a consistent set of independent homogeneous linear equations in random variables in R. For each $X _ { i }$ in R of positive indegree there is an equation in EQ of the form

$$
X _ {i} = \sum_ {X _ {j} \in \mathbf {P a r e n t s} (X _ {i})} a _ {i j} X _ {j}
$$

where each $a _ { i j }$ is a nonzero real number and each $X _ { i }$ is in R. This implies that each vertex $X _ { i }$ in R of positive indegree can be expressed as a linear function of all and only its parents. There are no other equations in EQ. A nonzero value of $a _ { i j }$ is the equation coefficient of $X _ { j }$ in the equation for $X _ { i }$ .

- (v) If vertices (random variables) $X _ { i }$ and $X _ { j }$ are exogenous, then $X _ { i }$ and $X _ { j }$ are pairwise statistically independent.
- (vi) L is a function with domain E such that for each e in $E , L ( e ) = a _ { i j }$ iff $h e a d ( e ) = X _ { i }$ and $t a i l ( e ) = X _ { j } . ~ L ( e )$ will be called the label of e. By extension, the product of labels of edges

in any acyclic undirected path U will be denoted by $L ( U )$ , and $L ( U )$ will be called the label of U. The label of an empty path is fixed at 1.

(vii) There is a subset of S of R called the error variables, each of indegree 0 and outdegree 1. Note that the variance of any endogenous variable I conditional on any set of variables that does not contain the error variable of I is not equal to zero.

Linear Representation: A directed acyclic graph G over V linearly represents a distribution P(V) if and only if there exists a a directed acyclic graph $G ^ { \prime }$ over $\mathbf { V } ^ { \prime }$ and a distribution $P ^ { \prime \prime } ( \mathbf { V } ^ { \prime } )$ such that

- (i) V is included in $\mathbf { V ^ { \prime } } ;$
- (ii) for each endogenous (that is, with positive indegree) variable X in V, there is a unique variable $\varepsilon _ { X }$ in $\mathbf { V } ^ { \pmb { \eta } } \mathbf { W }$ with zero indegree, positive variance, outdegree equal to one, and a directed edge from $\varepsilon _ { X }$ to $X ;$
- (iii) G is the subgraph of $G ^ { \prime }$ over V;
- (iv) each endogenous variable in G is a linear function of its parents in $G ^ { \prime } ;$
- (v) in $P ^ { \prime \prime } ( \mathbf { V } ^ { \prime } )$ the correlation between any two exogenous variables in $G ^ { \prime }$ is zero;
- (vi) $P ( \mathbf { V } )$ is the marginal of $P ^ { \prime \prime } ( \mathbf { V } ^ { \prime } )$ over V.

The members of V \V are called error variables and we call $G ^ { \prime }$ the expanded graph.

Linearly implies: A directed acyclic graph G linearly implies $\rho _ { A B . \mathbf { H } } = 0$ if and only if $\rho _ { A B . \mathbf { H } } = 0$ in all distributions linearly represented by $G .$ (We assume all partial correlations are defined for the distribution.)

Manipulate: See manipulation.

Manipulated graph: See manipulation.

Manipulation: If G is a directed acyclic graph over a set of variables $\mathbf { V } \cup \mathbf { W }$ , and $\textbf { V } _ { \bigcap }$ $\mathbf { W } = \varnothing$ , then W is exogenous with respect to V in G if and only if there is no directed edge from any member of V to any member of W. If $G _ { C o m b }$ is a directed acyclic graph over a set of variables $\mathbf { V } \cup \mathbf { W }$ , and $P ( \mathbf { V } \cup \mathbf { W } )$ satisfies the Markov condition for $G _ { C o m b } ,$ then changing the value of W from $\mathbf { w _ { 1 } }$ to ${ \bf w } _ { 2 }$ is a manipulation of $G _ { C o m b }$ with respect to V if and only if W is exogenous with respect to V, and $P ( \mathbf { V } | \mathbf { W } = \mathbf { w _ { 1 } } ) \neq P ( \mathbf { V } | \mathbf { W } = \mathbf { w } _ { 2 } )$ . We define $P _ { U n m a n ( \mathbf { W } ) } ( \mathbf { V } ) = P ( \mathbf { V } | \mathbf { W } = \mathbf { w _ { 1 } } )$ , and $P _ { M a n ( \mathbf { W } ) } ( \mathbf { V } ) = \mathrm { P } ( \mathbf { V } | \mathbf { W } = \mathbf { w } _ { 2 } )$ , and similarly for various marginal and conditional distributions formed from P(V). We refer to $G _ { C o m b }$ as the combined graph, and the subgraph of $G _ { C o m b }$ over V as the unmanipulated graph $G _ { U n m a n } .$ . V is in Manipulated(W) (that is, V is a variable directly influenced by one of the manipulation variables) if and only if V is in $\mathbf { C h i l d r e n ( W ) } \cap { \mathbf { V } } ;$ ; we will also say that the variables in Manipulated(W) have been directly manipulated. We will refer to the variables in W as policy variables. The manipulated graph, $G _ { M a n }$ is a subgraph of$G _ { U n m a n }$ for which $P _ { M a n ( \mathbf { W } ) } ( \mathbf { V } )$ satisfies the Markov Condition and which differs from $G _ { U n m a n }$ in at most the parents of members of Manipulated(W).

Minimal I-map: An acyclic graph G is a minimal I-map of probability distribution P iff G is an I-map of P, and no subgraph of G is an I-map of P. However, when minimal Imap is applied to the graph in an LCT, the quantifiers in the definitions apply only to sets of non-error variables.

Mod: If G is a directed acyclic graph over V, and Z is included in V, then $G ^ { \prime }$ is in Mod(G) relative to Deterministic(V) and Z if and only if for each V in V

(i) if there exists a set of vertices included in Z that are nondescendants of V in G and that determine V, then Parents $( G ^ { \prime } , V ) { = } \mathbf { X }$ , where X is some set of vertices included in Z that are nondescendants of V in G and that determine V;

(ii) if there is no set X of vertices included in Z that are nondescendants of V in G and that determine V, then Parents $( G ^ { \prime } , V ) = \mathbf { P a r e n t s } ( G , V )$ .

ND: In a directed acyclic graph G, ND(Y) is the set of all vertices that do not have a descendant in Y.

Nondescendants: In a directed acyclic graph G, X is in Nondescendants(Y) if and only if there is no directed path from any member of Y to X in G.

Observed: See inducing path graph, inducing path.

Out of: In a graph G, an edge between A and B is out of A if and only if the mark at the A endpoint is the empty mark. If an undirected path U between A and B contains an edge out of A we will say that U is out of A.

Parallel embedding: Directed acyclic graphs $G _ { 1 }$ and $G _ { 2 }$ with common vertex set O have a parallel embedding in directed acyclic graphs $H _ { 1 }$ and $H _ { 2 }$ having a common set U of vertices that includes O if and only if

(i) $G _ { 1 }$ is the subgraph of $H _ { 1 }$ over O and $G _ { 2 }$ is the subgraph of $H _ { 2 }$ over $\mathbf { o } ;$

(ii) every directed edge in $H _ { 1 }$ but not in $G _ { 1 }$ is in $H _ { 2 }$ and every directed edge in $H _ { 2 }$ but not in $G _ { 2 }$ is in $H _ { 1 }$ .

Path form: If G is a directed acyclic graph, let Let $\mathbf { P } _ { X Y }$ be the set of all directed paths in G from X to Y. In an LCF S, the path form of a product of covariances $\gamma _ { I J } \gamma _ { K L }$ is the distributed form of

$$
\left(\sum_ {U \in \mathbf {U} _ {I J}} \left(\sum_ {R \in \mathbf {P} _ {U I}} \sum_ {R ^ {\prime} \in \mathbf {P} _ {U J}} L (R) L (R ^ {\prime}) \sigma_ {U} ^ {2}\right)\right) \left(\sum_ {V \in \mathbf {U} _ {K L}} \left(\sum_ {R ^ {\prime \prime} \in \mathbf {P} _ {V K}} \sum_ {R ^ {\prime \prime \prime} \in \mathbf {P} _ {V L}} L (R ^ {\prime \prime}) L (R ^ {\prime \prime \prime}) \sigma_ {V} ^ {2}\right)\right)
$$

$\gamma _ { I J } \gamma _ { K L } - \gamma _ { I L } \gamma _ { J K }$ is in path form iff both terms are in path form. $\gamma _ { I J } \gamma _ { K L } - \gamma _ { I L } \gamma _ { J K }$ is in path form iff both terms are in path form.

Policy variables: See manipulate.

Possible-D-SEP(A,B): If A ≠ B in partially oriented inducing path graph , V is in Possible-D-Sep(A,B) in if and only if $V \neq A$ , and there is an undirected path U between A and V in such that for every subpath ${ < X , Y , Z > }$ of U either Y is a collider on the subpath, or Y is not a definite noncollider on U, and X, Y, and Z form a triangle in .

Possibly d-connecting: If A and B are not in Z, and A ≠ B, then an undirected path U between A and B in a partially oriented inducing path graph over O is a possibly dconnecting path of A and B given Z if and only if every collider on U is the source of a semidirected path to a member of Z, and every definite noncollider is not in Z.

Possibly-IP: If is a partially oriented inducing path graph of G over O, then X is in Possibly-IP(Y,Z) if and only if Y and Z are disjoint, X is in Z, and there is a possibly dconnecting path between X and some Y in Y given Z\{X} that is not out of X.

Possibly-IV: If is a partially oriented inducing path graph of G over O, then X is in Possibly-IV(Y,Z) if and only if X is not in Z, there is a possibly d-connecting path between X and some Y in Y given Z, and there is a semidirected path from X to a member of Y ∪ Z.

Possible-SP: For a partially oriented inducing path graph and ordering Ord acceptable for , let V be in Possible-SP(Ord,X) if and only if V ≠ X and there is an undirected path U in between V and X such that every vertex on U except for X is a predecessor of X in Ord, and no vertex on U except for the endpoints is a definite-noncollider on U.

Predecessors: For inducing path graph G and acceptable total ordering Ord, let Predecessors(Ord,V) equal the set of all variables that precede V (not including V) according to Ord.

Proper final segment: A path U of length n is a proper final segment of path V of length m iff U is a final segment of V and $U \neq V .$Proper initial segment: A path U of length n is a proper initial segment of path V of length m iff U is an initial segment of V and $U \neq V .$ .

$P _ { M a n ( \mathbf { W } ) } ( \mathbf { V } )$ : See manipulate.

$P _ { U n m a n ( \mathbf { W } ) } ( \mathbf { V } )$ : See manipulate.

Pure Latent Variable Graph: A pure latent variable graph is a directed acyclic graph in which each measured variable is a child of exactly one latent variable, and a parent of no other variable.

Random coefficient linear causal theory: The definition of a random coefficient linear causal theory is the same as that of a linear causal theory except that each linear coefficient is a random variable independent of the set of all other random variables in the model.

Rigidly statistically indistinguishable: If directed acyclic graphs G and $G ^ { \prime }$ are strongly statistically indistinguishable and every parallel embedding of $G$ and $G ^ { \prime }$ is strongly statistically indistinguishable then structures $G$ and $G ^ { \prime }$ are rigidly statistically indistinguishable (r.s.i.).

R.S.I.: See rigidly statistically indistinguishable.

Semi-directed: A semidirected path from A to B in partially oriented inducing path graph is an undirected path U from A to B in which no edge contains an arrowhead pointing toward A, that is, there is no arrowhead at A on $U ,$ and if X and Y are adjacent on the path, and X is between A and Y on the path, then there is no arrowhead at the X end of the edge between X and Y.

Source: See trek.

SP: For inducing path graph $G ^ { \prime }$ and acceptable total ordering Ord, W is in $\mathbf { S P } ( O r d , G ^ { \prime } , V )$ （号 (separating predecessors of V in $G ^ { \prime }$ for ordering $o r d )$ if and only if $W \neq V$ and there is an undirected path U between W and V such that each vertex on U except for V precedes V in Ord and every vertex on U except for the endpoints is a collider on U.

S.S.I.: See strongly statistically indistinguisable.

Strongly statistically indistinguishable: Two directed acyclic graphs $G , G ^ { \prime }$ are strongly statistically indistinguishable if and only if they have the same vertex set V and every distribution P on V satisfying the Minimality and Markov Conditions for G satisfies those conditions for $G ^ { \prime } ,$ , and vice-versa.

Substituable: In an inducing path or directed acyclic graph G that contains an undirected path U between X and Y, the edge between V and W is substitutable for $U ( V , W )$ in U if and only if V and W are on U, V is between X and W on U, G contains an edge between V and W, V is a collider on the concatenation of $U ( X , V )$ and the edge between V and W if and only if it is a collider on U, and W is a collider on the concatenation of $U ( Y , W )$ and the edge between V and W if and only if it is a collider on U.

T: See trek.

Termini: See trek.

Trek: A trek $T ( I , J )$ between two distinct vertices I and J is an unordered pair of acyclic directed paths from some vertex K to I and J respectively that intersect only at K. The source of the paths in the trek is called the source of the trek. I and J are called the termini of the trek. Given a trek $T ( I , J )$ between I and J, $I ( T ( I , J ) )$ will denote the path in $T ( I , J )$ from the source of $T ( I , J )$ to I and $J ( T ( I , J ) )$ will denote the path in $T ( I , J )$ from the source of $T ( I , J )$ to J. One of the paths in a trek may be an empty path. However, since the termini of a trek are distinct, only one path in a trek can be empty. $\mathbf { T } ( I , J )$ is the set of all treks between I and $J . \ T ( I , J )$ will represent a trek in $\mathbf { T } ( I , J ) . S ( T ( I , J ) )$ represents the source of the trek $T ( I , J )$ .

Undirected: In a graph G, Let V be in Undirected(X,Y) if and only if V lies on some undirected path between X and Y.

Unmanipulated graph: See manipulation.

${ \mathbf { U } } _ { X } { \mathbf { : } }$ In an LCF S, $\mathbf { U } _ { X }$ is the set of all independent variables that are the source of a directed path to X. (Note that if X is independent then $X \in \ \mathbf { U } _ { X }$ since there is an empty path from every vertex to itself.)

$\mathbf { U } _ { X Y } { \mathrm { : } }$ In an LCF S, $\mathbf { U } _ { X Y }$ is $\mathbf { U } _ { X } \cap \mathbf { U } _ { Y }$ .

Weakly faithfully indistinguishable: Two directed acyclic graphs are weakly faithfully indistinguishable (w.f.i.) if and only if there exists a probability distribution faithful to both of them.

Weakly statistically indistinguishable: Two directed acyclic graphs are weakly statistically indistinguishable (w.s.i.) if and only if there exists a probability distribution meeting the Minimality and Markov Conditions for both of them.

W.F.I.: See weakly faithfully indistinguishable.

W.S.I.: See weakly statistically indistinguishable.