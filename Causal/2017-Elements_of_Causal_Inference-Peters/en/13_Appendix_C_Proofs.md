# Appendix C Proofs

## C.1 Proof of Theorem 4.2

We first state a lemma; its proof can be found in Peters [2008], for example.

Lemma C.1 Let X and N be independent variables and assume that N is nondeterministic. Then $N \not \vdash \left( X + N \right)$ .

Proof of Theorem 4.2. If X and $N _ { Y }$ are normally distributed, we have

$$
\beta := \frac {\operatorname{cov} [ X , Y ]}{\operatorname{cov} [ Y , Y ]} = \frac {\alpha \operatorname{var} [ X ]}{\alpha^ {2} \operatorname{var} [ X ] + \operatorname{var} [ N _ {Y} ]}
$$

and define $N _ { X } : = X - \beta Y . ~ N _ { X }$ and Y are uncorrelated by construction and because $N _ { X }$ and Y are jointly Gaussian, it follows that they are independent, too.

To prove the “only $\mathrm { i f } ^ { \dag }$ statement, we assume that

$$
Y = \alpha X + N _ {Y}
$$

$$
\text { and } \quad N _ {X} = (1 - \alpha \beta) X - \beta N _ {Y}
$$

are independent. Distinguish between the following cases:

(i) $( 1 - \alpha \beta ) \neq 0$ and $\beta \neq 0 .$

Here, Theorem 4.3 implies that $X , N _ { Y }$ and thus also $Y , N _ { X }$ are normally distributed. Hence, $P _ { X , Y }$ is bivariate Gaussian, too.

(ii) $\beta = 0 .$

This implies

$$
X \perp \alpha X + N _ {Y},
$$

which is a contradiction to Lemma C.1.

(iii) $\left( 1 - \alpha \beta \right) = 0 .$ .

It follows $- \beta N _ { Y } \perp \perp \alpha X + N _ { Y }$ . Thus

$$
N _ {Y} \perp \alpha X + N _ {Y},
$$

which, again, contradicts Lemma C.1.

This concludes the proof.

![image_63](images/image_63.png)

## C.2 Proof of Proposition 6.3

Proof. Recall that our definition of an SCM includes the requirement that the underlying graph is acyclic. We can now substitute the structural assignments recursively into each other and can therefore write each node $X _ { j }$ as a unique function of all noise terms $( N _ { k } ) _ { k \in { \bf A N } _ { j } }$ that belong to the ancestors of $X _ { j }$ . That is,

$$
X _ {j} := g _ {j} \big ((N _ {k}) _ {k \in \mathbf {A N} _ {j}} \big).
$$

(The function does not necessarily depend on the noise terms of all ancestors.) 

## C.3 Proof of Remark 6.6

Proof. We will show that whenever we can remove a variable from $\mathbf { P A } _ { j }$ , we can still remove it from $\mathbf { P A } _ { j } ^ { * }$ in the reduced model.

Consider an input ${ X _ { k } } ^ { \prime } \in \mathbf { P A } _ { j } \cap \mathbf { P A } _ { j } ^ { * }$ that $f _ { j }$ does not depend on. That is, we have $f _ { j } ( \mathbf { p } \mathbf { a } _ { j , - k } , x _ { k } , n _ { j } ) = f _ { j } ( \mathbf { p } \mathbf { a } _ { j , - k } , x _ { k } ^ { \prime } , n _ { j } )$ for all $x _ { k } , x _ { k } ^ { \prime } , \mathbf { p } \mathbf { a } _ { j , - k }$ and $n _ { j }$ with $p ( n _ { j } ) > 0$ . Here, $\mathbf { P A } _ { j , - k } : = \mathbf { P A } _ { j } \setminus \{ k \}$ denotes the set of all input variables except for k. Then, g does not depend on this variable $x _ { k }$ either because $g ( \mathbf { p } \mathbf { a } _ { j , - k } ^ { * } , x _ { k } , n _ { j } ) = f _ { j } ( \mathbf { p } \mathbf { a } _ { j } , x _ { k } , n _ { j } )$ for all $x _ { k } , \mathbf { p } \mathbf { a } _ { j , - k } ^ { * }$ and $n _ { j }$ with $p ( n _ { j } ) > 0$ . 

## C.4 Proof of Proposition 6.13

Proof. To simplify notation we write $X _ { 1 }$ instead of X and $X _ { 2 }$ instead of Y . First,the truncated factorization formula (6.9) implies

$$
\begin{array}{l} p _ {X _ {2}} ^ {\mathfrak {C}; d o (X _ {1} := x _ {1})} (x _ {2}) = \int \prod_ {j \neq 1} p _ {j} (x _ {j} | x _ {p a (j)}) d x _ {3} \dots d x _ {d} \\ = \int \prod_ {j \neq 1} p _ {j} (x _ {j} | x _ {p a (j)}) \frac {\tilde {p} (x _ {1})}{\tilde {p} (x _ {1})} d x _ {3} \dots d x _ {d} \\ = p _ {X _ {2} \mid X _ {1} = x _ {1}} ^ {\mathfrak {C}; d o (X _ {1} := \tilde {N} _ {1})} (x _ {2}) \tag {C.1} \\ \end{array}
$$

if $\tilde { N } _ { 1 }$ puts positive mass on $x _ { 1 }$ , that is, $\tilde { p } ( x _ { 1 } ) > 0$ . We furthermore require that the following two statements hold for all distributions $Q _ { X _ { 1 } , X _ { 2 } }$ over $( X _ { 1 } , X _ { 2 } )$ with density q:

$$
X _ {2} \not \perp X _ {1} \text {   in   } Q \iff \exists x _ {1} ^ {\triangle}, x _ {1} ^ {\square} \text {   with   } q (x _ {1} ^ {\triangle}), q (x _ {1} ^ {\square}) > 0 \text {   and   } Q _ {X _ {2} | X _ {1} = x _ {1} ^ {\triangle}} \neq Q _ {X _ {2} | X _ {1} = x _ {1} ^ {\square}} \tag {C.2}
$$

and

$$
X _ {2} \not \perp X _ {1} \text {   in   } Q \iff \exists x _ {1} ^ {\triangle} \text {   with   } q (x _ {1} ^ {\triangle}) > 0 \text {   and   } Q _ {X _ {2} | X _ {1} = x _ {1} ^ {\triangle}} \neq Q _ {X _ {2}}. \tag {C.3}
$$

We then have for any $\hat { N } _ { 1 }$ with full support

$$
(i) \stackrel {{(\mathrm{C}. 2)}} {{\Longrightarrow}} \exists x _ {1} ^ {\triangle}, x _ {1} ^ {\square} \text { with   pos.   density   under } \tilde {N} _ {1} \text { s.t. } P _ {X _ {2} | X _ {1} = x _ {1} ^ {\triangle}} ^ {\mathfrak {C}; d o (X _ {1} := \tilde {N} _ {1})} \neq P _ {X _ {2} | X _ {1} = x _ {1} ^ {\square}} ^ {\mathfrak {C}; d o (X _ {1} := \tilde {N} _ {1})}
$$

$$
\stackrel {\text {(C.1)}} {\Longrightarrow} (i i)
$$

$$
\stackrel {\text {(C.1)}} {\Longrightarrow} \exists x _ {1} ^ {\triangle}, x _ {1} ^ {\square} \text {with pos. density under} \hat {N} _ {1} \text {s.t.} P _ {X _ {2} | X _ {1} = x _ {1} ^ {\triangle}} ^ {\mathfrak {C}; d o (X _ {1} := \hat {N} _ {1})} \neq P _ {X _ {2} | X _ {1} = x _ {1} ^ {\square}} ^ {\mathfrak {C}; d o (X _ {1} := \hat {N} _ {1})}
$$

$$
\stackrel {\text {(C.2)}} {\Longrightarrow} (i v)
$$

$$
\stackrel {\text {(trivial)}} {\Longrightarrow} (i)
$$

We further have $( i i )  { \stackrel { \mathrm { \scriptsize ~ ( t r i v i a l ) } } { = } } ( i i i )$ $P _ { X _ { 2 } } ^ { \mathrm { g } } = P _ { X _ { 2 } } ^ { \mathrm { g } ; d o ( X _ { 1 } : = N _ { 1 } ^ { * } ) }$ $N _ { 1 } ^ { * }$ distribution $P _ { X _ { 1 } } ^ { \mathrm { g } }$ . Together with $\neg ( i ) \Rightarrow \neg ( i i )$ , the latter implies

$$
\begin{array}{l} \neg (i) \implies X _ {2} \perp       \perp X _ {1} \text {   in   } P _ {\mathbf {X}} ^ {\mathfrak {C}; d o (X _ {1} := N _ {1} ^ {*})} \\ \stackrel {(C. 3)} {\Longrightarrow} P _ {X _ {2} \mid X _ {1} = x ^ {\triangle}} ^ {\mathfrak {C}; d o (X _ {1} := N _ {1} ^ {*})} = P _ {X _ {2}} ^ {\mathfrak {C}; d o (X _ {1} := N _ {1} ^ {*})} \text {   for   all   } x ^ {\triangle} \text {   with   } p _ {1} (x ^ {\triangle}) > 0 \\ \stackrel {\text {(C.1)}} {\Longrightarrow} P _ {X _ {2}} ^ {\mathfrak {C}; d o \left(X _ {1} := x ^ {\triangle}\right)} = P _ {X _ {2}} ^ {\mathfrak {C}} \text {   for   all   } x ^ {\triangle} \text {   with   } p _ {1} (x ^ {\triangle}) > 0 \\ \stackrel {\neg (i i)} {\Longrightarrow} P _ {X _ {2}} ^ {\mathfrak {C}; d o \left(X _ {1} := x ^ {\triangle}\right)} = P _ {X _ {2}} ^ {\mathfrak {C}} \text {   for   all   } x ^ {\triangle} \\ \Longrightarrow \neg (i i i) \\ \end{array}
$$

Here, the symbol “¬” denotes the negation of a statement.

## C.5 Proof of Proposition 6.14

Proof. Statement (i) follows directly from the Markov property of the interventional SCM. The intervention removes the incoming edges into $X$ , and if there is no direct path from X to Y in the original graph, X and Y are d-separated.

Statement (ii) can be proved by a counterexample (see, e.g., Example 6.34). 

## C.6 Proof of Proposition 6.36

Proof. $\mathrm { \Omega ^ { 6 6 } \mathrm { \Omega _ { 1 1 } ^ { 6 9 } } }$ : Assume that causal minimality is not satisfied. Then, there is an $X _ { j }$ and a $Y \in \mathbf { P A } _ { j } ^ { \mathcal { G } }$ , such that $P _ { \mathbf { X } }$ is also Markovian with respect to the graph obtained when removing the edge $Y  X _ { j }$ from ${ \mathcal { G } } .$ . This implies $X _ { j } \perp \perp Y | \mathbf { P A } _ { j } ^ { \mathcal { G } } \setminus \{ Y \}$ by the local Markov property.

“only $\mathrm { i f } ^ { \dag }$ : If $P _ { \mathbf { X } }$ has a density, the Markov condition is equivalent to the Markov factorization [Lauritzen, 1996, Theorem 3.27]. Assume now that $Y \in \mathbf { P A } _ { j } ^ { \mathcal { G } }$ and $X _ { j } \perp \perp Y | \mathbf { P A } _ { i } ^ { \mathcal { G } } \setminus \{ Y \}$ , which implies $p ( x _ { j } | \mathbf { p } \mathbf { a } _ { j } ^ { \mathcal { G } } ) = p ( x _ { j } | \mathbf { p } \mathbf { a } _ { j , - Y } ^ { \mathcal { G } } )$ where $\mathbf { P A } _ { j , - Y } ^ { \mathcal { G } }$ is defined as $\mathbf { P A } _ { j , - Y } ^ { \mathcal { G } } = \mathbf { P A } _ { j } ^ { \mathcal { G } } \setminus \{ Y \}$ . Then, $p ( \mathbf { x } ) = p ( x _ { j } | \mathbf { p a } _ { i , - Y } ^ { \mathcal { G } } ) \prod _ { k \neq j } p ( x _ { k } | \mathbf { p a } _ { k } ^ { \mathcal { G } } )$ , which implies that $P _ { \mathbf { X } }$ is Markovian with respect to $\mathcal { G }$ without $Y  X _ { j }$ . 

## C.7 Proof of Proposition 6.48

Proof. We assume that both models satisfy causal minimality and come with graphs $\mathcal { G }$ and $\mathcal { H } .$ . Intuitively, we can identify the children of a node X since they change after intervening on $X$ . Some of the children, however, may not change their distribution after an intervention due to two canceling paths, for example. We thus introduce the following notation. Given a DAG $\mathcal { G }$ , we call X a youngest parent of a node Y and write $X \in \mathbf { Y } \mathbf { P } \mathbf { A } _ { Y }$ if $X \in \mathbf { P A } _ { Y }$ and X is not an ancestor of any other parent of Y . A node Y may have several youngest parents. The proof requires two arguments:

(i) If $X \in \mathbf { Y } \mathbf { P } \mathbf { A } _ { Y } ^ { \mathcal { G } }$ , then there is a total causal effect from X to $Y _ { ; }$ , meaning that there are $x ^ { \triangle }$ and $x ^ { \square }$ , such that $P _ { Y } ^ { d o \left( X : = x ^ { \triangle } \right) } \neq P _ { Y } ^ { d o \left( X : = x ^ { \triangle } \right) }$ . This follows from causal minimality.

(ii) If $Z \in \mathbf { A } \mathbf { N } _ { Y } ^ { \mathcal { G } }$ , then there exist $X _ { 1 } , \ldots , X _ { k }$ , such that $X _ { 1 } = Z , X _ { k } = Y$ , and $X _ { i } \in$ $\mathbf { Y P A } _ { X _ { i + 1 } } ^ { \mathcal { G } } \mathrm { ~ f o r ~ } i \in \{ 1 , . . . , k - 1 \}$ .

Finally, we can combine these two statements and conclude that if $Z \in \mathbf { A } \mathbf { N } _ { Y } ^ { \mathcal { G } }$ , then there are $X _ { 1 } , \ldots , X _ { k }$ such that for $i \in \left\{ 1 , \ldots , k - 1 \right\}$ , $X _ { i }$ has a total causal effect on $X _ { i + 1 }$ , which implies that there must be a direct causal path from $X _ { i }$ to $X _ { i + 1 }$ also in H; see Proposition 6.13. But then $Z \in \mathbf { A } \mathbf { N } _ { Y } ^ { \mathcal { H } }$ , which implies that both $\mathcal { G }$ and H have the same ancestor relationships. Since both $\mathcal { G }$ and H satisfy causal minimality, this implies that $\mathcal { G } = \mathcal { H }$ and therefore the two models are equivalent as causal graphical models. 

## C.8 Proof of Proposition 6.49

Proof. According to the proof of Proposition 6.3, we can write for the first SCM $\mathbf { X } = \mathbf { g } ( \mathbf { N } )$ . But since

$$
\mathbf {g} (\mathbf {n}) = \mathbf {g} ^ {*} (\mathbf {n}) \quad \forall \mathbf {n} \text { with } p (\mathbf {n}) > 0,
$$

we clearly have that both SCMs induce the same observational distributions (and intervention distributions with the same argument). Regarding counterfactuals, we cover both the discrete and the continuous case by conditioning on $\mathbf { X } \in A$ with $P ( \mathbf { X } \in A ) > 0 ;$ ; see Definition 6.17. The new density over the noise variables satisfies

$$
\begin{array}{l} \tilde {p} (n _ {1}, \ldots , n _ {d}) = \left\{ \begin{array}{c l} \frac {p (n _ {1} , \ldots , n _ {d})}{P (X \in A)} & \text { if } \mathbf {g} (n _ {1}, \ldots , n _ {d}) \in A \\ 0 & \text { else } \end{array} \right. \\ = \left\{ \begin{array}{c l} \frac {p (n _ {1} , \ldots , n _ {d})}{P (\mathbf {g} (\mathbf {N}) \in A)} & \text { if } \mathbf {g} ^ {*} (n _ {1}, \ldots , n _ {d}) \in A \\ 0 & \text { else } \end{array} \right. \\ = \left\{ \begin{array}{c l} \frac {p (n _ {1} , \ldots , n _ {d})}{P (\mathbf {g} ^ {*} (\mathbf {N}) \in A)} & \text { if } \mathbf {g} ^ {*} (n _ {1}, \ldots , n _ {d}) \in A \\ 0 & \text { else } \end{array} \right. \\ = \tilde {p} ^ {*} (n _ {1}, \dots , n _ {d}). \\ \end{array}
$$

We still have

$$
\mathbf {g} (\mathbf {n}) = \mathbf {g} ^ {*} (\mathbf {n}) \quad \forall \mathbf {n} \text { with } \tilde {p} (\mathbf {n}) > 0,
$$

which implies that all counterfactual statements coincide.

## C.9 Proof of Proposition 7.1

Proof. Let $N _ { 1 } , \ldots , N _ { d }$ be independent and uniformly distributed between 0 and 1. We then define $X _ { j } : = f _ { j } ( X _ { \mathbf { P A } _ { j } } , N _ { j } )$ with

$$
f _ {j} \left(\mathbf {p a} _ {j}, n _ {j}\right) := F _ {X _ {j} \mid \mathbf {P A} _ {j} = \mathbf {p a} _ {j}} ^ {- 1} \left(n _ {j}\right) \tag {C.4}
$$

$F _ { X _ { j } | \mathbf { P A } _ { j } = \mathbf { p a } _ { j } } ^ { - 1 }$ is the generalized inverse cumulative distribution function from $X _ { j }$ given $\mathbf { P A } _ { j } = \mathbf { \dot { p } } \mathbf { a } _ { j }$ . The generalized inverse cumulative distribution function of a random variable Y is defined as $F _ { Y } ^ { - 1 } ( a ) : = \operatorname* { i n f } \{ y \in \mathbb { R } : F _ { Y } ( y ) \geq a \}$ . Equation (C.4) guarantees that in the constructed SCM, the conditional $X _ { j } | \mathbf { P } \mathbf { A } _ { j } = \mathbf { p } \mathbf { a } _ { j }$ has the correct distribution. The statement then follows from the Markov factorization, Definition 6.21(iii). 

## C.10 Proof of Proposition 7.4

Proof. Assume causal minimality is not satisfied. We can then find nodes $j$ and $i \in \mathbf { P } \mathbf { A } _ { j }$ with $X _ { j } = f _ { j } ( \mathbf { P A } _ { i } \backslash \{ i \} , X _ { i } ) + N _ { j }$ that does not depend on $X _ { i }$ if we condition on all other parents $A : = \mathbf { \bar { P } } \mathbf { A } _ { j } \backslash \{ i \}$ , that is $X _ { j } \perp \perp X _ { i } | X _ { A }$ (see Proposition 6.36). Here, we denote $\mathbf { P A } _ { j } \backslash \{ X _ { i } \}$ by $X _ { A }$ . For the function $f _ { j }$ , we will now show that $f _ { j } ( x _ { A } , x _ { i } ) =$ $c _ { x _ { A } }$ for $P _ { X _ { A } , X _ { i } - \mathrm { { a l m o s t } } }$ all $( x _ { A } , x _ { i } )$ . Indeed, assume without loss of generality that $\mathbb { E } [ N _ { j } ] = 0$ , then the mean of $X _ { j } | \mathbf { P A } _ { j } = \left( x _ { A } , x _ { i } \right)$ equals $f _ { j } ( x _ { A } , x _ { i } )$ . Equation (2b) from Dawid [1979] states that if $X _ { j } \perp \perp X _ { i } | X _ { A }$ , then the density of $X _ { j } | X _ { A } , X _ { i }$ does not depend on the argument of $X _ { i }$ . Therefore, also the conditional mean $f _ { j } ( x _ { A } , x _ { i } )$ does not depend on xi. It follows that $f _ { j } ( x _ { A } , x _ { i } ) = c _ { x _ { A } }$ . The continuity of $f _ { j }$ implies that $f _ { j }$ is constant in its last argument.

The converse statement follows from Proposition 6.36, too.

## C.11 Proof of Proposition 8.1

Proof. We use the Bellman optimality equation [e.g., Sutton and Barto, 2015,Chapter 3.8]. For all $s ^ { \circ }$ and s with $f ( s ^ { \circ } ) = f ( s )$ , we have

$$
\begin{array}{l} Q ^ {*} (s, a) = \sum_ {s ^ {\prime}} p (s ^ {\prime} | s, a) \left(\mathbb {E} [ R | s ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) \\ = \sum_ {f ^ {\prime}} \sum_ {s ^ {\prime}: f (s ^ {\prime}) = f ^ {\prime}} p (s ^ {\prime} \mid s, a) \left(\mathbb {E} [ R \mid s ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) \\ = \sum_ {f ^ {\prime}} p (f ^ {\prime} \mid s, a) \left(\mathbb {E} [ R \mid f ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) \\ = \sum_ {f ^ {\prime}} p (f ^ {\prime} | s ^ {\circ}, a) \left(\mathbb {E} [ R | f ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) = Q ^ {*} (s ^ {\circ}, a). \\ \end{array}
$$

This concludes the proof.

![image_64](images/image_64.png)

## C.12 Proof of Proposition 8.2

Proof. The first equation follows from the discussion in Section 8.2.1. The Markov factorization property implies

$$
p (\mathbf {x}) = p (a | s)   p (s | h)   p (h)   p (y | f, h)   p (f | a);
$$

see Figure 8.5. It now follows with $F \perp \perp S | A$ that

$$
\begin{array}{l} \int y \frac {\tilde {p} (a | s)}{p (a | s)} p (\mathbf {x}) d \mathbf {x} = \int y \tilde {p} (a | s) p (s | h) p (h) p (y | f, h) p (f | a, s) d a d f d h d s d y \\ = \int y \tilde {p} (f, a | s) p (s | h) p (h) p (y | f, h) d a d f d h d s d y \\ = \int y \frac {\tilde {p} (f | s)}{p (f | s)} p (s | h) p (h) p (y | f, h) p (f | s) d f d h d s d y \\ = \int y \frac {\tilde {p} (f | s)}{p (f | s)} p (s | h) p (h) p (y | f, h) p (f, a | s) d a d f d h d s d y \\ = \int y \frac {\tilde {p} (f | s)}{p (f | s)} p (\mathbf {x}) d \mathbf {x}. \\ \end{array}
$$

The last equality follows from $p ( f , a | s ) = p ( f | a , s ) p ( a | s )$ .

![image_65](images/image_65.png)

## C.13 Proof of Proposition 9.3

Proof. To show (i), we start with the SCM C over X and its entailed distribution $R _ { \mathbf { X } }$ . We then consider the structural assignments for variables $O \in { \mathbf { 0 } }$ and repeatedly plug in the assignments for the variables $X \in \mathbf { X } \backslash \mathbf { o }$ whenever these variables appear on the right-hand side. This leads to a new SCM in which each structural assignment for $O \in { \mathbf { 0 } }$ contains a multivariate error variable $\tilde { \mathbf { N } } _ { O }$ . It is apparent that this smaller SCM entails the same observational distribution $P _ { \mathbf { 0 } }$ and the same intervention distributions when intervening on any $O \in { \mathbf { 0 } }$ . From causal sufficiency, it follows that the new noise variables $( \tilde { \mathbf { N } } _ { O } ) _ { O \in \mathbf { O } }$ are jointly independent. As in the case of one-dimensional noise variables (Proposition 6.31), this again implies that the distribution $P _ { \mathbf { 0 } }$ is Markovian with respect to the induced graph structure. The statement now follows from the fact that this new SCM can be transformed to an SCM with one-dimensional error variables that entails the same observational and intervention distributions (exploiting the same construction as in Proposition 7.1). For a more formal description of this procedure, as well as for more details on these arguments, see Bongers et al. [2016].

Statement (ii) follows from Example 9.2.

## C.14 Proof of Theorem 10.3

Proof. If there is an arrow from $X _ { \mathrm { p a s t } ( t ) } ^ { j }$ to $X _ { t } ^ { k }$ , the dependence (10.3) follows immediately from faithfulness because two directly connected variables cannot be d-separated. Now assume that there is no edge from $X _ { \mathrm { p a s t } ( t ) } ^ { j }$ to $X _ { t } ^ { k }$ . Then, $X _ { t } ^ { k }$ is $d -$ separated from X jpa $X _ { \mathrm { p a s t } ( t ) } ^ { j }$ st(t) given X− jpast $\mathbf { X } _ { \mathrm { p a s t } ( t ) } ^ { - j }$ (t) . Any path leaving $X _ { t } ^ { k }$ with an outgoing edge is blocked because it will have a collider (and no node after with time index larger or equal to t is conditioned on); any path leaving $X _ { t } ^ { k }$ with an incoming edge is blocked because the next node is in the conditioning set $\mathbf { X } _ { \mathrm { p a s t } ( t ) } ^ { - j } .$ . 

## C.15 Proof of Theorem 10.4

Proof. To prove (i), consider a full time graph containing no arrow from X to $Y$ . Then, every path from $Y _ { t }$ to $X _ { \mathrm { p a s t } ( t ) }$ is blocked by $Y _ { \mathrm { p a s t } ( t ) }$ . Any path that starts with an outgoing edge from $Y _ { t }$ must contain a collider that is not in the conditioning set (neither is any of its descendants); any path starting with an incoming edge is blocked since the first node on this path is in Ypast(t). $Y _ { \mathrm { p a s t } ( t ) }$

To prove (ii), assume $Y _ { t }$ has parents from $X$ , denoted by $\mathbf { P A } _ { Y _ { t } } ^ { X }$ . Then (10.5) implies

$$
Y _ {t} \perp \perp \mathbf {P A} _ {Y _ {t}} ^ {X} \mid Y _ {\text { past } (t)}. \tag {C.5}
$$

For any $X _ { s } \in \mathbf { P } \mathbf { A } _ { Y _ { t } } ^ { X }$ , (C.5) implies by weak union (see Appendix A.1)

$$
Y _ {t} \perp X _ {s} \mid Y _ {\text { past } (t)} \cup (\mathbf {P A} _ {Y _ {t}} ^ {X} \setminus \{X _ {s} \}). \tag {C.6}
$$

Due to Peters et al. [2014, Lemma 38], minimality implies that $Y _ { t }$ is dependent of any parent A of $Y _ { t }$ , given any set of non-descendants of $Y _ { t }$ that includes the other parents of $Y _ { t }$ except A. Hence we have

$$
Y _ {t} \not \perp X _ {s} \mid Y _ {\text { past } (t)} \cup (\mathbf {P A} _ {Y _ {t}} ^ {X} \setminus \{X _ {s} \}),
$$

in contradiction to (C.6).

<!-- footnote -->

- We write $H ( X _ { j _ { 1 } } , \dots , X _ { j _ { k } } )$ instead of $H \big ( ( X _ { j _ { 1 } } , \dots , X _ { j _ { k } } ) \big )$ for notational convenience and again perform set operations on vectors.

<!-- footnote end -->

<!-- footnote -->

- Strictly speaking, we have introduced the causal DAG only for finitely many nodes so far. Here, however, we need infinite graphs and neglect this technical subtlety [see, e.g., Peters et al., 2013].

<!-- footnote end -->