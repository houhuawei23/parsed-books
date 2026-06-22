# 附录 C（Appendix C）

## C.1 定理 4.2 的证明（Proof of Theorem 4.2）

我们首先陈述一个引理；其证明可参见 Peters [2008]。

**引理 C.1** 设 $X$ 和 $N$ 是独立变量，且假设 $N$ 是非确定性的。则 $N \not \vdash \left( X + N \right)$。

**定理 4.2 的证明**。如果 $X$ 和 $N _ { Y }$ 服从正态分布，我们有

$$
\beta := \frac {\operatorname{cov} [ X , Y ]}{\operatorname{cov} [ Y , Y ]} = \frac {\alpha \operatorname{var} [ X ]}{\alpha^ {2} \operatorname{var} [ X ] + \operatorname{var} [ N _ {Y} ]}
$$

并定义 $N _ { X } : = X - \beta Y$。由于 $N _ { X }$ 和 $Y$ 在构造上是不相关的，且由于 $N _ { X }$ 和 $Y$ 是联合高斯的，因此它们也是独立的。

为了证明"仅当"陈述，我们假设

$$
Y = \alpha X + N _ {Y}
$$

$$
\text { 且 } \quad N _ {X} = (1 - \alpha \beta) X - \beta N _ {Y}
$$

是独立的。区分以下情况：

(i) $( 1 - \alpha \beta ) \neq 0$ 且 $\beta \neq 0$。

此时，**定理 4.3** 意味着 $X , N _ { Y }$ 以及 $Y , N _ { X }$ 都服从正态分布。因此，$P _ { X , Y }$ 也是二元高斯的。

(ii) $\beta = 0$。

这意味着

$$
X \perp \alpha X + N _ {Y},
$$

这与**引理 C.1** 矛盾。

(iii) $\left( 1 - \alpha \beta \right) = 0$。

由此可得 $- \beta N _ { Y } \perp \perp \alpha X + N _ { Y }$。因此

$$
N _ {Y} \perp \alpha X + N _ {Y},
$$

这同样与**引理 C.1** 矛盾。

证明完毕。

![image_63](images/image_63.png)

## C.2 命题 6.3 的证明（Proof of Proposition 6.3）

**证明**。回顾我们对**结构因果模型（Structural Causal Model, SCM）**的定义包含了底层图是无环的这一要求。我们现在可以递归地将结构赋值相互代入，因此可以将每个节点 $X _ { j }$ 写为属于 $X _ { j }$ 祖先的所有噪声项 $( N _ { k } ) _ { k \in { \bf A N } _ { j } }$ 的唯一函数。即，

$$
X _ {j} := g _ {j} \big ((N _ {k}) _ {k \in \mathbf {A N} _ {j}} \big).
$$

（该函数不一定依赖于所有祖先的噪声项。）

## C.3 注释 6.6 的证明（Proof of Remark 6.6）

**证明**。我们将证明，只要我们可以从 $\mathbf { P A } _ { j }$ 中移除一个变量，我们在简化模型中仍然可以将其从 $\mathbf { P A } _ { j } ^ { * }$ 中移除。

考虑一个输入 ${ X _ { k } } ^ { \prime } \in \mathbf { P A } _ { j } \cap \mathbf { P A } _ { j } ^ { * }$，$f _ { j }$ 不依赖于该输入。即，对于所有满足 $p ( n _ { j } ) > 0$ 的 $x _ { k } , x _ { k } ^ { \prime } , \mathbf { p } \mathbf { a } _ { j , - k }$ 和 $n _ { j }$，有 $f _ { j } ( \mathbf { p } \mathbf { a } _ { j , - k } , x _ { k } , n _ { j } ) = f _ { j } ( \mathbf { p } \mathbf { a } _ { j , - k } , x _ { k } ^ { \prime } , n _ { j } )$。这里，$\mathbf { P A } _ { j , - k } : = \mathbf { P A } _ { j } \setminus \{ k \}$ 表示除 $k$ 之外的所有输入变量的集合。那么，$g$ 也不依赖于这个变量 $x _ { k }$，因为对于所有满足 $p ( n _ { j } ) > 0$ 的 $x _ { k } , \mathbf { p } \mathbf { a } _ { j , - k } ^ { * }$ 和 $n _ { j }$，有 $g ( \mathbf { p } \mathbf { a } _ { j , - k } ^ { * } , x _ { k } , n _ { j } ) = f _ { j } ( \mathbf { p } \mathbf { a } _ { j } , x _ { k } , n _ { j } )$。

## C.4 命题 6.13 的证明（Proof of Proposition 6.13）

**证明**。为简化符号，我们用 $X _ { 1 }$ 代替 $X$，用 $X _ { 2 }$ 代替 $Y$。首先，**截断分解公式（truncated factorization formula）** (6.9) 意味着

$$
\begin{array}{l} p _ {X _ {2}} ^ {\mathfrak {C}; d o (X _ {1} := x _ {1})} (x _ {2}) = \int \prod_ {j \neq 1} p _ {j} (x _ {j} | x _ {p a (j)}) d x _ {3} \dots d x _ {d} \\ = \int \prod_ {j \neq 1} p _ {j} (x _ {j} | x _ {p a (j)}) \frac {\tilde {p} (x _ {1})}{\tilde {p} (x _ {1})} d x _ {3} \dots d x _ {d} \\ = p _ {X _ {2} \mid X _ {1} = x _ {1}} ^ {\mathfrak {C}; d o (X _ {1} := \tilde {N} _ {1})} (x _ {2}) \tag {C.1} \\ \end{array}
$$

如果 $\tilde { N } _ { 1 }$ 在 $x _ { 1 }$ 上赋予正质量，即 $\tilde { p } ( x _ { 1 } ) > 0$。我们进一步要求以下两个表述对所有具有密度 $q$ 的 $( X _ { 1 } , X _ { 2 } )$ 上的分布 $Q _ { X _ { 1 } , X _ { 2 } }$ 成立：

$$
X _ {2} \not \perp X _ {1} \text {   in   } Q \iff \exists x _ {1} ^ {\triangle}, x _ {1} ^ {\square} \text {   with   } q (x _ {1} ^ {\triangle}), q (x _ {1} ^ {\square}) > 0 \text {   and   } Q _ {X _ {2} | X _ {1} = x _ {1} ^ {\triangle}} \neq Q _ {X _ {2} | X _ {1} = x _ {1} ^ {\square}} \tag {C.2}
$$

且

$$
X _ {2} \not \perp X _ {1} \text {   in   } Q \iff \exists x _ {1} ^ {\triangle} \text {   with   } q (x _ {1} ^ {\triangle}) > 0 \text {   and   } Q _ {X _ {2} | X _ {1} = x _ {1} ^ {\triangle}} \neq Q _ {X _ {2}}. \tag {C.3}
$$

那么，对于任何具有完全支撑的 $\hat { N } _ { 1 }$，我们有

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

我们进一步有 $( i i )  { \stackrel { \mathrm { \scriptsize ~ ( t r i v i a l ) } } { = } } ( i i i )$，$P _ { X _ { 2 } } ^ { \mathrm { g } } = P _ { X _ { 2 } } ^ { \mathrm { g } ; d o ( X _ { 1 } : = N _ { 1 } ^ { * } ) }$，其中 $N _ { 1 } ^ { * }$ 的分布为 $P _ { X _ { 1 } } ^ { \mathrm { g } }$。结合 $\neg ( i ) \Rightarrow \neg ( i i )$，后者意味着

$$
\begin{array}{l} \neg (i) \implies X _ {2} \perp       \perp X _ {1} \text {   in   } P _ {\mathbf {X}} ^ {\mathfrak {C}; d o (X _ {1} := N _ {1} ^ {*})} \\ \stackrel {(C. 3)} {\Longrightarrow} P _ {X _ {2} \mid X _ {1} = x ^ {\triangle}} ^ {\mathfrak {C}; d o (X _ {1} := N _ {1} ^ {*})} = P _ {X _ {2}} ^ {\mathfrak {C}; d o (X _ {1} := N _ {1} ^ {*})} \text {   for   all   } x ^ {\triangle} \text {   with   } p _ {1} (x ^ {\triangle}) > 0 \\ \stackrel {\text {(C.1)}} {\Longrightarrow} P _ {X _ {2}} ^ {\mathfrak {C}; d o \left(X _ {1} := x ^ {\triangle}\right)} = P _ {X _ {2}} ^ {\mathfrak {C}} \text {   for   all   } x ^ {\triangle} \text {   with   } p _ {1} (x ^ {\triangle}) > 0 \\ \stackrel {\neg (i i)} {\Longrightarrow} P _ {X _ {2}} ^ {\mathfrak {C}; d o \left(X _ {1} := x ^ {\triangle}\right)} = P _ {X _ {2}} ^ {\mathfrak {C}} \text {   for   all   } x ^ {\triangle} \\ \Longrightarrow \neg (i i i) \\ \end{array}
$$

这里，符号"¬"表示对陈述的否定。

## C.5 命题 6.14 的证明（Proof of Proposition 6.14）

**证明**。陈述 (i) 直接来自**干预性结构因果模型（interventional SCM）**的**马尔可夫性质（Markov property）**。干预移除了进入 $X$ 的入边，如果在原始图中从 $X$ 到 $Y$ 没有直接路径，则 $X$ 和 $Y$ 是 **d-分离（d-separated）** 的。

陈述 (ii) 可以通过反例证明（参见，例如，**例 6.34**）。

## C.6 命题 6.36 的证明（Proof of Proposition 6.36）

**证明**。"$\Rightarrow$"：假设**因果最小性（causal minimality）**不满足。则存在一个 $X _ { j }$ 和一个 $Y \in \mathbf { P A } _ { j } ^ { \mathcal { G } }$，使得 $P _ { \mathbf { X } }$ 相对于从 ${ \mathcal { G } }$ 中移除边 $Y \to X _ { j }$ 后得到的图也是马尔可夫的。这意味着根据局部马尔可夫性质，$X _ { j } \perp \perp Y | \mathbf { P A } _ { j } ^ { \mathcal { G } } \setminus \{ Y \}$。

"仅当"：如果 $P _ { \mathbf { X } }$ 具有密度，则马尔可夫条件等价于**马尔可夫分解（Markov factorization）**[Lauritzen, 1996, Theorem 3.27]。现在假设 $Y \in \mathbf { P A } _ { j } ^ { \mathcal { G } }$ 且 $X _ { j } \perp \perp Y | \mathbf { P A } _ { i } ^ { \mathcal { G } } \setminus \{ Y \}$，这意味着 $p ( x _ { j } | \mathbf { p } \mathbf { a } _ { j } ^ { \mathcal { G } } ) = p ( x _ { j } | \mathbf { p } \mathbf { a } _ { j , - Y } ^ { \mathcal { G } } )$，其中 $\mathbf { P A } _ { j , - Y } ^ { \mathcal { G } }$ 定义为 $\mathbf { P A } _ { j , - Y } ^ { \mathcal { G } } = \mathbf { P A } _ { j } ^ { \mathcal { G } } \setminus \{ Y \}$。那么，$p ( \mathbf { x } ) = p ( x _ { j } | \mathbf { p a } _ { i , - Y } ^ { \mathcal { G } } ) \prod _ { k \neq j } p ( x _ { k } | \mathbf { p a } _ { k } ^ { \mathcal { G } } )$，这意味着 $P _ { \mathbf { X } }$ 相对于没有 $Y \to X _ { j }$ 的 $\mathcal { G }$ 是马尔可夫的。

## C.7 命题 6.48 的证明（Proof of Proposition 6.48）

**证明**。我们假设两个模型都满足因果最小性，并且分别带有图 $\mathcal { G }$ 和 $\mathcal { H }$。直观上，我们可以识别一个节点 $X$ 的子节点，因为它们在干预 $X$ 后会发生变化。然而，由于例如两条路径相互抵消，某些子节点在干预后可能不会改变其分布。因此，我们引入以下符号。给定一个有向无环图（DAG）$\mathcal { G }$，如果 $X \in \mathbf { P A } _ { Y }$ 且 $X$ 不是 $Y$ 的任何其他父节点的祖先，则我们称 $X$ 为节点 $Y$ 的最年轻父节点，记为 $X \in \mathbf { Y } \mathbf { P } \mathbf { A } _ { Y }$。一个节点 $Y$ 可能有多个最年轻父节点。该证明需要两个论证：

(i) 如果 $X \in \mathbf { Y } \mathbf { P } \mathbf { A } _ { Y } ^ { \mathcal { G } }$，则存在从 $X$ 到 $Y$ 的总因果效应，即存在 $x ^ { \triangle }$ 和 $x ^ { \square }$，使得 $P _ { Y } ^ { d o \left( X : = x ^ { \triangle } \right) } \neq P _ { Y } ^ { d o \left( X : = x ^ { \triangle } \right) }$。这由因果最小性推出。

(ii) 如果 $Z \in \mathbf { A } \mathbf { N } _ { Y } ^ { \mathcal { G } }$，则存在 $X _ { 1 } , \ldots , X _ { k }$，使得 $X _ { 1 } = Z , X _ { k } = Y$，且对于 $i \in \{ 1 , . . . , k - 1 \}$，有 $X _ { i } \in \mathbf { Y P A } _ { X _ { i + 1 } } ^ { \mathcal { G } }$。

最后，我们可以结合这两个陈述，得出结论：如果 $Z \in \mathbf { A } \mathbf { N } _ { Y } ^ { \mathcal { G } }$，则存在 $X _ { 1 } , \ldots , X _ { k }$，使得对于 $i \in \left\{ 1 , \ldots , k - 1 \right\}$，$X _ { i }$ 对 $X _ { i + 1 }$ 具有总因果效应，这意味着在 $\mathcal { H }$ 中也必须存在从 $X _ { i }$ 到 $X _ { i + 1 }$ 的直接因果路径；参见**命题 6.13**。但此时 $Z \in \mathbf { A } \mathbf { N } _ { Y } ^ { \mathcal { H } }$，这意味着 $\mathcal { G }$ 和 $\mathcal { H }$ 具有相同的祖先关系。由于 $\mathcal { G }$ 和 $\mathcal { H }$ 都满足因果最小性，这意味着 $\mathcal { G } = \mathcal { H }$，因此这两个模型作为**因果图模型（causal graphical models）**是等价的。

## C.8 命题 6.49 的证明（Proof of Proposition 6.49）

**证明**。根据命题 6.3 的证明，对于第一个 **结构因果模型（Structural Causal Model, SCM）**，我们可以写出 $\mathbf { X } = \mathbf { g } ( \mathbf { N } )$ 。但由于

$$
\mathbf {g} (\mathbf {n}) = \mathbf {g} ^ {*} (\mathbf {n}) \quad \forall \mathbf {n} \text { with } p (\mathbf {n}) > 0,
$$

我们显然有这两个 SCM 诱导出相同的观测分布（以及具有相同参数的干预分布）。关于**反事实（counterfactuals）**，我们通过以 $P ( \mathbf { X } \in A ) > 0$ 的 $\mathbf { X } \in A$ 为条件，涵盖了离散和连续两种情况；参见定义 6.17。噪声变量上的新密度满足

$$
\begin{array}{l} \tilde {p} (n _ {1}, \ldots , n _ {d}) = \left\{ \begin{array}{c l} \frac {p (n _ {1} , \ldots , n _ {d})}{P (X \in A)} & \text { if } \mathbf {g} (n _ {1}, \ldots , n _ {d}) \in A \\ 0 & \text { else } \end{array} \right. \\ = \left\{ \begin{array}{c l} \frac {p (n _ {1} , \ldots , n _ {d})}{P (\mathbf {g} (\mathbf {N}) \in A)} & \text { if } \mathbf {g} ^ {*} (n _ {1}, \ldots , n _ {d}) \in A \\ 0 & \text { else } \end{array} \right. \\ = \left\{ \begin{array}{c l} \frac {p (n _ {1} , \ldots , n _ {d})}{P (\mathbf {g} ^ {*} (\mathbf {N}) \in A)} & \text { if } \mathbf {g} ^ {*} (n _ {1}, \ldots , n _ {d}) \in A \\ 0 & \text { else } \end{array} \right. \\ = \tilde {p} ^ {*} (n _ {1}, \dots , n _ {d}). \\ \end{array}
$$

我们仍然有

$$
\mathbf {g} (\mathbf {n}) = \mathbf {g} ^ {*} (\mathbf {n}) \quad \forall \mathbf {n} \text { with } \tilde {p} (\mathbf {n}) > 0,
$$

这意味着所有反事实陈述都是一致的。

## C.9 命题 7.1 的证明（Proof of Proposition 7.1）

**证明**。令 $N _ { 1 } , \ldots , N _ { d }$ 独立且在 0 和 1 之间均匀分布。然后我们定义 $X _ { j } : = f _ { j } ( X _ { \mathbf { P A } _ { j } } , N _ { j } )$ ，其中

$$
f _ {j} \left(\mathbf {p a} _ {j}, n _ {j}\right) := F _ {X _ {j} \mid \mathbf {P A} _ {j} = \mathbf {p a} _ {j}} ^ {- 1} \left(n _ {j}\right) \tag {C.4}
$$

$F _ { X _ { j } | \mathbf { P A } _ { j } = \mathbf { p a } _ { j } } ^ { - 1 }$ 是给定 $\mathbf { P A } _ { j } = \mathbf { \dot { p } } \mathbf { a } _ { j }$ 时 $X _ { j }$ 的**广义逆累积分布函数（generalized inverse cumulative distribution function）**。一个随机变量 Y 的广义逆累积分布函数定义为 $F _ { Y } ^ { - 1 } ( a ) : = \operatorname* { i n f } \{ y \in \mathbb { R } : F _ { Y } ( y ) \geq a \}$ 。方程 (C.4) 保证了在所构建的 SCM 中，条件分布 $X _ { j } | \mathbf { P } \mathbf { A } _ { j } = \mathbf { p } \mathbf { a } _ { j }$ 具有正确的分布。该结论随后由**马尔可夫分解（Markov factorization）**（定义 6.21(iii)）得出。

## C.10 命题 7.4 的证明（Proof of Proposition 7.4）

**证明**。假设**因果极小性（causal minimality）**不成立。那么我们可以找到节点 $j$ 和 $i \in \mathbf { P } \mathbf { A } _ { j }$ ，使得 $X _ { j } = f _ { j } ( \mathbf { P A } _ { i } \backslash \{ i \} , X _ { i } ) + N _ { j }$ 不依赖于 $X _ { i }$ ，如果我们将所有其他父节点 $A : = \mathbf { \bar { P } } \mathbf { A } _ { j } \backslash \{ i \}$ 作为条件，即 $X _ { j } \perp \perp X _ { i } | X _ { A }$ （见命题 6.36）。这里，我们将 $\mathbf { P A } _ { j } \backslash \{ X _ { i } \}$ 记为 $X _ { A }$ 。对于函数 $f _ { j }$ ，我们现在将证明，对于 $P _ { X _ { A } , X _ { i } - \mathrm { { a l m o s t } } }$ 所有的 $( x _ { A } , x _ { i } )$ ，有 $f _ { j } ( x _ { A } , x _ { i } ) = c _ { x _ { A } }$ 。实际上，不失一般性假设 $\mathbb { E } [ N _ { j } ] = 0$ ，则 $X _ { j } | \mathbf { P A } _ { j } = \left( x _ { A } , x _ { i } \right)$ 的均值等于 $f _ { j } ( x _ { A } , x _ { i } )$ 。Dawid [1979] 中的方程 (2b) 指出，如果 $X _ { j } \perp \perp X _ { i } | X _ { A }$ ，则 $X _ { j } | X _ { A } , X _ { i }$ 的密度不依赖于 $X _ { i }$ 的参数。因此，条件均值 $f _ { j } ( x _ { A } , x _ { i } )$ 也不依赖于 xi。由此可得 $f _ { j } ( x _ { A } , x _ { i } ) = c _ { x _ { A } }$ 。$f _ { j }$ 的连续性意味着 $f _ { j }$ 在其最后一个参数上是常数。

反之，该结论也由命题 6.36 得出。

## C.11 命题 8.1 的证明（Proof of Proposition 8.1）

**证明**。我们使用**贝尔曼最优方程（Bellman optimality equation）**[例如，Sutton and Barto, 2015, 第 3.8 章]。对于所有满足 $f ( s ^ { \circ } ) = f ( s )$ 的 $s ^ { \circ }$ 和 s，我们有

$$
\begin{array}{l} Q ^ {*} (s, a) = \sum_ {s ^ {\prime}} p (s ^ {\prime} | s, a) \left(\mathbb {E} [ R | s ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) \\ = \sum_ {f ^ {\prime}} \sum_ {s ^ {\prime}: f (s ^ {\prime}) = f ^ {\prime}} p (s ^ {\prime} \mid s, a) \left(\mathbb {E} [ R \mid s ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) \\ = \sum_ {f ^ {\prime}} p (f ^ {\prime} \mid s, a) \left(\mathbb {E} [ R \mid f ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) \\ = \sum_ {f ^ {\prime}} p (f ^ {\prime} | s ^ {\circ}, a) \left(\mathbb {E} [ R | f ^ {\prime}, a ] + \max _ {a ^ {\prime}} Q ^ {*} (s ^ {\prime}, a ^ {\prime})\right) = Q ^ {*} (s ^ {\circ}, a). \\ \end{array}
$$

证明到此结束。

![image_64](images/image_64.png)

## C.12 命题 8.2 的证明（Proof of Proposition 8.2）

**证明**。第一个方程来自第 8.2.1 节的讨论。**马尔可夫分解（Markov factorization）**性质意味着

$$
p (\mathbf {x}) = p (a | s)   p (s | h)   p (h)   p (y | f, h)   p (f | a);
$$

见图 8.5。现在由 $F \perp \perp S | A$ 可得

$$
\begin{array}{l} \int y \frac {\tilde {p} (a | s)}{p (a | s)} p (\mathbf {x}) d \mathbf {x} = \int y \tilde {p} (a | s) p (s | h) p (h) p (y | f, h) p (f | a, s) d a d f d h d s d y \\ = \int y \tilde {p} (f, a | s) p (s | h) p (h) p (y | f, h) d a d f d h d s d y \\ = \int y \frac {\tilde {p} (f | s)}{p (f | s)} p (s | h) p (h) p (y | f, h) p (f | s) d f d h d s d y \\ = \int y \frac {\tilde {p} (f | s)}{p (f | s)} p (s | h) p (h) p (y | f, h) p (f, a | s) d a d f d h d s d y \\ = \int y \frac {\tilde {p} (f | s)}{p (f | s)} p (\mathbf {x}) d \mathbf {x}. \\ \end{array}
$$

最后一个等式由 $p ( f , a | s ) = p ( f | a , s ) p ( a | s )$ 得出。

![image_65](images/image_65.png)

## C.13 命题 9.3 的证明（Proof of Proposition 9.3）

**证明**。为了证明 (i)，我们从 X 上的 SCM C 及其蕴含的分布 $R _ { \mathbf { X } }$ 开始。然后，我们考虑变量 $O \in { \mathbf { 0 } }$ 的结构赋值，并反复代入变量 $X \in \mathbf { X } \backslash \mathbf { o }$ 的赋值（只要这些变量出现在右侧）。这产生了一个新的 SCM，其中每个 $O \in { \mathbf { 0 } }$ 的结构赋值都包含一个多元误差变量 $\tilde { \mathbf { N } } _ { O }$ 。很明显，这个较小的 SCM 蕴含了相同的观测分布 $P _ { \mathbf { 0 } }$，并且在干预任何 $O \in { \mathbf { 0 } }$ 时蕴含了相同的干预分布。由**因果充分性（causal sufficiency）**可知，新的噪声变量 $( \tilde { \mathbf { N } } _ { O } ) _ { O \in \mathbf { O } }$ 是联合独立的。与一维噪声变量的情况（命题 6.31）一样，这再次意味着分布 $P _ { \mathbf { 0 } }$ 相对于诱导出的图结构是**马尔可夫的（Markovian）**。该结论随后由以下事实得出：这个新的 SCM 可以转化为一个具有一维误差变量的 SCM，该 SCM 蕴含相同的观测分布和干预分布（利用与命题 7.1 相同的构造）。关于此过程的更正式描述以及关于这些论点的更多细节，请参见 Bongers et al. [2016]。

陈述 (ii) 由例 9.2 得出。

## C.14 定理 10.3 的证明（Proof of Theorem 10.3）

**证明**。如果存在一条从 $X _ { \mathrm { p a s t } ( t ) } ^ { j }$ 到 $X _ { t } ^ { k }$ 的箭头，则依赖性 (10.3) 立即由**忠实性（faithfulness）**得出，因为两个直接相连的变量不能被 d-分离。现在假设不存在从 $X _ { \mathrm { p a s t } ( t ) } ^ { j }$ 到 $X _ { t } ^ { k }$ 的边。那么，给定 $\mathbf { X } _ { \mathrm { p a s t } ( t ) } ^ { - j }$ (t) 时，$X _ { t } ^ { k }$ 与 $X _ { \mathrm { p a s t } ( t ) } ^ { j }$ 是 $d-$ 分离的。任何从 $X _ { t } ^ { k }$ 出发带有出边的路径都会被阻断，因为它会包含一个碰撞节点（并且之后没有时间索引大于或等于 t 的节点被作为条件）；任何进入 $X _ { t } ^ { k }$ 的带有入边的路径都会被阻断，因为路径上的下一个节点在条件集 $\mathbf { X } _ { \mathrm { p a s t } ( t ) } ^ { - j }$ 中。

## C.15 定理 10.4 的证明（Proof of Theorem 10.4）

**证明**。为了证明 (i)，考虑一个不包含从 X 到 $Y$ 的箭头的完整时间图。那么，从 $Y _ { t }$ 到 $X _ { \mathrm { p a s t } ( t ) }$ 的每条路径都被 $Y _ { \mathrm { p a s t } ( t ) }$ 阻断。任何以从 $Y _ { t }$ 出发的出边开始的路径必须包含一个不在条件集中的碰撞节点（其任何后代也不在条件集中）；任何以入边开始的路径都被阻断，因为该路径上的第一个节点在 Ypast(t) 中。$Y _ { \mathrm { p a s t } ( t ) }$

为了证明 (ii)，假设 $Y _ { t }$ 有来自 X 的父节点，记为 $\mathbf { P A } _ { Y _ { t } } ^ { X }$ 。那么 (10.5) 意味着

$$
Y _ {t} \perp \perp \mathbf {P A} _ {Y _ {t}} ^ {X} \mid Y _ {\text { past } (t)}. \tag {C.5}
$$

对于任何 $X _ { s } \in \mathbf { P } \mathbf { A } _ { Y _ { t } } ^ { X }$ ，(C.5) 通过**弱并（weak union）**性质（见附录 A.1）意味着

$$
Y _ {t} \perp X _ {s} \mid Y _ {\text { past } (t)} \cup (\mathbf {P A} _ {Y _ {t}} ^ {X} \setminus \{X _ {s} \}). \tag {C.6}
$$

根据 Peters et al. [2014, Lemma 38]，极小性意味着 $Y _ { t }$ 依赖于 $Y _ { t }$ 的任意父节点 A，给定包含除 A 以外的 $Y _ { t }$ 的其他父节点的 $Y _ { t }$ 的任何非后代集合。因此我们有

$$
Y _ {t} \not \perp X _ {s} \mid Y _ {\text { past } (t)} \cup (\mathbf {P A} _ {Y _ {t}} ^ {X} \setminus \{X _ {s} \}),
$$

这与 (C.6) 矛盾。

<!-- footnote -->

- 为了方便表示，我们写作 $H ( X _ { j _ { 1 } } , \dots , X _ { j _ { k } } )$ 而不是 $H \big ( ( X _ { j _ { 1 } } , \dots , X _ { j _ { k } } ) \big )$，并再次对向量进行集合运算。

<!-- footnote end -->

<!-- footnote -->

- 严格来说，我们目前仅针对有限多个节点引入了**因果有向无环图（causal DAG）**。然而，这里我们需要无限图，并忽略这一技术细节 [例如，参见 Peters et al., 2013]。

<!-- footnote end -->