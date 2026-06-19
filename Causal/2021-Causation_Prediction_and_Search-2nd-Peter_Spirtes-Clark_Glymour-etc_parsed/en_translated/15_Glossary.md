# 术语表（Glossary）

**A**：在图 $G$ 中，设 $\mathbf{A}(A, B)$ 为 $A$ 或 $B$ 的**祖先（ancestors）**的并集。

**可接受的（Acceptable）**：设图 $G'$ 中变量的全序 Ord 对 $G$ 是**可接受的**，当且仅当只要 $A \ne B$ 且在 $G'$ 中存在从 $A$ 到 $B$ 的有向路径，则 $A$ 在 Ord 中先于 $B$。

**之后（After）**：在图 $G$ 中，顶点 $X$ 在顶点 $Y$ **之后**，当且仅当在 $G$ 中存在从 $Y$ 到 $X$ 的有向路径。

**几乎纯净的（Almost Pure）**：如果测量变量中唯一的杂质类型是**共同原因杂质（common cause impurities）**，则称测量模型是**几乎纯净的**。一个**几乎纯净的潜变量图（almost pure latent variable graph）**是指其测量模型是几乎纯净的图。

**之前（Before）**：在图 $G$ 中，顶点 $X$ 在顶点 $Y$ **之前**，当且仅当在 $G$ 中存在从 $X$ 到 $Y$ 的有向路径。

**C.F**：见**常数因子（constant factor）**。

**瓶颈点（Choke point）**：在有向无环图 $G$ 中，如果对于 $\mathbf{T}(K, L)$ 中的所有 $T(K, L)$ 和 $\mathbf{T}(I, J)$ 中的所有 $T(I, J)$，$L(T(K, L))$ 和 $J(T(I, J))$ 相交于顶点 $Q$，则 $Q$ 是一个 $LJ(T(I, J), T(K, L))$ **瓶颈点**。类似地，如果对于 $\mathbf{T}(K, L)$ 中的所有 $T(K, L)$ 和 $\mathbf{T}(I, J)$ 中的所有 $T(I, J)$，$L(T(K, L))$ 和所有 $J(T(I, J))$ 相交于顶点 $Q$，并且对于 $\mathbf{T}(I, L)$ 中的所有 $T(I, L)$ 和 $\mathbf{T}(J, K)$ 中的所有 $T(J, K)$，$L(T(I, L))$ 和 $\mathcal{J}(T(J, K))$ 也相交于 $Q$，则 $Q$ 是一个 $LJ(T(I, J), T(K, L), T(I, L), T(J, K))$ **瓶颈点**。另见**路径（trek）**的定义。

**组合图（Combined graph）**：见**操作（manipulation）**。

**常数因子（Constant factor）**：在 LCF 或 LCT $T$ 中，如果一个表达式等于 $ce$，其中 $c$ 是非零常数，$e$ 是方程系数（equation coefficients）的乘积且各系数均为正整指数，则 $c$ 是 $ce$ 的**常数因子（c.f.）**。

**包含（Contains）**：在有向无环图中，有向路径 $R(U, I)$ 和 $R(U, J)$ **包含**路径 $T$，当且仅当 $I(T(I, J))$ 是 $R(U, I)$ 的**终段（final segment）**，且 $\mathcal{J}(T(I, J))$ 是 $R(U, J)$ 的**终段**。

**D**：给定有向无环图 $G$，$\mathbf{D}(X_i, X_j)$ 是从 $X_i$ 到 $X_j$ 的所有有向路径的集合。

**D-连接（D-connection）**：见**D-分离（D-separation）**。

**确定判别路径（Definite discriminating path）**：在**部分定向诱导路径图（partially oriented inducing path graph）**中，$U$ 是 $B$ 的**确定判别路径**，当且仅当 $U$ 是 $X$ 和 $Y$ 之间的无向路径，包含 $B$，$B \ne X$，$B \ne Y$，$U$ 上除 $B$ 和端点外的每个顶点都是 $U$ 上的**碰撞器（collider）**或**确定非碰撞器（definite noncollider）**，并且：
- (i) 如果 $V$ 和 $V'$ 在 $U$ 上相邻，且 $V$ 在 $U$ 上位于 $V$ 和 $B$ 之间，则在 $U$ 上 $V * \rightarrow V'$，
- (ii) 如果 $V$ 在 $U$ 上位于 $X$ 和 $B$ 之间，且 $V$ 是 $U$ 上的碰撞器，则在图中 $V \rightarrow Y$，否则在图中 $V * \rightarrow Y$，
- (iii) 如果 $V$ 在 $U$ 上位于 $Y$ 和 $B$ 之间，且 $V$ 是 $U$ 上的碰撞器，则在图中 $V \rightarrow X$，否则在图中 $V * \rightarrow X$，
- (iv) $X$ 和 $Y$ 在图中不相邻。

**确定非碰撞器（Definite noncollider）**：顶点 $B$ 是无向路径 $U$ 上的**确定非碰撞器**，当且仅当 $B$ 是 $U$ 的端点，或者存在顶点 $A$ 和 $C$ 使得 $U$ 包含以下子路径之一：$A \leftarrow B * \rightarrow C$、$A * \rightarrow B \leftarrow C$ 或 $A * \rightarrow B * \rightarrow C$。

**确定非后代（Definite nondescendant）**：如果图是 $G$ 在 $O$ 上的 FCI **部分定向诱导路径图（partially oriented inducing path graph）**，则 $X$ 在 **Definite-Nondescendants(Y)** 中，当且仅当在图中不存在从 $Y$ 的任何成员到 $X$ 的**半有向路径（semidirected path）**。

**确定-SP（Definite-SP）**：对于 $O$ 上的部分定向诱导路径图和可接受序 Ord，$V$ 在 **Definite-SP(Ord, X)** 中，当且仅当 $V \ne X$ 且在图中存在 $V$ 和 $X$ 之间的无向路径 $U$，使得 $U$ 上除 $X$ 外的每个顶点都是 Ord 中 $X$ 的前驱，且 $U$ 上除端点外的每个顶点都是 $U$ 上的碰撞器。

**依赖的（Dependent）**：在 LCT 或 LCF $S$ 中，变量 $X_i$ 是**依赖的**，当且仅当 $X_i$ 的入度（indegree）不为零。

**Det**：**Det(Z)** 是由 $Z$ 的任何子集确定的变量的集合。

**确定（Determines）**：变量集 $Z$ **确定**变量集 $A$，当 $A$ 中的每个变量都是 $Z$ 中变量的确定性函数，且 $A$ 中并非每个变量都是 $Z$ 的任何真子集的确定性函数。

**Det-连接（Det-connected）**：见**Det-分离（Det-separation）**。

**Det-分离的（Det-separated）**：如果 $G$ 是 $V$ 上的有向无环图，$Z$ 是不包含 $X$ 或 $Y$ 的 $V$ 的子集，且 $X \ne Y$，则 $X$ 和 $Y$ 在给定 $Z$ 和 **Deterministic(V)** 下是 **det-分离的**，当且仅当在某个相对于 Deterministic(V) 和 $Z$ 的 Mod(G) 中，$X$ 和 $Y$ 在给定 $\mathbf{Z} \cup \mathbf{Det}(\mathbf{Z})$ 下是 d-分离的，或者 $X$ 或 $Y$ 在 Det(Z) 中；否则如果 $X \ne Y$ 且 $X$ 和 $Y$ 不在 $Z$ 中，则 $X$ 和 $Y$ 在给定 $Z$ 和 Deterministic(V) 下是 **det-连接的**。如果 $X$、$Y$ 和 $Z$ 是 $V$ 中不相交的变量集，且 $X$ 和 $Y$ 非空，则 $X$ 和 $Y$ 在给定 $Z$ 下是 **det-分离的**，当且仅当 $X$ 的每个成员 $X$ 和 $Y$ 的每个成员 $Y$ 在给定 $Z$ 下是 det-分离的；否则如果 $X$、$Y$ 和 $Z$ 是 $V$ 中不相交的变量集，且 $X$ 和 $Y$ 非空，则 $X$ 和 $Y$ 在给定 $Z$ 和 Deterministic(V) 下是 **det-连接的**。

**判别路径（Discriminating path）**：在诱导路径图 $G$ 中，$U$ 是 $B$ 的**判别路径**，当且仅当 $U$ 是 $X$ 和 $Y$ 之间的无向路径，包含 $B$，$B \ne X$，$B \ne Y$，并且：
- (i) 如果 $V$ 和 $V'$ 在 $U$ 上相邻，且 $V$ 在 $U$ 上位于 $V$ 和 $B$ 之间，则在 $U$ 上 $V * \rightarrow V'$，
- (ii) 如果 $V$ 在 $U$ 上位于 $X$ 和 $B$ 之间，且 $V$ 是 $U$ 上的碰撞器，则在 $G'$ 中 $V \rightarrow Y$，否则在 $G'$ 中 $V * \rightarrow Y$，
- (iii) 如果 $V$ 在 $U$ 上位于 $Y$ 和 $B$ 之间，且 $V$ 是 $U$ 上的碰撞器，则在 $G'$ 中 $V \rightarrow X$，否则在 $G'$ 中 $V * \rightarrow X$，
- (iv) $X$ 和 $Y$ 在 $G'$ 中不相邻。

**展开形式（Distributed form）**：表达式或方程 $E$ 的**展开形式**是执行 $E$ 中所有乘法但不执行加法、减法或除法的结果。如果方程中没有除法，则其展开形式是项的和。例如，方程 $u = (a + b)(c + d)\nu$ 的展开形式是 $u = ac\nu + ad\nu + bc\nu + bd\nu$。

**D-图（D-map）**：$V$ 上的无环图 $G$ 是概率分布 $P(V)$ 的 **D-图**，当且仅当对于 $V$ 中任意不相交的随机变量集 $X$、$Y$ 和 $Z$，如果在 $G$ 中 $X$ 在给定 $Z$ 下不与 $Y$ d-分离，则在 $P(V)$ 中 $X$ 在给定 $Z$ 下不独立于 $Y$。然而，当 D-map 应用于 LCT 中的图时，定义中的量词仅适用于非误差变量集。

**D-Sep**：如果 $G'$ 是 $O$ 上的诱导路径图且 $A \ne B$，则 $V \in \mathbf{D-SEP}(A, B)$ 当且仅当 $A \ne V$ 且存在 $A$ 和 $V$ 之间的无向路径 $U$，使得 $U$ 上的每个顶点都是 $A$ 或 $B$ 的祖先，并且（除端点外）都是 $U$ 上的碰撞器。

**D-分离的（D-separated）**：如果 $G$ 是顶点集为 $V$ 的有向无环图，$Z$ 是不包含 $X$ 或 $Y$ 的顶点集，$X \ne Y$，且 $X$ 和 $Y$ 不在 $Z$ 中，则 $X$ 和 $Y$ 在给定 $Z$ 和 **Deterministic(V)** 下是 **D-分离的**，当且仅当在 $G$ 中不存在 $X$ 和 $Y$ 之间的无向路径 $U$，使得 $U$ 上的每个碰撞器在 $Z$ 中有一个后代，且 $U$ 上没有其他顶点在 Det(Z) 中；否则如果 $X \ne Y$ 且 $X$ 和 $Y$ 不在 $Z$ 中，则 $X$ 和 $Y$ 在给定 $Z$ 和 Deterministic(V) 下是 **D-连接的**。类似地，如果 $X$、$Y$ 和 $Z$ 是不相交的变量集，且 $X$ 和 $Y$ 非空，则 $X$ 和 $Y$ 在给定 $Z$ 和 Deterministic(V) 下是 **D-分离的**，当且仅当 $X$ 和 $Y$ 的笛卡尔积中的每一对 $\langle X, Y \rangle$ 在给定 $Z$ 和 Deterministic(V) 下是 D-分离的；否则如果 $X$、$Y$ 和 $Z$ 不相交，且 $X$ 和 $Y$ 非空，则 $X$ 和 $Y$ 在给定 $Z$ 和 Deterministic(V) 下是 **D-连接的**。（注意，这与以小写字母 "d" 开头的 **d-分离（d-separation）** 和 **d-连接（d-connection）** 不同。）

**e**：在 LCF $F$ 中，$e(S)$ 等于 $S$（如果 $S$ 是独立变量），或者等于进入 $S$ 的误差变量（如果 $S$ 不是独立变量）。

**E**：如果 $X$ 是随机变量，$E(X)$ 是 $X$ 的期望值。

**Equiv(G')**：如果 $G'$ 是 $O$ 上的诱导路径图，**Equiv(G')** 是在相同顶点上、具有与 $G'$ 相同 d-连接的诱导路径图的集合。

**E.C.F**：见**方程系数因子（equation coefficient factor）**。

**方程系数（Equation coefficient）**：见**线性因果理论（linear causal theory）**、**线性因果形式（linear causal form）**。

**方程系数因子（Equation coefficient factor）**：在 LCF 或 LCT $T$ 中，如果一个表达式等于 $ce$，其中 $c$ 是非零常数，$e$ 是方程系数的乘积且各系数均为正整指数，则 $e$ 是 $ce$ 的**方程系数因子（e.c.f.）**。

**等价于多项式（Equivalent to a polynomial）**：在 LCF 中，一个量（例如协方差）$X$ **等价于系数和外生变量方差的多项式**，当且仅当对于每个 LCF $F = \langle \langle \mathbf{R}, \mathbf{M}, \mathbf{E} \rangle, \mathbf{C}, \mathbf{V}, \mathbf{EQ}, \mathbf{L}, \mathbf{Err} \rangle$ 和每个作为 $F$ 实例的 LCT $S = \langle \langle \mathbf{R}', \mathbf{M}', \mathbf{E}' \rangle, (\Omega, f, P), \mathbf{EQ}, \mathbf{L}, \mathbf{Err} \rangle$，存在一个在 $\mathbf{C}$ 和 $\mathbf{V}$ 中变量的多项式，使得 $X$ 等于将 $S$ 的线性系数作为 $\mathbf{C}$ 中对应变量的值、并将 $S$ 中外生变量的方差作为 $\mathbf{V}$ 中对应变量的值代入后的结果。

**误差变量（Error variable）**：见**线性因果理论（linear causal theory）**、**线性因果形式（linear causal form）**。

**外生的（Exogenous）**：如果 $G$ 是变量集 $\mathbf{V} \cup \mathbf{W}$ 上的有向无环图，且 $\mathbf{V} \cap \mathbf{W} = \emptyset$，则在 $G$ 中 $\mathbf{W}$ 相对于 $\mathbf{V}$ 是**外生的**，当且仅当不存在从 $\mathbf{V}$ 的任何成员到 $\mathbf{W}$ 的任何成员的有向边。

**忠实不可区分（Faithfully indistinguishable）**：我们说两个有向无环图 $G$ 和 $G'$ 是**忠实不可区分（f.i.）**的，当且仅当每个忠实于 $G$ 的分布也忠实于 $G'$，反之亦然。

**F.I.**：见**忠实不可区分（faithfully indistinguishable）**。

**终段（Final segment）**：在图 $G$ 中，长度为 $n$ 的路径 $U$ 是长度为 $m$ 的路径 $V$ 的**终段**，当且仅当 $m \ge n$，且对于 $1 \le i \le n+1$，$V$ 的第 $i$ 个顶点等于 $U$ 的第 $(m-n+i)$ 个顶点。

**I-图（I-map）**：$V$ 上的无环有向图 $G$ 是概率分布 $P(V)$ 的 **I-图**，当且仅当对于 $V$ 中任意不相交的随机变量集 $X$、$Y$ 和 $Z$，如果在 $G$ 中 $X$ 在给定 $Z$ 下与 $Y$ d-分离，则在 $P(V)$ 中 $X$ 在给定 $Z$ 下独立于 $Y$。然而，当 I-map 应用于 LCT 中的图时，定义中的量词仅适用于非误差变量集。

**Ind**：对于有向无环图 $G$，**Ind** 是 $G$ 中独立变量的集合。

**$^{Ind}a_{IJ}$**：$^{Ind}a_{IJ}$ 是 $I$ 的**独立方程（independent equational）**中 $J$ 的系数。另见**独立方程**。

**独立的（Independent）**：在 LCT 或 LCF $S$ 中，变量 $X_i$ 是**独立的**，当且仅当 $X_i$ 的入度为零（即没有指向它的边）。注意，独立性的属性与统计独立性的关系完全不同。上下文将明确该术语的使用含义。

**独立方程（Independent equational）**：在 LCF $\langle \langle \mathbf{R}, \mathbf{M}, \mathbf{E} \rangle, \mathbf{C}, \mathbf{V}, \mathbf{EQ}, \mathbf{L}, \mathbf{S} \rangle$ 中，一个方程是因变量 $X_j$ 的**独立方程**，当且仅当它由 EQ 蕴含，且出现在右端的 $\mathbf{R}$ 中的变量是独立的且最多出现一次。

**诱导路径（Inducing path）**：如果 $G$ 是变量集 $V$ 上的有向无环图，$O$ 是包含 $A$ 和 $B$ 的 $V$ 的子集，且 $A \ne B$，则 $A$ 和 $B$ 之间的无向路径 $U$ 是相对于 $O$ 的**诱导路径**，当且仅当 $U$ 上除端点外的每个 $O$ 中的成员都是 $U$ 上的碰撞器，且 $U$ 上的每个碰撞器都是 $A$ 或 $B$ 的祖先。我们有时将 $O$ 中的成员称为**观测变量（observed variables）**。

**诱导路径图（Inducing path graph）**：$G'$ 是有向无环图 $G$ 在 $O$ 上的**诱导路径图**，当且仅当 $O$ 是 $G$ 中顶点的子集，变量 $A$ 和 $B$ 之间存在一条在 $A$ 端带有箭头标记的边，当且仅当 $A$ 和 $B$ 在 $O$ 中，且在 $G$ 中存在 $A$ 和 $B$ 之间相对于 $O$ 的、指向 $A$ 的诱导路径。（使用第 2 章的符号，诱导路径图中的标记集为 $\{>, \circ\}$。）

**初始段（Initial segment）**：在图 $G$ 中，长度为 $n$ 的路径 $U$ 是长度为 $m$ 的路径 $V$ 的**初始段**，当且仅当 $m \ge n$，且对于 $1 \le i \le n+1$，$V$ 的第 $i$ 个顶点等于 $U$ 的第 $i$ 个顶点。

**指向（Into）**：在图 $G$ 中，$A$ 和 $B$ 之间的边**指向** $A$，当且仅当在 $A$ 端的标记是 $\rightarrow$。如果 $A$ 和 $B$ 之间的无向路径 $U$ 包含一条指向 $A$ 的边，我们称 $U$ **指向** $A$。

**不变的（Invariant）**：如果 $G$ 是变量集 $\mathbf{V} \cup \mathbf{W}$ 上的有向无环图，$\mathbf{W}$ 在 $G$ 中相对于 $\mathbf{V}$ 是外生的，$\mathbf{Y}$ 和 $\mathbf{Z}$ 是 $\mathbf{V}$ 的不相交子集，$P(\mathbf{V} \cup \mathbf{W})$ 是满足 $G$ 的马尔可夫条件（Markov condition）的分布，且 **Manipulated(W)** = $\mathbf{X}$，则 $P(\mathbf{Y} | \mathbf{Z})$ 在 $G$ 中通过将 $\mathbf{W}$ 从 $\mathbf{w}_1$ 改变为 $\mathbf{w}_2$ 而对 $\mathbf{X}$ 的直接操作下是**不变的**，当且仅当 $P(\mathbf{Y} | \mathbf{Z}, \mathbf{W} = \mathbf{w}_1) = P(\mathbf{Y} | \mathbf{Z}, \mathbf{W} = \mathbf{w}_2)$ 在两者都有定义的地方成立。

**实例（Instance）**：LCT $S$ 是 LCF $F$ 的**实例**，当且仅当 $S$ 的图与 $F$ 的图同构。

**IP**：在有向无环图 $G$ 中，如果 $\mathbf{Y} \cap \mathbf{Z} = \emptyset$，则 $W$ 在 **IP(Y, Z)** 中（$W$ 有一个父节点是给定 $\mathbf{Z}$ 下 $\mathbf{Y}$ 的信息变量），当且仅当 $W$ 是 $\mathbf{Z}$ 的成员，且 $W$ 在 $\mathbf{IV}(\mathbf{Y}, \mathbf{Z}) \cup \mathbf{Y}$ 中有一个父节点。

**IV**：在有向无环图 $G$ 中，如果 $\mathbf{Y} \cap \mathbf{Z} = \emptyset$，则 $V$ 在 **IV(Y, Z)** 中（给定 $\mathbf{Z}$ 下 $\mathbf{Y}$ 的信息变量），当且仅当 $V$ 在给定 $\mathbf{Z}$ 下与 $\mathbf{Y}$ d-连接，且 $V$ 不在 **ND(YZ)** 中。（这意味着 $V$ 不在 $\mathbf{Y} \cup \mathbf{Z}$ 中。）

**标签（Label）**：见**线性因果理论（linear causal theory）**、**线性因果形式（linear causal form）**。

**长度（Length）**：在图 $G$ 中，路径的**长度**等于路径中顶点数减一。

**最后交点（Last point of intersection）**：在有向无环图 $G$ 中，有向路径 $R(U, I)$ 与有向路径 $R(V, J)$ 的**最后交点**是 $R(U, I)$ 上同时也是 $R(V, J)$ 上的最后一个顶点。注意，如果 $G$ 是有向无环图，则 $R(U, I)$ 与 $R(V, J)$ 的最后交点等于 $R(V, J)$ 与 $R(U, I)$ 的最后交点；这对于有向环路径不成立。

**LCF**：见**线性因果形式（linear causal form）**。

**LCT**：见**线性因果理论（linear causal theory）**。

**线性因果形式（Linear causal form）**：**线性因果形式**是一种未估计的 LCT，其中线性系数和外生变量的方差是实变量而非常数。这意味着 LCF 中的边标签是实变量而非常数（除了从误差变量出发的边的标签固定为 1）。更形式化地，设**线性因果形式（LCF）**为 $\langle \langle \mathbf{R}, \mathbf{M}, \mathbf{E} \rangle, \mathbf{C}, \mathbf{V}, \mathbf{EQ}, \mathbf{L}, \mathbf{Err} \rangle$。

(i) $< \mathbf { R , M , E } >$ 是一个**有向无环图（directed acyclic graph）**。Err 是 R 的一个子集，称为**误差变量（error variables）**。每个误差变量的入度为 0，出度为 1。对于 R 中入度 $\neq 0$ 的每个 $X _ { i }$，恰好存在一个误差变量，其有一条边指向 $X _ { i }$。

- (ii) $c _ { i j }$ 是一个与从 $X _ { j }$ 到 $X _ { i }$ 的边相关联的唯一实变量，C 是 $c _ { i j }$ 的集合。V 是变量 $\boldsymbol { \sigma } _ { i } ^ { 2 }$ 的集合，其中 $X _ { i }$ 是 $< \mathbf { R , M , E } >$ 中的一个**外生变量（exogenous variable）**，而 $\boldsymbol { \sigma } _ { i } ^ { 2 }$ 是一个取值范围为正实数的变量。
- (iii) L 是一个定义域为 E 的函数，对于 E 中的每个 e，当且仅当 $h e a d ( e ) = X _ { i }$ 且 $t a i l ( e ) = X _ { j }$ 时，$L ( e ) = c _ { i j }$。$L ( e )$ 将被称为 e 的**标签（label）**。推而广之，任何无环无向路径 U 中各边标签的乘积将记为 $L ( U )$，并且 $L ( U )$ 将被称为 U 的标签。空路径的标签固定为 1。
- (iv) EQ 是关于 R 中变量的一组一致的独立齐次线性方程。对于 R 中入度为正的每个 $X _ { i }$，EQ 中存在一个如下形式的方程：

$$
X _ {i} = \sum_ {X _ {j} \in \mathbf {P a r e n t s} (X _ {i})} c _ {i j} X _ {j}
$$

其中每个 $c _ { i j }$ 是 C 中的一个实变量，每个 $X _ { i }$ 属于 R。EQ 中不存在其他方程。$c _ { i j }$ 是 $X _ { i }$ 的方程中 $X _ { j }$ 的**方程系数（equation coefficient）**。

**线性因果理论（Linear causal theory）**：令一个线性因果理论（LCT）为 $< < \mathbf { R , M , E } >$ ，$( \Omega , f , P )$ ，EQ,L,Err>，其中：

- (i) $( \Omega , f , P )$ 是一个**概率空间（probability space）**，其中 $\Omega$ 是**样本空间（sample space）**，f 是 $\Omega$ 上的一个**西格玛域（sigma-field）**，P 是 f 上的一个**概率分布（probability distribution）**。
- (ii) $< \mathbf { R , M , E } >$ 是一个有向无环图。R 是 $( \Omega , f , P )$ 上的一组**随机变量（random variables）**。
- (iii) R 中的变量具有**联合分布（joint distribution）**。R 中的每个变量都具有非零方差。E 是 R 中变量之间的一组有向边。（M 是出现在有向图中的标记集合，即 $\{ \mathrm { E M } , > \}$。）
- (iv) EQ 是关于 R 中随机变量的一组一致的独立齐次线性方程。对于 R 中入度为正的每个 $X _ { i }$，EQ 中存在一个如下形式的方程：

$$
X _ {i} = \sum_ {X _ {j} \in \mathbf {P a r e n t s} (X _ {i})} a _ {i j} X _ {j}
$$

其中每个 $a _ { i j }$ 是一个非零实数，每个 $X _ { i }$ 属于 R。这意味着 R 中入度为正的每个顶点 $X _ { i }$ 都可以表示为其所有且仅为其**父节点（parents）**的线性函数。EQ 中不存在其他方程。$a _ { i j }$ 的非零值是 $X _ { i }$ 的方程中 $X _ { j }$ 的方程系数。

- (v) 如果顶点（随机变量）$X _ { i }$ 和 $X _ { j }$ 是外生的，则 $X _ { i }$ 和 $X _ { j }$ 是**两两统计独立（pairwise statistically independent）**的。
- (vi) L 是一个定义域为 E 的函数，对于 E 中的每个 e，当且仅当 $h e a d ( e ) = X _ { i }$ 且 $t a i l ( e ) = X _ { j }$ 时，$L ( e ) = a _ { i j }$。$L ( e )$ 将被称为 e 的标签。推而广之，任何无环无向路径 U 中各边标签的乘积将记为 $L ( U )$，并且 $L ( U )$ 将被称为 U 的标签。空路径的标签固定为 1。
- (vii) 存在 R 的一个子集 S，称为误差变量，每个误差变量的入度为 0，出度为 1。注意，任何**内生变量（endogenous variable）** I 在给定任何不包含 I 的误差变量的变量集时的条件方差不为零。

**线性表示（Linear Representation）**：一个定义在 V 上的有向无环图 G **线性表示**一个分布 P(V)，当且仅当存在一个定义在 $\mathbf { V } ^ { \prime }$ 上的有向无环图 $G ^ { \prime }$ 和一个分布 $P ^ { \prime \prime } ( \mathbf { V } ^ { \prime } )$，使得：

- (i) V 包含于 $\mathbf { V ^ { \prime } }$；
- (ii) 对于 V 中的每个内生（即入度为正）变量 X，在 $\mathbf { V } ^ { \pmb { \eta } } \mathbf { W }$ 中存在一个唯一的变量 $\varepsilon _ { X }$，其入度为 0，方差为正，出度为 1，并且存在一条从 $\varepsilon _ { X }$ 到 X 的有向边；
- (iii) G 是 $G ^ { \prime }$ 在 V 上的**子图（subgraph）**；
- (iv) G 中的每个内生变量是其父节点在 $G ^ { \prime }$ 中的线性函数；
- (v) 在 $P ^ { \prime \prime } ( \mathbf { V } ^ { \prime } )$ 中，$G ^ { \prime }$ 中任意两个外生变量之间的相关性为零；
- (vi) $P ( \mathbf { V } )$ 是 $P ^ { \prime \prime } ( \mathbf { V } ^ { \prime } )$ 在 V 上的**边缘分布（marginal）**。

V \ V 中的成员称为**误差变量**，我们称 $G ^ { \prime }$ 为**扩展图（expanded graph）**。

**线性蕴含（Linearly implies）**：一个有向无环图 G **线性蕴含** $\rho _ { A B . \mathbf { H } } = 0$，当且仅当在所有由 G 线性表示的分布中，$\rho _ { A B . \mathbf { H } } = 0$。（我们假设所有**偏相关系数（partial correlations）**对于该分布都是已定义的。）

**操作（Manipulate）**：参见**操作（Manipulation）**。

**操作图（Manipulated graph）**：参见**操作**。

**操作（Manipulation）**：如果 G 是定义在一组变量 $\mathbf { V } \cup \mathbf { W }$ 上的有向无环图，且 $\mathbf { V } \cap \mathbf { W } = \varnothing$，那么当且仅当不存在从 V 中任何成员指向 W 中任何成员的有向边时，W 在 G 中相对于 V 是外生的。如果 $G _ { C o m b }$ 是定义在一组变量 $\mathbf { V } \cup \mathbf { W }$ 上的有向无环图，且 $P ( \mathbf { V } \cup \mathbf { W } )$ 满足 $G _ { C o m b }$ 的**马尔可夫条件（Markov condition）**，那么将 W 的值从 $\mathbf { w _ { 1 } }$ 改变为 $\mathbf { w } _ { 2 }$ 是 $G _ { C o m b }$ 相对于 V 的一次操作，当且仅当 W 相对于 V 是外生的，且 $P ( \mathbf { V } | \mathbf { W } = \mathbf { w _ { 1 } } ) \neq P ( \mathbf { V } | \mathbf { W } = \mathbf { w } _ { 2 } )$。我们定义 $P _ { U n m a n ( \mathbf { W } ) } ( \mathbf { V } ) = P ( \mathbf { V } | \mathbf { W } = \mathbf { w _ { 1 } } )$，以及 $P _ { M a n ( \mathbf { W } ) } ( \mathbf { V } ) = \mathrm { P } ( \mathbf { V } | \mathbf { W } = \mathbf { w } _ { 2 } )$，并且类似地定义由 P(V) 形成的各种边缘和条件分布。我们将 $G _ { C o m b }$ 称为**组合图（combined graph）**，将 $G _ { C o m b }$ 在 V 上的子图称为**未操作图（unmanipulated graph）** $G _ { U n m a n }$。V 属于 **Manipulated(W)**（即，V 是直接受某个操作变量影响的变量），当且仅当 V 属于 $\mathbf { C h i l d r e n ( W ) } \cap \mathbf { V }$；我们也会说 Manipulated(W) 中的变量已被**直接操作（directly manipulated）**。我们将 W 中的变量称为**政策变量（policy variables）**。**操作图（manipulated graph）** $G _ { M a n }$ 是 $G _ { U n m a n }$ 的一个子图，$P _ { M a n ( \mathbf { W } ) } ( \mathbf { V } )$ 满足该子图的马尔可夫条件，且该子图与 $G _ { U n m a n }$ 的区别最多在于 Manipulated(W) 中成员的父节点。

**最小 I-映射（Minimal I-map）**：一个无环图 G 是概率分布 P 的**最小 I-映射**，当且仅当 G 是 P 的一个 I-映射，并且 G 的**没有**子图是 P 的 I-映射。然而，当最小 I-映射应用于 LCT 中的图时，定义中的量词仅适用于非误差变量的集合。

**Mod**：如果 G 是定义在 V 上的有向无环图，且 Z 包含于 V，那么 $G ^ { \prime }$ 相对于 **Deterministic(V)** 和 Z 属于 **Mod(G)**，当且仅当对于 V 中的每个 V：

- (i) 如果存在一组包含于 Z 的顶点，这些顶点在 G 中不是 V 的后代且能确定 V，则 Parents $( G ^ { \prime } , V ) = \mathbf { X }$，其中 X 是包含于 Z 且在 G 中不是 V 的后代并能确定 V 的某个顶点集；
- (ii) 如果不存在包含于 Z 且在 G 中不是 V 的后代并能确定 V 的顶点集 X，则 Parents $( G ^ { \prime } , V ) = \mathbf { P a r e n t s } ( G , V )$。

**ND**：在有向无环图 G 中，**ND(Y)** 是所有在 Y 中没有后代的顶点的集合。

**非后代（Nondescendants）**：在有向无环图 G 中，X 属于 **Nondescendants(Y)**，当且仅当在 G 中不存在从 Y 的任何成员到 X 的有向路径。

**已观测（Observed）**：参见**诱导路径图（inducing path graph）**、**诱导路径（inducing path）**。

**出自（Out of）**：在图 G 中，A 和 B 之间的一条边**出自** A，当且仅当 A 端点处的标记是空标记。如果 A 和 B 之间的一条无向路径 U 包含一条出自 A 的边，我们将说 U 是**出自** A 的。

**平行嵌入（Parallel embedding）**：具有共同顶点集 O 的有向无环图 $G _ { 1 }$ 和 $G _ { 2 }$ 在具有共同顶点集 U（包含 O）的有向无环图 $H _ { 1 }$ 和 $H _ { 2 }$ 中具有**平行嵌入**，当且仅当：

- (i) $G _ { 1 }$ 是 $H _ { 1 }$ 在 O 上的子图，且 $G _ { 2 }$ 是 $H _ { 2 }$ 在 O 上的子图；
- (ii) $H _ { 1 }$ 中但不在 $G _ { 1 }$ 中的每条有向边都在 $H _ { 2 }$ 中，且 $H _ { 2 }$ 中但不在 $G _ { 2 }$ 中的每条有向边都在 $H _ { 1 }$ 中。

**路径形式（Path form）**：如果 G 是一个有向无环图，令 $\mathbf { P } _ { X Y }$ 为 G 中从 X 到 Y 的所有有向路径的集合。在 LCF S 中，协方差乘积 $\gamma _ { I J } \gamma _ { K L }$ 的**路径形式**是以下表达式的展开形式：

$$
\left(\sum_ {U \in \mathbf {U} _ {I J}} \left(\sum_ {R \in \mathbf {P} _ {U I}} \sum_ {R ^ {\prime} \in \mathbf {P} _ {U J}} L (R) L (R ^ {\prime}) \sigma_ {U} ^ {2}\right)\right) \left(\sum_ {V \in \mathbf {U} _ {K L}} \left(\sum_ {R ^ {\prime \prime} \in \mathbf {P} _ {V K}} \sum_ {R ^ {\prime \prime \prime} \in \mathbf {P} _ {V L}} L (R ^ {\prime \prime}) L (R ^ {\prime \prime \prime}) \sigma_ {V} ^ {2}\right)\right)
$$

$\gamma _ { I J } \gamma _ { K L } - \gamma _ { I L } \gamma _ { J K }$ 是路径形式，当且仅当两项都是路径形式。$\gamma _ { I J } \gamma _ { K L } - \gamma _ { I L } \gamma _ { J K }$ 是路径形式，当且仅当两项都是路径形式。

**政策变量（Policy variables）**：参见**操作（manipulate）**。

**Possible-D-SEP(A,B)**：如果在**部分定向诱导路径图（partially oriented inducing path graph）** 中 A ≠ B，那么 V 属于 **Possible-D-Sep(A,B)**，当且仅当 $V \neq A$，并且在 中存在一条 A 和 V 之间的无向路径 U，使得对于 U 的每个子路径 ${ < X , Y , Z > }$，要么 Y 是该子路径上的一个**碰撞点（collider）**，要么 Y 不是 U 上的**确定非碰撞点（definite noncollider）**，并且 X、Y 和 Z 在 中形成一个三角形。

**可能 d-连接（Possibly d-connecting）**：如果 A 和 B 不在 Z 中，且 A ≠ B，那么在一个定义在 O 上的部分定向诱导路径图 中，A 和 B 之间的一条无向路径 U 是给定 Z 下 A 和 B 的**可能 d-连接路径**，当且仅当 U 上的每个碰撞点都是一条指向 Z 中某个成员的**半有向路径（semidirected path）**的源点，并且每个确定非碰撞点都不在 Z 中。

**Possibly-IP**：如果 是 G 在 O 上的部分定向诱导路径图，那么 X 属于 **Possibly-IP(Y, Z)**，当且仅当 Y 和 Z 不相交，X 在 Z 中，并且在给定 Z\{X} 下，存在一条 X 与 Y 中某个 Y 之间的可能 d-连接路径，且该路径不是出自 X 的。

**Possibly-IV**：如果 是 G 在 O 上的部分定向诱导路径图，那么 X 属于 **Possibly-IV(Y, Z)**，当且仅当 X 不在 Z 中，在给定 Z 下存在一条 X 与 Y 中某个 Y 之间的可能 d-连接路径，并且存在一条从 X 到 Y ∪ Z 中某个成员的半有向路径。

**Possible-SP**：对于一个部分定向诱导路径图 和可接受的排序 Ord，令 V 属于 **Possible-SP(Ord, X)**，当且仅当 V ≠ X，并且在 中存在一条 V 和 X 之间的无向路径 U，使得 U 上除 X 之外的每个顶点都是 Ord 中 X 的前驱，并且 U 上除端点之外的顶点都不是 U 上的确定非碰撞点。

**前驱（Predecessors）**：对于诱导路径图 G 和可接受的全序 Ord，令 **Predecessors(Ord, V)** 等于根据 Ord 排在 V 之前（不包括 V）的所有变量的集合。

**真后段（Proper final segment）**：长度为 n 的路径 U 是长度为 m 的路径 V 的**真后段**，当且仅当 U 是 V 的一个后段且 $U \neq V$。

**真前段（Proper initial segment）**：长度为 n 的路径 U 是长度为 m 的路径 V 的**真前段**，当且仅当 U 是 V 的一个前段且 $U \neq V$。

$P _ { M a n ( \mathbf { W } ) } ( \mathbf { V } )$：参见**操作（manipulate）**。

$P _ { U n m a n ( \mathbf { W } ) } ( \mathbf { V } )$：参见**操作**。

**纯潜变量图（Pure Latent Variable Graph）**：一个**纯潜变量图**是一个有向无环图，其中每个**测量变量（measured variable）**恰好是一个**潜变量（latent variable）**的子节点，并且不是任何其他变量的父节点。

**随机系数线性因果理论（Random coefficient linear causal theory）**：**随机系数线性因果理论**的定义与线性因果理论相同，只是每个线性系数是一个随机变量，并且与模型中所有其他随机变量的集合独立。

**刚性统计不可区分（Rigidly statistically indistinguishable）**：如果有向无环图 G 和 $G ^ { \prime }$ 是**强统计不可区分（strongly statistically indistinguishable）**的，并且 G 和 $G ^ { \prime }$ 的每一个平行嵌入都是强统计不可区分的，那么结构 G 和 $G ^ { \prime }$ 是**刚性统计不可区分（rigidly statistically indistinguishable, r.s.i.）**的。

**R.S.I.**：参见**刚性统计不可区分（rigidly statistically indistinguishable）**。

**半有向（Semi-directed）**：在部分定向诱导路径图 中，从 A 到 B 的**半有向路径**是一条从 A 到 B 的无向路径 U，其中没有边包含指向 A 的箭头，即 U 上 A 处没有箭头，并且如果 X 和 Y 在路径上相邻，且 X 在路径上位于 A 和 Y 之间，那么在 X 和 Y 之间的边的 X 端没有箭头。

**源点（Source）**：参见**路径（trek）**。

**SP**：对于诱导路径图 $G ^ { \prime }$ 和可接受的全序 Ord，W 属于 **$\mathbf { S P } ( O r d , G ^ { \prime } , V )$**（即 V 在 $G ^ { \prime }$ 中对于排序 Ord 的**分离前驱（separating predecessors）**），当且仅当 $W \neq V$ 并且存在一条 W 和 V 之间的无向路径 U，使得 U 上除 V 之外的每个顶点在 Ord 中都排在 V 之前，并且 U 上除端点之外的每个顶点都是 U 上的碰撞点。

**S.S.I.**：参见**强统计不可区分（strongly statistically indistinguishable）**。

**强统计不可区分（Strongly statistically indistinguishable）**：两个有向无环图 G 和 $G ^ { \prime }$ 是**强统计不可区分**的，当且仅当它们具有相同的顶点集 V，并且 V 上满足 G 的**最小性条件（Minimality）**和马尔可夫条件的每个分布 P 也满足 $G ^ { \prime }$ 的这些条件，反之亦然。

**可替代（Substituable）**：在包含 X 和 Y 之间无向路径 U 的诱导路径图或有向无环图 G 中，V 和 W 之间的边在 U 中对于 $U ( V , W )$ 是**可替代的**，当且仅当 V 和 W 在 U 上，V 在 U 上位于 X 和 W 之间，G 包含 V 和 W 之间的一条边，V 是 $U ( X , V )$ 与 V 和 W 之间边连接后的路径上的碰撞点当且仅当它是 U 上的碰撞点，并且 W 是 $U ( Y , W )$ 与 V 和 W 之间边连接后的路径上的碰撞点当且仅当它是 U 上的碰撞点。

**T**：参见**路径（trek）**。

**端点（Termini）**：参见**路径**。

**路径（Trek）**：两个不同顶点 I 和 J 之间的一个**路径** $T ( I , J )$ 是从某个顶点 K 分别到 I 和 J 的两条仅在 K 处相交的无环有向路径的无序对。路径中路径的源点称为该路径的**源点（source）**。I 和 J 称为该路径的**端点（termini）**。给定 I 和 J 之间的一个路径 $T ( I , J )$，$I ( T ( I , J ) )$ 将表示 $T ( I , J )$ 中从该路径的源点到 I 的路径，$J ( T ( I , J ) )$ 将表示 $T ( I , J )$ 中从该路径的源点到 J 的路径。路径中的一条路径可以是空路径。然而，由于路径的端点是不同的，路径中只能有一条路径是空路径。$\mathbf { T } ( I , J )$ 是 I 和 J 之间所有路径的集合。$T ( I , J )$ 将表示 $\mathbf { T } ( I , J )$ 中的一个路径。$S ( T ( I , J ) )$ 表示路径 $T ( I , J )$ 的源点。

**无向（Undirected）**：在图 G 中，令 V 属于 **Undirected(X, Y)**，当且仅当 V 位于 X 和 Y 之间的某条无向路径上。

**未操作图（Unmanipulated graph）**：参见**操作（manipulation）**。

${ \mathbf { U } } _ { X } { \mathbf { : } }$ 在 LCF S 中，$\mathbf { U } _ { X }$ 是所有独立变量的集合，这些变量是通向 X 的有向路径的源点。（注意，如果 X 是独立的，则 $X \in \mathbf { U } _ { X }$，因为从每个顶点到其自身存在一条空路径。）

$\mathbf { U } _ { X Y } { \mathrm { : } }$ 在 LCF S 中，$\mathbf { U } _ { X Y }$ 是 $\mathbf { U } _ { X } \cap \mathbf { U } _ { Y }$。

**弱忠实不可区分（Weakly faithfully indistinguishable）**：两个有向无环图是**弱忠实不可区分（weakly faithfully indistinguishable, w.f.i.）**的，当且仅当存在一个对两者都忠实的概率分布。

**弱统计不可区分（Weakly statistically indistinguishable）**：两个有向无环图是**弱统计不可区分（weakly statistically indistinguishable, w.s.i.）**的，当且仅当存在一个同时满足两者最小性条件和马尔可夫条件的概率分布。

**W.F.I.**：参见**弱忠实不可区分（weakly faithfully indistinguishable）**。

**W.S.I.**：参见**弱统计不可区分（weakly statistically indistinguishable）**。