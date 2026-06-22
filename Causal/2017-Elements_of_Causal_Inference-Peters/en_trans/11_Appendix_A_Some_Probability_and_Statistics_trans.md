# 附录A 概率论与数理统计基础（Appendix A Some Probability and Statistics）

## A.1 基本定义（Basic Definitions）

(i) 我们将底层概率空间记为 $( \Omega , { \mathcal { F } } , P )$ 。其中，$\Omega$、${ \mathcal { F } }$ 和 $P$ 分别表示集合、$\sigma$-代数和概率测度。

(ii) 我们用大写字母表示实值随机变量。例如，$X$ : $( \Omega , \mathcal { F } ) \to ( \mathbb { R } , B _ { \mathbb { R } } )$ 是一个关于**Borel $\sigma$-代数**的可测函数。随机向量是可测函数 $\mathbf { X } : ( \Omega , \mathcal { F } ) \to ( \mathbb { R } ^ { d } , B _ { \mathbb { R } ^ { d } } )$ 。如果不存在 $\mathbf { c } \in \mathbb { R } ^ { d }$ 使得 $P ( \mathbf { X } = \mathbf { c } ) = 1$ ，则称 $\mathbf{X}$ 为非退化的。关于测度论的介绍，可参见 Dudley [2002]。

(iii) 我们通常用粗体字母表示向量。为简便起见，我们将变量集合 $\mathbf { B } \subseteq \mathbf { X }$ 视为单个多元变量。

(iv) $P _ { \mathbf { X } }$ 是 $d$ 维随机向量 $\mathbf{X}$ 的分布，即 $( \mathbb { R } ^ { d } , B _ { \mathbb { R } ^ { d } } )$ 上的一个概率测度。

(v) 我们用 $x \mapsto p _ { X } ( x )$ 或简写为 $x \mapsto p ( x )$ 表示密度，即 $P _ { X }$ 关于某个乘积测度的 **Radon-Nikodym 导数**。我们（有时隐含地）假设其存在性或连续性。

(vi) 当且仅当对于所有 $x , y$ 有

$$
p (x, y) = p (x) p (y) \tag {A.1}
$$

时，称 $X$ 独立于 $Y$，记为 $X \perp \perp Y$。否则，$X$ 和 $Y$ 是**相依的**，记为 $X \not \vdash Y$。

(vii) 当且仅当对于所有 $\boldsymbol { x } _ { 1 } , \ldots , \boldsymbol { x } _ { d }$ 有

$$
p (x _ {1}, \dots , x _ {d}) = p (x _ {1}) \cdot \dots \cdot p (x _ {d}) \tag {A.2}
$$

时，称 $X _ { 1 } , \ldots , X _ { d }$ 是**联合（或相互）独立**的。如果 $X _ { 1 } , \ldots , X _ { d }$ 是联合独立的，那么任意一对 $X _ { i }$ 和 $X _ { j }$（$i \neq j$）也是独立的。反之则不一定成立：**两两独立并不意味着联合独立**。

(viii) 当且仅当对于所有满足 $p ( z ) > 0$ 的 $x , y , z$ 有

$$
p (x, y \mid z) = p (x \mid z) p (y \mid z) \tag {A.3}
$$

时，称 $X$ 在给定 $Z$ 的条件下独立于 $Y$，记为 $X \perp \perp Y \mid Z$。否则，$X$ 和 $Y$ 在给定 $Z$ 的条件下是**相依**的，记为 $X \not \vdash Y | Z$。

(ix) **条件独立性关系**遵循以下重要规则 [例如，Pearl, 2009, 第1.1.5节]：

$$
\begin{array}{l} X \perp Y | Z \Rightarrow Y \perp X | Z \quad (\text { 对称性 }) \\ X \perp Y, W | Z \Rightarrow X \perp Y | Z \quad (\text { 分解性 }) \\ X \perp Y, W | Z \Rightarrow X \perp Y | W, Z \quad (\text { 弱联合性 }) \\ X \perp Y | Z \text {  且  } X \perp W | Y, Z \Rightarrow X \perp Y, W | Z \quad (\text { 收缩性 }) \\ X \perp Y | W, Z \text {  且  } X \perp W | Y, Z \Rightarrow X \perp Y, W | Z \quad (\text { 交性 }). \\ \end{array}
$$

**严格正密度**的存在性足以使交性成立。离散情况的必要和充分条件由 Drton 等人 [2009b, 练习6.6] 和 Fink [2011] 提供。Peters [2014] 涵盖了连续情况。

(x) 随机变量 $X$ 的**方差**定义为

$$
\operatorname{var} [ X ] := \mathbb {E} \left[ (X - \mathbb {E} [ X ]) ^ {2} \right] = \mathbb {E} \left[ X ^ {2} \right] - \mathbb {E} [ X ] ^ {2}
$$

如果 $\mathbb { E } [ X ^ { 2 } ] < \infty$。

(xi) 如果 $\mathbb { E } [ X ^ { 2 } ] , \mathbb { E } [ Y ^ { 2 } ] < \infty$ 且

$$
\mathbb {E} [ X Y ] = \mathbb {E} [ X ] \mathbb {E} [ Y ],
$$

即

$$
\rho_ {X, Y} := \frac {\mathbb {E} [ X Y ] - \mathbb {E} [ X ] \mathbb {E} [ Y ]}{\sqrt {\operatorname{var} [ X ] \operatorname{var} [ Y ]}} = 0,
$$

则称 $X$ 和 $Y$ 是**不相关的**。否则，即 $\rho _ { X , Y } \neq 0$，则 $X$ 和 $Y$ 是**相关的**。$\rho _ { X , Y }$ 称为 $X$ 和 $Y$ 之间的**相关系数**。

(xii) 如果 $X$ 和 $Y$ 是独立的，那么它们是不相关的：

$$
X \perp Y \Rightarrow \rho_ {X, Y} = 0.
$$

反之则不一定成立（参见代码片段 A.1）。只有在特殊情况下，例如二元高斯分布或二元变量，反向关系也成立。

(xiii) 如果

$$
\rho_ {X, Y \mid Z} := \frac {\rho_ {X , Y} - \rho_ {X , Z} \rho_ {Z , Y}}{\sqrt {(1 - \rho_ {X , Z} ^ {2}) (1 - \rho_ {Z , Y} ^ {2})}} = 0,
$$

则称 $X$ 和 $Y$ 在给定 $Z$ 的条件下是**偏不相关**的。**偏相关系数**的以下解释很重要：$\rho _ { X , Y \mid Z }$ 等于将 $X$ 对 $Z$ 进行线性回归以及将 $Y$ 对 $Z$ 进行线性回归后所得残差之间的相关性。

(xiv) 一般来说，我们有（参见例 7.9）

$$
\rho_ {X, Y \mid Z} = 0 \quad \nRightarrow \quad X \perp \perp Y \mid Z \quad \text { 且 }
$$

$$
\rho_ {X, Y \mid Z} = 0 \quad \nLeftarrow \quad X \perp \perp Y \mid Z.
$$

(xv) 在**回归估计**中，我们通常给定来自联合分布 $P _ { \mathbf { X } , Y }$ 的独立同分布样本 $( \mathbf { X } _ { 1 } , Y _ { 1 } ) , . . ., ( \mathbf { X } _ { n } , Y _ { n } )$。我们的目标是从**协变量**或**预测变量** $\mathbf{X}$ 预测目标 $Y$。例如，在**最小二乘回归**中，我们寻找一个函数 $\hat { f }$，使得

$$
\hat {f} = \underset {f \in \mathcal {F}} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \left(Y _ {i} - f (\mathbf {X} _ {i})\right) ^ {2}.
$$

这里，我们在一个**函数类** $\mathcal { F }$ 上进行优化（参见第 A.3 节）。不同的回归技术使用不同的函数类。在**线性回归**中，我们只考虑线性函数 $f$；示例见代码片段 6.43。代码片段 4.14 展示了一个非线性回归技术的例子。

(xvi) 离散随机变量集合 $\mathbf{X}$ 和 $\mathbf{Y}$ 之间的**相依性**可以通过**香农互信息**（Shannon mutual information）[Cover and Thomas, 1991] 来衡量

$$
I (\mathbf {X}: \mathbf {Y}) := \sum_ {\mathbf {x}, \mathbf {y}} p (\mathbf {x}, \mathbf {y}) \log \frac {p (\mathbf {x} , \mathbf {y})}{p (\mathbf {x}) p (\mathbf {y})}.
$$

(xvii) 给定集合 $\mathbf { Z }$ 的条件下，离散随机变量集合 $\mathbf{X}$ 和 $\mathbf{Y}$ 之间的**条件相依性**通过**条件香农互信息**（conditional Shannon mutual information）[Cover and Thomas, 1991] 来衡量

$$
I (\mathbf {X}: \mathbf {Y} | \mathbf {Z}) := \sum_ {\mathbf {x}, \mathbf {y}, \mathbf {z}} p (\mathbf {x}, \mathbf {y}, \mathbf {z}) \log \frac {p (\mathbf {x} , \mathbf {y} | \mathbf {z})}{p (\mathbf {x} | \mathbf {z}) p (\mathbf {y} | \mathbf {z})}.
$$

(xviii) 对于连续变量，求和被替换为积分

$$
I (\mathbf {X}: \mathbf {Y}) := \int p (\mathbf {x}, \mathbf {y}) \log \frac {p (\mathbf {x} , \mathbf {y})}{p (\mathbf {x}) p (\mathbf {y})} d \mathbf {x} d \mathbf {y},
$$

以及

$$
I (\mathbf {X}: \mathbf {Y} | \mathbf {Z}) := \int p (\mathbf {x}, \mathbf {y}, \mathbf {z}) \log \frac {p (\mathbf {x} , \mathbf {y} | \mathbf {z})}{p (\mathbf {x} | \mathbf {z}) p (\mathbf {y} | \mathbf {z})} d \mathbf {x} d \mathbf {y} d \mathbf {z}.
$$

## A.2 独立性与条件独立性检验（Independence and Conditional Independence Testing）

在实践中，我们给定一个有限样本 $( X _ { 1 } , Y _ { 1 } ) , \ldots , ( X _ { n } , Y _ { n } ) \overset { \mathrm { i i d } } { \sim } P _ { X , Y }$，并希望判断潜在的随机变量是否独立。由于我们不期望经验相关性（或任何独立性度量）恰好为 0，我们需要考虑相依性度量的随机波动。这可以通过**统计假设检验**来实现。其思想是考虑**零假设** $H _ { 0 } : X \perp Y$ 和**备择假设** $H _ { A } : X \not \perp Y$。因此，通常构造一个**检验统计量** $T _ { n }$，它将任何有限样本映射到一个实数，并根据以下规则做出判断

$$
(x _ {1}, y _ {1}), \ldots , (x _ {n}, y _ {n}) \mapsto \left\{ \begin{array}{l l} H _ {0} & \text { 如果 } T _ {n} \leq c \\ H _ {A} & \text { 如果 } T _ {n} > c. \end{array} \right.
$$

这里，$T _ { n }$ 是 $T _ { n } { \big ( } ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) { \big ) }$ 的简写。阈值 $c \in \mathbb { R }$ 的选择使得我们可以控制**第一类错误**；即，对于任何满足 $H _ { 0 }$ 的 $P$，有 $P ( T _ { n } > c ) \leq \alpha$，其中 $\alpha$ 是用户指定的检验**显著性水平**。在实践中，我们获得数据并计算统计量 $T _ { n }$。如果 $T _ { n } > c$，则拒绝零假设，我们可以相对确信我们的决策是正确的；否则，不拒绝零假设，但这并不一定意味着太多（可能样本量 $n$ 太小而无法检测到 $X$ 和 $Y$ 之间的相依性）。检验的 **p 值** 是使得检验被拒绝的最小显著性水平。

我们现在简要提及 $T _ { n }$ 的几种选择。然而，还有更多的检验方法，我们并不声称此列表包含最优程序；实际例子见代码片段 A.1。

(i) 为了检验**相关性消失**，我们可以使用经验相关系数和 t 检验（针对高斯变量）或 Fisher 的 z 变换（例如，R Core Team [2016] 中的 cor.test）。

(ii) 作为**独立性检验**，我们可以对离散或离散化数据使用 $\chi ^ { 2 }$ 检验（例如，R Core Team [2016] 中的 chisq.test）。

(iii) 一个通用的**非参数独立性检验**例子是**希尔伯特-施密特独立性准则**（Hilbert-Schmidt Independence Criterion, HSIC）[参见 Gretton 等人，2008]。其思想基于到**再生核希尔伯特空间**（Reproducing Kernel Hilbert Spaces, RKHSs）[Scholkopf and Smola, 2002] 的**单射映射**。给定一个正定核，我们可以将概率分布映射到相应的 RKHS $\mathcal { H }$，即 $P _ { X , Y } \mapsto \mu ( P _ { X , Y } ) \in \mathcal { H }$。对于所谓的**特征核**（例如高斯核），该映射是单射的。特别地，我们有

$$
\mu (P _ {X, Y}) = \mu (P _ {X} \otimes P _ {Y}) \quad \text {  当  且  仅  当   } \quad P _ {X, Y} = P _ {X} \otimes P _ {Y},
$$

而后者成立当且仅当 $X$ 和 $Y$ 是独立的。HSIC 定义为联合分布与边缘乘积之间的**平方 RKHS 距离**：

$$
\operatorname{HSIC} \left(P _ {X, Y}\right) := \left\| \mu \left(P _ {X, Y}\right) - \mu \left(P _ {X} \otimes P _ {Y}\right) \right\| _ {\mathcal {H}} ^ {2}.
$$

作为检验统计量 $T _ { n }$，我们现在可以使用 $\mathrm { H S I C } ( P _ { X , Y } )$ 的估计量。如果 $X$ 和 $Y$ 是独立的，则 $\operatorname{HSIC} \left( P _ { X , Y } \right)$ 等于 0，我们期望其估计量 $T _ { n }$ 很小。Gretton 等人 [2008] 提供了如何选择阈值 $c$ 的方法。

或者，我们可以将 HSIC 表示为**协方差算子** $C _ { X Y }$ 的**希尔伯特-施密特范数**。后者定义为对于所有属于相应 RKHS 的 $f$ 和 $g$

$$
\langle f, C _ {X Y} g \rangle = \mathbb {E} [ f (X) g (Y) ] - \mathbb {E} [ f (X) ] \mathbb {E} [ g (Y) ].
$$

因此，**互协方差算子**是**协方差矩阵**的扩展。如果 $X$ 是 $d _ { X }$ 维的，$Y$ 是 $d _ { Y }$ 维的，并且相应的 RKHS 分别同构于 $\mathbb { R } ^ { d _ { X } }$ 和 $\mathbb { R } ^ { d _ { Y } }$，则 $C _ { X Y }$ 可以用 $d _ { X } \times d _ { Y }$ 维的互协方差矩阵来描述。当然，如果协方差矩阵消失，$X$ 和 $Y$ 不一定需要独立。然而，对于特征核，RKHS 是无限维的，并且不同构于 $\mathbb { R } ^ { d }$。当且仅当 $X$ 和 $Y$ 独立时，互协方差算子的范数为零。

Pfister 等人 [2017] 将该过程扩展到检验 $d$ 个变量之间的**联合独立性**。例如，这对于检验噪声变量的联合独立性是必要的。他们提供了双变量和多变量过程的代码（参见 R 包 dHSIC）。

在实践中，通常需要选择核参数。对于高斯核，许多实现根据通常所说的**中位数启发式**（median heuristic）[例如，Gretton 等人，2008] 选择带宽 $\sigma$。

(iv) **条件独立性检验** 是一个难题，尤其是在**条件集**很大的情况下。虽然获得该陈述的精确形式化是当前的研究课题，但我们提供一个表明该问题难度的例子。如果 $Z _ { 1 } , \ldots , Z _ { d }$ 是二元变量，那么

$$
\begin{array}{l} X \perp Y | Z _ {1}, \dots , Z _ {d} \\ \Leftrightarrow \quad \forall (z _ {1}, \dots , z _ {d}) \in \{0, 1 \} ^ {d}: \quad X \perp Y | Z _ {1} = z _ {1}, \dots , Z _ {d} = z _ {d}. \\ \end{array}
$$

如果我们不能对 $X$ 和 $Y$ 可能依赖于 $Z$ 变量的方式做任何假设，我们需要对 $2 ^ { d }$ 个赋值中的每一个执行无条件独立性检验（例如，$Z _ { d }$ 可能是 $X$ 和 $Y$ 的共同子节点，其相依性仅在其他 $Z _ { 1 } , \ldots , Z _ { d - 1 }$ 的特定赋值下才能检测到）。

对于连续变量，已经提出了 HSIC 检验的扩展。Fukumizu 等人 [2008] 将该思想扩展到**条件互协方差算子**以获得条件独立性检验。Zhang 等人 [2011] 进一步发展了这一点，他们还提供了零假设下检验统计量分布的近似。

**代码片段 A.1** 以下代码生成两个不相关但相依的变量的分布样本。

```r
library(dHSIC)
#
# generates a sample from two uncorrelated but dependent random variables
set.seed(1)
A <- runif(200)-0.5
B <- runif(200)-0.5
X <- t(c(cos(pi/4), -sin(pi/4)) %*% rbind(A, B))
Y <- t(c(sin(pi/4), cos(pi/4)) %*% rbind(A, B))
#
# performs the statistical test
cor.test(X,Y)$p.value
# 0.3979561
dhsic.test(X,Y)$p.value
# 1.970705e-08
```

## A.3 函数类的容量（Capacity of Function Classes）

这里，我们讨论**经验风险最小化**（empirical risk minimization）的函数序列 (1.3) 是否收敛于也最小化风险 (1.2) 的函数的问题；参见第 1.2 节。根据**大数定律**，我们知道对于任何固定的 $f \in { \mathcal { F } }$ 和 $\varepsilon > 0$，

$$
\lim _ {n \to \infty} P \left(\left| R [ f ] - R _ {\mathrm{emp}} ^ {n} [ f ] \right| > \varepsilon\right) = 0, \tag {A.4}
$$

其收敛速度由 **Chernov 界** [例如，Vapnik, 1998] 控制，呈指数级快速收敛。然而，这并不意味着经验风险最小化的**相合性**。这是因为我们通过最小化 (1.3) 来选择函数 $f$。这意味着即使 $( x _ { i } , y _ { i } )$ 是独立的，误差或损失 $\frac { 1 } { 2 } \vert f ( x _ { i } ) - y _ { i } \vert$ 却不是。在这种情况下，通常形式的大数定律不适用。事实证明，要获得相合性，我们需要一个**一致大数定律** [Vapnik, 1998]。这相当于对于所有 $\varepsilon > 0$，

$$
\lim _ {n \rightarrow \infty} P \left(\sup _ {f \in \mathcal {F}} (R [ f ] - R _ {\mathrm{emp}} ^ {n} [ f ]) > \varepsilon\right) = 0 \tag {A.5}
$$

这是一个取决于函数类 ${ \mathcal F }$ 的性质。

如果选择 $\mathcal { F } = \mathcal { V } ^ { \mathcal { X } }$，换句话说，所有从 $\mathcal { X }$ 到 $\mathcal { V }$ 的函数，结果会怎样？不幸的是，这不会导致 (A.5)，理由如下：假设基于可用样本 (1.1)，我们判定 $f ^ { * }$ 是一个好的解——例如，因为它对所有 $i$ 满足 $f ( x _ { i } ) = y _ { i }$。在这种情况下，让我们构造另一个函数 $f ^ { * * }$，它在样本上与 $f ^ { * }$ 一致，但在其他地方都不一致。如果我们的分布 $P _ { X , Y }$ 具有密度，那么未来再次精确遇到任何训练点的概率为零。因此，$f ^ { * }$ 和 $f ^ { * * }$ 几乎总是会产生分歧。然而，仅基于训练集，无法选择其中一个而放弃另一个。类似地，在 (A.5) 中，我们会发现，每当我们找到一个函数 $f ^ { * }$ 使得 $( R [ f ^ { * } ] - R _ { \mathrm { e m p } } ^ { n } [ f ^ { * } ] )$ 恰好很小时，我们可以构造另一个函数 $f ^ { * * }$ 使得 $( R [ f ^ { * * } ] - R _ { \mathrm { e m p } } ^ { n } [ f ^ { * * } ] )$ 很大，因此在我们考虑的 $\mathcal { F } = \mathcal { V } ^ { \mathcal { X } }$ 的情况下，一致收敛 (A.5) 是不可能实现的。

另一方面，随着我们使 $\mathcal { F }$ 变小，条件 (A.5) 变得更弱。如何衡量 $\mathcal { F }$ 的大小（或容量）超出了本书的范围，但事实证明，对于独立于底层分布的 $\mathcal { F }$ 大小的总结，一个数字就足够了。它被称为 $\mathcal { F }$ 的 **VC（Vapnik-Chervonenkis）维数**。它有时与自由参数的数量一致，但也可能大相径庭。如果 VC 维数是有限的，那么对于任何 $P _ { X , Y }$，我们都能得到经验风险最小化的相合性 [Vapnik, 1998]。VC 维数与**可证伪性**以及 Popper 的理论维数概念有关 [Corfield 等人，2009]。统计学习理论的一个典型风险界指出，对于所有 $\delta > 0$，以概率 $1 - \delta$，并且对于所有 $f \in { \mathcal { F } }$，我们有

$$
R [ f ] \leq R _ {\mathrm{emp}} ^ {n} [ f ] + \sqrt {\frac {h (\log (2 n / h) + 1) - \log (\delta / 4)}{n}}, \tag {A.6}
$$

其中 $h$ 是函数类 $\mathcal { F }$ 的 VC 维数。这意味着，如果我们能够提出一个具有小 VC 维数，但又包含足够适合给定任务以实现小 $R _ { \mathrm { e m p } } ^ { n } [ f ]$ 的函数的 $\mathcal { F }$，那么我们可以（以高概率）保证这些函数在来自同一分布的未来数据上具有较小的期望误差。这形成了一个非平凡的权衡：一方面，我们希望使用一个大的函数类以获得小的 $R _ { \mathrm { e m p } } ^ { n }$，但另一方面，我们希望该类很小以控制 $h$。