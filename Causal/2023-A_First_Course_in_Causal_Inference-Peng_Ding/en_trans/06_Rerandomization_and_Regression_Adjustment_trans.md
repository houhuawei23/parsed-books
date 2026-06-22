# 重随机化与回归调整（Rerandomization and Regression Adjustment）

第5章中的**分层（Stratification）**和**事后分层（Post-stratification）**是随机化实验中针对离散协变量的两种对偶方法。我们应如何处理多维且可能连续的协变量？我们可以将连续协变量离散化，但当协变量较多时，这并非理想策略。**重随机化（Rerandomization）**和**回归调整（Regression adjustment）**是针对一般协变量的对偶方法，这正是本章的主题。

下表总结了第5章和第6章的主题：

<table><tr><td></td><td>设计（design）</td><td>分析（analysis）</td></tr><tr><td>离散协变量（discrete covariate）</td><td>分层（stratification）</td><td>事后分层（post-stratification）</td></tr><tr><td>一般协变量（general covariate）</td><td>重随机化（rerandomization）</td><td>回归调整（regression adjustment）</td></tr></table>

## 6.1 重随机化（Rerandomization）

## 6.1.1 实验设计（Experimental design）

我们再次考虑一个包含 $n$ 个单元的有限总体，其中 $n _ { 1 }$ 个单元接受处理，$n _ { 0 }$ 个单元接受对照。令 $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ 为这些单元的处理向量。单元 $i$ 具有协变量 $X _ { i } \in \mathbb { R } ^ { K }$，该协变量可以包含连续或二元分量。将它们拼接为 $\pmb { X } = ( X _ { 1 } , \ldots , X _ { n } )$，并在不失一般性的情况下将其中心化为均值零 $\begin{array} { r } { \bar { X } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } = 0 } \end{array}$。

**完全随机化实验（Completely Randomized Experiment, CRE）**平均而言平衡了处理组和对照组中的协变量，例如，协变量均值之差

$$
\hat {\tau} _ {X} = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} X _ {i} - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) X _ {i}
$$

在CRE下均值为零。然而，在实际实现的处理分配中，它可能导致处理组和对照组之间出现不理想的协变量平衡，即 $\hat { \tau } _ { X }$ 的实际实现值通常不为零。利用问题4.6中Neyman (1923)的向量形式，我们可以证明

$$
\operatorname{cov} (\hat {\tau} _ {X}) = \frac {1}{n _ {1}} S _ {X} ^ {2} + \frac {1}{n _ {0}} S _ {X} ^ {2} = \frac {n}{n _ {1} n _ {0}} S _ {X} ^ {2},
$$

其中 $\begin{array} { r } { S _ { X } ^ { 2 } = ( n - 1 ) ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } X _ { i } ^ { \mathsf { T } } } \end{array}$。以下**马氏距离（Mahalanobis distance）**衡量处理组与对照组之间的差异：

$$
M = \hat {\tau} _ {X} ^ {\mathsf {T}} \mathrm{cov} (\hat {\tau} _ {X}) ^ {- 1} \hat {\tau} _ {X} = \hat {\tau} _ {X} ^ {\mathsf {T}} \left(\frac {n}{n _ {1} n _ {0}} S _ {X} ^ {2}\right) ^ {- 1} \hat {\tau} _ {X}.
$$

从技术上讲，上述 $M$ 的公式仅在 $S _ { X } ^ { 2 }$ 可逆时才有意义，这意味着协变量矩阵的列是线性独立的。如果某一列可以由其他列的线性组合表示，则该列是冗余的，应在实验前予以剔除。$M$ 的一个优良特性是它在对 $X$ 进行非退化线性变换时具有不变性。下面的引理6.1总结了这一结果，其证明留待问题6.2。

**引理 6.1** 如果我们将所有单元 $i = 1 , \ldots , n$ 的 $X _ { i }$ 变换为 $\alpha + B X _ { i }$，其中 $\alpha \in \mathbb { R } ^ { K }$ 且 $B \in \mathbb { R } ^ { K \times K }$ 可逆，则 $M$ 保持不变。

**有限总体中心极限定理（Finite population central limit theorem）** (Li and Ding, 2017) 保证了当 $n$ 较大时，在CRE下马氏距离 $M$ 近似服从 $\chi _ { K } ^ { 2 }$ 分布。因此，在CRE下，$M$ 的实际实现值很可能较大，其渐近期望为 $K$，方差为 $2K$。**重随机化（Rerandomization）**通过丢弃 $M$ 值较大的处理分配来避免协变量不平衡。下面给出使用马氏距离的重随机化（Rerandomization using the Mahalanobis distance, ReM）的正式定义，该定义由Cox (1982)以及Morgan和Rubin (2012)提出。

**定义 6.1 (ReM)** 从CRE中抽取 $Z$，当且仅当

$$
M \leq a,
$$

时接受它，其中 $a > 0$ 为某个预先确定的常数。

选择 $a$ 类似于在**分层随机化实验（Stratified Randomized Experiment, SRE）**中选择层数，这在实践中是一个重要问题。在一个极端情况下，$a = \infty$，我们只是进行CRE。在另一个极端，$a = 0$，可行的处理分配非常少，因此实验几乎没有随机性，使得基于随机化的推断无效。作为折中，我们选择一个小但并非极小的 $a$，例如 $a = 0.001$ 或 $\chi _ { K } ^ { 2 }$ 分布的某个上分位数。

ReM使用马氏距离作为平衡准则。我们可以考虑一般的重随机化，其平衡准则定义为 $Z$ 和 $X$ 的函数。例如，我们可以使用基于 $X _ { i } = ( x _ { i 1 } , \ldots , x _ { i K } ) ^ { \mathsf { T } }$ 所有坐标的边缘检验的以下准则。当且仅当

$$
\left| \frac {\hat {\tau} _ {x k}}{\sqrt {\frac {n}{n _ {1} n _ {0}} S _ {x k} ^ {2}}} \right| \leq a \quad (k = 1, \dots , K) \tag {6.1}
$$

时接受 $Z$，其中 $a > 0$ 为某个预先确定的常数。例如，$a$ 可以是标准正态分布的某个上分位数。ReM具有许多优良性质。如上所述，它对协变量的线性变换具有不变性。此外，它具有良好的几何性质和优雅的数学理论。本章将重点讨论ReM。关于基于准则(6.1)以及其他准则的重随机化理论，请参见Zhao和Ding (2021b)。

## 6.1.2 统计推断（Statistical inference）

一个重要问题是如何在ReM下分析数据。Bruhn和McKenzie (2009)以及Morgan和Rubin (2012)认为，只要我们模拟在约束 $M \leq a$ 下的 $Z$，就总是可以使用**费希尔随机化检验（Fisher Randomization Test, FRT）**。在**尖锐零假设（sharp null hypothesis）**下，这总能得到有限样本精确 $p$ 值。

在不假设尖锐零假设的情况下推导ReM的有限样本性质是一个具有挑战性的问题。Li等人 (2018b) 则在ReM和以下正则条件下推导出了结果均值之差 $\hat { \tau }$ 的渐近分布。

## 条件 6.1 当 $n \to \infty$ 时

1. $n _ { 1 } / n$ 和 $n _ { 0 } / n$ 具有正极限；  
2. $\{ X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) , \tau _ { i } \}$ 的有限总体协方差存在极限；  
3. $\max _{1 \leq i \leq n} | Y _ { i } ( 1 ) - \bar { Y } ( 1 ) | ^ { 2 } / n \to 0$，$\max _{1 \leq i \leq n} | Y _ { i } ( 0 ) - \bar { Y } ( 0 ) | ^ { 2 } / n \to 0$，且 $\max _{1 \leq i \leq n} \| X _ { i } \| ^ { 2 } / n \to 0$。

以下是ReM的主要定理。令

$$
L _ {K, a} \sim D _ {1} \mid \boldsymbol {D} ^ {\mathsf {T}} \boldsymbol {D} \leq a
$$

其中 $\pmb { \mathcal {D} } = ( D _ { 1 } , \ldots , D _ { K } )$ 服从 $K$ 维标准正态分布；令 $\varepsilon$ 服从一元标准正态分布；$L _ { K , a } \bot \varepsilon$。

**定理 6.1** 在 $M \leq a$ 的ReM和条件6.1下，我们有 $^1$

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\hat {\tau})} \left\{\sqrt {R ^ {2}} L _ {K, a} + \sqrt {1 - R ^ {2}} \varepsilon \right\},
$$

其中

$$
\mathrm{var} (\hat {\tau}) = \frac {S ^ {2} (1)}{n _ {1}} + \frac {S ^ {2} (0)}{n _ {0}} - \frac {S ^ {2} (\tau)}{n}
$$

是第4章中证明的Neyman (1923)方差公式，且

$$
R ^ {2} = \mathrm{corr} ^ {2} (\hat {\tau}, \hat {\tau} _ {X})
$$

![image_06](images/image_06.png)

重随机化（Rerandomization）
区域（area）
O
θ
√R²Lₖ,ₐ
τ̂ - τ
√1 - R²ε
τ̂ₓ

图6.1：ReM的几何表示

是在CRE下 $\hat { \tau }$ 与 $\hat { \tau } _ { X }$ 之间的**平方多重相关系数（squared multiple correlation coefficient）** $^2$。

尽管Li等人 (2018b) 的证明技术性很强，但定理6.1中的渐近分布具有清晰的几何解释，如图6.1所示。该图表明，$\hat { \tau }$ 可以分解为一个与 $\hat { \tau } _ { X }$ 成线性组合的分量和一个与 $\hat { \tau } _ { X }$ 正交的分量。从几何上看，$\cos ^ { 2 } \theta = R ^ { 2 }$，其中 $\theta$ 是 $\hat { \tau }$ 与 $\hat { \tau } _ { X }$ 之间的夹角。ReM影响第一个分量，但不改变第二个分量。截断正态分布 $L _ { K , a }$ 是由于ReM对第一个分量的限制所致。

当 $a = \infty$ 时，渐近分布简化为CRE下的分布：

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\hat {\tau})} \varepsilon .
$$

当阈值 $a$ 接近于零时，渐近分布简化为

$$
\hat {\tau} - \tau \dot {\sim} \sqrt {\mathrm{var} (\hat {\tau}) (1 - R ^ {2})} \varepsilon .
$$

因此，对于较小的阈值 $a$，ReM带来的效率增益取决于 $R ^ { 2 }$，后者具有以下等价形式。

**命题 6.1** 在CRE下，

$$
R ^ {2} = \mathrm{corr} ^ {2} (\hat {\tau}, \hat {\tau} _ {X}) = \frac {n _ {1} ^ {- 1} S ^ {2} (1 \mid x) + n _ {0} ^ {- 1} S ^ {2} (0 \mid x) - n ^ {- 1} S ^ {2} (\tau \mid x)}{n _ {1} ^ {- 1} S ^ {2} (1) + n _ {0} ^ {- 1} S ^ {2} (0) - n ^ {- 1} S ^ {2} (\tau)},
$$

其中 $\{ S ^ { 2 } ( 1 ) , S ^ { 2 } ( 0 ) , S ^ { 2 } ( \tau ) \}$ 是 $\{ Y _ { i } ( 1 ) , Y _ { i } ( 0 ) , \tau _ { i } \} _ { i = 1 } ^ { n }$ 的有限总体方差，而 $\{ \hat { S ^ { 2 } } ( 1 \mid x ) , \hat { S ^ { 2 } } ( 0 \mid x ) , \hat { S ^ { 2 } } ( \tau \mid x ) \}$ 是它们在 $( 1 , X _ { i } )$ 上的线性投影的相应有限总体方差。$^3$ 在 $\tau _ { i } = \tau$ 的**常因果效应（constant causal effect）**假设下，$R ^ { 2 }$ 简化为 $Y _ { i } ( 0 )$ 与 $X _ { i }$ 之间的有限总体平方多重相关系数。

我将命题6.1的证明留待问题6.4。

当 $0 < a < \infty$ 时，渐近分布具有更复杂的形式，并且更集中于 $\tau$，因此在ReM下均值之差比在CRE下更精确。

如果我们忽略ReM的设计，仍然使用基于Neyman (1923)方差公式和正态近似的置信区间，那么即使个体因果效应是常数，该区间也会过于保守并对 $\tau$ 过度覆盖。Li等人 (2018b) 描述了如何基于定理6.1构建置信区间。我们在此省略讨论，但将在第6.3节回到推断问题。

<!-- 脚注 -->

- 它成为Salsburg (2001)关于现代统计学史的著作的书名。

<!-- 脚注结束 -->

<!-- 脚注 -->

- 在因果推断中，如果 $X _ { i }$ 不受处理影响，则称其为协变量。也就是说，如果协变量有两个潜在结果 $X _ { i } ( 1 )$ 和 $X _ { i } ( 0 )$，那么它们必须满足 $X _ { i } ( 1 ) = X _ { i } ( 0 )$。标准统计学教科书通常不区分处理变量和协变量，因为它们通常出现在结果回归模型的右侧。在这些统计模型中，它们都被称为协变量。本书区分处理变量和协变量，因为它们在因果推断中扮演着不同的角色。

<!-- 脚注结束 -->

<!-- 脚注 -->

- 这里使用除数 $n - 1$ 使公式更简洁。将除数改为 $n$ 会使公式复杂化，但不会从根本上改变结果。当 $n$ 较大时，差异很小。

<!-- 脚注结束 -->

<!-- 脚注 -->

- 在经典的两样本问题中，处理组的结果是来自均值为 $\mu _ { 1 }$、方差为 $\sigma _ { 1 } ^ { 2 }$ 的分布的独立同分布（IID）样本，对照组的结果是来自均值为 $\mu _ { 0 }$、方差为 $\sigma _ { 0 } ^ { 2 }$ 的分布的独立同分布样本。在此假设下，我们有
- $\mathrm { v a r } ( \hat { \tau } ) = \frac { \sigma _ { 1 } ^ { 2 } } { n _ { 1 } } + \frac { \sigma _ { 0 } ^ { 2 } } { n _ { 0 } } .$
- 这里，var(·) 是针对结果随机性的。该方差公式不涉及依赖于个体因果效应方差的第三项。

<!-- 脚注结束 -->

<!-- 脚注 -->

- 他最著名的名言是“所有模型都是错误的，但有些是有用的。”

<!-- 脚注结束 -->

<!-- 脚注 -->

- 符号“$A \dot { \sim } B$”表示 $A$ 和 $B$ 具有相同的渐近分布。

<!-- 脚注结束 -->

<!-- 脚注 -->

- 随机变量 $y$ 与随机向量 $X$ 之间的平方多重相关系数定义为

$$
R _ {y X} ^ {2} = \mathrm{corr} ^ {2} (y, X) = \frac {\mathrm{cov} (y , X) \mathrm{cov} (X) ^ {- 1} \mathrm{cov} (X , y)}{\mathrm{var} (y)}.
$$

它扩展了皮尔逊相关系数（Pearson correlation coefficient）的定义，衡量了 $y$ 对 $X$ 的线性依赖程度。

<!-- 脚注结束 -->

## 6.2 回归调整（Regression adjustment）

如果我们不在设计阶段进行**再随机化（rerandomization）**，而是希望在**完全随机化实验（Completely Randomized Experiment, CRE）**的分析阶段对协变量不平衡进行调整，该怎么办？我们将讨论几种回归调整策略。

## 6.2.1 协变量调整后的费希尔随机化检验（Covariate-adjusted FRT）

协变量 $X$ 都是固定的，此外，在 $H _ { \mathrm { 0 F } }$ 下，观测结果也都是固定的。因此，我们可以模拟任何检验统计量 $T ( Z , Y , X )$ 的分布并计算 p 值。在存在额外协变量的情况下，**费希尔随机化检验（Fisher Randomization Test, FRT）**的基本思想保持不变。

根据 Zhao 和 Ding (2021a) 的总结，构建检验统计量有两种通用策略。问题 3.6 对这两种策略都有所提示。我将其总结如下：

• **第一种策略**是基于拟合统计模型的残差构建检验统计量。我们可以将 $Y _ { i }$ 对 $X _ { i }$ 进行回归，得到残差 $\varepsilon _ { i }$，然后将 $\varepsilon _ { i }$ 作为伪结果来构建检验统计量。

• **第二种策略**是使用回归系数作为检验统计量。我们可以将 $Y _ { i }$ 对 $( Z _ { i } , X _ { i } )$ 进行回归，得到 $Z _ { i }$ 的系数作为检验统计量。本节剩余部分将回顾一些基于**普通最小二乘法（Ordinary Least Squares, OLS）**的检验统计量。

在第一种策略中，我们只需运行一次回归，但在第二种策略中，我们需要多次运行回归。在上述内容中，“回归”是一个通用术语，可以是线性回归、逻辑回归，甚至是机器学习算法。使用这两种策略中的任何检验统计量进行的 FRT，在 $H _ { \mathrm { 0 F } }$ 下都是**有限样本精确（finite-sample exact）**的，尽管它们在备择假设下有所不同。

## 6.2.2 协方差分析（Analysis of covariance）及其扩展

现在我们转向直接估计调整了观测协变量的**平均因果效应（Average Causal Effect）** $\tau$。

历史上，Fisher (1925) 提出使用**协方差分析（Analysis of Covariance, ANCOVA）**来提高估计效率。这在许多领域仍然是标准策略。他建议将 $Y _ { i }$ 对 $( Z _ { i } , X _ { i } )$ 进行 OLS 回归，并将 $Z _ { i }$ 的系数作为 $\tau$ 的估计量。令 $\hat { \tau } _ { \mathrm { F } }$ 表示 Fisher 的 ANCOVA 估计量。

前加州大学伯克利分校统计学教授 David Freedman 在 Neyman (1923) 的**潜在结果框架（potential outcomes framework）**下重新分析了 Fisher 的 ANCOVA。Freedman (2008a,b) 发现了以下负面结果：

1. $\hat { \tau } _ { \mathrm { F } }$ 是有偏的，但简单的均值差 $\hat { \tau }$ 是无偏的。
2. $\hat { \tau } _ { \mathrm { F } }$ 的渐近方差可能比 $\hat { \tau }$ 的更大。
3. 在 CRE 下，来自 OLS 的标准误对于 $\hat { \tau } _ { \mathrm { F } }$ 的真实标准误是不一致的。

加州大学伯克利分校的博士生 Winston Lin 撰写了一篇论文来回应 Freedman 的批评。Lin (2013) 发现了以下正面结果：

1. $\hat { \tau } _ { \mathrm { F } }$ 的偏误在大样本下很小，并且随着样本量趋近于无穷大而趋近于零。
2. 通过使用 $Y _ { i }$ 对 $( Z _ { i } , X _ { i } , Z _ { i } \times X _ { i } )$ 进行 OLS 回归中 $Z _ { i }$ 的系数，我们可以同时提高 $\hat { \tau }$ 和 $\hat { \tau } _ { \mathrm { F } }$ 的渐近效率。令 $\hat { \tau } _ { \mathrm { L } }$ 表示 Lin (2013) 的估计量。此外，在 CRE 下，**EHW（Eicker-Huber-White）标准误**是 $\hat { \tau } _ { \mathrm { L } }$ 真实标准误的一个保守估计量。
3. 在 $Y _ { i }$ 对 $( Z _ { i } , X _ { i } )$ 的 OLS 拟合中，$\hat { \tau } _ { \mathrm { F } }$ 的 EHW 标准误 $^4$ 是 CRE 下 $\hat { \tau } _ { \mathrm { F } }$ 真实标准误的一个保守估计量。

## 6.2.2.1 Lin (2013) 结果的一些启发式说明

Neyman (1923) 的结果表明，均值差估计量的方差取决于潜在结果的方差。直观地说，我们可以通过减小结果变量的方差来减小估计量的方差。一个简单的线性调整估计量族是：

$$
\begin{array}{l} \hat {\tau} \left(\beta_ {1}, \beta_ {0}\right) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \left(Y _ {i} - \beta_ {1} ^ {\mathsf {T}} X _ {i}\right) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} \left(1 - Z _ {i}\right) \left(Y _ {i} - \beta_ {0} ^ {\mathsf {T}} X _ {i}\right) (6. 2) \\ = \left\{\hat {\bar {Y}} (1) - \beta_ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \beta_ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\}, \tag {6.3} \\ \end{array}
$$

其中 $\{ \hat { \bar { Y } } ( 1 ) , \hat { \bar { Y } } ( 0 ) \}$ 是结果变量的样本均值，$\{ \hat { \bar { X } } ( 1 ) , \hat { \bar { X } } ( 0 ) \}$ 是协变量的样本均值。这个协变量调整后的估计量 $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ 试图通过对潜在结果进行残差化来减小 $\hat { \tau }$ 的方差。当 $\beta _ { 1 } = \beta _ { 0 } = 0$ 时，它简化为 $\hat { \tau }$。对于 $\beta _ { 1 }$ 和 $\beta _ { 0 }$ 的任何固定值，由于 $\bar { X } = 0$，它的均值都是 $\tau$。我们感兴趣的是找到使 $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ 的方差最小化的 $( \beta _ { 1 } , \beta _ { 0 } )$。这个估计量本质上是调整后潜在结果 $\{ Y _ { i } ( 1 ) - \beta _ { 1 } ^ { \mathsf { T } } X _ { i } , Y _ { i } ( 0 ) - \beta _ { 0 } ^ { \mathsf { T } } X _ { i } \} _ { i = 1 } ^ { n }$ 的均值差。应用 Neyman (1923) 的结果，该估计量的方差为：

$$
\operatorname{var} \{\hat {\tau} (\beta_ {1}, \beta_ {0}) \} = \frac {S ^ {2} (1 ; \beta_ {1})}{n _ {1}} + \frac {S ^ {2} (0 ; \beta_ {1})}{n _ {0}} - \frac {S ^ {2} (\tau ; \beta_ {1} , \beta_ {0})}{n},
$$

其中 $S ^ { 2 } ( z ; \beta _ { 1 } ) ~ ( z = 1 , 0 )$ 和 $S ^ { 2 } ( \tau ; \beta _ { 1 } , \beta _ { 0 } )$ 分别是调整后潜在结果和个体效应的有限总体方差；此外，一个保守的方差估计量为：

$$
\hat {V} (\beta_ {1}, \beta_ {0}) = \frac {\hat {S} ^ {2} (1 ; \beta_ {1})}{n _ {1}} + \frac {\hat {S} ^ {2} (0 ; \beta_ {1})}{n _ {0}},
$$

其中：

$$
\hat {S} ^ {2} (1; \beta_ {1}) = (n _ {1} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \gamma_ {1} - \beta_ {1} ^ {\mathsf {T}} X _ {i} \} ^ {2},
$$

$$
\hat {S} ^ {2} (0; \beta_ {0}) = (n _ {0} - 1) ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \gamma_ {0} - \beta_ {0} ^ {\mathsf {T}} X _ {i} \} ^ {2}
$$

是调整后潜在结果的样本方差，其中 $\gamma _ { 1 }$ 和 $\gamma _ { 0 }$ 分别是处理组中 $Y _ { i } - \beta _ { 1 } ^ { \mathsf { T } } X _ { i }$ 和对照组中 $Y _ { i } - \beta _ { 0 } ^ { \mathsf { T } } X _ { i }$ 的样本均值。为了最小化 $\hat { V } ( \beta _ { 1 } , \beta _ { 0 } )$，我们需要求解两个 OLS 问题：

$$
\min _ {\gamma_ {1}, \beta_ {1}} \sum_ {i = 1} ^ {n} Z _ {i} \{Y _ {i} - \gamma_ {1} - \beta_ {1} ^ {\mathsf {T}} X _ {i} \} ^ {2}, \quad \min _ {\gamma_ {0}, \beta_ {0}} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{Y _ {i} - \gamma_ {0} - \beta_ {0} ^ {\mathsf {T}} X _ {i} \} ^ {2}.
$$

我们分别对处理组和对照组运行 $Y _ { i }$ 对 $X _ { i }$ 的 OLS 回归，得到 $( \hat { \gamma } _ { 1 } , \hat { \beta } _ { 1 } )$ 和 $( \hat { \gamma } _ { 0 } , \hat { \beta } _ { 0 } )$。最终的估计量是：

$$
\begin{array}{l} \hat {\tau} (\hat {\beta} _ {1}, \hat {\beta} _ {0}) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\beta} _ {1} ^ {\mathsf {T}} X _ {i}) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - \hat {\beta} _ {0} ^ {\mathsf {T}} X _ {i}) \\ = \left\{\hat {\bar {Y}} (1) - \hat {\beta} _ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1) \right\} - \left\{\hat {\bar {Y}} (0) - \hat {\beta} _ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0) \right\}. \\ \end{array}
$$

根据 OLS 拟合的性质（见 (A2.3)），我们知道：

$$
\hat {\bar {Y}} (1) = \hat {\gamma} _ {1} + \hat {\beta} _ {1} ^ {\mathsf {T}} \hat {\bar {X}} (1), \quad \hat {\bar {Y}} (0) = \hat {\gamma} _ {0} + \hat {\beta} _ {0} ^ {\mathsf {T}} \hat {\bar {X}} (0).
$$

因此，我们可以将估计量重写为：

$$
\hat {\tau} \left(\hat {\beta} _ {1}, \hat {\beta} _ {0}\right) = \hat {\gamma} _ {1} - \hat {\gamma} _ {0} \tag {6.4}
$$

(6.4) 中的等价形式表明，我们可以通过下面单一的 OLS 拟合得到 $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$。

**命题 6.2** (6.4) 中的估计量 $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ 等于 $Y _ { i }$ 对 $( Z _ { i } , X _ { i } , Z _ { i } \times X _ { i } )$ 进行 OLS 拟合中 $Z _ { i }$ 的系数，即之前介绍的 $\hat { \tau } _ { \mathrm { L } }$。

我将命题 6.2 的证明留作问题 6.5，这纯粹是一个代数事实。

基于以上讨论，$\hat { \tau } _ { \mathrm { L } }$ 的一个保守方差估计量是：

$$
\begin{array}{l} \hat {V} (\hat {\beta} _ {1}, \hat {\beta} _ {0}) = \frac {1}{n _ {1} (n _ {1} - 1)} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - \hat {\gamma} _ {1} - \hat {\beta} _ {1} ^ {\mathsf {T}} X _ {i}) ^ {2} \\ + \frac {1}{n _ {0} (n _ {0} - 1)} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - \hat {\gamma} _ {0} - \hat {\beta} _ {0} ^ {\mathsf {T}} X _ {i}) ^ {2}. \\ \end{array}
$$

基于相当技术性的计算，Lin (2013) 进一步表明，来自命题 6.2 中 OLS 的 EHW 标准误几乎等同于 $\hat { V } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$，而后者是 CRE 下 $\scriptstyle { \hat { \tau } } _ { \mathrm { L } }$ 真实标准误的一个保守估计量。直观地说，这是因为我们没有假设线性模型被正确设定，而 EHW 标准误对模型误设是稳健的。

上述讨论存在一个微妙的问题。方差公式 $\operatorname{var} \{ \hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } ) \}$ 适用于固定的 $( \beta _ { 1 } , \beta _ { 0 } )$，但估计量 $\hat { \tau } ( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$ 使用了两个估计的系数 $( \hat { \beta } _ { 1 } , \hat { \beta } _ { 0 } )$。估计系数中的额外不确定性可能导致最终估计量存在有限样本偏误。Lin (2013) 表明，这个问题在渐近意义上会消失。然而，他的理论需要较大的样本量以及关于潜在结果和协变量的一些正则条件。

**表 6.1：预测潜在结果**

<table><tr><td>X</td><td>Z</td><td>Y(1)</td><td>Y(0)</td><td> $\hat{Y}(1)$ </td><td> $\hat{Y}(0)$ </td></tr><tr><td> $X_1$ </td><td>1</td><td> $Y_1(1)$ </td><td>?</td><td> $\hat{\mu}_1(X_1)$ </td><td> $\hat{\mu}_0(X_1)$ </td></tr><tr><td> $\vdots$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $X_{n_1}$ </td><td>1</td><td> $Y_{n_1}(1)$ </td><td>?</td><td> $\hat{\mu}_1(X_{n_1})$ </td><td> $\hat{\mu}_0(X_{n_1})$ </td></tr><tr><td> $X_{n_1+1}$ </td><td>0</td><td>?</td><td> $Y_{n_1+1}(0)$ </td><td> $\hat{\mu}_1(X_{n_1+1})$ </td><td> $\hat{\mu}_0(X_{n_1+1})$ </td></tr><tr><td> $\vdots$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $X_n$ </td><td>0</td><td>?</td><td> $Y_n(0)$ </td><td> $\hat{\mu}_1(X_n)$ </td><td> $\hat{\mu}_0(X_n)$ </td></tr></table>

## 6.2.2.2 通过预测潜在结果理解 Lin (2013) 的估计量

我们可以将 Lin (2013) 的估计量视为基于潜在结果的最小二乘法（OLS）拟合的**预测估计量（predictive estimator）**。我们使用处理组的数据，基于 $X$ 为 $Y(1)$ 构建一个预测模型：

$$
\hat {\mu} _ {1} (x) = \hat {\gamma} _ {1} + \hat {\beta} _ {1} ^ {\mathsf {T}} x. \tag {6.5}
$$

类似地，我们使用对照组的数据，基于 $X$ 为 $Y(0)$ 构建一个预测模型：

$$
\hat {\mu} _ {0} (x) = \hat {\gamma} _ {0} + \hat {\beta} _ {0} ^ {\mathsf {T}} x. \tag {6.6}
$$

如果我们预测缺失的潜在结果，那么我们得到以下预测估计量：

$$
\hat {\tau} _ {\text { pred }} = n ^ {- 1} \left\{\sum_ {Z _ {i} = 1} Y _ {i} + \sum_ {Z _ {i} = 0} \hat {\mu} _ {1} (X _ {i}) - \sum_ {Z _ {i} = 1} \hat {\mu} _ {0} (X _ {i}) - \sum_ {Z _ {i} = 0} Y _ {i} \right\}. \tag {6.7}
$$

我们可以验证，使用 (6.5) 和 (6.6)，该预测估计量等于 Lin (2013) 的估计量：

$$
\hat {\tau} _ {\mathrm{pred}} = \hat {\tau} _ {\mathrm{L}}. (6. 8)
$$

如果我们预测所有潜在结果（即使它们已被观测到），我们得到以下**投影估计量（projective estimator）**：

$$
\hat {\tau} _ {\text { proj }} = n ^ {- 1} \sum_ {i = 1} ^ {n} \{\hat {\mu} _ {1} (X _ {i}) - \hat {\mu} _ {0} (X _ {i}) \}. \tag {6.9}
$$

我们可以验证，使用 (6.5) 和 (6.6)，该投影估计量等于 Lin (2013) 的估计量：

$$
\hat {\tau} _ {\mathrm{proj}} = \hat {\tau} _ {\mathrm{L}}. \tag {6.10}
$$

我将 (6.8) 和 (6.10) 的证明留作问题 6.6。

更一般的公式 (6.7) 和 (6.9) 对于其他潜在结果预测器也是良好定义的。为了与 Lin (2013) 的估计量建立联系，我在此关注线性预测器。它们可以是非常通用的，包括更复杂的机器学习算法。然而，构建点估计量只是分析**完全随机实验（Completely Randomized Experiment, CRE）**的第一步。更重要的第二步是量化与估计量相关的不确定性，这取决于潜在结果预测器的性质。尽管如此，无需进行额外的理论分析，我们总是可以在**费希尔随机化检验（Fisher Randomization Test, FRT）**中，将 (6.7) 和 (6.9) 用作检验统计量。

## 6.2.2.3 通过调整协变量不平衡理解 Lin (2013) 的估计量

线性调整估计量有一个等价形式

$$
\hat {\tau} (\beta_ {1}, \beta_ {0}) = \hat {\tau} - \gamma^ {\mathsf {T}} \hat {\tau} _ {X} \tag {6.11}
$$

其中 $\begin{array} { r } { \gamma = \frac { n _ { 0 } } { n } \beta _ { 1 } + \frac { n _ { 1 } } { n } \beta _ { 0 } } \end{array}$ ，所以我们也可以将其写为 $\hat { \tau } ( \gamma ) = \hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ 。类似地，Lin (2013) 的估计量有一个等价形式

$$
\hat {\tau} _ {\mathrm{L}} = \hat {\tau} - \hat {\gamma} ^ {\mathsf {T}} \hat {\tau} _ {X}, \tag {6.12}
$$

其中 $\begin{array} { r } { \hat { \gamma } = \frac { n _ { 0 } } { n } \hat { \beta } _ { 1 } + \frac { n _ { 1 } } { n } \hat { \beta } _ { 0 } } \end{array}$ 。我将 (6.11) 和 (6.12) 的证明留作问题 6.7。形式 (6.11) 和 (6.12) 是“调整协变量不平衡”的数学表述。它们本质上是减去了协变量均值差异的某些线性组合。由于 $\hat {\tau}$ 和 $\hat { \tau } _ { X }$ 是相关的，使用适当的 $\gamma$ 进行协变量调整会减小 $\hat {\tau}$ 的方差。(6.11) 和 (6.12) 的另一个有趣特征是，最终的估计量仅依赖于 $\gamma$ 或 $\hat { \gamma }$ ，因此 $\beta$ 系数的选择并非唯一。因此，Lin (2013) 的估计量只是最优估计量之一，但它可以通过带有**EHW标准误（EHW standard error）**的标准 OLS 轻松实现。

## 6.2.3 关于回归调整的一些补充说明

## 6.2.3.1 再随机化（ReM）与回归调整之间的对偶性

Li et al. (2018b) 指出，**再随机化（Rerandomization, ReM）**和 Lin (2013) 的回归调整在实验的设计阶段和分析阶段使用协变量方面是对偶的。更具体地说，当 $a$ 很小时，在 ReM 下 $\hat {\tau}$ 的渐近分布与在 CRE 下 $\hat { \tau } _ { \mathrm { L } }$ 的渐近分布几乎相同。因此，ReM 在设计阶段使用协变量，而 Lin (2013) 的回归调整在分析阶段使用协变量，当 $a$ 很小时，两者实现了几乎相同的渐近效率增益。

## 6.2.3.2 回归调整与事后分层（Post-stratification）的等价性

如果我们有离散协变量 $C _ { i }$ ，它有 $K$ 个类别，我们可以创建 $K - 1$ 个中心化的虚拟变量

$$
X _ {i} = (I (C _ {i} = 1) - \pi_ {[ 1 ]}, \ldots , I (C _ {i} = K - 1) - \pi_ {[ K - 1 ]}).
$$

在这种情况下，Lin (2013) 的回归调整等价于**事后分层（post-stratification）**，如下述命题所述。

**命题 6.3** 基于 $X _ { i }$ 的 $\hat { \tau } _ { \mathrm { L } }$ 在数值上与基于 $C _ { i }$ 的事后分层估计量相同。

我将命题 6.3 的证明留作问题 6.9。

## 6.2.3.3 双重差分（Difference-in-Difference）作为协变量调整 $\hat { \tau } ( \beta _ { 1 } , \beta _ { 0 } )$ 的一个特例

许多研究中一个重要的协变量 $X$ 是处理前的滞后结果。例如，在教育研究中，如果结果 $Y$ 是后测分数，则协变量 $X$ 是前测分数；如果结果 $Y$ 是职业培训项目后的对数工资，则协变量 $X$ 是职业培训项目前的对数工资。以滞后结果 $X$ 作为协变量，一个流行的估计量是**增益分数（gain score）**或**双重差分（difference-in-difference, DID）**估计量，其中 (6.2) 和 (6.3) 中的 $\beta _ { 1 } = \beta _ { 0 } = 1$：

$$
\begin{array}{l} \hat {\tau} (1, 1) = n _ {1} ^ {- 1} \sum_ {i = 1} ^ {n} Z _ {i} (Y _ {i} - X _ {i}) - n _ {0} ^ {- 1} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) (Y _ {i} - X _ {i}) \\ { = } { \left\{ \hat { \bar { Y } } ( 1 ) - \hat { \bar { Y } } ( 0 ) \right\} - \left\{ \hat { \bar { X } } ( 1 ) - \hat { \bar { X } } ( 0 ) \right\} . } \\ \end{array}
$$

$\hat { \tau } ( 1 , 1 )$ 的第一种形式证明了“增益分数”这一名称的合理性，因为它本质上是增益分数 $g _ { i } = Y _ { i } - X _ { i }$ 的均值之差。$\hat { \tau } ( 1 , 1 )$ 的第二种形式证明了“双重差分”这一名称的合理性，因为它是两个均值之差之间的差。该估计量不同于 Lin (2013) 的估计量：它预先固定了 $\beta _ { 1 } = \beta _ { 0 } = 1$ ，而 Lin (2013) 的估计量涉及两个估计的 $\beta$ 系数。它是无偏的，并具有一个保守的方差估计量

$$
\begin{array}{l} \hat {V} (1, 1) = \frac {1}{n _ {1} (n _ {1} - 1)} \sum_ {i = 1} ^ {n} Z _ {i} \{g _ {i} - \hat {\bar {g}} (1) \} ^ {2} \\ + \frac {1}{n _ {0} (n _ {0} - 1)} \sum_ {i = 1} ^ {n} (1 - Z _ {i}) \{g _ {i} - \hat {\bar {g}} (0) \} ^ {2}, \\ \end{array}
$$

其中 $\hat { \bar { g } } ( 1 )$ 和 $\hat { \bar { g } } ( 0 )$ 分别是处理组和对照组中增益分数 $g _ { i } = Y _ { i } - X _ { i }$ 的样本均值。当滞后结果是结果的强预测因子时，增益分数 $g _ { i } = Y _ { i } - X _ { i }$ 的方差通常远小于结果本身的方差。在这种情况下，$\hat { \tau } ( 1 , 1 )$ 通常能大幅减小结果简单均值差的方差。

**表 6.2：实验的设计与分析**

<table><tr><td></td><td colspan="4">分析</td></tr><tr><td rowspan="3">设计</td><td>CRE</td><td> $\hat{\tau}$  (Neyman, 1923)</td><td> $\stackrel{1}{\longrightarrow}$ </td><td> $\hat{\tau}_{\text{L}}$  (Lin, 2013)</td></tr><tr><td></td><td> $2 \Big\downarrow$ </td><td></td><td> $\Big\downarrow 4$ </td></tr><tr><td>ReM</td><td> $\hat{\tau}$  (Li et al., 2018b)</td><td> $\stackrel{3}{\longrightarrow}$ </td><td> $\hat{\tau}_{\text{L}}$  (Li and Ding, 2020)</td></tr></table>

## 6.2.4 推广到分层随机实验（SRE）

有可能我们在一个离散变量 $C$ 上进行了分层实验，并观测到了额外的协变量 $X$ 。如果所有层都很大，那么我们可以获得层内的 Lin (2013) 估计量 $\hat { \tau } _ { \mathrm { L } , [ k ] }$ ，并将最终估计量计算为

$$
\hat {\tau} _ {\mathrm{L,S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} \hat {\tau} _ {\mathrm{L}, [ k ]}.
$$

一个保守的方差估计量是

$$
\hat {V} _ {\mathrm{L,S}} = \sum_ {k = 1} ^ {K} \pi_ {[ k ]} ^ {2} \hat {V} _ {\mathrm{EHW}, [ k ]},
$$

其中 $\hat { V } _ { \mathrm { E H W } , [ k ] }$ 是在层 $k$ 内，将结果对处理指示变量、协变量及其交互项进行 OLS 拟合得到的 EHW 方差估计量。重要的是，我们需要按层特定的均值对协变量进行中心化。

## 6.3 统一、结合与比较

Li and Ding (2020) 统一了相关文献，并表明我们可以结合再随机化和回归调整。也就是说，如果我们在设计阶段进行再随机化，我们可以在分析阶段使用带有 EHW 标准误的 Lin (2013) 估计量。再随机化与回归调整的结合改善了设计阶段的协变量平衡和分析阶段的估计效率。

表 6.2 总结了从 Neyman (1923) 到 Li and Ding (2020) 的文献。箭头 1 说明了 CRE 中协变量调整的效率增益：渐近地，$\hat { \tau } _ { \mathrm { L } }$ 的方差小于 $\hat {\tau}$ 的方差。箭头 2 说明了 ReM 的效率增益：渐近地，在 ReM 下 $\hat {\tau}$ 的分位数范围比在 CRE 下更窄。箭头 3 和 4 说明了结合的好处。

## 6.4 模拟

Angrist et al. (2009) 进行了一项实验，以评估提高大学新生学业成绩的不同策略。这里我使用了原始数据的一个子集，重点关注对照组和接受学业支持服务及成绩优异经济激励的处理组。结果是第一年末的**平均绩点（Grade Point Average, GPA）**，两个协变量是性别和基线 GPA。下表总结了基于未调整和调整估计量的结果。调整后的估计量具有较小的标准误，尽管它给出了与未调整估计量一样不显著的结果。

<table><tr><td></td><td>estimate</td><td>s.e.</td><td>t-stat</td><td>p-value</td></tr><tr><td>Neyman</td><td>0.054</td><td>0.076</td><td>0.719</td><td>0.472</td></tr><tr><td>Lin</td><td>0.075</td><td>0.072</td><td>1.036</td><td>0.300</td></tr></table>

我还使用该数据集进行模拟研究，以评估表 6.2 中总结的四种设计和分析策略。我拟合了结果对协变量的二次函数，并使用它们分别对处理组和对照组填补所有缺失的潜在结果。为了展示 ReM 和回归调整的改进，我还将误差项按 0.1 和 0.25 的比例缩放，以增加信噪比。使用填补后的科学表，我生成了 2000 个处理分配，获取观测数据，并计算估计量。在模拟中，“真实”结果模型是非线性的，但我们仍然使用线性调整进行估计。通过这样做，我们可以评估当线性模型被错误设定时估计量的性质。

图 6.2 显示了四种组合的小提琴图，从估计值中减去了真实的 $\tau$ 。正如理论所预测的，所有估计量几乎都是无偏的，并且 ReM 和回归调整都提高了效率。当噪声水平较低时，它们更有效。

## 6.5 总结评论

对于连续结果，Fisher 的**协方差分析（Analysis of Covariance, ANCOVA）**多年来一直是标准方法。即使线性模型被错误设定，Lin (2013) 的改进也具有更好的理论性质。对于二元结果，通常使用将观测结果对处理指示变量和协变量进行逻辑回归（logistic regression）中处理的系数来估计因果效应。然而，Freedman (2008c) 表明，这种逻辑回归在潜在结果框架下不具有良好的性质。即使逻辑模型是正确的，该系数估计的是条件优势比（conditional odds ratio），这可能不是感兴趣的参数；当逻辑模型不正确时，解释该系数就更加困难。从上面的讨论来看，如果感兴趣的参数是平均因果效应，我们仍然可以使用 Lin (2013) 的估计量来分析 CRE 中的二元结果数据。Guo and Basse (2023) 扩展了 Lin (2013) 的理论，允许使用广义线性模型（Generalized Linear Models, GLMs）在潜在结果框架下为平均因果效应构建估计量。

Lin (2013) 理论的其他扩展侧重于高维协变量。Bloniarz et al. (2016) 关注协变量数量多于样本量的情形，并且在稀疏性假设下，他们建议用结果对处理、协变量及其交互项的**最小绝对收缩和选择算子（Least Absolute Shrinkage and Selection Operator, LASSO）**拟合 (Tibshirani, 1996) 来替代 OLS 拟合。Lei and Ding (2021) 关注协变量数量发散但不假设稀疏性的情形，并且在某些正则条件下，他们证明了 Lin (2013) 的估计量仍然是一致的且渐近正态的。Wager et al. (2016) 提出使用机器学习方法来分析高维实验数据。

## 6.6 课后作业

## 6.1 ReM 下的 FRT

描述 ReM 下的 FRT。

## 6.6 课后作业

## 6.2 马氏距离（Mahalanobis Distance）的不变性

证明引理 6.1。

## 6.3 再随机化下均值差估计量的偏误

假设我们从 CRE 中抽取 $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { n } )$ ，并且当且仅当 $\phi ( Z , X ) = 1$ 时接受它，其中 $\phi$ 是一个预定的平衡准则。证明如果 $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ 且

$$
\phi (\mathbf {Z}, \mathbf {X}) = \phi (\mathbf {1} _ {n} - \mathbf {Z}, \mathbf {X}), \tag {6.13}
$$

那么 $\hat { \tau }$ 对于 $\tau$ 是无偏的。验证如果 $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ ，使用马氏距离的再随机化满足 (6.13)。给出当这两个条件不成立时 $\hat {\tau}$ 对 $\tau$ 有偏的反例。

## 6.4 CRE 中 $R ^ { 2 }$ 的等价形式

证明命题 6.1。

## 6.5 用于协变量调整的 Lin 估计量

证明命题 6.2。

## 6.6 预测估计量与投影估计量

证明 (6.8) 和 (6.10)。

## 6.7 协变量调整估计量的等价形式

证明 (6.11) 和 (6.12)。

## 6.8 ANCOVA 也调整协变量不平衡

这个问题给出了一个类似于 (6.12) 的 ANCOVA 结果。

证明

$$
\hat {\tau} _ {\mathrm{F}} = \hat {\tau} - \hat {\gamma} _ {\mathrm{F}} ^ {\mathsf {T}} \hat {\tau} _ {X},
$$

其中 $\hat { \gamma } _ { \mathrm { F } }$ 是将 $Y _ { i }$ 对 $( 1 , Z _ { i } , X _ { i } )$ 进行 OLS 拟合时 $X _ { i }$ 的系数。

## 6.9 CRE 的回归调整与事后分层

证明命题 6.3。

提示：有时 $\hat { \tau } _ { \mathrm { { P S } } }$ 或 $\hat { \tau } _ { \mathrm { L } }$ 可能不是良好定义的。在这些情况下，我们视 $\hat { \tau } _ { \mathrm { { P S } } }$ 和 $\hat { \tau } _ { \mathrm { L } }$ 相等。你可以在证明中忽略这种复杂性。

## 6.10 更多关于 CRE 中双重差分估计量的内容

这个问题提供了第 6.2.3.3 节中 CRE 内双重差分估计量的更多细节。

证明 $\hat { \tau } ( 1 , 1 )$ 对于 $\tau$ 是无偏的，计算其方差，并证明 $\hat { V } ( 1 , 1 )$ 是 $\hat { \tau } ( 1 , 1 )$ 真实方差的保守估计量。何时 $E \{ \hat { V } ( 1 , 1 ) \} = \operatorname { v a r } \{ \hat { \tau } ( 1 , 1 ) \}$ 成立？

比较 ${ \hat { \tau } } ( 0 , 0 )$ 和 $\hat { \tau } ( 1 , 1 )$ 的方差，以证明

$$
\operatorname{var} \{\hat {\tau} (0, 0) \} \geq \operatorname{var} \{\hat {\tau} (1, 1) \}
$$

当且仅当

$$
2 \frac {n _ {0}}{n} \beta_ {1} + 2 \frac {n _ {1}}{n} \beta_ {0} \geq 1,
$$

其中

$$
\beta_ {1} = \frac {\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) \{Y _ {i} (1) - \bar {Y} (1) \}}{\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) ^ {2}}, \quad \beta_ {0} = \frac {\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) \{Y _ {i} (0) - \bar {Y} (0) \}}{\sum_ {i = 1} ^ {n} (X _ {i} - \bar {X}) ^ {2}}
$$

分别是将 $Y _ { i } ( 1 )$ 和 $Y _ { i } ( 0 )$ 对 $( 1 , X _ { i } )$ 进行 OLS 拟合时 $X _ { i }$ 的系数。

备注：Gerber and Green (2012, 第 28 页) 讨论了当 $n _ { \mathrm { 1 } } = n _ { \mathrm { 0 } }$ 时该问题的一个特例。

## 6.11 数据再分析

重新分析 SRE Neyman penn.R 中使用的数据。第 5 章的分析使用了处理指示变量、结果和区组指示变量。现在我们想要使用所有其他协变量。

在实验的层内进行回归调整，然后将这些调整后的估计量组合起来，以估计平均因果效应。报告点估计量、估计标准误和 95% 置信区间。将它们与未进行回归调整的结果进行比较。

## 6.12 推荐阅读

本章的标题与 Li and Ding (2020) 的标题相同，该研究分别探讨了再随机化和回归调整在随机实验设计阶段和分析阶段中的作用。